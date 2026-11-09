"""Synthetic counterexamples for the original-benchmark answer audit.

These fixtures deliberately contain no benchmark coordinates or answer keys.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pdbthink.chem import parent_of
from pdbthink.config import DatasetConfig
from pdbthink.generators import GenerationContext, get_generator
from pdbthink.generators.base import Analysis, Rejection
from pdbthink.mechanistic.episodes import EpisodeSpec, FieldSpec, StateSpec
from pdbthink.mechanistic.pipeline import (
    EpisodeRejected,
    EpisodeResult,
    _changed_residue,
    _changed_residues,
    _gained_interactions,
    _gained_pair,
    _initial_residues,
    _new_contact,
    _packing_change,
    _render_episode,
    _within_protomer,
)
from pdbthink.preprocessing.model import EntityType
from pdbthink.preprocessing.transform import display, identity_transform
from pdbthink.prompts.library import MECHANISTIC_QUESTIONS
from pdbthink.prompts.render import StructureBlock, build_prompt
from pdbthink.schemas import SemanticInstance
from pdbthink.scoring import score_response
from tests.conftest import make_residue, make_structure, place_atom


def test_modified_residue_letters_are_visible_and_scoreable(definitions):
    first = make_structure([make_residue("A", 1, "ALA", {"CA": (0, 0, 0)})])
    second = make_structure([
        *first.residues,
        make_residue("B", 2, "DLE", {"CA": (3, 0, 0)}),
        make_residue("B", 3, "SNN", {"CA": (0, 3, 0)}),
        make_residue("B", 4, "UNK", {"CA": (0, 0, 10)}),
    ])
    gold = _new_contact(None, None, first, second, {}, definitions)["gold"]
    for answer in ("A:A1--B:L2", "B:N3--A:A1"):
        assert score_response(f"FINAL: {answer}", "residue_pair", gold)["score"]["correct"]
    prompt = build_prompt(
        representation="minimal_pdb", blocks=[StructureBlock("Structure", second)],
        context="Synthetic interface.", question="Give one contact.", answer_schema="residue_pair",
    )
    for mapping in ("DLE -> L", "SNN -> N", "UNK -> X"):
        assert mapping in prompt.user_prompt
    # SNN's cyclic chemistry must not inherit the ASN atom-name topology.
    assert parent_of("SNN") == "SNN"


def test_all_interface_contacts_are_accepted_in_multi_field_scoring(definitions):
    first = make_structure([make_residue("A", 1, "ALA", {"CA": (0, 0, 0)})])
    second = make_structure([
        *first.residues,
        make_residue("B", 2, "GLY", {"CA": (3, 0, 0)}),
        make_residue("B", 3, "SER", {"CA": (0, 4, 0)}),
        make_residue("B", 4, "VAL", {"CA": (0, 0, 6)}),
    ])
    gold = {"fields": {"contact": _new_contact(None, None, first, second, {}, definitions)["gold"]}}
    for answer, expected in (("B:G2--A:A1", True), ("A:A1--B:S3", True), ("A:A1--B:V4", False)):
        result = score_response(
            f"FINAL\ncontact: {answer}", "multi_field", gold,
            parameters={"field_schemas": {"contact": "residue_pair"}},
        )
        assert result["score"]["correct"] is expected


def test_gained_contact_uses_the_stated_cutoff_without_hidden_hysteresis(definitions):
    def state(distance):
        return make_structure([
            make_residue("A", 1, "TRP", {"CG": (0, 0, 0)}),
            make_residue("A", 8, "ILE", {"CB": (distance, 0, 0)}),
            make_residue("A", 9, "ALA", {"CB": (0, 3, 0)}),
        ])
    outcome = _gained_interactions(
        None, None, state(4.2), state(3.3), {"changed_residue": ["A:W1"]}, definitions,
    )
    assert outcome["gold"]["value"] == ["A:I8--A:W1"]


def test_interface_set_includes_contacts_outside_curated_examples(definitions):
    structure = make_structure([
        make_residue("A", 1, "ALA", {"CA": (0, 0, 0)}),
        make_residue("B", 2, "GLY", {"CA": (3, 0, 0)}),
        make_residue("B", 3, "SER", {"CA": (0, 4, 0)}),
        make_residue("B", 4, "VAL", {"CA": (0, 0, 6)}),
    ])
    spec = FieldSpec("initial_residues", "", "residue_set")
    outcome = _initial_residues(
        None, spec, structure, structure, {"initial_residues": ["B:G2"]}, definitions,
    )
    assert outcome["gold"]["value"] == ["B:G2", "B:S3"]


def charged_pair_state(distance, *, mixed_chains=False):
    return make_structure([
        make_residue("A", 1, "ARG", {"NH1": (0, 0, 0)}),
        make_residue("A", 8, "GLU", {"OE1": (distance, 0, 0)}),
        make_residue("B", 1, "ARG", {"NH1": (0, 20, 0)}),
        make_residue("C" if mixed_chains else "B", 8, "GLU", {"OE1": (distance, 20, 0)}),
        make_residue("D", 1, "ARG", {"NH1": (0, 40, 0)}),
        make_residue("D", 8, "GLU", {"OE1": (3, 40, 0)}),
    ])


def test_equivalent_protomers_are_valid_and_retained_contacts_are_not(definitions):
    first, second = charged_pair_state(6), charged_pair_state(3)
    gold = _gained_pair(None, None, first, second, {}, definitions)["gold"]
    for pair, correct in (("A:E8--A:R1", True), ("B:R1--B:E8", True), ("D:R1--D:E8", False)):
        assert score_response(f"FINAL: {pair}", "residue_pair", gold)["score"]["correct"] is correct
    assert _within_protomer(None, None, first, second, {}, definitions)["gold"]["value"] is True


def test_mixed_intra_and_interprotomer_answers_are_rejected(definitions):
    with pytest.raises(EpisodeRejected, match="protomer_relationship_ambiguous"):
        _within_protomer(
            None, None, charged_pair_state(6, mixed_chains=True),
            charged_pair_state(3, mixed_chains=True), {}, definitions,
        )


def rotamer_state(angles):
    residues = []
    for seq, angle in enumerate(angles, 1):
        n, ca, cb = np.array([0., 1., 0.]), np.zeros(3), np.array([1., 0., 0.])
        cg = place_atom(n, ca, cb, 1.5, 110, angle)
        residues.append(make_residue("A", seq, "PHE", {"N": n, "CA": ca, "CB": cb, "CG": cg}))
    return make_structure(residues)


def test_candidate_rotamers_are_measured_not_copied_from_claims(definitions):
    spec = FieldSpec("changed_residues", "", "residue_set")
    resolved = {"changed_residues": ["A:F1", "A:F2"]}
    outcome = _changed_residues(
        None, spec, rotamer_state([170, 0]), rotamer_state([-170, 100]), resolved, definitions,
    )
    assert outcome["gold"]["value"] == ["A:F2"]
    with pytest.raises(EpisodeRejected, match="chi1_change_near_cutoff"):
        _changed_residues(None, spec, rotamer_state([0, 0]), rotamer_state([61, 100]), resolved, definitions)


def test_packing_measures_sidechains_not_backbone(definitions):
    def state(sidechain_distance):
        return make_structure([
            make_residue("A", 1, "PHE", {"CA": (0, 0, 0), "CG": (0, 10, 0)}),
            make_residue("A", 2, "TRP", {"CA": (1, 0, 0), "CG": (sidechain_distance, 10, 0)}),
        ])
    outcome = _packing_change(
        None, None, state(3), state(5), {"changed_residues": ["A:F1", "A:W2"]}, definitions,
    )
    assert outcome["gold"]["value"] == "farther"


def test_render_recomputes_rounded_gold_and_reverses_question_state_names(definitions):
    first = make_structure([make_residue("A", 1, "ALA", {"CA": (0, 0, 0)})])
    second = make_structure([
        *first.residues,
        make_residue("B", 2, "GLY", {"CA": (3, 0, 0)}),
        make_residue("B", 3, "SER", {"CA": (4.0001, 0, 0)}),
    ])
    episode = EpisodeSpec(
        "synthetic", "synthetic", StateSpec("fixture1", ["A"]), StateSpec("fixture2", ["A", "B"]),
        "Structure 1 is unbound; Structure 2 is bound.",
        [FieldSpec("new_contact", "In Structure 2: " + MECHANISTIC_QUESTIONS["new_contact"], "residue_pair")],
        {},
    )
    instance = SemanticInstance(
        semantic_instance_id="synthetic", question_family="MECH", question_version="test",
        protein_group_id="synthetic", source_type="pdb", source_entries=[], source_file_sha256s=[],
        answer_schema="multi_field", gold_answer={"fields": {}}, definition_version=definitions.version,
    )
    processed = EpisodeResult(
        episode, {"new_contact": {"schema": "residue_pair", "value": "A:A1--B:G2"}},
        {"resolved_claims": {}}, {}, (first, second), (None, None), None,
    )
    config = DatasetConfig.load()
    # Seed 3 rotates the 4.0001-A separation to a displayed distance below 4 A.
    render, _ = _render_episode(
        episode, instance, processed, "minimal_pdb", 3, False, True, definitions, config,
    )
    assert "Structure 2 is unbound; Structure 1 is bound." in render.user_prompt
    question = render.user_prompt.split("Question:", 1)[1]
    assert "Structure 1" in question and "Structure 2" not in question
    atoms = [line for line in render.user_prompt.splitlines() if line.startswith("ATOM")]
    xyz = [np.array([float(line[i:i + 8]) for i in (30, 38, 46)]) for line in atoms]
    assert np.linalg.norm(xyz[2] - xyz[0]) <= 4
    assert render.gold_answer["fields"]["new_contact"]["alternatives"] == ["A:A1--B:S3"]


def test_ligand_rotamer_ranking_excludes_larger_changes_outside_the_site(definitions):
    first, second = rotamer_state([0, 0]), rotamer_state([100, 170])
    for structure in (first, second):
        for atom in structure.residues[1].atoms:
            atom.pos += np.array([20., 0., 0.])
        structure.invalidate()
    second.residues.append(make_residue(
        "L", 1, "L2", {"C1": (0, 0, 2)}, entity=EntityType.LIGAND, polymer_kind=None,
    ))
    second.invalidate()
    spec = FieldSpec("changed_residue", "", "residue", {"ligand": "L2"})
    outcome = _changed_residue(
        None, spec, first, second, {"changed_residue": ["A:F1"]}, definitions,
    )
    assert outcome["gold"]["value"] == "A:F1"
    with pytest.raises(EpisodeRejected, match="rotamer_maximum_not_unique"):
        _changed_residue(None, spec, first, second, {"changed_residue": ["A:F2"]}, definitions)


def test_protein_interface_scope_excludes_ligands_sharing_chain_identifiers(definitions):
    structure = make_structure([
        make_residue("A", 1, "DLE", {"CA": (0, 0, 0)}),
        make_residue("A", 2, "ALA", {"CA": (20, 0, 0)}),
        make_residue("A", 3, "L1", {"C1": (0, 0, 2)}, entity=EntityType.LIGAND, polymer_kind=None),
        make_residue("B", 1, "GLY", {"CA": (0, 3, 0)}),
        make_residue("B", 2, "L2", {"C1": (20, 0, 2)}, entity=EntityType.LIGAND, polymer_kind=None),
    ])
    generator = get_generator("I01")
    parameters = {"chain_a": "A", "chain_b": "B"}
    outcome = generator.oracle(structure, parameters, definitions, Analysis(structure, definitions))
    assert outcome.gold_answer["value"] == ["A:L1"]
    question = generator.question(parameters, structure)
    assert "protein residue in chain A" in question
    assert "protein heavy atom in chain B" in question
    assert "Exclude non-polymer ligands" in question


def test_residue_identity_does_not_certify_clash_topology(crambin, definitions):
    structure = make_structure([
        make_residue("A", 1, "SNN", {"N1": (0, 0, 0)}),
        make_residue("A", 2, "GLY", {"CA": (1.4, 0, 0)}),
    ])
    processed = replace(crambin, structure=structure)
    context = GenerationContext(
        spec=processed.spec, processed=processed,
        displayed=display(structure, identity_transform(), definitions),
        definitions=definitions, rng=np.random.default_rng(0),
    )
    generator = get_generator("G04")
    proposals = list(generator.propose(context))
    assert len(proposals) == 1 and isinstance(proposals[0], Rejection)
    assert proposals[0].reason == "clash_unknown_protein_topology"
    with pytest.raises(ValueError, match="clash_unknown_protein_topology"):
        generator.oracle(structure, {}, definitions, Analysis(structure, definitions))
