"""Golden tests for the question generators and their oracles.

Each test drives a generator over a committed, hand-checkable structure and
verifies the gold answer against an independently known fact about that
structure, then re-runs the oracle on a rotated copy to confirm the label is
rotation invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdbthink.generators import Analysis, GenerationContext, all_generators, get_generator
from pdbthink.generators.base import Proposal, Rejection
from pdbthink.preprocessing import build_transform, display, identity_transform
from pdbthink.util import rng_for

ROTATION_INVARIANT = tuple(f for f in all_generators() if f != "P03")


def context_for(processed, definitions, seed: int = 0):
    transform = (
        identity_transform() if seed == 0 else build_transform([processed.structure], seed, definitions)
    )
    displayed = display(processed.structure, transform, definitions)
    return GenerationContext(
        spec=processed.spec,
        processed=processed,
        displayed=displayed,
        definitions=definitions,
        rng=rng_for("test", seed),
    )


def first_proposal(family: str, processed, definitions) -> tuple[Proposal, GenerationContext]:
    generator = get_generator(family)
    ctx = context_for(processed, definitions)
    proposals = [p for p in generator.propose(ctx) if isinstance(p, Proposal)]
    assert proposals, f"{family}: no proposals for {processed.spec.id}"
    proposals.sort(key=lambda p: (p.rank, p.key()))
    return proposals[0], ctx


def run_oracle(family: str, parameters, structure, definitions):
    generator = get_generator(family)
    return generator.oracle(structure, parameters, definitions, Analysis(structure, definitions))


class TestParsingFamilies:
    def test_p01_lists_every_chain(self, crambin, definitions):
        result = run_oracle("P01", {}, crambin.structure, definitions)
        assert result.gold_answer["value"] == ["A"]

    def test_p02_counts_crambin_residues(self, crambin, definitions):
        result = run_oracle("P02", {"chain": "A"}, crambin.structure, definitions)
        assert result.gold_answer["value"] == 46      # crambin is a 46-residue protein

    def test_p03_reports_displayed_coordinates(self, crambin, definitions):
        atom = crambin.structure.find("A:C3").atom("SG")
        result = run_oracle("P03", {"residue": "A:C3", "atom": "SG"}, crambin.structure, definitions)
        assert result.gold_answer["value"] == pytest.approx(list(atom.pos))

    def test_p03_gold_follows_the_rotation(self, crambin, definitions):
        rotated = display(
            crambin.structure, build_transform([crambin.structure], 5, definitions), definitions
        ).structure
        first = run_oracle("P03", {"residue": "A:C3", "atom": "SG"}, crambin.structure, definitions)
        second = run_oracle("P03", {"residue": "A:C3", "atom": "SG"}, rotated, definitions)
        assert first.gold_answer["value"] != second.gold_answer["value"]


class TestGeometryFamilies:
    def test_g01_distance_matches_direct_computation(self, crambin, definitions):
        proposal, _ = first_proposal("G01", crambin, definitions)
        result = run_oracle("G01", proposal.parameters, crambin.structure, definitions)
        a = crambin.structure.atom_by_label(proposal.parameters["atom1"])[1]
        b = crambin.structure.atom_by_label(proposal.parameters["atom2"])[1]
        assert result.gold_answer["value"] == pytest.approx(
            float(np.linalg.norm(a.pos - b.pos)), abs=1e-3
        )

    def test_g02_answer_is_not_covalently_bonded(self, crambin, definitions):
        proposal, ctx = first_proposal("G02", crambin, definitions)
        result = run_oracle("G02", proposal.parameters, crambin.structure, definitions)
        answer = result.gold_answer["value"]
        assert answer.count(":") == 2
        assert result.evidence["distance"] < result.evidence["runner_up_distance"]
        gap = result.evidence["runner_up_distance"] - result.evidence["distance"]
        assert gap >= definitions.nearest_margin

    def test_g03_winner_beats_the_runner_up_by_the_margin(self, crambin, definitions):
        proposal, _ = first_proposal("G03", crambin, definitions)
        result = run_oracle("G03", proposal.parameters, crambin.structure, definitions)
        assert result.gold_answer["value"] in proposal.parameters["candidates"]
        assert result.evidence["gap_to_runner_up"] >= definitions.nearest_margin


class TestLocalFamilies:
    def test_s04_secondary_structure_is_a_three_state_class(self, crambin, definitions):
        proposal, _ = first_proposal("S04", crambin, definitions)
        result = run_oracle("S04", proposal.parameters, crambin.structure, definitions)
        assert result.gold_answer["value"] in ("helix", "strand", "coil")

    def test_s05_crambin_is_not_rejected(self, crambin, definitions):
        result = run_oracle("S05", {"chain": "A"}, crambin.structure, definitions)
        assert result.gold_answer["value"] in (
            "predominantly alpha helical",
            "predominantly beta sheet",
            "mixed alpha/beta",
        )

    def test_s08_recovers_a_known_crambin_disulfide(self, crambin, definitions):
        result = run_oracle("S08", {"residue": "A:C3"}, crambin.structure, definitions)
        assert result.gold_answer["value"] == "A:C40"    # crambin Cys3-Cys40
        assert result.evidence["sg_sg_distance"] <= 2.3

    def test_s09_chi1_is_outside_the_boundary_margin(self, crambin, definitions):
        proposal, _ = first_proposal("S09", crambin, definitions)
        result = run_oracle("S09", proposal.parameters, crambin.structure, definitions)
        assert result.gold_answer["value"] in ("g+", "t", "g-")
        assert result.evidence["boundary_distance"] >= float(
            definitions.get("chi1.boundary_margin")
        )

    def test_s07_zinc_donors_are_all_within_the_cutoff(self, zinc_site, definitions):
        metal = next(r for r in zinc_site.structure.residues if r.name == "ZN")
        result = run_oracle(
            "S07",
            {"metal": metal.label, "element": "ZN", "cutoff": 2.6},
            zinc_site.structure,
            definitions,
        )
        assert len(result.gold_answer["value"]) >= 3
        assert all(d["distance"] <= 2.6 for d in result.evidence["donors"])

    def test_s03_is_rejected_for_intermediate_burial(self, crambin, definitions):
        ctx = context_for(crambin, definitions)
        rejections = [r for r in get_generator("S03").propose(ctx) if isinstance(r, Rejection)]
        assert any(r.reason == "intermediate_burial" for r in rejections)
        assert all(
            "inside_ambiguity_margin" in r.criteria_failed
            for r in rejections
            if r.reason == "intermediate_burial"
        )


class TestRotationInvariance:
    @pytest.mark.parametrize(
        "family,parameters",
        [
            ("P01", {}),
            ("P02", {"chain": "A"}),
            ("S05", {"chain": "A"}),
            ("S08", {"residue": "A:C3"}),
        ],
    )
    def test_gold_is_unchanged_under_rotation(self, crambin, definitions, family, parameters):
        rotated = display(
            crambin.structure, build_transform([crambin.structure], 42, definitions), definitions
        ).structure
        first = run_oracle(family, parameters, crambin.structure, definitions)
        second = run_oracle(family, parameters, rotated, definitions)
        assert first.gold_answer == second.gold_answer

    def test_distance_labels_survive_rotation(self, crambin, definitions):
        proposal, _ = first_proposal("G01", crambin, definitions)
        rotated = display(
            crambin.structure, build_transform([crambin.structure], 11, definitions), definitions
        ).structure
        first = run_oracle("G01", proposal.parameters, crambin.structure, definitions)
        second = run_oracle("G01", proposal.parameters, rotated, definitions)
        assert first.gold_answer["value"] == pytest.approx(second.gold_answer["value"], abs=0.005)


class TestGeneratorContract:
    def test_every_v1_family_is_registered(self):
        from pdbthink.generators import V1_FAMILIES

        registered = set(all_generators()) | {"T01"}
        assert set(V1_FAMILIES) <= registered

    def test_every_generator_declares_a_known_answer_schema(self):
        from pdbthink.schemas import ANSWER_SCHEMAS

        for family, generator in all_generators().items():
            assert generator.answer_schema in ANSWER_SCHEMAS, family
            assert generator.version, family

    def test_rejections_carry_machine_readable_reasons(self, crambin, definitions):
        ctx = context_for(crambin, definitions)
        rejections = [
            (family, r)
            for family, generator in all_generators().items()
            for r in generator.propose(ctx)
            if isinstance(r, Rejection)
        ]
        assert rejections, "no family rejected anything on crambin"
        for family, rejection in rejections:
            assert rejection.reason, family
            assert isinstance(rejection.detail, dict), family
            assert isinstance(rejection.criteria_failed, list), family
