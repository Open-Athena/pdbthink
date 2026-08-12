"""Two-state family T01 (specification section 8).

Two-state questions need a different context from the single-structure
generators: two experimentally observed structures of the same protein, mapped
residue by residue, superposed on a conserved core, and then rendered with one
shared random transformation (A.27, section 6).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ..config import Definitions, StatePairSpec
from ..geometry.align import (
    CONTACT_GAINED,
    CONTACT_LOST,
    AlignmentResult,
    ResidueMapping,
    classify_contact_change,
)
from ..geometry.contacts import min_heavy_distance
from ..preprocessing.loader import ProcessedStructure
from ..preprocessing.model import Structure
from ..prompts.library import QUESTION_TEMPLATES
from .base import Proposal, Rejection

T01_MIN_CANDIDATES = 6
T01_MAX_CANDIDATES = 8


@dataclass
class TwoStateContext:
    """One aligned pair of states."""

    pair: StatePairSpec
    processed1: ProcessedStructure
    processed2: ProcessedStructure
    structure1: Structure               # state 1, deposited frame
    structure2: Structure               # state 2, superposed onto state 1
    mapping: ResidueMapping
    alignment: AlignmentResult
    definitions: Definitions
    notes: list[str] = field(default_factory=list)

    @property
    def shared_labels(self) -> list[tuple[str, int, int]]:
        """Mapped residues whose identifier is identical in both states."""
        out: list[tuple[str, int, int]] = []
        for i, j in self.mapping.pairs:
            a, b = self.structure1.residues[i], self.structure2.residues[j]
            if a.label == b.label and a.one_letter == b.one_letter:
                out.append((a.label, i, j))
        return out


class T01ContactChange:
    """Classify supplied candidate contacts as gained or lost in Structure 2."""

    family = "T01"
    version = "1.0.0"
    answer_schema = "two_interaction_sets"
    level = "two_state"
    two_state = True
    # The answer is defined over a supplied candidate list, so a local excerpt
    # containing every candidate residue is admissible (section 6).
    croppable = True
    crop_radius = 8.0

    def propose(self, ctx: TwoStateContext) -> Iterator[Proposal | Rejection]:
        structure1, structure2 = ctx.structure1, ctx.structure2
        shared = ctx.shared_labels
        index_by_label_1 = {label: i for label, i, _ in shared}
        index_by_label_2 = {label: j for label, _, j in shared}
        if len(shared) < 30:
            yield Rejection(
                "too_few_consistently_labelled_residues",
                {"n_shared": len(shared)},
                ["uncertain_state_mapping"],
            )
            return

        # Only pairs that are in contact in at least one state can change.
        candidates: list[tuple[float, str, str, str, float, float]] = []
        from ..geometry.contacts import residue_contacts

        seen: set[tuple[str, str]] = set()
        for source, structure in ((1, structure1), (2, structure2)):
            for contact in residue_contacts(structure, ctx.definitions):
                label_i = structure.residues[contact.i].label
                label_j = structure.residues[contact.j].label
                if label_i not in index_by_label_1 or label_j not in index_by_label_1:
                    continue
                key = tuple(sorted((label_i, label_j)))
                if key in seen:
                    continue
                seen.add(key)
                d1, _, _ = min_heavy_distance(
                    structure1.residues[index_by_label_1[key[0]]],
                    structure1.residues[index_by_label_1[key[1]]],
                )
                d2, _, _ = min_heavy_distance(
                    structure2.residues[index_by_label_2[key[0]]],
                    structure2.residues[index_by_label_2[key[1]]],
                )
                verdict = classify_contact_change(d1, d2, ctx.definitions)
                if verdict not in (CONTACT_GAINED, CONTACT_LOST):
                    continue
                candidates.append((abs(d1 - d2), key[0], key[1], verdict, d1, d2))

        gained = sorted([c for c in candidates if c[3] == CONTACT_GAINED], key=lambda c: -c[0])
        lost = sorted([c for c in candidates if c[3] == CONTACT_LOST], key=lambda c: -c[0])
        if not gained or not lost:
            yield Rejection(
                "state_pair_lacks_both_gained_and_lost_contacts",
                {"n_gained": len(gained), "n_lost": len(lost)},
            )
            return

        half = T01_MAX_CANDIDATES // 2
        chosen = gained[:half] + lost[:half]
        if len(chosen) < T01_MIN_CANDIDATES:
            chosen = (gained + lost)[:T01_MAX_CANDIDATES]
        if len(chosen) < T01_MIN_CANDIDATES:
            yield Rejection(
                "too_few_unambiguous_contact_changes",
                {"n_candidates": len(chosen), "required": T01_MIN_CANDIDATES},
            )
            return

        chosen.sort(key=lambda c: (c[1], c[2]))
        pairs = [f"{c[1]}--{c[2]}" for c in chosen]
        yield Proposal(
            parameters={"candidates": pairs},
            margins={
                "distances": {
                    f"{c[1]}--{c[2]}": {"state1": round(c[4], 3), "state2": round(c[5], 3)}
                    for c in chosen
                },
                "n_gained": sum(1 for c in chosen if c[3] == CONTACT_GAINED),
                "n_lost": sum(1 for c in chosen if c[3] == CONTACT_LOST),
                "alignment_rmsd": ctx.alignment.rmsd_after,
                "sequence_identity": ctx.mapping.identity,
            },
            reasons=[
                f"{sum(1 for c in chosen if c[3] == CONTACT_GAINED)} gained and "
                f"{sum(1 for c in chosen if c[3] == CONTACT_LOST)} lost contacts, all outside "
                "the 4.0-4.5 A hysteresis band"
            ],
            criteria_passed=["contact_change_hysteresis", "same_residue_identity", "aligned_core"],
            required_labels=sorted({c[1] for c in chosen} | {c[2] for c in chosen}),
            crop_centers=sorted({c[1] for c in chosen} | {c[2] for c in chosen}),
            rank=-len(chosen),
        )

    def oracle(
        self,
        structures: tuple[Structure, Structure],
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Any = None,
    ) -> dict[str, Any]:
        """Recompute gained/lost from the two displayed structures, in shown order."""
        first, second = structures
        gained: list[str] = []
        lost: list[str] = []
        evidence: dict[str, Any] = {"distances": {}}
        for pair in parameters["candidates"]:
            a, b = pair.split("--")
            ia, ib = first.find_index(a), first.find_index(b)
            ja, jb = second.find_index(a), second.find_index(b)
            if None in (ia, ib, ja, jb):
                raise ValueError(f"candidate pair {pair} is not present in both structures")
            d1, atom_a1, atom_b1 = min_heavy_distance(first.residues[ia], first.residues[ib])
            d2, atom_a2, atom_b2 = min_heavy_distance(second.residues[ja], second.residues[jb])
            verdict = classify_contact_change(d1, d2, definitions)
            evidence["distances"][pair] = {
                "state1": round(d1, 3),
                "state2": round(d2, 3),
                "verdict": verdict,
                "atoms_state1": [atom_a1, atom_b1],
                "atoms_state2": [atom_a2, atom_b2],
            }
            if verdict == CONTACT_GAINED:
                gained.append(_canonical_pair(a, b))
            elif verdict == CONTACT_LOST:
                lost.append(_canonical_pair(a, b))
            else:
                raise ValueError(
                    f"candidate pair {pair} is {verdict} in the displayed coordinates; "
                    "T01 requires every candidate to be unambiguously gained or lost"
                )
        return {
            "gold_answer": {"gained": sorted(gained), "lost": sorted(lost)},
            "evidence": evidence,
        }

    def question(self, parameters: dict[str, Any], structure: Structure | None = None) -> str:
        return QUESTION_TEMPLATES["T01"].format(candidates=", ".join(parameters["candidates"]))

    def context(self, parameters: dict[str, Any]) -> str:
        return (
            "You are given two molecular structures of the same protein in two different "
            "states. Every candidate pair below is either gained or lost; no candidate is "
            "unchanged."
        )


def _canonical_pair(a: str, b: str) -> str:
    first, second = sorted((a, b))
    return f"{first}--{second}"


def build_two_state_context(
    pair: StatePairSpec,
    processed1: ProcessedStructure,
    processed2: ProcessedStructure,
    definitions: Definitions,
    *,
    exclude_from_core: list[str] = (),
) -> TwoStateContext:
    """Map and superpose two states, following A.27 steps 1-5."""
    from ..geometry.align import apply_superposition, map_residues, superpose_states

    structure1 = processed1.structure
    raw2 = processed2.structure
    chain_map = pair.chain_map or None
    mapping = map_residues(structure1, raw2, definitions, chain_map=chain_map)

    exclude_indices = [
        i for label in exclude_from_core if (i := structure1.find_index(label)) is not None
    ]
    alignment = superpose_states(
        structure1, raw2, mapping, definitions, exclude=exclude_indices
    )
    structure2 = apply_superposition(raw2, alignment.superposition)

    notes = [
        f"sequence identity {mapping.identity}",
        f"core {len(alignment.core_pairs)} CA pairs, RMSD {alignment.rmsd_after:.3f} A",
    ]
    if alignment.excluded_pairs:
        notes.append(f"{len(alignment.excluded_pairs)} core pairs removed as outliers")
    return TwoStateContext(
        pair=pair,
        processed1=processed1,
        processed2=processed2,
        structure1=structure1,
        structure2=structure2,
        mapping=mapping,
        alignment=alignment,
        definitions=definitions,
        notes=notes,
    )


T01 = T01ContactChange()
