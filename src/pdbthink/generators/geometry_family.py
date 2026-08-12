"""Geometry families G01-G04 (specification section 8)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..chem import has_known_topology
from ..config import Definitions
from ..geometry.contacts import min_distance_to_residues, nearest_nonbonded_atom
from ..geometry.core import distance
from ..preprocessing.model import Structure
from ..prompts.library import QUESTION_TEMPLATES
from ..scoring.scorers import DEFAULT_DISTANCE_TOLERANCE
from .base import (
    Analysis,
    GenerationContext,
    Generator,
    OracleResult,
    Proposal,
    Rejection,
    register,
)

G01_MIN_DISTANCE = 3.0
G01_MAX_DISTANCE = 25.0
G03_N_CANDIDATES = 5


class G01AtomDistance(Generator):
    family = "G01"
    version = "1.0.0"
    answer_schema = "distance"
    level = "geometry"
    croppable = True
    crop_radius = 16.0

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        residues = [
            (i, r)
            for i, r in enumerate(structure.residues)
            if r.is_protein and r.has_atoms("CA")
        ]
        if len(residues) < 8:
            yield Rejection("too_few_residues", {"n": len(residues)})
            return

        # Deterministic spread of pairs across the polymer.
        picks = np.linspace(0, len(residues) - 1, num=min(12, len(residues)), dtype=int)
        seen: set[tuple[str, str]] = set()
        rank = 0
        for a in range(len(picks)):
            for b in range(a + 1, len(picks)):
                ri, res_i = residues[int(picks[a])]
                rj, res_j = residues[int(picks[b])]
                atom_i = _informative_atom(res_i)
                atom_j = _informative_atom(res_j)
                if atom_i is None or atom_j is None:
                    continue
                d = distance(atom_i.pos, atom_j.pos)
                if not (G01_MIN_DISTANCE <= d <= G01_MAX_DISTANCE):
                    continue
                label_i = f"{res_i.label}:{atom_i.name}"
                label_j = f"{res_j.label}:{atom_j.name}"
                if (label_i, label_j) in seen:
                    continue
                seen.add((label_i, label_j))
                yield Proposal(
                    parameters={"atom1": label_i, "atom2": label_j},
                    margins={"distance": d},
                    reasons=[f"{label_i} to {label_j} is {d:.3f} A"],
                    criteria_passed=["atoms_present", "distance_in_range"],
                    required_labels=[res_i.label, res_j.label],
                    crop_centers=[res_i.label, res_j.label],
                    rank=rank,
                )
                rank += 1

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        first = structure.atom_by_label(parameters["atom1"])
        second = structure.atom_by_label(parameters["atom2"])
        if first is None or second is None:
            raise ValueError(f"missing query atom(s) for {parameters}")
        d = distance(first[1].pos, second[1].pos)
        return OracleResult(
            gold_answer={"value": round(d, 3)},
            evidence={
                "atom1": parameters["atom1"],
                "atom2": parameters["atom2"],
                "distance": d,
                "tolerance": DEFAULT_DISTANCE_TOLERANCE,
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["G01"].format(
            atom1=parameters["atom1"], atom2=parameters["atom2"]
        )

    def prompt_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {"tolerance": DEFAULT_DISTANCE_TOLERANCE}


class G02NearestNonbondedAtom(Generator):
    family = "G02"
    version = "1.0.0"
    answer_schema = "atom"
    level = "geometry"
    croppable = True
    crop_radius = 16.0

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        definitions = ctx.definitions
        margin = definitions.nearest_margin
        topology = ctx.analysis.topology
        rank = 0
        # A spread sample rather than every side chain: the family needs a handful
        # of instances and every candidate costs a neighbourhood scan.
        protein_indices = [i for i, r in enumerate(structure.residues) if r.is_protein]
        sampled = set(
            int(i) for i in np.linspace(0, len(protein_indices) - 1, num=min(60, len(protein_indices)))
        )
        for position, ri in enumerate(protein_indices):
            if position not in sampled:
                continue
            res = structure.residues[ri]
            if not res.is_standard_aa or not has_known_topology(res.orig_name):
                continue
            for atom in res.sidechain_atoms:
                ranked = nearest_nonbonded_atom(structure, ri, atom.name, topology, definitions)
                if len(ranked) < 2:
                    continue
                best, runner = ranked[0], ranked[1]
                gap = runner[2] - best[2]
                target_res = structure.residues[best[0]]
                if not target_res.is_standard_aa:
                    continue
                if best[2] > 5.0:
                    continue
                if gap < margin:
                    if gap >= margin / 2:
                        yield Rejection(
                            "nearest_atom_margin",
                            {
                                "query": f"{res.label}:{atom.name}",
                                "best": f"{target_res.label}:{best[1]}",
                                "gap": round(gap, 3),
                                "required": margin,
                            },
                            ["inside_ambiguity_margin"],
                        )
                    continue
                yield Proposal(
                    parameters={"residue": res.label, "atom": atom.name},
                    margins={
                        "nearest_distance": best[2],
                        "runner_up_distance": runner[2],
                        "gap": gap,
                    },
                    reasons=[
                        f"{target_res.label}:{best[1]} at {best[2]:.3f} A beats the runner-up "
                        f"by {gap:.3f} A"
                    ],
                    criteria_passed=["nearest_neighbour_margin", "standard_topology"],
                    required_labels=[res.label, target_res.label],
                    crop_centers=[res.label],
                    rank=-gap,
                )
                rank += 1

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        ri = structure.find_index(parameters["residue"])
        if ri is None:
            raise ValueError(f"residue {parameters['residue']} absent from displayed structure")
        ranked = nearest_nonbonded_atom(
            structure, ri, parameters["atom"], analysis.topology, definitions
        )
        if not ranked:
            raise ValueError(f"no candidate atoms for {parameters}")
        best = ranked[0]
        answer = f"{structure.residues[best[0]].label}:{best[1]}"
        return OracleResult(
            gold_answer={"value": answer},
            evidence={
                "query_atom": f"{parameters['residue']}:{parameters['atom']}",
                "nearest": answer,
                "distance": best[2],
                "runner_up": (
                    f"{structure.residues[ranked[1][0]].label}:{ranked[1][1]}"
                    if len(ranked) > 1
                    else None
                ),
                "runner_up_distance": ranked[1][2] if len(ranked) > 1 else None,
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["G02"].format(
            atom=f"{parameters['residue']}:{parameters['atom']}"
        )


class G03ClosestCandidate(Generator):
    family = "G03"
    version = "1.0.0"
    answer_schema = "residue"
    level = "geometry"
    croppable = True
    crop_radius = 14.0

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        margin = ctx.definitions.nearest_margin
        protein = [i for i, r in enumerate(structure.residues) if r.is_protein]
        if len(protein) < G03_N_CANDIDATES + 2:
            yield Rejection("too_few_residues", {"n": len(protein)})
            return

        sampled = {
            protein[int(i)]
            for i in np.linspace(0, len(protein) - 1, num=min(20, len(protein)))
        }
        for target in sorted(sampled):
            per_residue = min_distance_to_residues(structure, protein, [target])
            distances = []
            for other, d in per_residue.items():
                if other == target:
                    continue
                sep = _polymer_gap(structure, target, other)
                if sep is not None and sep <= 2:
                    continue
                distances.append((d, other))
            if len(distances) < G03_N_CANDIDATES:
                continue
            distances.sort(key=lambda t: (t[0], t[1]))
            winner_d, winner = distances[0]
            runner_d, _ = distances[1]
            if runner_d - winner_d < margin:
                yield Rejection(
                    "closest_candidate_margin",
                    {
                        "target": structure.residues[target].label,
                        "gap": round(runner_d - winner_d, 3),
                        "required": margin,
                    },
                    ["inside_ambiguity_margin"],
                )
                continue
            # Decoys spread over increasing distance so the task needs real geometry.
            decoy_pool = [idx for _, idx in distances[1:]]
            step = max(1, len(decoy_pool) // (G03_N_CANDIDATES - 1))
            decoys = decoy_pool[: step * (G03_N_CANDIDATES - 1) : step][: G03_N_CANDIDATES - 1]
            if len(decoys) < G03_N_CANDIDATES - 1:
                continue
            candidates = sorted(
                structure.residues[i].label for i in ([winner] + list(decoys))
            )
            yield Proposal(
                parameters={
                    "target": structure.residues[target].label,
                    "candidates": candidates,
                },
                margins={
                    "winner_distance": winner_d,
                    "runner_up_distance": runner_d,
                    "gap": runner_d - winner_d,
                },
                reasons=[
                    f"{structure.residues[winner].label} at {winner_d:.3f} A beats the "
                    f"next candidate by {runner_d - winner_d:.3f} A"
                ],
                criteria_passed=["nearest_neighbour_margin"],
                required_labels=[structure.residues[target].label, *candidates],
                crop_centers=[structure.residues[target].label, *candidates],
                rank=-(runner_d - winner_d),
            )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        target = structure.find_index(parameters["target"])
        if target is None:
            raise ValueError(f"target {parameters['target']} absent")
        rows = []
        for label in parameters["candidates"]:
            idx = structure.find_index(label)
            if idx is None:
                raise ValueError(f"candidate {label} absent from displayed structure")
            d, ai, aj = _min_distance(structure, target, idx)
            rows.append((d, label, ai, aj))
        rows.sort(key=lambda t: (t[0], t[1]))
        return OracleResult(
            gold_answer={"value": rows[0][1]},
            evidence={
                "target": parameters["target"],
                "distances": {label: d for d, label, _, _ in rows},
                "winning_atom_pair": [rows[0][2], rows[0][3]],
                "gap_to_runner_up": rows[1][0] - rows[0][0] if len(rows) > 1 else None,
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["G03"].format(
            target=parameters["target"], candidates=", ".join(parameters["candidates"])
        )


class G04SevereClash(Generator):
    family = "G04"
    version = "1.0.0"
    answer_schema = "residue_pair"
    level = "geometry"
    croppable = False          # uniqueness is a property of the whole structure

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        clashes = ctx.analysis.clashes
        required = float(ctx.definitions.get("clash.uniqueness_margin"))
        if not clashes:
            yield Rejection("no_clashes", {})
            return
        if len(structure.protein_residues) > 600:
            yield Rejection(
                "structure_too_large_for_uncroppable_family",
                {"protein_residues": len(structure.protein_residues)},
            )
            return
        best = clashes[0]
        runner_up = clashes[1].overlap if len(clashes) > 1 else 0.0
        gap = best.overlap - runner_up
        if gap < required:
            yield Rejection(
                "clash_uniqueness_margin",
                {
                    "best_overlap": round(best.overlap, 3),
                    "runner_up_overlap": round(runner_up, 3),
                    "required": required,
                },
                ["multiple_valid_answers", "inside_ambiguity_margin"],
            )
            return
        labels = sorted((structure.residues[best.i].label, structure.residues[best.j].label))
        yield Proposal(
            parameters={"pair": f"{labels[0]}--{labels[1]}"},
            margins={
                "overlap": best.overlap,
                "runner_up_overlap": runner_up,
                "gap": gap,
                "distance": best.distance,
            },
            reasons=[
                f"{labels[0]}--{labels[1]} overlaps by {best.overlap:.3f} A, "
                f"{gap:.3f} A more than the runner-up"
            ],
            criteria_passed=["clash_uniqueness_margin"],
            required_labels=list(labels),
            rank=-gap,
        )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        clashes = analysis.clashes
        if not clashes:
            raise ValueError("no clashes found in the displayed structure")
        best = clashes[0]
        labels = sorted((structure.residues[best.i].label, structure.residues[best.j].label))
        return OracleResult(
            gold_answer={"value": f"{labels[0]}--{labels[1]}"},
            evidence={
                "atoms": [best.atom_i, best.atom_j],
                "distance": best.distance,
                "overlap": best.overlap,
                "runner_up_overlap": clashes[1].overlap if len(clashes) > 1 else None,
                "top_clashes": [
                    {
                        "pair": f"{structure.residues[c.i].label}--{structure.residues[c.j].label}",
                        "overlap": round(c.overlap, 3),
                    }
                    for c in clashes[:5]
                ],
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["G04"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _informative_atom(res):
    """Prefer a distinctive side-chain atom, falling back to CA."""
    for name in ("NZ", "NE2", "OD1", "OE1", "SG", "OG", "OH", "CZ", "CG", "CB"):
        atom = res.atom(name)
        if atom is not None:
            return atom
    return res.atom("CA")


def _min_distance(structure: Structure, i: int, j: int) -> tuple[float, str, str]:
    from ..geometry.contacts import min_heavy_distance

    return min_heavy_distance(structure.residues[i], structure.residues[j])


def _polymer_gap(structure: Structure, i: int, j: int) -> int | None:
    from ..geometry.contacts import polymer_separation

    return polymer_separation(structure, i, j)


register(G01AtomDistance())
register(G02NearestNonbondedAtom())
register(G03ClosestCandidate())
register(G04SevereClash())
