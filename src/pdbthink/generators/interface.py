"""Interface and contact-network families I01 and N01 (specification section 8)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..config import Definitions
from ..geometry.contacts import (
    bridging_residues,
    interface_contacts,
    min_distance_to_residues,
    min_heavy_distance,
)
from ..preprocessing.model import Structure
from ..prompts.library import QUESTION_TEMPLATES
from .base import (
    MAX_UNCROPPABLE_ATOMS,
    Analysis,
    GenerationContext,
    Generator,
    OracleResult,
    Proposal,
    Rejection,
    register,
)


class I01InterfaceResidues(Generator):
    family = "I01"
    version = "1.0.0"
    answer_schema = "residue_set"
    level = "interface"
    croppable = False        # the answer is defined over a complete chain pair

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        if ctx.spec.source_type == "afdb":
            yield Rejection("afdb_is_single_chain", {}, ["uncertain_biological_assembly"])
            return
        if not ctx.spec.uses_assembly:
            yield Rejection(
                "interface_requires_validated_biological_assembly",
                {"protein": ctx.spec.id},
                ["uncertain_biological_assembly"],
            )
            return
        if structure.atom_count > MAX_UNCROPPABLE_ATOMS:
            yield Rejection(
                "structure_too_large_for_uncroppable_family",
                {"atoms": structure.atom_count, "limit": MAX_UNCROPPABLE_ATOMS},
            )
            return

        chains = structure.protein_chains
        cutoff = float(ctx.definitions.get("interface.heavy_atom_cutoff"))
        margin = float(ctx.definitions.get("distance.set_question_negative_margin"))
        for a in chains:
            for b in chains:
                if a == b:
                    continue
                contacts = interface_contacts(structure, a, b, ctx.definitions)
                if not 4 <= len(contacts) <= 40:
                    yield Rejection(
                        "interface_size_out_of_range",
                        {"chains": f"{a}/{b}", "n_residues": len(contacts)},
                    )
                    continue
                nearest_excluded = self._nearest_excluded(structure, a, b, cutoff)
                if nearest_excluded is not None and nearest_excluded < cutoff + margin:
                    yield Rejection(
                        "interface_negative_margin",
                        {
                            "chains": f"{a}/{b}",
                            "nearest_excluded": round(nearest_excluded, 3),
                            "required": cutoff + margin,
                        },
                        ["inside_ambiguity_margin"],
                    )
                    continue
                labels = sorted({structure.residues[c.i].label for c in contacts})
                yield Proposal(
                    parameters={"chain_a": a, "chain_b": b},
                    margins={
                        "n_interface_residues": len(labels),
                        "closest_pair_distance": contacts[0].min_distance,
                        "nearest_excluded_distance": nearest_excluded,
                    },
                    reasons=[
                        f"chain {a} contributes {len(labels)} residues to the {a}/{b} interface"
                    ],
                    criteria_passed=["validated_assembly", "negative_example_margin"],
                    required_labels=labels,
                    tag=f"{a}{b}",
                    rank=-len(labels),
                )

    @staticmethod
    def _nearest_excluded(
        structure: Structure, chain_a: str, chain_b: str, cutoff: float
    ) -> float | None:
        """Closest chain-A residue that is *not* an interface residue (A.4 margin)."""
        residues_a = [
            i for i, r in enumerate(structure.residues) if r.is_protein and r.chain == chain_a
        ]
        residues_b = [
            i for i, r in enumerate(structure.residues) if r.is_protein and r.chain == chain_b
        ]
        distances = min_distance_to_residues(structure, residues_a, residues_b)
        excluded = [d for d in distances.values() if d > cutoff]
        return min(excluded) if excluded else None

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        contacts = interface_contacts(
            structure, parameters["chain_a"], parameters["chain_b"], definitions
        )
        return OracleResult(
            gold_answer={"value": sorted({structure.residues[c.i].label for c in contacts})},
            evidence={
                "contacts": [
                    {
                        "residue": structure.residues[c.i].label,
                        "partner": structure.residues[c.j].label,
                        "distance": round(c.min_distance, 3),
                        "atoms": [c.atom_i, c.atom_j],
                    }
                    for c in contacts
                ],
                "closest_pair": (
                    f"{structure.residues[contacts[0].i].label}--"
                    f"{structure.residues[contacts[0].j].label}"
                    if contacts
                    else None
                ),
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["I01"].format(
            chain_a=parameters["chain_a"], chain_b=parameters["chain_b"]
        )


class N01BridgingResidue(Generator):
    family = "N01"
    version = "1.0.0"
    answer_schema = "residue"
    level = "network"
    croppable = True
    crop_radius = 16.0

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        graph = ctx.analysis.graph
        cutoff = float(ctx.definitions.get("residue_contact.heavy_atom_cutoff"))
        margin = ctx.definitions.negative_margin

        # Enumerate anchor pairs through their shared neighbours rather than over
        # all residue pairs: only pairs with a common contact can have a bridge.
        common: dict[tuple[int, int], list[int]] = {}
        for node in sorted(graph.adjacency):
            if not structure.residues[node].is_protein:
                continue
            neighbours = sorted(
                n for n in graph.neighbours(node) if structure.residues[n].is_protein
            )
            for i, a in enumerate(neighbours):
                for b in neighbours[i + 1:]:
                    common.setdefault((a, b), []).append(node)

        for (a, b), bridges in sorted(common.items()):
            if b in graph.neighbours(a):
                continue                        # anchors must not contact each other
            label_a = structure.residues[a].label
            label_b = structure.residues[b].label
            if len(bridges) > 1:
                yield Rejection(
                    "multiple_bridging_residues",
                    {
                        "anchors": [label_a, label_b],
                        "bridges": [structure.residues[i].label for i in bridges],
                    },
                    ["multiple_valid_answers"],
                )
                continue
            bridge = bridges[0]
            d_a, _, _ = min_heavy_distance(structure.residues[bridge], structure.residues[a])
            d_b, _, _ = min_heavy_distance(structure.residues[bridge], structure.residues[b])
            anchor_distance, _, _ = min_heavy_distance(
                structure.residues[a], structure.residues[b]
            )
            if anchor_distance < cutoff + margin:
                yield Rejection(
                    "anchors_too_close",
                    {"anchors": [label_a, label_b], "distance": round(anchor_distance, 3)},
                    ["inside_ambiguity_margin"],
                )
                continue
            near_miss = self._closest_non_bridge(structure, a, b, bridge, cutoff + margin)
            if near_miss is not None and near_miss < cutoff + margin:
                yield Rejection(
                    "near_miss_bridging_residue",
                    {
                        "anchors": [label_a, label_b],
                        "closest_non_bridge": round(near_miss, 3),
                        "required": cutoff + margin,
                    },
                    ["inside_ambiguity_margin"],
                )
                continue
            yield Proposal(
                parameters={"anchor_a": label_a, "anchor_b": label_b},
                margins={
                    "bridge_to_anchor_a": d_a,
                    "bridge_to_anchor_b": d_b,
                    "anchor_anchor_distance": anchor_distance,
                    "closest_non_bridge_max_distance": near_miss,
                },
                reasons=[
                    f"{structure.residues[bridge].label} is the unique residue contacting "
                    f"both {label_a} ({d_a:.2f} A) and {label_b} ({d_b:.2f} A)"
                ],
                criteria_passed=["unique_bridge", "anchors_not_in_contact"],
                required_labels=[label_a, label_b, structure.residues[bridge].label],
                crop_centers=[label_a, label_b],
                tag="interchain"
                if structure.residues[a].chain != structure.residues[b].chain
                else "intrachain",
                rank=max(d_a, d_b),
            )

    @staticmethod
    def _closest_non_bridge(
        structure: Structure, a: int, b: int, bridge: int, search: float
    ) -> float | None:
        """How close the best near-miss residue comes to bridging both anchors.

        A residue that contacts one anchor and sits just beyond the cutoff of the
        other would make the singular question fragile, so the generator measures
        ``max(d_to_a, d_to_b)`` for the best non-bridge candidate. Only residues
        within ``search`` of both anchors can compete, so the scan is restricted
        to those.
        """
        near_a = _residues_within(structure, a, search)
        near_b = _residues_within(structure, b, search)
        candidates = (near_a & near_b) - {a, b, bridge}
        best: float | None = None
        for ri in sorted(candidates):
            if not structure.residues[ri].is_protein:
                continue
            d_a, _, _ = min_heavy_distance(structure.residues[ri], structure.residues[a])
            d_b, _, _ = min_heavy_distance(structure.residues[ri], structure.residues[b])
            worst = max(d_a, d_b)
            if best is None or worst < best:
                best = worst
        return best

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        a = structure.find_index(parameters["anchor_a"])
        b = structure.find_index(parameters["anchor_b"])
        if a is None or b is None:
            raise ValueError(f"anchor residues missing for {parameters}")
        bridges = [
            i
            for i in bridging_residues(analysis.graph, a, b)
            if structure.residues[i].is_protein
        ]
        if len(bridges) != 1:
            raise ValueError(
                f"{len(bridges)} residues contact both anchors; the question requires one"
            )
        bridge = bridges[0]
        d_a, atom_a, _ = min_heavy_distance(structure.residues[bridge], structure.residues[a])
        d_b, atom_b, _ = min_heavy_distance(structure.residues[bridge], structure.residues[b])
        return OracleResult(
            gold_answer={"value": structure.residues[bridge].label},
            evidence={
                "distance_to_anchor_a": round(d_a, 3),
                "distance_to_anchor_b": round(d_b, 3),
                "bridge_atoms": [atom_a, atom_b],
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["N01"].format(
            anchor_a=parameters["anchor_a"], anchor_b=parameters["anchor_b"]
        )


def _residues_within(structure: Structure, residue_index: int, radius: float) -> set[int]:
    """Residues with any heavy atom within ``radius`` of the given residue."""
    index = structure.index
    out: set[int] = set()
    for atom in structure.residues[residue_index].atoms:
        for ai in index.within(atom.pos, radius):
            out.add(int(index.residue_of[ai]))
    return out


register(I01InterfaceResidues())
register(N01BridgingResidue())
