"""Local- and global-structure families S01-S09 (specification section 8)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..chem import PHOSPHO_COMPONENTS, chi1_atom4, metal_element, parent_of
from ..config import Definitions
from ..geometry.contacts import (
    ligand_contacts,
    metal_coordination,
    min_charged_distance,
    nearest_excluded_ligand_contact,
)
from ..geometry.dssp import fold_class
from ..geometry.rotamer import compute_chi1
from ..geometry.sasa import BURIED, EXPOSED
from ..preprocessing.model import EntityType, Structure
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

CHARGED = ("ASP", "GLU", "LYS", "ARG", "HIS")
HYDROPHOBIC = ("LEU", "ILE", "VAL", "PHE", "MET", "TRP", "ALA")


def _polarity(parent: str) -> str:
    """Diversity tag so S03 covers charged and hydrophobic residues alike (A.13)."""
    if parent in CHARGED:
        return "charged"
    if parent in HYDROPHOBIC:
        return "hydrophobic"
    return "polar"


class S01SaltBridgePartner(Generator):
    family = "S01"
    version = "1.0.0"
    answer_schema = "residue"
    level = "local"
    croppable = True
    crop_radius = 14.0

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        cutoff = float(ctx.definitions.get("salt_bridge.cutoff"))
        margin = ctx.definitions.negative_margin
        partners: dict[int, list[tuple[int, float]]] = {}
        for bridge in ctx.analysis.salt_bridges:
            partners.setdefault(bridge.i, []).append((bridge.j, bridge.distance))
            partners.setdefault(bridge.j, []).append((bridge.i, bridge.distance))

        for ri, found in sorted(partners.items()):
            label = structure.residues[ri].label
            if len(found) != 1:
                yield Rejection(
                    "multiple_salt_bridge_partners",
                    {"residue": label, "n_partners": len(found)},
                    ["multiple_valid_answers"],
                )
                continue
            partner, distance = found[0]
            runner_up = self._closest_non_partner(structure, ri, partner)
            if runner_up is not None and runner_up < cutoff + margin:
                yield Rejection(
                    "salt_bridge_negative_margin",
                    {
                        "residue": label,
                        "closest_non_partner": round(runner_up, 3),
                        "required": cutoff + margin,
                    },
                    ["inside_ambiguity_margin"],
                )
                continue
            partner_label = structure.residues[partner].label
            yield Proposal(
                parameters={"residue": label},
                margins={
                    "salt_bridge_distance": distance,
                    "closest_non_partner": runner_up,
                    "required_non_partner_distance": cutoff + margin,
                },
                reasons=[
                    f"{label} pairs with {partner_label} at {distance:.3f} A; the next "
                    f"oppositely charged residue is "
                    f"{'absent' if runner_up is None else format(runner_up, '.3f') + ' A away'}"
                ],
                criteria_passed=["unique_partner", "negative_example_margin"],
                required_labels=[label, partner_label],
                crop_centers=[label, partner_label],
                tag="interchain"
                if structure.residues[ri].chain != structure.residues[partner].chain
                else "intrachain",
                rank=distance,
            )

    @staticmethod
    def _closest_non_partner(structure: Structure, ri: int, partner: int) -> float | None:
        best: float | None = None
        for rj, other in enumerate(structure.residues):
            if rj in (ri, partner) or not other.is_protein:
                continue
            d = min_charged_distance(structure, ri, rj)
            if d is not None and (best is None or d < best):
                best = d
        return best

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        ri = structure.find_index(parameters["residue"])
        if ri is None:
            raise ValueError(f"residue {parameters['residue']} absent")
        found = [b for b in analysis.salt_bridges if ri in (b.i, b.j)]
        if len(found) != 1:
            raise ValueError(
                f"{parameters['residue']} has {len(found)} salt-bridge partners in the "
                "displayed structure; the question requires exactly one"
            )
        bridge = found[0]
        partner = bridge.j if bridge.i == ri else bridge.i
        return OracleResult(
            gold_answer={"value": structure.residues[partner].label},
            evidence={
                "distance": bridge.distance,
                "atoms": [bridge.positive_atom, bridge.negative_atom],
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["S01"].format(residue=parameters["residue"])


class S02PhosphorylatedResidue(Generator):
    family = "S02"
    version = "1.0.0"
    answer_schema = "residue"
    level = "local"
    croppable = False           # uniqueness is a property of the whole structure

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        phospho = [
            i
            for i, r in enumerate(structure.residues)
            if r.is_protein and r.orig_name in PHOSPHO_COMPONENTS
        ]
        if not phospho:
            yield Rejection("no_phosphorylated_residue", {})
            return
        if len(phospho) > 1:
            yield Rejection(
                "multiple_phosphorylated_residues",
                {"residues": [structure.residues[i].label for i in phospho]},
                ["multiple_valid_answers"],
            )
            return
        if structure.atom_count > MAX_UNCROPPABLE_ATOMS:
            yield Rejection(
                "structure_too_large_for_uncroppable_family",
                {"atoms": structure.atom_count, "limit": MAX_UNCROPPABLE_ATOMS},
            )
            return
        res = structure.residues[phospho[0]]
        if res.atom("P") is None:
            yield Rejection(
                "phosphate_atom_missing", {"residue": res.label}, ["missing_required_atoms"]
            )
            return
        yield Proposal(
            parameters={},
            margins={"n_phospho": 1, "component": res.orig_name},
            reasons=[f"exactly one covalently modified residue ({res.orig_name}) is present"],
            criteria_passed=["unique_answer", "component_dictionary_match"],
            required_labels=[res.label],
            rank=0.0,
        )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        components = set(definitions.get("phosphorylation.components"))
        phospho = [
            i
            for i, r in enumerate(structure.residues)
            if r.is_protein and r.orig_name in components
        ]
        if len(phospho) != 1:
            raise ValueError(f"expected exactly one phosphorylated residue, found {len(phospho)}")
        res = structure.residues[phospho[0]]
        return OracleResult(
            gold_answer={"value": res.label},
            evidence={"component": res.orig_name, "atoms": [a.name for a in res.atoms]},
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["S02"]


class S03Burial(Generator):
    family = "S03"
    version = "1.0.0"
    answer_schema = "category"
    level = "local"
    croppable = False           # SASA depends on every occluding atom

    CATEGORIES = [BURIED, EXPOSED]

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        if structure.atom_count > MAX_UNCROPPABLE_ATOMS:
            yield Rejection(
                "structure_too_large_for_uncroppable_family",
                {"atoms": structure.atom_count, "limit": MAX_UNCROPPABLE_ATOMS},
            )
            return
        sasa = ctx.analysis.sasa
        buried_max = float(ctx.definitions.get("solvent_accessibility.buried_max_rasa"))
        exposed_min = float(ctx.definitions.get("solvent_accessibility.exposed_min_rasa"))
        for ri, res in enumerate(structure.residues):
            if not res.is_protein or not res.is_standard_aa:
                continue
            rasa = sasa.residue_rasa.get(ri)
            if rasa is None:
                continue
            parent = parent_of(res.orig_name)
            if rasa <= buried_max:
                category, distance = BURIED, buried_max - rasa
            elif rasa >= exposed_min:
                category, distance = EXPOSED, rasa - exposed_min
            else:
                yield Rejection(
                    "intermediate_burial",
                    {"residue": res.label, "rasa": round(rasa, 3)},
                    ["inside_ambiguity_margin"],
                )
                continue
            yield Proposal(
                parameters={"residue": res.label},
                margins={
                    "rasa": rasa,
                    "absolute_sasa": sasa.residue_sasa.get(ri),
                    "distance_from_threshold": distance,
                },
                reasons=[f"{res.label} has rASA {rasa:.3f}, classified {category}"],
                criteria_passed=["outside_intermediate_band"],
                required_labels=[res.label],
                tag=f"{category}:{_polarity(parent)}",
                rank=-distance,
            )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        ri = structure.find_index(parameters["residue"])
        if ri is None:
            raise ValueError(f"residue {parameters['residue']} absent")
        category = analysis.sasa.classify(ri, definitions)
        if category not in (BURIED, EXPOSED):
            raise ValueError(
                f"{parameters['residue']} is in the intermediate burial band and cannot "
                "be asked as a binary question"
            )
        return OracleResult(
            gold_answer={"value": category},
            evidence={
                "rasa": analysis.sasa.residue_rasa.get(ri),
                "absolute_sasa": analysis.sasa.residue_sasa.get(ri),
                "probe_radius": analysis.sasa.probe_radius,
                "n_sphere_points": analysis.sasa.n_points,
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["S03"].format(residue=parameters["residue"])

    def prompt_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {"categories": self.CATEGORIES}


class S04SecondaryStructure(Generator):
    family = "S04"
    version = "1.0.0"
    answer_schema = "category"
    level = "local"
    croppable = False           # DSSP needs the surrounding backbone

    CATEGORIES = ["helix", "strand", "coil"]

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        if structure.atom_count > MAX_UNCROPPABLE_ATOMS:
            yield Rejection(
                "structure_too_large_for_uncroppable_family",
                {"atoms": structure.atom_count, "limit": MAX_UNCROPPABLE_ATOMS},
            )
            return
        cfg = ctx.definitions.get("secondary_structure")
        min_run = int(cfg["min_stable_run"])
        min_margin = int(cfg["min_run_edge_margin"])
        dssp = ctx.analysis.dssp
        runs = ctx.analysis.secondary_runs
        for ri, res in enumerate(structure.residues):
            if not res.is_protein:
                continue
            if ri in dssp.failed:
                yield Rejection(
                    "dssp_missing_backbone", {"residue": res.label}, ["missing_required_atoms"]
                )
                continue
            klass = dssp.three_state.get(ri)
            if klass is None:
                continue
            run_length, edge_distance = runs.get(ri, (0, 0))
            if run_length < min_run or edge_distance < min_margin:
                yield Rejection(
                    "unstable_secondary_structure_run",
                    {
                        "residue": res.label,
                        "class": klass,
                        "run_length": run_length,
                        "distance_to_run_edge": edge_distance,
                    },
                    ["inside_ambiguity_margin"],
                )
                continue
            yield Proposal(
                parameters={"residue": res.label},
                margins={
                    "dssp_code": dssp.codes.get(ri),
                    "run_length": run_length,
                    "distance_to_run_edge": edge_distance,
                },
                reasons=[
                    f"{res.label} is {klass} ({dssp.codes.get(ri)}), {edge_distance} residues "
                    f"inside a run of {run_length}"
                ],
                criteria_passed=["stable_run", "backbone_complete"],
                required_labels=[res.label],
                tag=klass,
                rank=-(run_length + edge_distance),
            )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        ri = structure.find_index(parameters["residue"])
        if ri is None:
            raise ValueError(f"residue {parameters['residue']} absent")
        klass = analysis.dssp.three_state.get(ri)
        if klass is None:
            raise ValueError(f"no DSSP assignment for {parameters['residue']}")
        return OracleResult(
            gold_answer={"value": klass},
            evidence={
                "dssp_code": analysis.dssp.codes.get(ri),
                "algorithm": analysis.dssp.algorithm,
                "run": analysis.secondary_runs.get(ri),
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["S04"].format(residue=parameters["residue"])

    def prompt_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {"categories": self.CATEGORIES}


class S05FoldClass(Generator):
    family = "S05"
    version = "1.0.0"
    answer_schema = "category"
    level = "global"
    croppable = False           # defined over a complete chain

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        if structure.atom_count > MAX_UNCROPPABLE_ATOMS:
            yield Rejection(
                "structure_too_large_for_uncroppable_family",
                {"atoms": structure.atom_count, "limit": MAX_UNCROPPABLE_ATOMS},
            )
            return
        for chain in structure.protein_chains:
            fractions = ctx.analysis.dssp.chain_fractions(structure, chain)
            if fractions["n_assigned"] < 40:
                yield Rejection(
                    "chain_too_short_for_fold_class",
                    {"chain": chain, "assigned": fractions["n_assigned"]},
                )
                continue
            label = fold_class(fractions, ctx.definitions)
            if label is None:
                yield Rejection(
                    "fold_class_undefined",
                    {
                        "chain": chain,
                        "helix_fraction": round(fractions["helix_fraction"], 3),
                        "beta_fraction": round(fractions["beta_fraction"], 3),
                    },
                    ["inside_ambiguity_margin"],
                )
                continue
            yield Proposal(
                parameters={"chain": chain},
                margins=fractions,
                reasons=[
                    f"chain {chain}: helix {fractions['helix_fraction']:.2f}, "
                    f"beta {fractions['beta_fraction']:.2f} -> {label}"
                ],
                criteria_passed=["fold_class_defined"],
                tag=label,
                rank=0.0,
            )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        chain = parameters["chain"]
        fractions = analysis.dssp.chain_fractions(structure, chain)
        label = fold_class(fractions, definitions)
        if label is None:
            raise ValueError(f"chain {chain} satisfies none of the A.16 fold classes")
        return OracleResult(gold_answer={"value": label}, evidence=fractions)

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["S05"].format(chain=parameters["chain"])

    def prompt_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "categories": [
                "predominantly alpha helical",
                "predominantly beta sheet",
                "mixed alpha/beta",
            ]
        }


class S06LigandContacts(Generator):
    family = "S06"
    version = "1.0.0"
    answer_schema = "residue_set"
    level = "local"
    croppable = True
    crop_radius = 15.0
    needs_ligand = True

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        cutoff = float(ctx.definitions.get("ligand_contact.heavy_atom_cutoff"))
        margin = float(ctx.definitions.get("distance.set_question_negative_margin"))
        if ctx.spec.source_type == "afdb":
            yield Rejection(
                "afdb_not_eligible_for_ligand_questions", {}, ["uncertain_biological_assembly"]
            )
            return
        for ri, res in enumerate(structure.residues):
            if res.entity is not EntityType.LIGAND:
                continue
            if len(res.atoms) < 6:
                yield Rejection(
                    "ligand_too_small", {"ligand": res.label, "atoms": len(res.atoms)}
                )
                continue
            contacts = ligand_contacts(structure, ri, ctx.definitions)
            if not 3 <= len(contacts) <= 35:
                yield Rejection(
                    "ligand_contact_count_out_of_range",
                    {"ligand": res.label, "n_contacts": len(contacts)},
                )
                continue
            nearest_excluded = nearest_excluded_ligand_contact(structure, ri, ctx.definitions)
            if nearest_excluded is not None and nearest_excluded < cutoff + margin:
                yield Rejection(
                    "ligand_contact_negative_margin",
                    {
                        "ligand": res.label,
                        "nearest_excluded": round(nearest_excluded, 3),
                        "required": cutoff + margin,
                    },
                    ["inside_ambiguity_margin"],
                )
                continue
            labels = [structure.residues[c.i].label for c in contacts]
            yield Proposal(
                parameters={"ligand": res.label},
                margins={
                    "n_contacts": len(contacts),
                    "max_contact_distance": max(c.min_distance for c in contacts),
                    "nearest_excluded_distance": nearest_excluded,
                },
                reasons=[
                    f"{res.label} contacts {len(contacts)} residues; nearest excluded residue "
                    f"at {nearest_excluded if nearest_excluded is None else round(nearest_excluded, 3)} A"
                ],
                criteria_passed=["negative_example_margin", "curator_selected_ligand"],
                required_labels=[res.label, *labels],
                crop_centers=[res.label],
                rank=len(contacts),
            )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        ri = structure.find_index(parameters["ligand"])
        if ri is None:
            raise ValueError(f"ligand {parameters['ligand']} absent")
        contacts = ligand_contacts(structure, ri, definitions)
        return OracleResult(
            gold_answer={"value": sorted({structure.residues[c.i].label for c in contacts})},
            evidence={
                "contacts": [
                    {
                        "residue": structure.residues[c.i].label,
                        "distance": round(c.min_distance, 3),
                        "atoms": [c.atom_i, c.atom_j],
                    }
                    for c in contacts
                ],
                "ligand_atoms": [a.name for a in structure.residues[ri].atoms],
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["S06"].format(ligand=parameters["ligand"])


class S07MetalCoordination(Generator):
    family = "S07"
    version = "1.0.0"
    answer_schema = "residue_set"
    level = "local"
    croppable = True
    crop_radius = 14.0
    needs_metal = True

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        cfg = ctx.definitions.get("metal_coordination")
        eligible = set(cfg["eligible_metals"])
        inside = float(cfg["inside_margin"])
        outside = float(cfg["outside_margin"])
        if ctx.spec.source_type == "afdb":
            yield Rejection("afdb_not_eligible_for_metal_questions", {})
            return
        for ri, res in enumerate(structure.residues):
            if res.entity is not EntityType.METAL:
                continue
            element = metal_element(res.name)
            if element not in eligible:
                yield Rejection("metal_not_eligible", {"metal": res.label, "element": element})
                continue
            coordination = metal_coordination(structure, ri, ctx.definitions)
            if not coordination.donors:
                yield Rejection("metal_without_protein_donors", {"metal": res.label})
                continue
            worst_accepted = max(d for _, _, d in coordination.donors)
            if worst_accepted > coordination.cutoff - inside:
                yield Rejection(
                    "metal_donor_inside_margin",
                    {
                        "metal": res.label,
                        "worst_accepted": round(worst_accepted, 3),
                        "required_max": round(coordination.cutoff - inside, 3),
                    },
                    ["inside_ambiguity_margin"],
                )
                continue
            if coordination.rejected:
                best_rejected = min(d for _, _, d in coordination.rejected)
                if best_rejected < coordination.cutoff + outside:
                    yield Rejection(
                        "metal_nondonor_inside_margin",
                        {
                            "metal": res.label,
                            "closest_rejected": round(best_rejected, 3),
                            "required_min": round(coordination.cutoff + outside, 3),
                        },
                        ["inside_ambiguity_margin"],
                    )
                    continue
            else:
                best_rejected = None
            labels = [structure.residues[i].label for i in coordination.residues]
            yield Proposal(
                parameters={"metal": res.label, "element": element, "cutoff": coordination.cutoff},
                margins={
                    "cutoff": coordination.cutoff,
                    "worst_accepted_donor": worst_accepted,
                    "closest_rejected_donor": best_rejected,
                    "coordination_number": coordination.coordination_number,
                },
                reasons=[
                    f"{res.label} has {coordination.coordination_number} protein donor atoms "
                    f"from {len(labels)} residues"
                ],
                criteria_passed=["donor_inside_margin", "nondonor_outside_margin"],
                required_labels=[res.label, *labels],
                crop_centers=[res.label],
                tag=element,
                rank=-coordination.coordination_number,
            )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        ri = structure.find_index(parameters["metal"])
        if ri is None:
            raise ValueError(f"metal {parameters['metal']} absent")
        coordination = metal_coordination(structure, ri, definitions)
        return OracleResult(
            gold_answer={
                "value": sorted({structure.residues[i].label for i in coordination.residues})
            },
            evidence={
                "cutoff": coordination.cutoff,
                "donors": [
                    {"residue": structure.residues[i].label, "atom": name, "distance": round(d, 3)}
                    for i, name, d in coordination.donors
                ],
                "rejected": [
                    {"residue": structure.residues[i].label, "atom": name, "distance": round(d, 3)}
                    for i, name, d in coordination.rejected[:5]
                ],
                "coordination_number": coordination.coordination_number,
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        element = parameters["element"]
        return QUESTION_TEMPLATES["S07"].format(
            metal=parameters["metal"],
            metal_name=element.capitalize(),
            cutoff=f"{float(parameters['cutoff']):.1f}",
        )


class S08DisulfidePartner(Generator):
    family = "S08"
    version = "1.0.0"
    answer_schema = "residue"
    level = "local"
    croppable = True
    crop_radius = 14.0

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        negative_min = float(ctx.definitions.get("disulfide.negative_min"))
        bonded: dict[int, tuple[int, float]] = {}
        for ss in ctx.analysis.disulfides:
            bonded[ss.i] = (ss.j, ss.distance)
            bonded[ss.j] = (ss.i, ss.distance)
        for ri, (partner, d) in sorted(bonded.items()):
            label = structure.residues[ri].label
            partner_label = structure.residues[partner].label
            third = self._closest_other_sg(structure, ri, partner)
            if third is not None and third < negative_min:
                yield Rejection(
                    "disulfide_third_cysteine_too_close",
                    {"residue": label, "closest_other_sg": round(third, 3), "required": negative_min},
                    ["inside_ambiguity_margin"],
                )
                continue
            yield Proposal(
                parameters={"residue": label},
                margins={
                    "sg_sg_distance": d,
                    "closest_other_sg": third,
                    "required_other_sg": negative_min,
                },
                reasons=[f"{label}-{partner_label} SG-SG distance {d:.3f} A"],
                criteria_passed=["single_disulfide_edge", "negative_example_margin"],
                required_labels=[label, partner_label],
                crop_centers=[label, partner_label],
                tag="interchain"
                if structure.residues[ri].chain != structure.residues[partner].chain
                else "intrachain",
                rank=d,
            )

    @staticmethod
    def _closest_other_sg(structure: Structure, ri: int, partner: int) -> float | None:
        """Closest SG other than the bonded partner's, for the negative margin."""
        query = structure.residues[ri].atom("SG")
        if query is None:
            return None
        best: float | None = None
        for rj, other in enumerate(structure.residues):
            if rj in (ri, partner) or not other.is_protein:
                continue
            if parent_of(other.orig_name) != "CYS":
                continue
            atom = other.atom("SG")
            if atom is None:
                continue
            d = float(np.linalg.norm(query.pos - atom.pos))
            if best is None or d < best:
                best = d
        return best

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        ri = structure.find_index(parameters["residue"])
        if ri is None:
            raise ValueError(f"residue {parameters['residue']} absent")
        found = [s for s in analysis.disulfides if ri in (s.i, s.j)]
        if len(found) != 1:
            raise ValueError(
                f"{parameters['residue']} participates in {len(found)} disulfides"
            )
        ss = found[0]
        partner = ss.j if ss.i == ri else ss.i
        return OracleResult(
            gold_answer={"value": structure.residues[partner].label},
            evidence={"sg_sg_distance": ss.distance, "interchain": ss.interchain},
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["S08"].format(residue=parameters["residue"])


class S09Chi1Rotamer(Generator):
    family = "S09"
    version = "1.0.0"
    answer_schema = "category"
    level = "local"
    croppable = True
    crop_radius = 12.0

    CATEGORIES = ["g+", "t", "g-"]

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        margin = float(ctx.definitions.get("chi1.boundary_margin"))
        for ri, res in enumerate(structure.residues):
            if not res.is_protein or not res.is_standard_aa:
                continue
            chi = compute_chi1(structure, ri, ctx.definitions)
            if chi is None:
                continue
            if not chi.rotamer:
                yield Rejection(
                    "chi1_within_boundary_margin",
                    {
                        "residue": res.label,
                        "chi1": round(chi.angle, 2),
                        "boundary_distance": round(chi.boundary_distance, 2),
                        "required": margin,
                    },
                    ["inside_ambiguity_margin"],
                )
                continue
            yield Proposal(
                parameters={"residue": res.label, "atom4": chi.atom4},
                margins={"chi1": chi.angle, "boundary_distance": chi.boundary_distance},
                reasons=[
                    f"{res.label} chi1 {chi.angle:.1f} deg is {chi.boundary_distance:.1f} deg "
                    f"from the nearest bin boundary -> {chi.rotamer}"
                ],
                criteria_passed=["outside_boundary_margin", "required_atoms_present"],
                required_labels=[res.label],
                crop_centers=[res.label],
                tag=chi.rotamer,
                rank=-chi.boundary_distance,
            )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        ri = structure.find_index(parameters["residue"])
        if ri is None:
            raise ValueError(f"residue {parameters['residue']} absent")
        chi = compute_chi1(structure, ri, definitions)
        if chi is None or not chi.rotamer:
            raise ValueError(f"chi1 for {parameters['residue']} is undefined or ambiguous")
        return OracleResult(
            gold_answer={"value": chi.rotamer},
            evidence={
                "chi1": chi.angle,
                "atom4": chi.atom4,
                "boundary_distance": chi.boundary_distance,
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        residue = parameters["residue"]
        atom4 = parameters.get("atom4")
        if atom4 is None:
            found = structure.find(residue)
            atom4 = chi1_atom4(found.orig_name) if found else "CG"
        return QUESTION_TEMPLATES["S09"].format(residue=residue, atom4=atom4)

    def prompt_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {"categories": self.CATEGORIES}


register(S01SaltBridgePartner())
register(S02PhosphorylatedResidue())
register(S03Burial())
register(S04SecondaryStructure())
register(S05FoldClass())
register(S06LigandContacts())
register(S07MetalCoordination())
register(S08DisulfidePartner())
register(S09Chi1Rotamer())
