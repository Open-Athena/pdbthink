"""Parsing families P01-P03 (specification section 8).

These questions test only that a model can read the supplied file format, so
none of them may be cropped: the answer is defined over a complete chain or the
complete structure.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..config import Definitions
from ..preprocessing.model import Structure
from ..prompts.library import QUESTION_TEMPLATES
from ..scoring.scorers import DEFAULT_COORDINATE_TOLERANCE
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


class P01ChainList(Generator):
    family = "P01"
    version = "1.0.0"
    answer_schema = "string_set"
    level = "parsing"
    croppable = False

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        chains = structure.chains
        if len(chains) < 1:
            yield Rejection("no_chains", {}, ["multiple_valid_answers"])
            return
        if len({c.upper() for c in chains}) != len(chains):
            yield Rejection(
                "chain_ids_differ_only_by_case", {"chains": chains}, ["multiple_valid_answers"]
            )
            return
        if structure.atom_count > MAX_UNCROPPABLE_ATOMS:
            yield Rejection(
                "structure_too_large_for_uncroppable_family",
                {"atoms": structure.atom_count, "limit": MAX_UNCROPPABLE_ATOMS},
            )
            return
        yield Proposal(
            parameters={},
            margins={"n_chains": len(chains)},
            reasons=[f"{len(chains)} distinct chain identifiers are visible"],
            criteria_passed=["unique_answer", "no_crop"],
            rank=-len(chains),
        )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        chains = structure.chains
        return OracleResult(
            gold_answer={"value": sorted(chains)},
            evidence={
                "chains_in_file_order": chains,
                "residues_per_chain": {c: len(structure.chain_residues(c)) for c in chains},
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["P01"]


class P02ResidueCount(Generator):
    family = "P02"
    version = "1.0.0"
    answer_schema = "integer"
    level = "parsing"
    croppable = False

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        if structure.atom_count > MAX_UNCROPPABLE_ATOMS:
            yield Rejection(
                "structure_too_large_for_uncroppable_family",
                {"atoms": structure.atom_count, "limit": MAX_UNCROPPABLE_ATOMS},
            )
            return
        for chain in structure.protein_chains:
            count = len(structure.protein_chain_residues(chain))
            if count < 20:
                yield Rejection(
                    "chain_too_short", {"chain": chain, "residues": count}
                )
                continue
            yield Proposal(
                parameters={"chain": chain},
                margins={"residues": count},
                reasons=[f"chain {chain} has {count} retained protein residues"],
                criteria_passed=["unique_answer", "no_crop"],
                rank=-count,
            )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        chain = parameters["chain"]
        residues = structure.protein_chain_residues(chain)
        return OracleResult(
            gold_answer={"value": len(residues)},
            evidence={
                "chain": chain,
                "first_residue": residues[0].label if residues else None,
                "last_residue": residues[-1].label if residues else None,
                "non_protein_residues_in_chain": [
                    r.label for r in structure.chain_residues(chain) if not r.is_protein
                ],
            },
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["P02"].format(chain=parameters["chain"])


class P03AtomCoordinates(Generator):
    family = "P03"
    version = "1.0.0"
    answer_schema = "numeric_triple"
    level = "parsing"
    croppable = False

    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        structure = ctx.structure
        if structure.atom_count > MAX_UNCROPPABLE_ATOMS:
            yield Rejection(
                "structure_too_large_for_uncroppable_family",
                {"atoms": structure.atom_count, "limit": MAX_UNCROPPABLE_ATOMS},
            )
            return
        # Prefer side-chain atoms deep in the file: reading them requires
        # locating the correct record rather than the first or last line.
        candidates: list[tuple[float, str, str]] = []
        total = len(structure.residues)
        for ri, res in enumerate(structure.residues):
            if not res.is_protein or res.poly_index is None:
                continue
            position = ri / max(total - 1, 1)
            if not 0.15 <= position <= 0.85:
                continue
            for atom in res.sidechain_atoms:
                if atom.name in ("CB",):
                    continue
                candidates.append((abs(position - 0.5), res.label, atom.name))
        if not candidates:
            yield Rejection("no_suitable_atom", {})
            return
        candidates.sort()
        for rank, (offset, residue, atom) in enumerate(candidates[:24]):
            yield Proposal(
                parameters={"residue": residue, "atom": atom},
                margins={"file_position_offset": offset},
                reasons=[f"transcription of {residue}:{atom} from the middle of the file"],
                criteria_passed=["unique_answer", "no_crop"],
                required_labels=[residue],
                rank=rank,
            )

    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        label = f"{parameters['residue']}:{parameters['atom']}"
        found = structure.atom_by_label(label)
        if found is None:
            raise ValueError(f"atom {label} is absent from the displayed structure")
        _, atom = found
        return OracleResult(
            gold_answer={"value": [float(v) for v in atom.pos]},
            evidence={"atom": label, "element": atom.element},
        )

    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        return QUESTION_TEMPLATES["P03"].format(
            atom=f"{parameters['residue']}:{parameters['atom']}"
        )

    def prompt_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        # Transcription, not calculation: score each component tightly (section 8).
        return {"tolerance": DEFAULT_COORDINATE_TOLERANCE}


register(P01ChainList())
register(P02ResidueCount())
register(P03AtomCoordinates())
