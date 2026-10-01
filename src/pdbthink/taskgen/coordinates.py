"""Reconstruct oracle inputs using only the serialised coordinate records."""

from __future__ import annotations

import numpy as np

from ..chem import is_amino_acid
from ..config import Definitions
from ..generators import T01, Analysis, get_generator
from ..preprocessing.model import Atom, EntityType, Residue, Structure


def parse_structures(prompt: str, definitions: Definitions) -> list[Structure]:
    structures, residues, by_key = [], [], {}
    metals = set(definitions.get("structure_processing.metal_components"))
    for line in prompt.splitlines():
        if line == "END" and residues:
            structure = Structure(residues)
            structure.assign_polymer_indices()
            structures.append(structure)
            residues, by_key = [], {}
        elif line.startswith(("ATOM  ", "HETATM")):
            name, chain, number, icode = line[17:20].strip(), line[21], int(line[22:26]), line[26]
            key = chain, number, icode
            if key not in by_key:
                protein = is_amino_acid(name)
                entity = (
                    EntityType.PROTEIN
                    if protein
                    else (EntityType.METAL if name in metals else EntityType.LIGAND)
                )
                residue = Residue(
                    chain,
                    number,
                    name,
                    entity,
                    orig_name=name,
                    icode=icode,
                    polymer_kind="protein" if protein else None,
                )
                by_key[key] = residue
                residues.append(residue)
            if float(line[60:66]) != 0.0 or float(line[54:60]) != 1.0:
                raise ValueError("non-sanitised coordinate record")
            by_key[key].atoms.append(
                Atom(
                    name=line[12:16].strip(),
                    element=line[76:78].strip(),
                    pos=np.array([float(line[x : x + 8]) for x in (30, 38, 46)]),
                    serial=int(line[6:11]),
                    is_hetatm=line.startswith("HETATM"),
                )
            )
    if residues or not structures:
        raise ValueError("missing END record or no coordinate structures")
    return structures


def recompute(prompt: str, family: str, parameters: dict, definitions: Definitions) -> dict:
    structures = parse_structures(prompt, definitions)
    if family == "T01":
        if len(structures) != 2:
            raise ValueError("T01 requires exactly two structures")
        return T01.oracle(tuple(structures), parameters, definitions)["gold_answer"]
    if len(structures) != 1:
        raise ValueError("single-state task contains multiple structures")
    structure = structures[0]
    return (
        get_generator(family)
        .oracle(structure, parameters, definitions, Analysis(structure, definitions))
        .gold_answer
    )
