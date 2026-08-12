"""Internal structure model.

A :class:`Structure` is a flat, ordered list of :class:`Residue` objects, each
holding heavy :class:`Atom` records. It deliberately carries only what the
benchmark needs, because everything else is stripped by A.2 before a structure
becomes model-visible.

Coordinates in a *displayed* structure are already rounded to three decimals and
already rotated, so every oracle computes gold answers from exactly the numbers
the model sees (specification A.1, "definition of done").
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ..chem import AA3_TO_1, is_amino_acid, metal_element, parent_of, three_to_one


class EntityType(str, Enum):
    PROTEIN = "protein"
    NUCLEIC = "nucleic"
    LIGAND = "ligand"
    METAL = "metal"
    WATER = "water"


@dataclass
class Atom:
    """One retained heavy atom."""

    name: str
    element: str
    pos: np.ndarray                 # shape (3,), Angstrom
    altloc: str = ""
    occupancy: float = 1.0
    bfactor: float = 0.0
    serial: int = 0
    is_hetatm: bool = False

    def copy(self) -> Atom:
        return replace(self, pos=np.array(self.pos, dtype=float))

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() in ("H", "D")


@dataclass
class Residue:
    """One residue or non-polymer component."""

    chain: str
    seq_id: int
    name: str                       # model-visible component code
    entity: EntityType
    atoms: list[Atom] = field(default_factory=list)
    icode: str = " "
    orig_name: str = ""             # pre-anonymisation code (private metadata)
    polymer_kind: str | None = None  # "protein" | "DNA" | "RNA" | None
    poly_index: int | None = None   # position within the chain polymer (A.5)

    def __post_init__(self) -> None:
        if not self.orig_name:
            self.orig_name = self.name

    # -- identity ----------------------------------------------------------
    @property
    def is_protein(self) -> bool:
        return self.entity is EntityType.PROTEIN

    @property
    def one_letter(self) -> str:
        return three_to_one(self.orig_name if self.is_protein else self.name)

    @property
    def label(self) -> str:
        """Model-visible identifier, e.g. ``A:V22``, ``L:L2401``, ``M:ZN501`` (A.5)."""
        if self.is_protein:
            return f"{self.chain}:{self.one_letter}{self.seq_id}"
        return f"{self.chain}:{self.name}{self.seq_id}"

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.seq_id, self.icode)

    @property
    def is_standard_aa(self) -> bool:
        return parent_of(self.orig_name) in AA3_TO_1

    # -- atoms -------------------------------------------------------------
    def atom(self, name: str) -> Atom | None:
        for a in self.atoms:
            if a.name == name:
                return a
        return None

    def require(self, *names: str) -> list[Atom] | None:
        """All named atoms, or ``None`` if any is missing (A.34)."""
        out = []
        for n in names:
            a = self.atom(n)
            if a is None:
                return None
            out.append(a)
        return out

    def has_atoms(self, *names: str) -> bool:
        return self.require(*names) is not None

    @property
    def sidechain_atoms(self) -> list[Atom]:
        return [a for a in self.atoms if a.name not in ("N", "CA", "C", "O", "OXT")]

    @property
    def element_symbol(self) -> str:
        """Element of a monatomic component (metals)."""
        return metal_element(self.name)

    def copy(self) -> Residue:
        return replace(self, atoms=[a.copy() for a in self.atoms])

    def coords(self) -> np.ndarray:
        return np.array([a.pos for a in self.atoms], dtype=float).reshape(-1, 3)


@dataclass
class AtomIndex:
    """Vectorised view over every atom of a structure."""

    coords: np.ndarray              # (N, 3)
    residue_of: np.ndarray          # (N,) int, index into Structure.residues
    atom_of: np.ndarray             # (N,) int, index into Residue.atoms
    names: list[str]
    elements: list[str]
    entity: list[EntityType]

    def __len__(self) -> int:
        return len(self.names)

    @cached_property
    def tree(self) -> cKDTree:
        return cKDTree(self.coords)

    def within(self, point: Iterable[float], radius: float) -> list[int]:
        return sorted(self.tree.query_ball_point(np.asarray(point, dtype=float), radius))


class Structure:
    """An ordered collection of residues plus private provenance metadata."""

    def __init__(self, residues: list[Residue], meta: dict[str, Any] | None = None) -> None:
        self.residues = residues
        self.meta: dict[str, Any] = dict(meta or {})

    # -- construction ------------------------------------------------------
    def copy(self) -> Structure:
        return Structure([r.copy() for r in self.residues], dict(self.meta))

    def subset(self, keep: Iterable[int]) -> Structure:
        keep = sorted(set(keep))
        sub = Structure([self.residues[i].copy() for i in keep], dict(self.meta))
        sub.assign_polymer_indices()
        return sub

    def assign_polymer_indices(self) -> None:
        """Number polymer residues within each chain in file order (A.5)."""
        counters: dict[str, int] = {}
        for res in self.residues:
            if res.polymer_kind is None:
                res.poly_index = None
                continue
            idx = counters.get(res.chain, 0)
            res.poly_index = idx
            counters[res.chain] = idx + 1

    # -- iteration ---------------------------------------------------------
    def __iter__(self) -> Iterator[Residue]:
        return iter(self.residues)

    def __len__(self) -> int:
        return len(self.residues)

    @property
    def chains(self) -> list[str]:
        """Unique chain IDs in file order (A.5: as visible in the rendering)."""
        seen: list[str] = []
        for r in self.residues:
            if r.chain not in seen:
                seen.append(r.chain)
        return seen

    def chain_residues(self, chain: str) -> list[Residue]:
        return [r for r in self.residues if r.chain == chain]

    @property
    def protein_residues(self) -> list[Residue]:
        return [r for r in self.residues if r.is_protein]

    def protein_chain_residues(self, chain: str) -> list[Residue]:
        return [r for r in self.residues if r.is_protein and r.chain == chain]

    @property
    def protein_chains(self) -> list[str]:
        seen: list[str] = []
        for r in self.residues:
            if r.is_protein and r.chain not in seen:
                seen.append(r.chain)
        return seen

    @property
    def ligands(self) -> list[Residue]:
        return [r for r in self.residues if r.entity is EntityType.LIGAND]

    @property
    def metals(self) -> list[Residue]:
        return [r for r in self.residues if r.entity is EntityType.METAL]

    @property
    def atom_count(self) -> int:
        return sum(len(r.atoms) for r in self.residues)

    # -- lookup ------------------------------------------------------------
    @cached_property
    def _by_label(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i, r in enumerate(self.residues):
            out.setdefault(r.label, i)
        return out

    @cached_property
    def _by_key(self) -> dict[tuple[str, int, str], int]:
        return {r.key: i for i, r in enumerate(self.residues)}

    def find(self, label: str) -> Residue | None:
        i = self._by_label.get(label)
        return None if i is None else self.residues[i]

    def find_index(self, label: str) -> int | None:
        return self._by_label.get(label)

    def by_number(self, chain: str, seq_id: int) -> Residue | None:
        i = self._by_key.get((chain, seq_id, " "))
        if i is not None:
            return self.residues[i]
        for r in self.residues:
            if r.chain == chain and r.seq_id == seq_id:
                return r
        return None

    def atom_by_label(self, label: str) -> tuple[Residue, Atom] | None:
        """Resolve ``A:H57:NE2`` to its residue and atom."""
        parts = label.split(":")
        if len(parts) != 3:
            return None
        res = self.find(f"{parts[0]}:{parts[1]}")
        if res is None:
            return None
        atom = res.atom(parts[2])
        return None if atom is None else (res, atom)

    # -- vectorised access -------------------------------------------------
    @cached_property
    def index(self) -> AtomIndex:
        coords, res_of, atom_of, names, elements, entities = [], [], [], [], [], []
        for ri, res in enumerate(self.residues):
            for ai, atom in enumerate(res.atoms):
                coords.append(atom.pos)
                res_of.append(ri)
                atom_of.append(ai)
                names.append(atom.name)
                elements.append(atom.element)
                entities.append(res.entity)
        return AtomIndex(
            coords=np.array(coords, dtype=float).reshape(-1, 3),
            residue_of=np.array(res_of, dtype=int),
            atom_of=np.array(atom_of, dtype=int),
            names=names,
            elements=elements,
            entity=entities,
        )

    def invalidate(self) -> None:
        """Drop cached views after an in-place mutation."""
        for attr in ("index", "_by_label", "_by_key", "sequences"):
            self.__dict__.pop(attr, None)

    # -- sequences ---------------------------------------------------------
    @cached_property
    def sequences(self) -> dict[str, str]:
        """One-letter sequence per protein chain, in polymer order."""
        out: dict[str, list[str]] = {}
        for r in self.residues:
            if r.is_protein:
                out.setdefault(r.chain, []).append(r.one_letter)
        return {k: "".join(v) for k, v in out.items()}

    def polymer_neighbours(self, res: Residue, offset: int) -> Residue | None:
        """Residue ``offset`` positions away in polymer order within the chain."""
        if res.poly_index is None:
            return None
        target = res.poly_index + offset
        for r in self.residues:
            if r.chain == res.chain and r.poly_index == target:
                return r
        return None

    # -- transforms --------------------------------------------------------
    def transformed(self, rotation: np.ndarray, translation: np.ndarray) -> Structure:
        """Apply ``x' = R (x + t)`` and return a new structure."""
        out = self.copy()
        for res in out.residues:
            for atom in res.atoms:
                atom.pos = rotation @ (atom.pos + translation)
        out.invalidate()
        return out

    def rounded(self, decimals: int = 3) -> Structure:
        """Round every coordinate to the displayed precision (A.4)."""
        out = self.copy()
        fmt = f"%.{decimals}f"
        for res in out.residues:
            for atom in res.atoms:
                atom.pos = np.array([float(fmt % v) for v in atom.pos], dtype=float)
        out.invalidate()
        return out

    def centroid(self) -> np.ndarray:
        return self.index.coords.mean(axis=0)

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "chains": self.chains,
            "residues": len(self.residues),
            "protein_residues": len(self.protein_residues),
            "atoms": self.atom_count,
            "ligands": [r.label for r in self.ligands],
            "metals": [r.label for r in self.metals],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Structure(chains={self.chains}, residues={len(self.residues)}, "
            f"atoms={self.atom_count})"
        )


def polymer_kind_for(resname: str, gemmi_kind: str | None = None) -> str | None:
    """Classify a component as protein / DNA / RNA polymer, or non-polymer."""
    from ..chem import DNA_RESIDUES, RNA_RESIDUES

    name = resname.strip().upper()
    if is_amino_acid(name):
        return "protein"
    if name in DNA_RESIDUES:
        return "DNA"
    if name in RNA_RESIDUES and gemmi_kind in ("RNA", "polyribonucleotide"):
        return "RNA"
    return None
