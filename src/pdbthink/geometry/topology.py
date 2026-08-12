"""Covalent topology from a versioned component dictionary (A.8).

Bonds come from three sources:

1. intra-residue bonds in :mod:`pdbthink.chem` (``ccd_lite_v1``),
2. standard polymer links, accepted only when the linking atoms are actually
   within bonding distance so chain breaks are not bridged,
3. geometrically identified disulfides (A.20), which A.8 treats as covalent.

Components outside the dictionary (anonymised ligands) contribute no bonds; A.8
restricts nearest-nonbonded questions to standard residues for exactly this
reason.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..chem import CCD_LITE_VERSION, component_bonds, has_known_topology
from ..preprocessing.model import Structure

PEPTIDE_BOND_MAX = 2.0          # C(i)-N(i+1)
NUCLEIC_BOND_MAX = 2.2          # O3'(i)-P(i+1)


@dataclass
class Topology:
    """Atom-level covalent graph over the global atom indices of a structure."""

    neighbours: dict[int, set[int]] = field(default_factory=dict)
    dictionary_version: str = CCD_LITE_VERSION
    unknown_components: set[str] = field(default_factory=set)
    disulfide_pairs: list[tuple[int, int]] = field(default_factory=list)

    def bonded(self, i: int, j: int) -> bool:
        return j in self.neighbours.get(i, ())

    def separation(self, i: int, j: int, max_depth: int = 3) -> int | None:
        """Number of bonds between two atoms, or ``None`` beyond ``max_depth``."""
        if i == j:
            return 0
        seen = {i}
        frontier = deque([(i, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            for nb in self.neighbours.get(node, ()):
                if nb == j:
                    return depth + 1
                if nb not in seen:
                    seen.add(nb)
                    frontier.append((nb, depth + 1))
        return None

    def within(self, i: int, max_depth: int) -> set[int]:
        """All atoms at most ``max_depth`` bonds away, excluding ``i`` itself."""
        seen = {i}
        out: set[int] = set()
        frontier = deque([(i, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            for nb in self.neighbours.get(node, ()):
                if nb not in seen:
                    seen.add(nb)
                    out.add(nb)
                    frontier.append((nb, depth + 1))
        return out


def build_topology(
    structure: Structure, disulfides: list[tuple[int, int]] | None = None
) -> Topology:
    """Build the covalent graph for a structure.

    ``disulfides`` are residue-index pairs previously identified by
    :func:`pdbthink.geometry.contacts.find_disulfides`.
    """
    index = structure.index
    atom_id: dict[tuple[int, str], int] = {}
    for global_i, (res_i, atom_i) in enumerate(zip(index.residue_of, index.atom_of)):
        atom_id[(int(res_i), structure.residues[int(res_i)].atoms[int(atom_i)].name)] = global_i

    topo = Topology()
    unknown: set[str] = set()

    def link(a: int, b: int) -> None:
        topo.neighbours.setdefault(a, set()).add(b)
        topo.neighbours.setdefault(b, set()).add(a)

    for ri, res in enumerate(structure.residues):
        if not has_known_topology(res.orig_name):
            if res.polymer_kind == "protein" or res.entity.value == "ligand":
                unknown.add(res.orig_name)
            continue
        for a, b in component_bonds(res.orig_name):
            ia, ib = atom_id.get((ri, a)), atom_id.get((ri, b))
            if ia is not None and ib is not None:
                link(ia, ib)

    # Polymer links, validated by distance so chain breaks stay unbridged.
    by_chain: dict[str, list[int]] = {}
    for ri, res in enumerate(structure.residues):
        if res.polymer_kind is not None:
            by_chain.setdefault(res.chain, []).append(ri)
    for chain_residues in by_chain.values():
        ordered = sorted(chain_residues, key=lambda i: structure.residues[i].poly_index or 0)
        for first, second in zip(ordered, ordered[1:]):
            r1, r2 = structure.residues[first], structure.residues[second]
            if r1.polymer_kind != r2.polymer_kind:
                continue
            if r1.polymer_kind == "protein":
                a, b, limit = "C", "N", PEPTIDE_BOND_MAX
            else:
                a, b, limit = "O3'", "P", NUCLEIC_BOND_MAX
            atom_a, atom_b = r1.atom(a), r2.atom(b)
            if atom_a is None or atom_b is None:
                continue
            if float(np.linalg.norm(atom_a.pos - atom_b.pos)) <= limit:
                link(atom_id[(first, a)], atom_id[(second, b)])

    for ri, rj in disulfides or []:
        ia, ib = atom_id.get((ri, "SG")), atom_id.get((rj, "SG"))
        if ia is not None and ib is not None:
            link(ia, ib)
            topo.disulfide_pairs.append((ri, rj))

    topo.unknown_components = unknown
    return topo
