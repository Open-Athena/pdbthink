"""Contacts, salt bridges, disulfides, metal coordination, clashes and graphs.

Implements A.9, A.10, A.11, A.18, A.19, A.20, A.23, A.24, A.25 and A.26. Every
function takes the *displayed* structure so that gold labels are computed from
the coordinates the model actually reads.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..chem import (
    SALT_BRIDGE_NEGATIVE,
    SALT_BRIDGE_POSITIVE,
    metal_element,
    parent_of,
    vdw_radius,
)
from ..config import Definitions
from ..preprocessing.model import EntityType, Residue, Structure
from .core import pairwise_min_distance
from .topology import Topology


@dataclass(frozen=True)
class Contact:
    """A residue-residue (or residue-entity) contact with its witness atoms."""

    i: int
    j: int
    min_distance: float
    atom_i: str
    atom_j: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.i, self.j) if self.i <= self.j else (self.j, self.i)


@dataclass(frozen=True)
class SaltBridge:
    i: int
    j: int
    positive_atom: str
    negative_atom: str
    distance: float


@dataclass(frozen=True)
class Disulfide:
    i: int
    j: int
    distance: float
    interchain: bool


@dataclass
class MetalCoordination:
    metal_index: int
    donors: list[tuple[int, str, float]] = field(default_factory=list)   # residue, atom, distance
    rejected: list[tuple[int, str, float]] = field(default_factory=list)
    cutoff: float = 0.0

    @property
    def residues(self) -> list[int]:
        seen: list[int] = []
        for ri, _, _ in self.donors:
            if ri not in seen:
                seen.append(ri)
        return seen

    @property
    def coordination_number(self) -> int:
        """Protein donor *atoms*, not residues (A.19)."""
        return len(self.donors)


@dataclass(frozen=True)
class ClashPair:
    i: int
    j: int
    atom_i: str
    atom_j: str
    distance: float
    overlap: float


# --------------------------------------------------------------------------- #
# Basic distances
# --------------------------------------------------------------------------- #

def min_heavy_distance(a: Residue, b: Residue) -> tuple[float, str, str]:
    """Minimum heavy-atom distance between two residues and the atom pair."""
    ca, cb = a.coords(), b.coords()
    d, i, j = pairwise_min_distance(ca, cb)
    return d, a.atoms[i].name, b.atoms[j].name


def _atom_pairs_within(structure: Structure, cutoff: float) -> Iterable[tuple[int, int, float]]:
    index = structure.index
    pairs = index.tree.query_pairs(cutoff, output_type="ndarray")
    if len(pairs) == 0:
        return []
    d = np.linalg.norm(index.coords[pairs[:, 0]] - index.coords[pairs[:, 1]], axis=1)
    return zip(pairs[:, 0].tolist(), pairs[:, 1].tolist(), d.tolist())


def _residue_pair_minima(
    structure: Structure, cutoff: float, *, entities: tuple[EntityType, ...] | None = None
) -> dict[tuple[int, int], tuple[float, str, str]]:
    """Minimum distance per residue pair over all atom pairs within ``cutoff``."""
    index = structure.index
    out: dict[tuple[int, int], tuple[float, str, str]] = {}
    for ai, aj, d in _atom_pairs_within(structure, cutoff):
        ri, rj = int(index.residue_of[ai]), int(index.residue_of[aj])
        if ri == rj:
            continue
        if entities is not None:
            if structure.residues[ri].entity not in entities:
                continue
            if structure.residues[rj].entity not in entities:
                continue
        key = (ri, rj) if ri < rj else (rj, ri)
        name_i, name_j = index.names[ai], index.names[aj]
        if key == (rj, ri):
            name_i, name_j = name_j, name_i
        prev = out.get(key)
        if prev is None or d < prev[0]:
            out[key] = (d, name_i, name_j)
    return out


# --------------------------------------------------------------------------- #
# A.9 generic residue-residue contact
# --------------------------------------------------------------------------- #

def polymer_separation(structure: Structure, i: int, j: int) -> int | None:
    """``|i - j|`` in polymer positions when both residues share a chain."""
    a, b = structure.residues[i], structure.residues[j]
    if a.chain != b.chain or a.poly_index is None or b.poly_index is None:
        return None
    return abs(a.poly_index - b.poly_index)


def residue_contacts(
    structure: Structure,
    definitions: Definitions,
    *,
    apply_sequence_exclusion: bool = True,
) -> list[Contact]:
    """All protein-protein residue contacts (A.9)."""
    cfg = definitions.get("residue_contact")
    cutoff = float(cfg["heavy_atom_cutoff"])
    min_sep = int(cfg["intrachain_min_polymer_separation"])
    minima = _residue_pair_minima(structure, cutoff, entities=(EntityType.PROTEIN,))
    out: list[Contact] = []
    for (i, j), (d, ai, aj) in minima.items():
        if apply_sequence_exclusion:
            sep = polymer_separation(structure, i, j)
            if sep is not None and sep < min_sep:
                continue
        out.append(Contact(i=i, j=j, min_distance=d, atom_i=ai, atom_j=aj))
    out.sort(key=lambda c: (c.i, c.j))
    return out


def residue_pair_distance(structure: Structure, i: int, j: int) -> tuple[float, str, str]:
    return min_heavy_distance(structure.residues[i], structure.residues[j])


def min_distance_to_residues(
    structure: Structure, sources: Sequence[int], targets: Sequence[int]
) -> dict[int, float]:
    """Minimum heavy-atom distance from each source residue to any target residue.

    A single KD-tree query rather than a pairwise scan, which matters for
    interface and bridging margins on large assemblies.
    """
    if not sources or not targets:
        return {}
    target_coords = np.vstack([structure.residues[i].coords() for i in targets])
    tree = cKDTree(target_coords)
    out: dict[int, float] = {}
    for ri in sources:
        coords = structure.residues[ri].coords()
        distances, _ = tree.query(coords, k=1)
        out[ri] = float(np.min(distances))
    return out


# --------------------------------------------------------------------------- #
# A.23 interface contacts
# --------------------------------------------------------------------------- #

def interface_contacts(
    structure: Structure, chain_a: str, chain_b: str, definitions: Definitions
) -> list[Contact]:
    """Cross-chain protein contacts between two chains (A.23)."""
    cutoff = float(definitions.get("interface.heavy_atom_cutoff"))
    minima = _residue_pair_minima(structure, cutoff, entities=(EntityType.PROTEIN,))
    out: list[Contact] = []
    for (i, j), (d, ai, aj) in minima.items():
        ci, cj = structure.residues[i].chain, structure.residues[j].chain
        if {ci, cj} != {chain_a, chain_b}:
            continue
        if ci == chain_a:
            out.append(Contact(i=i, j=j, min_distance=d, atom_i=ai, atom_j=aj))
        else:
            out.append(Contact(i=j, j=i, min_distance=d, atom_i=aj, atom_j=ai))
    out.sort(key=lambda c: (c.min_distance, c.i, c.j))
    return out


# --------------------------------------------------------------------------- #
# A.10 protein-ligand contact
# --------------------------------------------------------------------------- #

def ligand_contacts(
    structure: Structure, ligand_index: int, definitions: Definitions
) -> list[Contact]:
    """Protein residues contacting a specified ligand, nearest first (A.10)."""
    cutoff = float(definitions.get("ligand_contact.heavy_atom_cutoff"))
    ligand = structure.residues[ligand_index]
    lig_coords = ligand.coords()
    index = structure.index
    hits: dict[int, tuple[float, str, str]] = {}
    for li, lig_atom in enumerate(ligand.atoms):
        for ai in index.within(lig_atom.pos, cutoff):
            ri = int(index.residue_of[ai])
            if ri == ligand_index:
                continue
            res = structure.residues[ri]
            if not res.is_protein:
                continue
            d = float(np.linalg.norm(index.coords[ai] - lig_coords[li]))
            prev = hits.get(ri)
            if prev is None or d < prev[0]:
                hits[ri] = (d, index.names[ai], lig_atom.name)
    out = [
        Contact(i=ri, j=ligand_index, min_distance=d, atom_i=an, atom_j=ln)
        for ri, (d, an, ln) in hits.items()
    ]
    out.sort(key=lambda c: (c.min_distance, c.i))
    return out


def nearest_excluded_ligand_contact(
    structure: Structure, ligand_index: int, definitions: Definitions, *, search: float = 6.0
) -> float | None:
    """Distance of the closest protein residue that is *not* within the cutoff.

    Used to prove the negative-example margin of A.4 for list questions.
    """
    cutoff = float(definitions.get("ligand_contact.heavy_atom_cutoff"))
    ligand = structure.residues[ligand_index]
    index = structure.index
    best: float | None = None
    per_residue: dict[int, float] = {}
    for li, lig_atom in enumerate(ligand.atoms):
        for ai in index.within(lig_atom.pos, search):
            ri = int(index.residue_of[ai])
            if ri == ligand_index or not structure.residues[ri].is_protein:
                continue
            d = float(np.linalg.norm(index.coords[ai] - ligand.coords()[li]))
            if ri not in per_residue or d < per_residue[ri]:
                per_residue[ri] = d
    for d in per_residue.values():
        if d > cutoff and (best is None or d < best):
            best = d
    return best


# --------------------------------------------------------------------------- #
# A.11 salt bridges
# --------------------------------------------------------------------------- #

def salt_bridges(structure: Structure, definitions: Definitions) -> list[SaltBridge]:
    cfg = definitions.get("salt_bridge")
    cutoff = float(cfg["cutoff"])
    positives = {k: tuple(v) for k, v in cfg["positive_atoms"].items()}
    negatives = {k: tuple(v) for k, v in cfg["negative_atoms"].items()}

    pos_atoms: list[tuple[int, str, np.ndarray]] = []
    neg_atoms: list[tuple[int, str, np.ndarray]] = []
    for ri, res in enumerate(structure.residues):
        if not res.is_protein:
            continue
        parent = parent_of(res.orig_name)
        for name in positives.get(parent, ()):
            atom = res.atom(name)
            if atom is not None:
                pos_atoms.append((ri, name, atom.pos))
        for name in negatives.get(parent, ()):
            atom = res.atom(name)
            if atom is not None:
                neg_atoms.append((ri, name, atom.pos))

    best: dict[tuple[int, int], SaltBridge] = {}
    for ri, pname, ppos in pos_atoms:
        for rj, nname, npos in neg_atoms:
            if ri == rj:
                continue
            d = float(np.linalg.norm(ppos - npos))
            if d > cutoff:
                continue
            key = (ri, rj)
            prev = best.get(key)
            if prev is None or d < prev.distance:
                best[key] = SaltBridge(
                    i=ri, j=rj, positive_atom=pname, negative_atom=nname, distance=d
                )
    out = sorted(best.values(), key=lambda s: (s.distance, s.i, s.j))
    return out


def min_charged_distance(structure: Structure, i: int, j: int) -> float | None:
    """Minimum qualifying positive-negative atom distance between two residues."""
    a, b = structure.residues[i], structure.residues[j]
    best: float | None = None
    for first, second in ((a, b), (b, a)):
        pos = SALT_BRIDGE_POSITIVE.get(parent_of(first.orig_name), ())
        neg = SALT_BRIDGE_NEGATIVE.get(parent_of(second.orig_name), ())
        for pn in pos:
            pa = first.atom(pn)
            if pa is None:
                continue
            for nn in neg:
                na = second.atom(nn)
                if na is None:
                    continue
                d = float(np.linalg.norm(pa.pos - na.pos))
                if best is None or d < best:
                    best = d
    return best


# --------------------------------------------------------------------------- #
# A.20 disulfides
# --------------------------------------------------------------------------- #

def find_disulfides(structure: Structure, definitions: Definitions) -> list[Disulfide]:
    """Geometric disulfide detection; ``SSBOND`` records are never consulted."""
    cfg = definitions.get("disulfide")
    cutoff = float(cfg["sg_sg_max"])
    sgs: list[tuple[int, np.ndarray]] = []
    for ri, res in enumerate(structure.residues):
        if res.is_protein and parent_of(res.orig_name) == "CYS":
            atom = res.atom("SG")
            if atom is not None:
                sgs.append((ri, atom.pos))

    candidates: list[Disulfide] = []
    for a in range(len(sgs)):
        for b in range(a + 1, len(sgs)):
            ri, pi = sgs[a]
            rj, pj = sgs[b]
            d = float(np.linalg.norm(pi - pj))
            if d <= cutoff:
                candidates.append(
                    Disulfide(
                        i=ri,
                        j=rj,
                        distance=d,
                        interchain=structure.residues[ri].chain != structure.residues[rj].chain,
                    )
                )

    if not cfg["one_edge_per_cysteine"]:
        return sorted(candidates, key=lambda s: (s.i, s.j))

    used: set[int] = set()
    out: list[Disulfide] = []
    for ss in sorted(candidates, key=lambda s: (s.distance, s.i, s.j)):
        if ss.i in used or ss.j in used:
            continue
        used.update((ss.i, ss.j))
        out.append(ss)
    return sorted(out, key=lambda s: (s.i, s.j))


def free_cysteine_min_sg_distance(structure: Structure, residue_index: int) -> float | None:
    """Closest other ``SG`` to a cysteine, for negative-example margins (A.20)."""
    res = structure.residues[residue_index]
    sg = res.atom("SG")
    if sg is None:
        return None
    best: float | None = None
    for ri, other in enumerate(structure.residues):
        if ri == residue_index or not other.is_protein:
            continue
        if parent_of(other.orig_name) != "CYS":
            continue
        atom = other.atom("SG")
        if atom is None:
            continue
        d = float(np.linalg.norm(sg.pos - atom.pos))
        if best is None or d < best:
            best = d
    return best


# --------------------------------------------------------------------------- #
# A.19 metal coordination
# --------------------------------------------------------------------------- #

def metal_coordination(
    structure: Structure, metal_index: int, definitions: Definitions
) -> MetalCoordination:
    cfg = definitions.get("metal_coordination")
    metal = structure.residues[metal_index]
    element = metal_element(metal.name)
    cutoff = cfg["max_donor_distance"].get(element)
    if cutoff is None:
        raise ValueError(f"{metal.label}: element {element} is not an eligible metal (A.19)")
    cutoff = float(cutoff)
    donors = tuple(cfg["donor_elements"])
    search = cutoff + 1.5
    pos = metal.atoms[0].pos
    index = structure.index

    result = MetalCoordination(metal_index=metal_index, cutoff=cutoff)
    for ai in index.within(pos, search):
        ri = int(index.residue_of[ai])
        if ri == metal_index:
            continue
        res = structure.residues[ri]
        if not res.is_protein:
            continue
        if index.elements[ai].upper() not in donors:
            continue
        d = float(np.linalg.norm(index.coords[ai] - pos))
        entry = (ri, index.names[ai], d)
        (result.donors if d <= cutoff else result.rejected).append(entry)
    result.donors.sort(key=lambda t: (t[2], t[0], t[1]))
    result.rejected.sort(key=lambda t: (t[2], t[0], t[1]))
    return result


# --------------------------------------------------------------------------- #
# A.18 severe nonbonded clashes
# --------------------------------------------------------------------------- #

def find_clashes(
    structure: Structure,
    topology: Topology,
    definitions: Definitions,
    *,
    exclude_metal_contacts: bool = True,
) -> list[ClashPair]:
    """Residue-pair clashes, most severe first (A.18)."""
    cfg = definitions.get("clash")
    tolerance = float(cfg["overlap_tolerance"])
    max_sep = int(cfg["exclude_bond_separation_upto"])
    index = structure.index
    # The widest possible pair is 2 * max(radius) - tolerance.
    search = 2 * max(vdw_radius(e) for e in set(index.elements)) - tolerance
    worst: dict[tuple[int, int], ClashPair] = {}

    for ai, aj, d in _atom_pairs_within(structure, search):
        ri, rj = int(index.residue_of[ai]), int(index.residue_of[aj])
        if ri == rj:
            continue
        res_i, res_j = structure.residues[ri], structure.residues[rj]
        ei, ej = index.elements[ai].upper(), index.elements[aj].upper()
        overlap = vdw_radius(ei) + vdw_radius(ej) - tolerance - d
        if overlap <= 0:
            continue
        if topology.separation(ai, aj, max_depth=max_sep) is not None:
            continue
        if exclude_metal_contacts and cfg["exclude_metal_coordination"]:
            if res_i.entity is EntityType.METAL or res_j.entity is EntityType.METAL:
                continue
        if cfg["exclude_disulfide_sulfurs"] and ei == "S" and ej == "S":
            if index.names[ai] == "SG" and index.names[aj] == "SG":
                continue
        key = (ri, rj) if ri < rj else (rj, ri)
        prev = worst.get(key)
        if prev is None or overlap > prev.overlap:
            worst[key] = ClashPair(
                i=key[0],
                j=key[1],
                atom_i=index.names[ai] if key[0] == ri else index.names[aj],
                atom_j=index.names[aj] if key[0] == ri else index.names[ai],
                distance=d,
                overlap=overlap,
            )
    return sorted(worst.values(), key=lambda c: (-c.overlap, c.i, c.j))


# --------------------------------------------------------------------------- #
# A.8 nearest non-covalently-bonded atom
# --------------------------------------------------------------------------- #

def nearest_nonbonded_atom(
    structure: Structure,
    residue_index: int,
    atom_name: str,
    topology: Topology,
    definitions: Definitions,
    *,
    search_radius: float = 8.0,
) -> list[tuple[int, str, float]]:
    """Candidate answers for G02, nearest first, after A.8 exclusions.

    Only atoms within ``search_radius`` are ranked; that is far more than any
    plausible nearest neighbour and keeps the scan linear in the local
    neighbourhood rather than in the whole structure.
    """
    cfg = definitions.get("covalent_topology.nearest_nonbonded")
    max_sep = int(cfg["exclude_bond_separation_upto"])
    index = structure.index
    query_global = _global_atom_index(structure, residue_index, atom_name)
    if query_global is None:
        return []

    excluded = topology.within(query_global, max_sep) | {query_global}
    qpos = index.coords[query_global]
    out: list[tuple[int, str, float]] = []
    for gi in index.within(qpos, search_radius):
        if gi in excluded:
            continue
        ri = int(index.residue_of[gi])
        if cfg["restrict_to_protein_heavy_atoms"] and not structure.residues[ri].is_protein:
            continue
        d = float(np.linalg.norm(index.coords[gi] - qpos))
        out.append((ri, index.names[gi], d))
    out.sort(key=lambda t: (t[2], t[0], t[1]))
    return out


def _global_atom_index(structure: Structure, residue_index: int, atom_name: str) -> int | None:
    offset = 0
    for ri, res in enumerate(structure.residues):
        if ri == residue_index:
            for ai, atom in enumerate(res.atoms):
                if atom.name == atom_name:
                    return offset + ai
            return None
        offset += len(res.atoms)
    return None


# --------------------------------------------------------------------------- #
# A.24 - A.26 contact graph
# --------------------------------------------------------------------------- #

@dataclass
class ContactGraph:
    """Undirected residue-contact graph with minimum distances as metadata."""

    adjacency: dict[int, set[int]] = field(default_factory=dict)
    distances: dict[tuple[int, int], float] = field(default_factory=dict)

    def neighbours(self, node: int) -> set[int]:
        return self.adjacency.get(node, set())

    def add(self, i: int, j: int, distance: float) -> None:
        self.adjacency.setdefault(i, set()).add(j)
        self.adjacency.setdefault(j, set()).add(i)
        key = (i, j) if i < j else (j, i)
        prev = self.distances.get(key)
        if prev is None or distance < prev:
            self.distances[key] = distance

    def shortest_paths(self, source: int, target: int, max_edges: int) -> list[list[int]]:
        """All shortest paths up to ``max_edges`` (A.26)."""
        if source == target:
            return [[source]]
        level = {source: 0}
        parents: dict[int, list[int]] = {}
        frontier = deque([source])
        found_depth: int | None = None
        while frontier:
            node = frontier.popleft()
            depth = level[node]
            if found_depth is not None and depth >= found_depth:
                continue
            if depth >= max_edges:
                continue
            for nb in sorted(self.neighbours(node)):
                if nb not in level:
                    level[nb] = depth + 1
                    parents.setdefault(nb, []).append(node)
                    frontier.append(nb)
                elif level[nb] == depth + 1:
                    parents.setdefault(nb, []).append(node)
                if nb == target:
                    found_depth = level[nb]
        if target not in level or level[target] > max_edges:
            return []

        paths: list[list[int]] = []

        def walk(node: int, acc: list[int]) -> None:
            if node == source:
                paths.append([source, *reversed(acc)])
                return
            for parent in parents.get(node, []):
                walk(parent, [*acc, node])

        walk(target, [])
        return sorted(paths)


def build_contact_graph(
    structure: Structure,
    definitions: Definitions,
    *,
    include_ligands: bool | None = None,
    include_metals: bool | None = None,
) -> ContactGraph:
    """Residue contact graph following A.24."""
    cfg = definitions.get("contact_graph")
    include_ligands = cfg["include_ligand_nodes"] if include_ligands is None else include_ligands
    include_metals = cfg["include_metal_nodes"] if include_metals is None else include_metals

    graph = ContactGraph()
    for contact in residue_contacts(structure, definitions):
        graph.add(contact.i, contact.j, contact.min_distance)
    if include_ligands:
        for ri, res in enumerate(structure.residues):
            if res.entity is EntityType.LIGAND:
                for contact in ligand_contacts(structure, ri, definitions):
                    graph.add(contact.i, ri, contact.min_distance)
    if include_metals:
        eligible = set(definitions.get("metal_coordination.eligible_metals"))
        for ri, res in enumerate(structure.residues):
            # A.19 defines direct coordination only for the listed metals. Others
            # (Cd, Hg, alkali ions) stay in the structure as retained entities but
            # contribute no coordination edges rather than failing the build.
            if res.entity is EntityType.METAL and metal_element(res.name) in eligible:
                coordination = metal_coordination(structure, ri, definitions)
                for rj, _, d in coordination.donors:
                    graph.add(rj, ri, d)
    return graph


def bridging_residues(
    graph: ContactGraph, anchor_a: int, anchor_b: int, *, exclude: Sequence[int] = ()
) -> list[int]:
    """Residues directly contacting both anchors (A.25)."""
    common = graph.neighbours(anchor_a) & graph.neighbours(anchor_b)
    return sorted(common - {anchor_a, anchor_b} - set(exclude))
