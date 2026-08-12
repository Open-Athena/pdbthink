"""Two-state residue mapping and structural superposition (A.27-A.29)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..config import Definitions
from ..preprocessing.model import Structure
from .core import Superposition, kabsch
from .seqalign import align

CONTACT_GAINED = "gained"
CONTACT_LOST = "lost"
CONTACT_RETAINED = "retained_contact"
CONTACT_RETAINED_NONCONTACT = "retained_noncontact"
CONTACT_AMBIGUOUS = "ambiguous"


@dataclass
class ResidueMapping:
    """Residue-index correspondence between two states of the same protein."""

    pairs: list[tuple[int, int]] = field(default_factory=list)
    identity: dict[str, float] = field(default_factory=dict)
    chain_map: dict[str, str] = field(default_factory=dict)
    unmapped_1: list[int] = field(default_factory=list)
    unmapped_2: list[int] = field(default_factory=list)

    def to_second(self) -> dict[int, int]:
        return dict(self.pairs)

    def to_first(self) -> dict[int, int]:
        return {b: a for a, b in self.pairs}


def map_residues(
    state1: Structure,
    state2: Structure,
    definitions: Definitions,
    *,
    chain_map: dict[str, str] | None = None,
    require_same_identity: bool = True,
) -> ResidueMapping:
    """Map polymer residues by sequence alignment, chain by chain (A.27.1)."""
    cfg = definitions.get("two_state_alignment")
    min_identity = float(cfg["sequence_identity_min"])

    if chain_map is None:
        chains1, chains2 = state1.protein_chains, state2.protein_chains
        if len(chains1) != len(chains2):
            raise ValueError(
                f"cannot infer a chain map: {chains1} vs {chains2}; supply chain_map explicitly"
            )
        chain_map = dict(zip(chains1, chains2))

    mapping = ResidueMapping(chain_map=dict(chain_map))
    for chain1, chain2 in chain_map.items():
        res1 = sorted(
            [i for i, r in enumerate(state1.residues) if r.is_protein and r.chain == chain1],
            key=lambda i: state1.residues[i].poly_index or 0,
        )
        res2 = sorted(
            [i for i, r in enumerate(state2.residues) if r.is_protein and r.chain == chain2],
            key=lambda i: state2.residues[i].poly_index or 0,
        )
        seq1 = "".join(state1.residues[i].one_letter for i in res1)
        seq2 = "".join(state2.residues[i].one_letter for i in res2)
        alignment = align(seq1, seq2)
        mapping.identity[f"{chain1}->{chain2}"] = alignment.identity
        if alignment.identity < min_identity:
            raise ValueError(
                f"chains {chain1}/{chain2} are only {alignment.identity:.2%} identical, "
                f"below the A.27 minimum of {min_identity:.0%}"
            )
        matched_1: set[int] = set()
        matched_2: set[int] = set()
        for a, b in alignment.matched:
            ri, rj = res1[a], res2[b]
            if require_same_identity and seq1[a] != seq2[b]:
                continue
            mapping.pairs.append((ri, rj))
            matched_1.add(ri)
            matched_2.add(rj)
        mapping.unmapped_1.extend(i for i in res1 if i not in matched_1)
        mapping.unmapped_2.extend(i for i in res2 if i not in matched_2)
    mapping.pairs.sort()
    return mapping


@dataclass
class AlignmentResult:
    superposition: Superposition
    core_pairs: list[tuple[int, int]]
    excluded_pairs: list[tuple[int, int]]
    rmsd_before: float
    rmsd_after: float
    displacements: dict[int, float] = field(default_factory=dict)   # state1 index -> CA distance


def superpose_states(
    state1: Structure,
    state2: Structure,
    mapping: ResidueMapping,
    definitions: Definitions,
    *,
    exclude: Sequence[int] = (),
) -> AlignmentResult:
    """Superpose state 2 onto state 1 over a conserved C-alpha core (A.27.2-5).

    ``exclude`` lists state-1 residue indices belonging to the queried flexible
    region, which A.27.3 keeps out of the alignment core.
    """
    cfg = definitions.get("two_state_alignment")
    excluded = set(exclude)

    core: list[tuple[int, int]] = []
    for i, j in mapping.pairs:
        if i in excluded:
            continue
        if state1.residues[i].atom("CA") is None or state2.residues[j].atom("CA") is None:
            continue
        core.append((i, j))
    if len(core) < 3:
        raise ValueError("fewer than three usable C-alpha pairs in the alignment core")

    def coords(pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        a = np.array([state2.residues[j].atom("CA").pos for _, j in pairs])
        b = np.array([state1.residues[i].atom("CA").pos for i, _ in pairs])
        return a, b

    mobile, target = coords(core)
    fit = kabsch(mobile, target)
    rmsd_before = fit.rmsd
    excluded_pairs: list[tuple[int, int]] = []

    if cfg["refit_once"]:
        cutoff = float(cfg["outlier_residual_cutoff"])
        moved = fit.apply(mobile)
        residuals = np.linalg.norm(moved - target, axis=1)
        keep = [p for p, r in zip(core, residuals) if r <= cutoff]
        excluded_pairs = [p for p, r in zip(core, residuals) if r > cutoff]
        if len(keep) >= 3 and excluded_pairs:
            core = keep
            mobile, target = coords(core)
            fit = kabsch(mobile, target)

    displacements: dict[int, float] = {}
    for i, j in mapping.pairs:
        ca1 = state1.residues[i].atom("CA")
        ca2 = state2.residues[j].atom("CA")
        if ca1 is None or ca2 is None:
            continue
        moved = fit.apply(np.atleast_2d(ca2.pos))[0]
        displacements[i] = float(np.linalg.norm(moved - ca1.pos))

    return AlignmentResult(
        superposition=fit,
        core_pairs=core,
        excluded_pairs=excluded_pairs,
        rmsd_before=rmsd_before,
        rmsd_after=fit.rmsd,
        displacements=displacements,
    )


def apply_superposition(structure: Structure, fit: Superposition) -> Structure:
    """Return a copy of ``structure`` moved onto the reference frame."""
    out = structure.copy()
    for res in out.residues:
        for atom in res.atoms:
            atom.pos = fit.rotation @ atom.pos + fit.translation
    out.invalidate()
    return out


def classify_contact_change(
    distance_1: float, distance_2: float, definitions: Definitions
) -> str:
    """Hysteresis classification of a mapped pair distance (A.28)."""
    cfg = definitions.get("contact_change")
    contact = float(cfg["contact_cutoff"])
    noncontact = float(cfg["noncontact_cutoff"])
    if distance_2 <= contact and distance_1 >= noncontact:
        return CONTACT_GAINED
    if distance_1 <= contact and distance_2 >= noncontact:
        return CONTACT_LOST
    if distance_1 <= contact and distance_2 <= contact:
        return CONTACT_RETAINED
    if distance_1 >= noncontact and distance_2 >= noncontact:
        return CONTACT_RETAINED_NONCONTACT
    return CONTACT_AMBIGUOUS


def classify_displacement(displacement: float, definitions: Definitions) -> str:
    """A.29 three-way classification of mapped C-alpha displacement."""
    cfg = definitions.get("backbone_displacement")
    if displacement <= float(cfg["unchanged_max"]):
        return "approximately unchanged"
    if displacement >= float(cfg["displaced_min"]):
        return "substantially displaced"
    return "intermediate"
