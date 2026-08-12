"""Side-chain and backbone torsions: chi1 rotamers (A.21-A.22) and omega (A.17)."""

from __future__ import annotations

from dataclasses import dataclass

from ..chem import chi1_atom4, parent_of
from ..config import Definitions
from ..preprocessing.model import Residue, Structure
from .core import circular_difference, dihedral

G_PLUS = "g+"
TRANS = "t"
G_MINUS = "g-"
CIS = "cis"
TRANS_OMEGA = "trans"


@dataclass
class Chi1:
    residue_index: int
    angle: float
    rotamer: str
    atom4: str
    boundary_distance: float          # degrees to the nearest bin boundary

    @property
    def is_ambiguous(self) -> bool:
        return self.rotamer == ""


def chi1_bin(angle: float) -> str:
    """Classify a normalised chi1 angle (A.21)."""
    if 0.0 <= angle < 120.0:
        return G_PLUS
    if 120.0 <= angle < 180.0 or -180.0 <= angle < -120.0:
        return TRANS
    return G_MINUS


def boundary_distance(angle: float) -> float:
    """Smallest circular distance to a chi1 bin boundary (A.21).

    The boundaries are -120, 0 and 120 degrees only: the ``t`` bin is continuous
    across +/-180, so that wrap point is not a boundary.
    """
    return min(circular_difference(angle, b) for b in (-120.0, 0.0, 120.0))


def compute_chi1(
    structure: Structure, residue_index: int, definitions: Definitions
) -> Chi1 | None:
    """chi1 for one residue, or ``None`` when the residue is excluded (A.21)."""
    res = structure.residues[residue_index]
    if not res.is_protein:
        return None
    atom4_name = chi1_atom4(res.orig_name)
    if atom4_name is None:
        return None
    atoms = res.require("N", "CA", "CB", atom4_name)
    if atoms is None:
        return None
    angle = dihedral(atoms[0].pos, atoms[1].pos, atoms[2].pos, atoms[3].pos)
    margin = float(definitions.get("chi1.boundary_margin"))
    dist = boundary_distance(angle)
    return Chi1(
        residue_index=residue_index,
        angle=angle,
        rotamer=chi1_bin(angle) if dist >= margin else "",
        atom4=atom4_name,
        boundary_distance=dist,
    )


@dataclass
class Chi1Change:
    label: str
    angle1: float
    angle2: float
    rotamer1: str
    rotamer2: str
    difference: float
    changed: bool
    reasons: list[str]


def chi1_change(
    chi_a: Chi1, chi_b: Chi1, definitions: Definitions, *, ca_displacement: float | None = None
) -> Chi1Change:
    """Decide whether a mapped residue changed chi1 state (A.22)."""
    cfg = definitions.get("chi1_change")
    margin = float(cfg["boundary_margin"])
    min_diff = float(cfg["min_circular_difference"])
    diff = circular_difference(chi_a.angle, chi_b.angle)
    reasons: list[str] = []
    bin_a, bin_b = chi1_bin(chi_a.angle), chi1_bin(chi_b.angle)
    if bin_a == bin_b:
        reasons.append("same_rotamer_bin")
    if diff < min_diff:
        reasons.append(f"circular_difference_{diff:.1f}_below_{min_diff}")
    if chi_a.boundary_distance < margin or chi_b.boundary_distance < margin:
        reasons.append("within_boundary_margin")
    if ca_displacement is not None:
        limit = float(cfg["max_ca_displacement_for_pure_rotamer_change"])
        if ca_displacement > limit:
            reasons.append(f"ca_displacement_{ca_displacement:.2f}_exceeds_{limit}")
    return Chi1Change(
        label="",
        angle1=chi_a.angle,
        angle2=chi_b.angle,
        rotamer1=bin_a,
        rotamer2=bin_b,
        difference=diff,
        changed=not reasons,
        reasons=reasons,
    )


def proline_omega(
    structure: Structure, residue_index: int, definitions: Definitions
) -> tuple[float, str] | None:
    """Omega ``CA(i-1)-C(i-1)-N(i)-CA(i)`` and its cis/trans state (A.17)."""
    res = structure.residues[residue_index]
    if parent_of(res.orig_name) != "PRO":
        return None
    prev = structure.polymer_neighbours(res, -1)
    if prev is None:
        return None
    prev_atoms = prev.require("CA", "C")
    this_atoms = res.require("N", "CA")
    if prev_atoms is None or this_atoms is None:
        return None
    omega = dihedral(prev_atoms[0].pos, prev_atoms[1].pos, this_atoms[0].pos, this_atoms[1].pos)
    cfg = definitions.get("proline_omega")
    if abs(omega) <= float(cfg["cis_max_abs"]):
        return omega, CIS
    if abs(abs(omega) - 180.0) <= float(cfg["trans_max_deviation"]):
        return omega, TRANS_OMEGA
    return omega, ""


def sidechain_rmsd(a: Residue, b: Residue) -> float | None:
    """RMSD over shared side-chain heavy atoms, used as mechanistic evidence."""
    import numpy as np

    names = [x.name for x in a.sidechain_atoms]
    shared = [n for n in names if b.atom(n) is not None]
    if len(shared) < 2:
        return None
    pa = np.array([a.atom(n).pos for n in shared])
    pb = np.array([b.atom(n).pos for n in shared])
    return float(np.sqrt(((pa - pb) ** 2).sum(axis=1).mean()))
