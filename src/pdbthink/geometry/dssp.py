"""Secondary structure by the Kabsch-Sander DSSP rules (A.14-A.16).

A self-contained implementation is used instead of an external ``mkdssp`` binary
so that assignments are reproducible from the repository alone and versioned
together with the rest of Appendix A (``kabsch_sander_v1``).

Electrostatic hydrogen-bond energy, following Kabsch & Sander (1983):

    E = 0.084 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332  kcal/mol

with a bond declared when ``E < -0.5``. The amide hydrogen is placed on the
N of residue *i* along the C=O direction of residue *i-1*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..chem import parent_of
from ..config import Definitions
from ..preprocessing.model import Structure
from .core import angle

ALGORITHM = "kabsch_sander_v1"
Q_FACTOR = 0.084 * 332.0
MIN_HBOND_DISTANCE = 0.5      # guard against singularities
CA_SEARCH_RADIUS = 9.0        # pairs beyond this cannot hydrogen bond
BEND_ANGLE = 70.0

HELIX = "helix"
STRAND = "strand"
COIL = "coil"


@dataclass
class DsspResult:
    """Eight-state and three-state assignments keyed by residue index."""

    codes: dict[int, str] = field(default_factory=dict)          # H G I E B T S or " "
    three_state: dict[int, str] = field(default_factory=dict)    # helix/strand/coil
    failed: set[int] = field(default_factory=set)                # missing backbone
    hbonds: dict[tuple[int, int], float] = field(default_factory=dict)
    algorithm: str = ALGORITHM

    def code(self, residue_index: int) -> str | None:
        return self.codes.get(residue_index)

    def klass(self, residue_index: int) -> str | None:
        return self.three_state.get(residue_index)

    def chain_fractions(self, structure: Structure, chain: str) -> dict[str, float]:
        """Helix and strand fractions over residues with valid assignments (A.16)."""
        codes = [
            self.codes[i]
            for i, res in enumerate(structure.residues)
            if res.is_protein and res.chain == chain and i in self.codes
        ]
        if not codes:
            return {"helix_fraction": 0.0, "beta_fraction": 0.0, "n_assigned": 0}
        helix = sum(1 for c in codes if c in "HGI")
        beta = sum(1 for c in codes if c in "EB")
        return {
            "helix_fraction": helix / len(codes),
            "beta_fraction": beta / len(codes),
            "n_assigned": len(codes),
        }


def _three_state(code: str, mapping: dict[str, list[str]]) -> str:
    if code in mapping["helix"]:
        return HELIX
    if code in mapping["strand"]:
        return STRAND
    return COIL


def assign_dssp(structure: Structure, definitions: Definitions) -> DsspResult:
    """Assign secondary structure over every protein chain of a structure."""
    cfg = definitions.get("secondary_structure")
    if cfg["algorithm"] != ALGORITHM:
        raise ValueError(f"unsupported secondary-structure algorithm {cfg['algorithm']!r}")
    energy_cutoff = float(cfg["hbond_energy_cutoff"])

    residues = [(i, r) for i, r in enumerate(structure.residues) if r.is_protein]
    result = DsspResult()
    if not residues:
        return result

    # Chain-ordered arrays of backbone atoms.
    order: list[int] = []
    n_pos: list[np.ndarray] = []
    ca_pos: list[np.ndarray] = []
    c_pos: list[np.ndarray] = []
    o_pos: list[np.ndarray] = []
    chain_of: list[str] = []
    for ri, res in sorted(residues, key=lambda t: (t[1].chain, t[1].poly_index or 0)):
        atoms = res.require("N", "CA", "C", "O")
        if atoms is None:
            result.failed.add(ri)
            continue
        order.append(ri)
        n_pos.append(atoms[0].pos)
        ca_pos.append(atoms[1].pos)
        c_pos.append(atoms[2].pos)
        o_pos.append(atoms[3].pos)
        chain_of.append(res.chain)

    n = len(order)
    if n == 0:
        return result
    N = np.array(n_pos)
    CA = np.array(ca_pos)
    C = np.array(c_pos)
    O = np.array(o_pos)

    # Consecutive residues in the same chain, actually peptide bonded.
    connected = np.zeros(n, dtype=bool)     # connected[k]: k and k+1 are bonded
    for k in range(n - 1):
        if chain_of[k] != chain_of[k + 1]:
            continue
        if float(np.linalg.norm(C[k] - N[k + 1])) <= 2.0:
            connected[k] = True

    # Amide hydrogen positions; proline and chain starts cannot donate.
    has_h = np.zeros(n, dtype=bool)
    H = np.zeros((n, 3), dtype=float)
    for k in range(n):
        if k == 0 or not connected[k - 1]:
            continue
        if parent_of(structure.residues[order[k]].orig_name) == "PRO":
            continue
        direction = C[k - 1] - O[k - 1]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        H[k] = N[k] + direction / norm
        has_h[k] = True

    # Hydrogen bonds: hbond[(donor, acceptor)] with donor N-H and acceptor C=O.
    tree = cKDTree(CA)
    energies: dict[tuple[int, int], float] = {}
    for donor, acceptor in tree.query_pairs(CA_SEARCH_RADIUS):
        for d, a in ((donor, acceptor), (acceptor, donor)):
            if not has_h[d]:
                continue
            if abs(d - a) < 2 and chain_of[d] == chain_of[a]:
                continue
            e = _hbond_energy(N[d], H[d], C[a], O[a])
            if e < energy_cutoff:
                energies[(d, a)] = e

    def hbond(donor: int, acceptor: int) -> bool:
        return (donor, acceptor) in energies

    # ---- n-turns ---------------------------------------------------------
    turns = {3: np.zeros(n, dtype=bool), 4: np.zeros(n, dtype=bool), 5: np.zeros(n, dtype=bool)}
    for step in (3, 4, 5):
        for k in range(n - step):
            if not all(connected[k + t] for t in range(step)):
                continue
            if hbond(k + step, k):        # N-H of k+step to C=O of k
                turns[step][k] = True

    codes = [" "] * n

    def set_code(k: int, code: str) -> None:
        if codes[k] == " ":
            codes[k] = code

    # ---- bridges and ladders --------------------------------------------
    def prev_ok(k: int) -> bool:
        return k - 1 >= 0 and connected[k - 1]

    def next_ok(k: int) -> bool:
        return k + 1 < n and connected[k]

    bridges: dict[tuple[int, int], str] = {}
    for i, j in tree.query_pairs(CA_SEARCH_RADIUS):
        if abs(i - j) <= 2 and chain_of[i] == chain_of[j]:
            continue
        if not (prev_ok(i) and next_ok(i) and prev_ok(j) and next_ok(j)):
            continue
        parallel = (hbond(j, i - 1) and hbond(i + 1, j)) or (hbond(i, j - 1) and hbond(j + 1, i))
        anti = (hbond(j, i) and hbond(i, j)) or (hbond(j + 1, i - 1) and hbond(i + 1, j - 1))
        if parallel:
            bridges[(min(i, j), max(i, j))] = "P"
        elif anti:
            bridges[(min(i, j), max(i, j))] = "A"

    ladder_size: dict[tuple[int, int], int] = {}
    for (i, j), kind in bridges.items():
        length = 1
        # extend forward
        step = 1
        while True:
            nxt = (i + step, j + step) if kind == "P" else (i + step, j - step)
            if bridges.get((min(nxt), max(nxt))) != kind:
                break
            length += 1
            step += 1
        step = 1
        while True:
            prv = (i - step, j - step) if kind == "P" else (i - step, j + step)
            if bridges.get((min(prv), max(prv))) != kind:
                break
            length += 1
            step += 1
        ladder_size[(i, j)] = length

    # ---- helices (A.14 precedence H > B/E > G > I > T > S) ---------------
    for k in range(1, n):
        if turns[4][k - 1] and turns[4][k]:
            for t in range(k, min(k + 4, n)):
                set_code(t, "H")

    for (i, j), length in ladder_size.items():
        code = "E" if length >= 2 else "B"
        set_code(i, code)
        set_code(j, code)

    for k in range(1, n):
        if turns[3][k - 1] and turns[3][k]:
            for t in range(k, min(k + 3, n)):
                set_code(t, "G")
    for k in range(1, n):
        if turns[5][k - 1] and turns[5][k]:
            for t in range(k, min(k + 5, n)):
                set_code(t, "I")

    for step in (3, 4, 5):
        for k in range(n - step):
            if turns[step][k]:
                for t in range(k + 1, min(k + step, n)):
                    set_code(t, "T")

    for k in range(2, n - 2):
        if not all(connected[t] for t in range(k - 2, k + 2)):
            continue
        try:
            # kappa is the angle between the CA(i-2)->CA(i) and CA(i)->CA(i+2)
            # directions; a straight backbone gives an interior angle of 180.
            kappa = 180.0 - angle(CA[k - 2], CA[k], CA[k + 2])
        except ValueError:  # pragma: no cover - degenerate backbone
            continue
        if kappa > BEND_ANGLE:
            set_code(k, "S")

    mapping = definitions.get("secondary_structure.three_state_map")
    for k, ri in enumerate(order):
        result.codes[ri] = codes[k]
        result.three_state[ri] = _three_state(codes[k], mapping)
    result.hbonds = {(order[d], order[a]): e for (d, a), e in energies.items()}
    return result


def _hbond_energy(n: np.ndarray, h: np.ndarray, c: np.ndarray, o: np.ndarray) -> float:
    r_on = float(np.linalg.norm(o - n))
    r_ch = float(np.linalg.norm(c - h))
    r_oh = float(np.linalg.norm(o - h))
    r_cn = float(np.linalg.norm(c - n))
    if min(r_on, r_ch, r_oh, r_cn) < MIN_HBOND_DISTANCE:
        return 0.0
    return Q_FACTOR * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)


def fold_class(fractions: dict[str, float], definitions: Definitions) -> str | None:
    """Classify a chain as predominantly alpha, beta, or mixed (A.16)."""
    cfg = definitions.get("fold_class")
    helix = float(fractions["helix_fraction"])
    beta = float(fractions["beta_fraction"])
    labels = cfg["labels"]

    if helix >= float(cfg["alpha"]["min_helix_fraction"]) and helix > float(
        cfg["alpha"]["helix_over_beta_ratio"]
    ) * beta:
        return labels["alpha"]
    if beta >= float(cfg["beta"]["min_beta_fraction"]) and beta > float(
        cfg["beta"]["beta_over_helix_ratio"]
    ) * helix:
        return labels["beta"]
    lo, hi = cfg["mixed"]["ratio_range"]
    min_each = float(cfg["mixed"]["min_each_fraction"])
    if helix >= min_each and beta >= min_each:
        ratio = helix / beta if beta > 0 else float("inf")
        if float(lo) <= ratio <= float(hi):
            return labels["mixed"]
    return None


def three_ten_runs(
    structure: Structure, result: DsspResult, chain: str, definitions: Definitions
) -> list[list[int]]:
    """Contiguous runs of ``G`` of at least the configured length (A.15)."""
    min_run = int(definitions.get("three_ten_helix.min_run_length"))
    ordered = [
        i
        for i, r in enumerate(structure.residues)
        if r.is_protein and r.chain == chain
    ]
    ordered.sort(key=lambda i: structure.residues[i].poly_index or 0)
    runs: list[list[int]] = []
    current: list[int] = []
    for ri in ordered:
        if result.codes.get(ri) == "G":
            current.append(ri)
        else:
            if len(current) >= min_run:
                runs.append(current)
            current = []
    if len(current) >= min_run:
        runs.append(current)
    return runs
