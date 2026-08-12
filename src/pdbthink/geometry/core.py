"""Elementary geometry: distances, angles, dihedrals, rigid superposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle ``p1-p2-p3`` in degrees."""
    v1 = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    v2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        raise ValueError("degenerate angle")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Torsion ``p1-p2-p3-p4`` in degrees, normalised to ``[-180, 180)``."""
    b0 = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    b1 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    b2 = np.asarray(p4, dtype=float) - np.asarray(p3, dtype=float)
    n1 = np.linalg.norm(b1)
    if n1 < 1e-9:
        raise ValueError("degenerate dihedral")
    b1 = b1 / n1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return normalise_angle(float(np.degrees(np.arctan2(y, x))))


def normalise_angle(deg: float) -> float:
    """Wrap to ``[-180, 180)`` (A.17, A.21)."""
    wrapped = (float(deg) + 180.0) % 360.0 - 180.0
    # ``(-180 + 180) % 360 - 180`` already returns -180; guard the +180 edge.
    return -180.0 if wrapped == 180.0 else wrapped


def circular_difference(a: float, b: float) -> float:
    """Minimum absolute circular difference between two angles, in degrees."""
    diff = abs(normalise_angle(a) - normalise_angle(b)) % 360.0
    return float(min(diff, 360.0 - diff))


@dataclass
class Superposition:
    rotation: np.ndarray            # (3, 3), proper rotation
    translation: np.ndarray         # (3,), applied after rotation
    rmsd: float
    n_pairs: int

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return (self.rotation @ np.atleast_2d(coords).T).T + self.translation


def kabsch(mobile: np.ndarray, target: np.ndarray) -> Superposition:
    """Least-squares rigid superposition of ``mobile`` onto ``target``.

    Returns a proper rotation (determinant +1); reflections are corrected in the
    standard way by flipping the sign of the least significant singular vector.
    """
    p = np.asarray(mobile, dtype=float).reshape(-1, 3)
    q = np.asarray(target, dtype=float).reshape(-1, 3)
    if p.shape != q.shape or len(p) < 3:
        raise ValueError(f"need matching coordinate sets with >= 3 points, got {p.shape} {q.shape}")
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    p0, q0 = p - pc, q - qc
    h = p0.T @ q0
    u, _, vt = np.linalg.svd(h)
    d = float(np.sign(np.linalg.det(vt.T @ u.T)))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T
    translation = qc - rotation @ pc
    fitted = (rotation @ p.T).T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - q) ** 2, axis=1))))
    return Superposition(rotation=rotation, translation=translation, rmsd=rmsd, n_pairs=len(p))


def pairwise_min_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Minimum distance between two coordinate sets and the winning indices."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    diff = a[:, None, :] - b[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    flat = int(np.argmin(d))
    i, j = divmod(flat, d.shape[1])
    return float(d[i, j]), int(i), int(j)


def centroid(coords: np.ndarray) -> np.ndarray:
    return np.asarray(coords, dtype=float).reshape(-1, 3).mean(axis=0)
