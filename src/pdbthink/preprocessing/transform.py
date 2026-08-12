"""Reproducible rigid-body transformation and coordinate rounding (A.2.10-A.2.12).

The transform applied to every model-visible structure is

    x' = R (x - c)

where ``c`` is the centroid of the retained atoms (the union centroid for paired
states) and ``R`` is a proper rotation derived deterministically from the
rotation seed. The stored ``translation_vector`` is ``-c``, so a consumer can
reproduce the displayed coordinates exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Definitions
from ..util import rng_for
from .model import Structure


class TransformError(RuntimeError):
    """Raised when a rendering would violate the PDB coordinate field limits."""


@dataclass
class RigidTransform:
    rotation: np.ndarray            # (3, 3), determinant +1
    translation: np.ndarray         # (3,), applied before rotation
    seed: int
    algorithm: str

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return (self.rotation @ (np.atleast_2d(coords) + self.translation).T).T

    def as_dict(self) -> dict[str, Any]:
        return {
            "rotation_matrix": [[float(v) for v in row] for row in self.rotation],
            "translation_vector": [float(v) for v in self.translation],
            "rotation_seed": int(self.seed),
            "algorithm": self.algorithm,
        }


@dataclass
class DisplayedStructure:
    """A structure exactly as the model will see it: transformed and rounded."""

    structure: Structure
    transform: RigidTransform
    label: str = "structure"
    crop: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def atom_count(self) -> int:
        return self.structure.atom_count


def random_rotation(seed: int, definitions: Definitions) -> np.ndarray:
    """Uniformly distributed proper rotation, reproducible from ``seed``."""
    cfg = definitions.get("structure_processing.rotation")
    algorithm = cfg["algorithm"]
    if algorithm != "pcg64_unit_quaternion_v1":
        raise TransformError(f"unsupported rotation algorithm {algorithm!r}")
    rng = rng_for("rotation", algorithm, int(seed))
    q = rng.standard_normal(4)
    norm = float(np.linalg.norm(q))
    while norm < 1e-9:  # pragma: no cover - astronomically unlikely
        q = rng.standard_normal(4)
        norm = float(np.linalg.norm(q))
    w, x, y, z = q / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    det = float(np.linalg.det(rotation))
    tol = float(cfg["determinant_tolerance"])
    if abs(det - float(cfg["require_determinant"])) > tol:
        raise TransformError(f"rotation determinant {det!r} is not +1 (reflection)")
    return rotation


def identity_transform() -> RigidTransform:
    return RigidTransform(np.eye(3), np.zeros(3), seed=0, algorithm="identity")


def build_transform(
    structures: Sequence[Structure], seed: int, definitions: Definitions
) -> RigidTransform:
    """One shared transform for one or more already-aligned structures."""
    coords = np.vstack([s.index.coords for s in structures])
    if coords.size == 0:
        raise TransformError("cannot build a transform for an empty structure")
    centroid = coords.mean(axis=0)
    rotation = random_rotation(seed, definitions)
    return RigidTransform(
        rotation=rotation,
        translation=-centroid,
        seed=int(seed),
        algorithm=definitions.get("structure_processing.rotation.algorithm"),
    )


def display(
    structure: Structure,
    transform: RigidTransform,
    definitions: Definitions,
    *,
    label: str = "structure",
    crop: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> DisplayedStructure:
    """Apply the transform, round to three decimals and check field limits."""
    decimals = int(definitions.get("structure_processing.coordinate_decimals"))
    limit = float(definitions.get("structure_processing.rotation.max_abs_coordinate"))
    moved = structure.transformed(transform.rotation, transform.translation).rounded(decimals)
    coords = moved.index.coords
    if coords.size and float(np.abs(coords).max()) > limit:
        raise TransformError(
            f"transformed coordinate magnitude {float(np.abs(coords).max()):.3f} "
            f"exceeds the PDB field limit {limit}"
        )
    return DisplayedStructure(
        structure=moved,
        transform=transform,
        label=label,
        crop=crop,
        provenance=dict(provenance or {}),
    )


def display_pair(
    first: Structure,
    second: Structure,
    seed: int,
    definitions: Definitions,
    *,
    labels: tuple[str, str] = ("Structure 1", "Structure 2"),
) -> tuple[DisplayedStructure, DisplayedStructure]:
    """Render two aligned states with one shared transform (A.2, section 6)."""
    transform = build_transform([first, second], seed, definitions)
    return (
        display(first, transform, definitions, label=labels[0]),
        display(second, transform, definitions, label=labels[1]),
    )
