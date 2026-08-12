"""Local structure excerpts (section 6, "Input budget and cropping").

A crop is only ever produced for question families that are explicitly allowed
to use one, the prompt always states that the structure is a local excerpt, and
the crop is verified to retain every atom the oracle needs. Paired
representations and paired states reuse the same retained atom set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .model import EntityType, Residue, Structure


class CropError(RuntimeError):
    """Raised when a crop would remove atoms required by the oracle."""


@dataclass
class CropInfo:
    """Machine-readable record of what a crop retained."""

    mode: str
    radius: float | None
    centers: list[str] = field(default_factory=list)
    kept_labels: list[str] = field(default_factory=list)
    removed_residues: int = 0
    boundary_labels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "radius": self.radius,
            "centers": self.centers,
            "kept_residues": len(self.kept_labels),
            "removed_residues": self.removed_residues,
        }


def crop_around(
    structure: Structure,
    centers: Sequence[int],
    radius: float,
    *,
    required_labels: Iterable[str] = (),
    keep_entities: Sequence[EntityType] = (EntityType.LIGAND, EntityType.METAL),
) -> tuple[Structure, CropInfo]:
    """Whole-residue crop within ``radius`` of any atom of the center residues."""
    if not centers:
        raise CropError("a crop needs at least one center residue")
    index = structure.index
    keep: set[int] = set(centers)

    for ci in centers:
        for atom in structure.residues[ci].atoms:
            for ai in index.within(atom.pos, radius):
                keep.add(int(index.residue_of[ai]))

    for ri, res in enumerate(structure.residues):
        if res.entity in keep_entities and _near(structure, res, centers, radius * 1.5):
            keep.add(ri)

    cropped = structure.subset(keep)
    kept_labels = {r.label for r in cropped.residues}
    missing = [label for label in required_labels if label not in kept_labels]
    if missing:
        raise CropError(f"crop removes atoms required by the oracle: {missing}")

    info = CropInfo(
        mode="radius",
        radius=radius,
        centers=[structure.residues[c].label for c in centers],
        kept_labels=sorted(kept_labels),
        removed_residues=len(structure.residues) - len(cropped.residues),
        boundary_labels=sorted(_boundary_labels(structure, cropped)),
    )
    return cropped, info


def crop_to_selection(
    structure: Structure,
    selection: Sequence[tuple[str, int, int]],
    *,
    keep_labels: Iterable[str] = (),
    required_labels: Iterable[str] = (),
) -> tuple[Structure, CropInfo]:
    """Crop to explicit ``(chain, first, last)`` author-numbering ranges."""
    extra = set(keep_labels)
    keep: set[int] = set()
    for ri, res in enumerate(structure.residues):
        if res.label in extra:
            keep.add(ri)
            continue
        for chain, first, last in selection:
            if res.chain == chain and first <= res.seq_id <= last:
                keep.add(ri)
                break
    if not keep:
        raise CropError(f"selection {selection} retained no residues")

    cropped = structure.subset(keep)
    kept_labels = {r.label for r in cropped.residues}
    missing = [label for label in required_labels if label not in kept_labels]
    if missing:
        raise CropError(f"crop removes atoms required by the oracle: {missing}")

    info = CropInfo(
        mode="selection",
        radius=None,
        centers=[f"{c}:{a}-{b}" for c, a, b in selection],
        kept_labels=sorted(kept_labels),
        removed_residues=len(structure.residues) - len(cropped.residues),
        boundary_labels=sorted(_boundary_labels(structure, cropped)),
    )
    return cropped, info


def crop_like(structure: Structure, template: CropInfo) -> Structure:
    """Reproduce a crop on a paired state or paired representation."""
    keep = {i for i, r in enumerate(structure.residues) if r.label in set(template.kept_labels)}
    if not keep:
        raise CropError("paired crop retained no residues")
    return structure.subset(keep)


def _near(structure: Structure, res: Residue, centers: Sequence[int], radius: float) -> bool:
    coords = res.coords()
    for ci in centers:
        other = structure.residues[ci].coords()
        d = np.linalg.norm(coords[:, None, :] - other[None, :, :], axis=-1)
        if float(d.min()) <= radius:
            return True
    return False


def _boundary_labels(original: Structure, cropped: Structure) -> set[str]:
    """Residues at a crop edge, whose local environment is now incomplete.

    DSSP and burial answers must never be asked about these (A.14, A.34).
    """
    kept = {r.label for r in cropped.residues}
    boundary: set[str] = set()
    for res in cropped.residues:
        if res.polymer_kind is None:
            continue
        source = original.find(res.label)
        if source is None:
            continue
        for offset in (-1, 1):
            neighbour = original.polymer_neighbours(source, offset)
            if neighbour is not None and neighbour.label not in kept:
                boundary.add(res.label)
    return boundary
