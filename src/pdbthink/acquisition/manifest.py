"""Source manifests: the reproducible list of structures a dataset depends on."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import DatasetConfig


@dataclass
class SourceManifest:
    """A declarative list of PDB entries and AlphaFold DB accessions."""

    version: str = "1"
    pdb_entries: list[str] = field(default_factory=list)
    afdb_entries: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> SourceManifest:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            version=str(raw.get("version", "1")),
            pdb_entries=[str(e).upper() for e in raw.get("pdb_entries", [])],
            afdb_entries=[str(e).upper() for e in raw.get("afdb_entries", [])],
            notes={str(k): str(v) for k, v in (raw.get("notes") or {}).items()},
        )

    def save(self, path: str | Path) -> None:
        payload: dict[str, Any] = {
            "version": self.version,
            "pdb_entries": sorted(set(self.pdb_entries)),
            "afdb_entries": sorted(set(self.afdb_entries)),
        }
        if self.notes:
            payload["notes"] = self.notes
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def merge(self, other: SourceManifest) -> SourceManifest:
        return SourceManifest(
            version=self.version,
            pdb_entries=sorted(set(self.pdb_entries) | set(other.pdb_entries)),
            afdb_entries=sorted(set(self.afdb_entries) | set(other.afdb_entries)),
            notes={**self.notes, **other.notes},
        )

    def __len__(self) -> int:
        return len(self.pdb_entries) + len(self.afdb_entries)


def manifest_from_dataset(config: DatasetConfig) -> SourceManifest:
    """Everything a dataset configuration needs, including mechanistic episodes."""
    from ..mechanistic.episodes import EPISODE_SOURCES

    pdb: set[str] = set()
    afdb: set[str] = set()
    for spec in config.proteins:
        (pdb if spec.source_type == "pdb" else afdb).add(spec.entry.upper())
    for pair in config.state_pairs:
        for spec in (pair.state1, pair.state2):
            (pdb if spec.source_type == "pdb" else afdb).add(spec.entry.upper())
    for episode_id in config.episodes:
        pdb.update(EPISODE_SOURCES.get(episode_id, ()))
    return SourceManifest(pdb_entries=sorted(pdb), afdb_entries=sorted(afdb))
