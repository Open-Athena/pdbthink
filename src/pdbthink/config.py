"""Loading of versioned configuration: operational definitions and dataset specs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .util import REPO_ROOT, sha256_text

DEFAULT_DEFINITIONS = REPO_ROOT / "configs" / "definitions_v1.yaml"
DEFAULT_DATASET = REPO_ROOT / "configs" / "dataset_v1.yaml"


class ConfigError(RuntimeError):
    """Raised for malformed or internally inconsistent configuration."""


_MISSING = object()


@dataclass(frozen=True)
class Definitions:
    """Appendix A operational definitions, loaded from a versioned YAML file."""

    data: dict[str, Any]
    version: str
    source_path: str
    content_sha256: str

    @classmethod
    def load(cls, path: str | Path | None = None) -> Definitions:
        path = Path(path) if path is not None else DEFAULT_DEFINITIONS
        if not path.exists():
            raise ConfigError(f"definitions file not found: {path}")
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "definition_version" not in data:
            raise ConfigError(f"{path}: missing 'definition_version'")
        return cls(
            data=data,
            version=str(data["definition_version"]),
            source_path=str(path),
            content_sha256=sha256_text(text),
        )

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Fetch ``a.b.c`` from the definitions, failing loudly when absent."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not _MISSING:
                    return default
                raise ConfigError(f"{self.source_path}: missing definition '{dotted}'")
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, dict | list) else node

    # Frequently used scalars, named for readability at call sites.
    @property
    def contact_cutoff(self) -> float:
        return float(self.get("residue_contact.heavy_atom_cutoff"))

    @property
    def negative_margin(self) -> float:
        return float(self.get("distance.negative_example_margin"))

    @property
    def nearest_margin(self) -> float:
        return float(self.get("distance.nearest_neighbour_margin"))


@dataclass
class ProteinSpec:
    """One model-visible structure source in the dataset configuration."""

    id: str                                   # stable protein_group_id
    source_type: str                          # "pdb" | "afdb"
    entry: str                                # PDB ID or UniProt accession
    assembly_id: str | None = None            # None -> asymmetric unit / AFDB model
    chains: list[str] | None = None           # None -> all polymer chains
    keep_components: list[str] = field(default_factory=list)   # task-relevant HETATMs
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    #: Bootstrap clustering key (section 13). Structures of the same protein or
    #: from the same paper share a cluster so they are resampled together and so
    #: the per-protein instance cap counts them as one protein.
    cluster: str = ""
    #: chain -> allowed author-numbering ranges. Used to drop crystallisation
    #: fusions (T4 lysozyme, BRIL, flavodoxin) that are not part of the protein
    #: under study and would otherwise dominate an alignment.
    residue_ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    #: chain -> [(first, last, new_chain)]. Moves a fused peptide into its own
    #: chain so it can be excluded from residue mapping (A.27).
    split_chains: dict[str, list[tuple[int, int, str]]] = field(default_factory=dict)

    @property
    def uses_assembly(self) -> bool:
        return self.assembly_id is not None


@dataclass
class StatePairSpec:
    """A matched pair of experimentally observed structures for two-state tasks."""

    id: str
    state1: ProteinSpec
    state2: ProteinSpec
    cluster: str = ""
    #: state-1 chain -> state-2 chain; inferred from chain order when omitted.
    chain_map: dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class DatasetConfig:
    """The versioned dataset build configuration."""

    name: str
    version: str
    seed: int
    token_budget_automatic: int
    token_budget_mechanistic: int
    tokenizer: str
    rotation_variant_fraction: float
    representation_variant_fraction: float
    max_instances_per_protein: int
    family_targets: dict[str, int]
    proteins: list[ProteinSpec]
    state_pairs: list[StatePairSpec]
    family_proteins: dict[str, list[str]]
    episodes: list[str]
    crop: dict[str, Any]
    source_path: str
    content_sha256: str
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> DatasetConfig:
        path = Path(path) if path is not None else DEFAULT_DATASET
        if not path.exists():
            raise ConfigError(f"dataset config not found: {path}")
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        for key in ("name", "version", "seed", "proteins"):
            if key not in raw:
                raise ConfigError(f"{path}: missing '{key}'")

        proteins = [_protein_from_dict(p, path) for p in raw["proteins"]]
        by_id = {p.id: p for p in proteins}
        if len(by_id) != len(proteins):
            raise ConfigError(f"{path}: duplicate protein ids")

        state_pairs = []
        for sp in raw.get("state_pairs", []):
            state_pairs.append(
                StatePairSpec(
                    id=sp["id"],
                    state1=_protein_from_dict(sp["state1"], path),
                    state2=_protein_from_dict(sp["state2"], path),
                    cluster=str(sp.get("cluster") or sp["id"]),
                    chain_map={str(k): str(v) for k, v in (sp.get("chain_map") or {}).items()},
                    notes=sp.get("notes", ""),
                )
            )

        family_proteins = {k: list(v) for k, v in raw.get("family_proteins", {}).items()}
        unknown = {
            pid
            for members in family_proteins.values()
            for pid in members
            if pid not in by_id
        }
        if unknown:
            raise ConfigError(f"{path}: family_proteins references unknown proteins {sorted(unknown)}")

        budgets = raw.get("token_budget", {})
        return cls(
            name=raw["name"],
            version=str(raw["version"]),
            seed=int(raw["seed"]),
            token_budget_automatic=int(budgets.get("automatic", 64000)),
            token_budget_mechanistic=int(budgets.get("mechanistic", 96000)),
            tokenizer=str(budgets.get("tokenizer", "cl100k_base")),
            rotation_variant_fraction=float(raw.get("rotation_variant_fraction", 0.2)),
            representation_variant_fraction=float(raw.get("representation_variant_fraction", 0.35)),
            max_instances_per_protein=int(raw.get("max_instances_per_protein", 6)),
            family_targets={str(k): int(v) for k, v in raw.get("family_targets", {}).items()},
            proteins=proteins,
            state_pairs=state_pairs,
            family_proteins=family_proteins,
            episodes=list(raw.get("episodes", [])),
            crop=raw.get("crop", {}),
            source_path=str(path),
            content_sha256=sha256_text(text),
            raw=raw,
        )

    def protein(self, protein_id: str) -> ProteinSpec:
        for p in self.proteins:
            if p.id == protein_id:
                return p
        raise ConfigError(f"unknown protein id '{protein_id}' in {self.source_path}")

    def proteins_for(self, family: str) -> list[ProteinSpec]:
        """Protein candidates for a family, in configuration order."""
        ids = self.family_proteins.get(family)
        if ids is None:
            return list(self.proteins)
        return [self.protein(i) for i in ids]

    def state_pair(self, pair_id: str) -> StatePairSpec:
        for sp in self.state_pairs:
            if sp.id == pair_id:
                return sp
        raise ConfigError(f"unknown state pair '{pair_id}' in {self.source_path}")


def _protein_from_dict(d: dict[str, Any], path: Path) -> ProteinSpec:
    for key in ("id", "source_type", "entry"):
        if key not in d:
            raise ConfigError(f"{path}: protein entry missing '{key}': {d}")
    if d["source_type"] not in ("pdb", "afdb"):
        raise ConfigError(f"{path}: source_type must be 'pdb' or 'afdb', got {d['source_type']!r}")
    assembly = d.get("assembly_id")
    return ProteinSpec(
        id=str(d["id"]),
        cluster=str(d.get("cluster") or d["id"]),
        source_type=str(d["source_type"]),
        entry=str(d["entry"]),
        assembly_id=str(assembly) if assembly is not None else None,
        chains=[str(c) for c in d["chains"]] if d.get("chains") else None,
        keep_components=[str(c).upper() for c in d.get("keep_components", [])],
        notes=str(d.get("notes", "")),
        tags=[str(t) for t in d.get("tags", [])],
    )
