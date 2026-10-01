"""Freeze benchmark identities before acquiring any post-training structures.

The exclusion unit is deliberately broader than a rendered question: all
configured benchmark sources, their polymer sequences and their RCSB 30%
sequence clusters are excluded, including partners in heteromeric entries.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import gemmi
import yaml

from ..acquisition.cache import StructureCache
from ..util import sha256_bytes, stable_hash, write_json


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def polymer_sequences(path: str | Path) -> dict[str, str]:
    """Capture both complete entity sequences and the observed chain sequences."""
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    sequences = {}
    for entity in st.entities:
        if entity.polymer_type not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            continue
        seq = "".join(gemmi.find_tabulated_residue(x).one_letter_code.upper() for x in entity.full_sequence)
        if seq:
            sequences[f"entity:{entity.name}"] = seq
    if len(st):
        for chain in st[0]:
            seq = "".join(
                gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                for r in chain
                if gemmi.find_tabulated_residue(r.name).is_amino_acid()
            )
            if seq:
                sequences[f"chain:{chain.name}"] = seq
    return sequences


def clusters(path: Path) -> dict[str, str]:
    return {
        entity: "rcsb30-" + hashlib.sha256(" ".join(sorted(line.split())).encode()).hexdigest()[:20]
        for line in path.read_text().splitlines()
        for entity in line.split()
    }


def _identities(value: Any, entries: set[str], sequences: set[str], files: set[str]) -> None:
    if isinstance(value, dict):
        source = value.get("source_type")
        if source in ("pdb", "afdb") and value.get("entry"):
            entries.add(f"{source}:{str(value['entry']).upper()}")
        if source in ("pdb", "afdb"):
            entries.update(f"{source}:{str(x).upper()}" for x in value.get("source_entries", []))
        for key, item in value.items():
            if key in ("sequence", "target_sequence", "amino_acid_sequence") and isinstance(item, str):
                seq = re.sub(r"\s", "", item).upper()
                if seq and re.fullmatch("[A-Z]+", seq):
                    sequences.add(seq)
            if key == "source_file_sha256s" and isinstance(item, list):
                files.update(item)
            _identities(item, entries, sequences, files)
    elif isinstance(value, list):
        for item in value:
            _identities(item, entries, sequences, files)


def freeze_exclusions(repo: Path, destination: Path, cluster_file: Path) -> dict:
    """Fail closed if any frozen benchmark source cannot be sequence-audited."""
    entries: set[str] = set()
    sequences: set[str] = set()
    files: set[str] = set()
    inventory = {}
    paths = set(repo.glob("configs/*.yaml")) | set(repo.glob("configs/*panel.json"))
    paths |= set(repo.glob("data/manifests/*.yaml"))
    paths |= set(repo.glob("data/datasets/*/instances.jsonl"))
    paths |= set(repo.glob("data/prediction_sources/*/sources.json"))
    for path in sorted(paths):
        raw = path.read_bytes()
        inventory[str(path.relative_to(repo))] = sha256_bytes(raw)
        if path.suffix == ".jsonl":
            value = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            value = yaml.safe_load(raw)
        _identities(value, entries, sequences, files)
    cache = StructureCache(repo / "data/cache", offline=True)
    missing = []
    for source in sorted(entries):
        kind, entry = source.split(":", 1)
        try:
            record = cache.get(kind, entry)
            path = Path(record.path)
            if not path.exists():
                path = cache.files / path.name
            files.add(sha256_bytes(path.read_bytes()))
            sequences.update(polymer_sequences(path).values())
        except Exception as exc:
            missing.append({"source": source, "error": str(exc)})
    if missing:
        write_json(destination.with_suffix(".missing.json"), missing)
        raise ValueError(
            f"cannot audit {len(missing)} benchmark sources; see {destination.with_suffix('.missing.json')}"
        )
    mapping = clusters(cluster_file)
    excluded_clusters = sorted(
        {cluster for entity, cluster in mapping.items() if f"pdb:{entity.split('_')[0]}" in entries}
    )
    result = {
        "version": "1.0.0",
        "policy": "all-configured-sources+exact-entity-and-observed-sequences+rcsb30",
        "sources": sorted(entries),
        "source_file_sha256s": sorted(files),
        "sequence_sha256s": sorted({sequence_hash(s) for s in sequences}),
        "rcsb30_clusters": excluded_clusters,
        "cluster_file_sha256": sha256_bytes(cluster_file.read_bytes()),
        "inventory": inventory,
        "missing_sources": [],
    }
    result["fingerprint"] = stable_hash(result)
    write_json(destination, result)
    return result


def audit_source(record, exclusion: dict, entity_clusters: dict[str, str]) -> dict:
    """Audit every protein partner, even when the eventual prompt selects one chain."""
    reasons = []
    entry = record.entry.upper()
    if f"{record.source_type}:{entry}" in exclusion["sources"]:
        reasons.append("benchmark_source_entry")
    raw_hash = sha256_bytes(Path(record.path).read_bytes())
    if raw_hash in exclusion["source_file_sha256s"]:
        reasons.append("benchmark_source_file")
    seqs = polymer_sequences(record.path)
    if not seqs:
        reasons.append("no_auditable_protein_sequence")
    old = set(exclusion["sequence_sha256s"])
    overlap = sorted(key for key, seq in seqs.items() if sequence_hash(seq) in old)
    if overlap:
        reasons.append("benchmark_protein_sequence")
    entry_clusters = sorted(
        {
            entity_clusters[f"{entry}_{key.split(':', 1)[1]}"]
            for key in seqs
            if key.startswith("entity:") and f"{entry}_{key.split(':', 1)[1]}" in entity_clusters
        }
    )
    if not entry_clusters:
        reasons.append("missing_rcsb30_assignment")
    if set(entry_clusters) & set(exclusion["rcsb30_clusters"]):
        reasons.append("benchmark_rcsb30_cluster")
    return {
        "entry": entry,
        "allowed": not reasons,
        "reasons": reasons,
        "sequence_overlaps": overlap,
        "sequence_sha256s": sorted({sequence_hash(s) for s in seqs.values()}),
        "rcsb30_clusters": entry_clusters,
        "source_file_sha256": raw_hash,
        "exclusion_fingerprint": exclusion["fingerprint"],
    }
