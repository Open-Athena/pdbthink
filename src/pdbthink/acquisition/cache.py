"""Content-addressed cache of downloaded PDB and AlphaFold DB structures.

Network access is required here and nowhere else in the build path: once a file
is cached, ``build``, ``validate``, ``score`` and ``report`` run entirely
offline (specification section 3).
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..util import REPO_ROOT, sha256_bytes

RCSB_FILE_URL = "https://files.rcsb.org/download/{entry}.cif.gz"
RCSB_ENTRY_API = "https://data.rcsb.org/rest/v1/core/entry/{entry}"
AFDB_FILE_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F{frag}-model_v{ver}.cif"
AFDB_VERSIONS = (6, 4, 3)

USER_AGENT = "pdbthink/0.1 (structural reasoning benchmark; +https://github.com/Open-Athena/pdbthink)"
DEFAULT_CACHE = REPO_ROOT / "data" / "cache"


class AcquisitionError(RuntimeError):
    """Raised when a source structure cannot be obtained."""


@dataclass
class SourceRecord:
    """Provenance for one cached source file. Private: never model-visible."""

    key: str                      # "pdb:4MQS" or "afdb:P69905"
    source_type: str              # "pdb" | "afdb"
    entry: str
    path: str
    sha256: str
    bytes: int
    url: str
    release_date: str | None = None
    deposit_date: str | None = None
    experimental_method: str | None = None
    resolution: float | None = None
    title: str | None = None
    publications: list[str] = field(default_factory=list)
    assembly_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructureCache:
    """Downloads structures once and remembers where they came from."""

    def __init__(self, root: str | Path | None = None, *, offline: bool = False) -> None:
        self.root = Path(root) if root else DEFAULT_CACHE
        self.files = self.root / "structures"
        self.metadata = self.root / "metadata"
        self.offline = offline
        self.files.mkdir(parents=True, exist_ok=True)
        self.metadata.mkdir(parents=True, exist_ok=True)

    # -- public API --------------------------------------------------------
    def get_pdb(self, entry: str) -> SourceRecord:
        entry = entry.strip().upper()
        record_path = self.metadata / f"pdb_{entry}.json"
        if record_path.exists():
            return SourceRecord(**json.loads(record_path.read_text()))
        if self.offline:
            raise AcquisitionError(f"pdb:{entry} not cached and cache is offline")

        url = RCSB_FILE_URL.format(entry=entry)
        raw = _download(url)
        text = gzip.decompress(raw)
        dest = self.files / f"{entry}.cif"
        dest.write_bytes(text)

        record = SourceRecord(
            key=f"pdb:{entry}",
            source_type="pdb",
            entry=entry,
            path=str(dest),
            sha256=sha256_bytes(text),
            bytes=len(text),
            url=url,
        )
        _annotate_from_rcsb(record, self._entry_metadata(entry))
        record_path.write_text(json.dumps(record.to_dict(), indent=2))
        return record

    def get_afdb(self, accession: str, *, fragment: int = 1) -> SourceRecord:
        accession = accession.strip().upper()
        record_path = self.metadata / f"afdb_{accession}.json"
        if record_path.exists():
            return SourceRecord(**json.loads(record_path.read_text()))
        if self.offline:
            raise AcquisitionError(f"afdb:{accession} not cached and cache is offline")

        last_error: Exception | None = None
        for version in AFDB_VERSIONS:
            url = AFDB_FILE_URL.format(acc=accession, frag=fragment, ver=version)
            try:
                text = _download(url)
            except AcquisitionError as exc:
                last_error = exc
                continue
            dest = self.files / f"AF-{accession}-F{fragment}-v{version}.cif"
            dest.write_bytes(text)
            record = SourceRecord(
                key=f"afdb:{accession}",
                source_type="afdb",
                entry=accession,
                path=str(dest),
                sha256=sha256_bytes(text),
                bytes=len(text),
                url=url,
                experimental_method="PREDICTED (AlphaFold DB)",
                extra={"afdb_version": version, "fragment": fragment},
            )
            record_path.write_text(json.dumps(record.to_dict(), indent=2))
            return record
        raise AcquisitionError(f"afdb:{accession} not available: {last_error}")

    def get(self, source_type: str, entry: str) -> SourceRecord:
        if source_type == "pdb":
            return self.get_pdb(entry)
        if source_type == "afdb":
            return self.get_afdb(entry)
        raise AcquisitionError(f"unknown source_type {source_type!r}")

    def records(self) -> list[SourceRecord]:
        out = []
        for p in sorted(self.metadata.glob("*.json")):
            out.append(SourceRecord(**json.loads(p.read_text())))
        return out

    # -- internals ---------------------------------------------------------
    def _entry_metadata(self, entry: str) -> dict[str, Any]:
        path = self.metadata / f"rcsb_api_{entry}.json"
        if path.exists():
            return json.loads(path.read_text())
        try:
            payload = json.loads(_download(RCSB_ENTRY_API.format(entry=entry)))
        except (AcquisitionError, json.JSONDecodeError):
            return {}
        path.write_text(json.dumps(payload, indent=2))
        return payload


def _download(url: str, *, attempts: int = 4, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                raise AcquisitionError(f"{url}: HTTP {exc.code}") from exc
            last = exc
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise AcquisitionError(f"{url}: {last}")


def _annotate_from_rcsb(record: SourceRecord, payload: dict[str, Any]) -> None:
    if not payload:
        return
    accession = payload.get("rcsb_accession_info", {})
    record.release_date = _date_only(accession.get("initial_release_date"))
    record.deposit_date = _date_only(accession.get("deposit_date"))
    methods = payload.get("exptl") or []
    if methods:
        record.experimental_method = methods[0].get("method")
    entry_info = payload.get("rcsb_entry_info", {})
    resolutions = entry_info.get("resolution_combined") or []
    if resolutions:
        record.resolution = float(resolutions[0])
    record.title = (payload.get("struct") or {}).get("title")
    pubs = []
    primary = payload.get("rcsb_primary_citation") or {}
    for key in ("pdbx_database_id_DOI", "pdbx_database_id_PubMed"):
        value = primary.get(key)
        if value:
            pubs.append(f"{key}:{value}")
    if primary.get("year"):
        pubs.append(f"year:{primary['year']}")
    record.publications = pubs
    assemblies = payload.get("rcsb_entry_container_identifiers", {}).get("assembly_ids") or []
    record.assembly_ids = [str(a) for a in assemblies]
    record.extra["polymer_entity_count"] = entry_info.get("polymer_entity_count")
    record.extra["deposited_atom_count"] = entry_info.get("deposited_atom_count")


def _date_only(value: str | None) -> str | None:
    if not value:
        return None
    return value.split("T")[0]
