"""Small shared helpers: hashing, deterministic RNG, JSON/JSONL I/O, paths."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(*parts: Any) -> str:
    """Deterministic hash of arbitrary JSON-serialisable parts."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)


def derive_seed(*parts: Any) -> int:
    """Derive a reproducible 63-bit seed from arbitrary identifiers.

    Used so that every rendered variant has a seed that depends only on its
    identity, never on iteration order or wall-clock time.
    """
    return int(stable_hash(*parts)[:16], 16) & ((1 << 63) - 1)


def rng_for(*parts: Any) -> np.random.Generator:
    """Versioned RNG (``pcg64_unit_quaternion_v1`` uses PCG64 seeded this way)."""
    return np.random.Generator(np.random.PCG64(derive_seed(*parts)))


def write_json(path: str | os.PathLike[str], obj: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=indent, sort_keys=False, default=_json_default)
    path.write_text(text + "\n", encoding="utf-8")


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=False, default=_json_default) + "\n")
            count += 1
    return count


def append_jsonl(path: str | os.PathLike[str], row: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=False, default=_json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - corrupt file
                raise ValueError(f"{p}:{line_no}: malformed JSON line") from exc


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def round3(value: float) -> float:
    """Round to the three decimals used by every model-visible representation.

    ``float(f"{v:.3f}")`` is used rather than :func:`round` so that the stored
    value matches the rendered text exactly (banker's rounding in :func:`round`
    can disagree with ``%.3f`` formatting).
    """
    return float(f"{value:.3f}")


def round3_array(arr: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of :func:`round3` for coordinate arrays."""
    return np.array([[round3(v) for v in row] for row in np.atleast_2d(arr)], dtype=float)
