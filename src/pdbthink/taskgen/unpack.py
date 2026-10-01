"""Unpack selected Parquet rows as local Harbor tasks without exposing solutions."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path


def unpack_release(
    dataset: Path,
    output: Path,
    *,
    split: str = "train",
    families: list[str] | None = None,
    limit: int | None = None,
    include_solutions: bool = False,
) -> int:
    import pyarrow.parquet as pq

    files = sorted((dataset / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise ValueError(f"no Parquet files for split {split}")
    count = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                if families and row["family"] not in families:
                    continue
                name = row["path"]
                if Path(name).name != name or name in (".", ".."):
                    raise ValueError("unsafe task directory name")
                destination = output / name
                destination.mkdir(parents=True, exist_ok=False)
                columns = ["task_binary", "solution_binary"] if include_solutions else ["task_binary"]
                for column in columns:
                    with tarfile.open(fileobj=io.BytesIO(row[column]), mode="r:gz") as archive:
                        archive.extractall(destination, filter="data")
                count += 1
                if limit is not None and count >= limit:
                    return count
    return count
