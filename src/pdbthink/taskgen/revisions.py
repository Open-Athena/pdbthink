"""Retain published task identities and splits when correcting or extending a release."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from ..prompts.library import PROMPT_VERSION, QUESTION_TEMPLATES, answer_format
from ..util import sha256_bytes


def read_parent(directory: Path) -> list[dict]:
    import pyarrow.parquet as pq

    manifest = json.loads((directory / "manifest.json").read_text())
    tasks = []
    for path in sorted((directory / "data").glob("*.parquet")):
        if sha256_bytes(path.read_bytes()) != manifest["data_hashes"][str(path.relative_to(directory))]:
            raise ValueError("parent Parquet checksum mismatch")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                if sha256_bytes(row["task_binary"]) != row["task_sha256"]:
                    raise ValueError("parent task checksum mismatch")
                task = json.loads(row["provenance"])
                with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:gz") as archive:
                    task["render"].update(json.load(archive.extractfile("prompt.json")))
                task.update(
                    semantic_key=row["semantic_key"],
                    parent_split=row["split"],
                    parent_source_group=row["source_group"],
                )
                tasks.append(task)
    if len(tasks) != manifest["task_count"]:
        raise ValueError("parent task count mismatch")
    return tasks


def update_prompt(task: dict) -> None:
    """Reject historical contracts that need a full prompt and oracle rebuild.

    The v6 revision changes the system instruction, residue legends and interface
    wording. The old suffix-only migration must not label those prompts as v6.
    Released tasks remain readable through ``read_parent``.
    """
    from ..prompts.library import SYSTEM_PROMPT

    render = task["render"]
    if render["prompt_version"] != PROMPT_VERSION:
        raise ValueError(
            f"parent prompt {render['prompt_version']} requires a full {PROMPT_VERSION} "
            "prompt and oracle rebuild; historical tasks cannot be relabelled"
        )
    if render["system_prompt"] != SYSTEM_PROMPT:
        raise ValueError("system prompt differs from the declared scientific contract")
    family = task["instance"]["question_family"]
    expected = answer_format(render["answer_schema"], family) + "\n"
    if not render["user_prompt"].endswith("\n\n" + expected):
        raise ValueError("prompt has an unexpected answer format")
    if family == "G04" and not render["user_prompt"].endswith(
        QUESTION_TEMPLATES["G04"] + "\n\n" + expected
    ):
        raise ValueError("clash prompt has an unexpected question")
