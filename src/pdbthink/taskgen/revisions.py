"""Retain published task identities and splits when correcting or extending a release."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from ..prompts.library import G04_QUESTION_V4, PROMPT_VERSION, QUESTION_TEMPLATES, answer_format
from ..representations.tokens import count_tokens
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
    """Migrate known prompt suffixes while preserving coordinates, gold and task identity."""
    render = task["render"]
    family = task["instance"]["question_family"]
    expected = answer_format(render["answer_schema"], family) + "\n"
    version = render["prompt_version"]
    original = render["user_prompt"]
    if version not in ("v3", "v4", PROMPT_VERSION):
        raise ValueError("unsupported parent prompt version")
    if version == "v3" and render["answer_schema"] == "category":
        old = "Answer with exactly one of the listed categories.\nExample: FINAL: helix\n"
        if not render["user_prompt"].endswith("\n\n" + old):
            raise ValueError("parent category prompt has an unexpected suffix")
        render["user_prompt"] = render["user_prompt"][: -len(old)] + expected
    elif not render["user_prompt"].endswith("\n\n" + expected):
        raise ValueError("prompt has an unexpected answer format")
    if family == "G04":
        question = QUESTION_TEMPLATES["G04"] if version == PROMPT_VERSION else G04_QUESTION_V4
        suffix = question + "\n\n" + expected
        if not render["user_prompt"].endswith(suffix):
            raise ValueError("clash prompt has an unexpected question")
        render["user_prompt"] = (
            render["user_prompt"][: -len(suffix)] + QUESTION_TEMPLATES["G04"] + "\n\n" + expected
        )
    if render["user_prompt"] != original or (version == "v3" and render["answer_schema"] == "category"):
        render["input_token_count"], render["tokenizer"] = count_tokens(
            render["system_prompt"] + "\n" + render["user_prompt"], render["tokenizer"]
        )
    render["prompt_version"] = PROMPT_VERSION
