"""Freeze teacher inputs and score with the verifier shipped in the task release."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import io
import json
import sqlite3
import sys
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DATASET_REVISION = "fbd07fe7255f1f65d4d860c7561d5ae03110d1e9"
TEACHER_MODEL = "zai-org/GLM-5.3"
TEACHER_REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
STUDENT_MODEL = "open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38"
STUDENT_REVISION = "cfc1d845dae89b067cdc7250d0164abefa5a69cf"
CONTEXT = 262144
STUDENT_CONTEXT = 32768
MAX_ATTEMPTS = 10
EFFORT = "high"
TOKENIZER = None
COHORT = None


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Status readers and the controller may publish progress concurrently.
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def connect(root: Path) -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(root / "state.sqlite")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    try:
        with db:
            yield db
    finally:
        db.close()


def messages(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": task["user_prompt"]},
    ]


def init_prepare(cohort: dict) -> None:
    from transformers import AutoTokenizer

    global TOKENIZER, COHORT
    TOKENIZER = AutoTokenizer.from_pretrained(TEACHER_MODEL, revision=TEACHER_REVISION, local_files_only=True)
    COHORT = cohort


def prepare_shard(path: Path) -> tuple[list[dict], dict[str, bytes]]:
    import pyarrow.parquet as pq

    rows, verifier = [], {}
    columns = ["path", "family", "source_group", "prompt_sha256", "task_sha256", "task_binary"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=columns):
        for row in batch.to_pylist():
            if row["path"] not in COHORT:
                continue
            if digest(row["task_binary"]) != row["task_sha256"]:
                raise ValueError("Task archive checksum mismatch")
            with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:gz") as archive:
                prompt = json.load(archive.extractfile("prompt.json"))
                gold = json.load(archive.extractfile("tests/gold.json"))
                for member in archive.getmembers():
                    if (
                        member.name.startswith("tests/coordinate_scoring/")
                        or member.name == "tests/verify.py"
                    ):
                        content = archive.extractfile(member).read()
                        if member.name in verifier and verifier[member.name] != content:
                            raise ValueError("Mixed native verifier implementations")
                        verifier[member.name] = content
            prompt_hash = digest((prompt["system_prompt"] + "\n" + prompt["user_prompt"]).encode())
            if prompt_hash != row["prompt_sha256"] or prompt_hash != COHORT[row["path"]]["prompt_sha256"]:
                raise ValueError("Frozen prompt identity mismatch")
            encoded = TOKENIZER.apply_chat_template(
                messages(prompt),
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                tools=[],
                reasoning_effort=EFFORT,
            )["input_ids"]
            rows.append(
                {
                    "path": row["path"],
                    "family": row["family"],
                    "source_group": row["source_group"],
                    "prompt_sha256": prompt_hash,
                    "task_sha256": row["task_sha256"],
                    **prompt,
                    "gold_json": json.dumps(gold, sort_keys=True),
                    "teacher_input_tokens": len(encoded),
                    "student_input_tokens": COHORT[row["path"]]["input_tokens"],
                    "split": "train",
                }
            )
    return rows, verifier


def prepare(dataset: Path, root: Path, workers: int = 8) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        print("Prepared cohort already frozen", flush=True)
        return
    release = json.loads((dataset / "manifest.json").read_text())
    if release["task_set_fingerprint"] != "fca36af64aeb6b92e671062b7415948508caac785de845148f60bfc5f1f3d76d":
        raise ValueError("Expected the pinned v1.2.0 task population")
    train_shards = sorted((dataset / "data").glob("train-*.parquet"))
    for shard in train_shards:
        if digest(shard.read_bytes()) != release["data_hashes"][str(shard.relative_to(dataset))]:
            raise ValueError("Source Parquet differs from the frozen release")
    save_json(
        root / "source-validation.json",
        {
            "all_hashes_match": True,
            "dataset_revision": DATASET_REVISION,
            "task_set_fingerprint": release["task_set_fingerprint"],
            "verified_train_shards": len(train_shards),
        },
    )
    context = pq.read_table(dataset / "snowball_context.parquet").to_pylist()
    cohort = {
        r["path"]: r for r in context if r["split"] == "train" and r["input_tokens"] + 8192 <= STUDENT_CONTEXT
    }
    if len(cohort) != 28045:
        raise ValueError(f"Expected 28045 training tasks, found {len(cohort)}")
    all_rows, native_files = [], {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=init_prepare, initargs=(cohort,)
    ) as pool:
        for rows, files in pool.map(prepare_shard, train_shards):
            all_rows.extend(rows)
            for name, content in files.items():
                if name in native_files and native_files[name] != content:
                    raise ValueError("Mixed native verifier implementations")
                native_files[name] = content
            print("prepared", len(all_rows), "/", len(cohort), flush=True)
    all_rows.sort(key=lambda r: r["path"])
    if {r["path"] for r in all_rows} != set(cohort):
        raise ValueError("Missing or unexpected cohort task")
    for name, content in native_files.items():
        target = root / "native_verifier" / name.removeprefix("tests/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    pq.write_table(pa.Table.from_pylist(all_rows), root / "tasks.parquet", compression="zstd")
    with connect(root) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            path TEXT PRIMARY KEY, family TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            solved INTEGER NOT NULL DEFAULT 0, active_batch TEXT, infra_errors INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS batches (
            id TEXT PRIMARY KEY, directory TEXT NOT NULL, status TEXT NOT NULL, submitted TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts (
            path TEXT NOT NULL, attempt INTEGER NOT NULL, reward REAL NOT NULL, result_path TEXT NOT NULL,
            PRIMARY KEY(path, attempt)
        );
        """)
        db.executemany(
            "INSERT OR IGNORE INTO tasks(path,family) VALUES (?,?)",
            [(r["path"], r["family"]) for r in all_rows],
        )
    init_prepare(cohort)
    manifest = {
        "version": "1.0.0",
        "dataset": "open-athena/pdbthink-coordinate-tasks",
        "dataset_revision": DATASET_REVISION,
        "task_count": len(all_rows),
        "family_counts": dict(sorted(Counter(r["family"] for r in all_rows).items())),
        "cohort": "train split; exact Snowball prompt tokens + 8192 <= 32768",
        "teacher_model": "glm-5.3",
        "teacher_weights": "FP8",
        "teacher_weight_revision": None,
        "teacher_tokenizer": TEACHER_MODEL,
        "teacher_tokenizer_revision": TEACHER_REVISION,
        "teacher_chat_template_sha256": digest(TOKENIZER.chat_template.encode()),
        "teacher_context_window": CONTEXT,
        "reasoning_effort": EFFORT,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_attempts": MAX_ATTEMPTS,
        "retry_policy": (
            "Fresh independent sample of the unchanged prompt; stop at first native-verifier success. "
            "Infrastructure errors do not consume a scored attempt."
        ),
        "output_budget_policy": (
            "Full remaining teacher context: 262144 minus exact input tokens. "
            "Reasoning and answer share the budget."
        ),
        "student_model": STUDENT_MODEL,
        "student_tokenizer_revision": STUDENT_REVISION,
        "student_context_window": STUDENT_CONTEXT,
        "tasks_sha256": digest((root / "tasks.parquet").read_bytes()),
        "native_verifier_hashes": {k: digest(v) for k, v in sorted(native_files.items())},
        "seed_policy": "SHA256(task path, attempt, version) truncated to 31 bits",
        "tools": None,
        "tool_choice": "none",
    }
    save_json(root / "manifest.json", manifest)


def load_native_scorer(root: Path):
    package = root / "native_verifier/coordinate_scoring"
    spec = importlib.util.spec_from_file_location(
        "teacher_native_scoring", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.score_response


def request_for(task: dict, attempt: int) -> dict:
    seed = int(digest(f"{task['path']}:{attempt}:teacher-v1".encode())[:8], 16) % (2**31)
    return {
        "model": "glm-5.3",
        "messages": messages(task),
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": seed,
        "max_tokens": CONTEXT - task["teacher_input_tokens"],
        "chat_template_kwargs": {"reasoning_effort": EFFORT},
        "tools": None,
        "tool_choice": "none",
        "prompt_cache_key": "pdbthink-teacher-" + task["prompt_sha256"],
    }


def score_completion(task: dict, body: dict, scorer) -> dict:
    from .protocol import tool_events

    choice = body["choices"][0]
    message = choice["message"]
    content, reasoning = (
        message.get("content") or "",
        message.get("reasoning_content") or message.get("reasoning") or "",
    )
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise ValueError("Expected text answer and text reasoning")
    contract = json.loads(task["gold_json"])
    outcome = scorer(
        content,
        contract["answer_schema"],
        contract["gold_answer"],
        parameters=contract["parameters"],
        truncated=choice.get("finish_reason") == "length",
        provider_refusal=bool(message.get("refusal")),
    )
    events = tool_events(body)
    return {
        "reward": float(outcome["score"].get("correct", False) and not events),
        "outcome": outcome,
        "tool_events": events,
        "content": content,
        "reasoning": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.dataset, args.root, args.workers)


if __name__ == "__main__":
    main()
