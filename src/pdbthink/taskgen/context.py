"""Count exact model inputs, including the pinned tokenizer's chat template."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from ..util import sha256_bytes, write_json

TOKENIZER = None


def _init(model: str, revision: str) -> None:
    from transformers import AutoTokenizer

    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)


def _count(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:gz") as archive:
                prompt = json.load(archive.extractfile("prompt.json"))
            tokens = TOKENIZER.apply_chat_template(
                [
                    {"role": "system", "content": prompt["system_prompt"]},
                    {"role": "user", "content": prompt["user_prompt"]},
                ],
                tokenize=True,
                truncation=False,
                return_dict=True,
                add_generation_prompt=True,
                enable_thinking=True,
                tools=[],
            )
            rows.append(
                {
                    "path": row["path"],
                    "family": row["family"],
                    "split": row["split"],
                    "source_group": row["source_group"],
                    "prompt_sha256": row["prompt_sha256"],
                    "input_tokens": len(tokens["input_ids"]),
                }
            )
    return rows


def audit_context(dataset: Path, *, model: str, revision: str, workers: int = 8) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _init(model, revision)
    rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_init, initargs=(model, revision)
    ) as executor:
        for result in executor.map(_count, sorted((dataset / "data").glob("*.parquet"))):
            rows.extend(result)
            print("tokenised", len(rows), flush=True)
    context = 32768
    families, splits = defaultdict(Counter), defaultdict(Counter)
    groups = defaultdict(set)
    for row in rows:
        row["available_output_tokens"] = max(0, context - row["input_tokens"])
        flags = {"total": True, "prompt_under_32k": row["input_tokens"] < context}
        flags.update(
            {
                f"room_for_{reserve}": row["input_tokens"] + reserve <= context
                for reserve in (1024, 4096, 8192, 16384)
            }
        )
        for name, selected in flags.items():
            families[row["family"]][name] += int(selected)
            splits[row["split"]][name] += int(selected)
            if selected:
                groups[name].add(row["source_group"])
    totals = sum(families.values(), Counter())
    report = {
        "model_id": model,
        "model_revision": revision,
        "context_window": context,
        "enable_thinking": True,
        "add_generation_prompt": True,
        "chat_template_sha256": sha256_bytes(TOKENIZER.chat_template.encode()),
        "tokenizer_sha256": sha256_bytes(TOKENIZER.backend_tokenizer.to_str().encode()),
        "software": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
        "counts": dict(totals),
        "families": dict(families),
        "splits": dict(splits),
        "source_groups": {k: len(v) for k, v in groups.items()},
        "note": "Exact input tokens. SFT fit also depends on the actual teacher completion length. "
        "Output allowances include reasoning and the final answer; these are cohort filters, "
        "not imposed model generation caps.",
    }
    pq.write_table(pa.Table.from_pylist(rows), dataset / "snowball_context.parquet", compression="zstd")
    write_json(dataset / "snowball_context.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = audit_context(args.dataset, model=args.model, revision=args.revision, workers=args.workers)
    print(json.dumps({k: result[k] for k in ("counts", "families", "splits")}, indent=2))


if __name__ == "__main__":
    main()
