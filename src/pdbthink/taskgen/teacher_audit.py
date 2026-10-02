"""Independently check a completed teacher release against its saved run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .teacher_data import STUDENT_MODEL, STUDENT_REVISION, request_for


def audit(root: Path, release: Path, output: Path) -> dict:
    if not __debug__:
        raise RuntimeError("Release audits require Python assertions; do not use python -O")
    manifest = json.loads((release / "manifest.json").read_text())
    summary = json.loads((release / "summary.json").read_text())
    assert summary["complete"] and not summary["pending"]
    tasks = {r["path"]: r for r in pq.read_table(root / "tasks.parquet").to_pylist()}
    assert len(tasks) == manifest["task_count"] == 28045
    request_rows = 0
    for directory in sorted((root / "batches").iterdir()):
        if not (directory / "job.json").exists():
            continue
        selection = json.loads((directory / "selection.json").read_text())
        submitted = [
            json.loads(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(submitted) == len(selection)
        for i, (item, request) in enumerate(zip(selection, submitted)):
            assert request["custom_id"] == str(i)
            assert request["method"] == "POST" and request["url"] == "/v1/chat/completions"
            assert request["body"] == request_for(tasks[item["path"]], item["attempt"])
            request_rows += 1
    assert request_rows == summary["attempt_count"] + summary["infrastructure_errors"]
    attempts = defaultdict(list)
    sample = {}
    metrics = Counter()
    for shard in sorted((release / "attempts").glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=32):
            for r in batch.to_pylist():
                t = tasks[r["task_id"]]
                assert r["messages"][:2] == [
                    {"role": "system", "content": t["system_prompt"]},
                    {"role": "user", "content": t["user_prompt"]},
                ]
                joined = r["reasoning"] + "\n\n" + r["answer"] if r["reasoning"] else r["answer"]
                assert r["messages"][-1] == {"role": "assistant", "content": joined}
                assert r["prompt_sha256"] == t["prompt_sha256"]
                assert r["task_sha256"] == t["task_sha256"]
                assert r["dataset_revision"] == manifest["dataset_revision"]
                assert r["source_group"] == t["source_group"] and r["family"] == t["family"]
                assert r["source_split"] == "train"
                assert r["student_input_tokens"] <= 32768 - 8192
                assert r["fits_student_context"] == (r["student_total_tokens"] <= 32768)
                assert r["completion_within_8k"] == (r["student_completion_tokens"] <= 8192)
                assert r["reward"] in (0.0, 1.0)
                if r["reward"]:
                    assert not r["tool_violation"] and r["finish_reason"] != "length"
                assert r["request_max_tokens"] + r["teacher_input_tokens"] == 262144
                body = json.loads(r["raw_response_json"])
                msg = body["choices"][0]["message"]
                assert r["answer"] == (msg.get("content") or "")
                assert r["reasoning"] == (msg.get("reasoning_content") or msg.get("reasoning") or "")
                assert body["usage"]["prompt_tokens"] == r["teacher_input_tokens"]
                score = json.loads(r["score_json"])
                assert r["reward"] == float(bool(score["score"]["correct"]) and not r["tool_violation"])
                attempts[r["task_id"]].append(
                    {
                        k: r[k]
                        for k in (
                            "attempt",
                            "reward",
                            "fits_student_context",
                            "student_total_tokens",
                            "student_completion_tokens",
                            "completion_within_8k",
                            "has_reasoning",
                        )
                    }
                )
                metrics["attempts"] += 1
                metrics["solved"] += bool(r["reward"])
                metrics["first_try"] += bool(r["reward"]) and r["attempt"] == 1
                metrics["correct_over_context"] += bool(r["reward"]) and not r["fits_student_context"]
                if (
                    r["family"] not in sample
                    or r["student_total_tokens"] > sample[r["family"]]["student_total_tokens"]
                ):
                    sample[r["family"]] = r
    assert set(attempts) == set(tasks)
    expected_sft = set()
    for path, rows in attempts.items():
        rows.sort(key=lambda r: r["attempt"])
        assert [r["attempt"] for r in rows] == list(range(1, len(rows) + 1))
        assert 1 <= len(rows) <= 10
        correct = [r for r in rows if r["reward"]]
        assert len(correct) <= 1
        if correct:
            assert rows[-1]["reward"] == 1
            if rows[-1]["fits_student_context"]:
                expected_sft.add(path)
        else:
            assert len(rows) == 10
            metrics["unsolved_after_10"] += 1
    seen = set()
    for shard in sorted((release / "sft").glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=32):
            for r in batch.to_pylist():
                assert r["task_id"] not in seen
                seen.add(r["task_id"])
                assert r["task_id"] in expected_sft
                assert r["reward"] == 1 and r["student_total_tokens"] <= 32768
                expected = attempts[r["task_id"]][-1]
                for key, value in expected.items():
                    assert r[key] == value
                metrics["sft_traces"] += 1
                metrics["sft_traces_with_reasoning"] += r["has_reasoning"]
                metrics["sft_traces_completion_within_8k"] += r["completion_within_8k"]
    assert seen == expected_sft
    outcomes = pq.read_table(release / "outcomes").to_pylist()
    assert len(outcomes) == len(tasks) and {r["task_id"] for r in outcomes} == set(tasks)
    for r in outcomes:
        rows = attempts[r["task_id"]]
        assert r["attempts"] == len(rows)
        solved = bool(rows[-1]["reward"])
        assert r["status"] == ("solved" if solved else "unsolved_after_10")
        assert r["first_correct_attempt"] == (len(rows) if solved else None)
        assert r["sft_eligible"] == (r["task_id"] in expected_sft)
    for key in (
        "solved",
        "first_try",
        "correct_over_context",
        "unsolved_after_10",
        "sft_traces",
        "sft_traces_with_reasoning",
        "sft_traces_completion_within_8k",
    ):
        assert metrics[key] == summary[key], (key, metrics[key], summary[key])
    assert metrics["attempts"] == summary["attempt_count"]
    assert metrics["solved"] + metrics["unsolved_after_10"] == 28045
    # Independently render then encode the longest full response from every represented family.
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, revision=STUDENT_REVISION, local_files_only=True)
    for family, r in sample.items():
        text = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=True, tools=[]
        )
        count = len(tokenizer.encode(text, add_special_tokens=False))
        assert count == r["student_total_tokens"], (family, count, r["student_total_tokens"])
    for relative, expected in json.loads((release / "checksums.json").read_text()).items():
        assert hashlib.sha256((release / relative).read_bytes()).hexdigest() == expected
    receipt = {
        "passed": True,
        "cohort_tasks": len(tasks),
        "scored_attempts": metrics["attempts"],
        "submitted_request_rows": request_rows,
        "checks": [
            "actual_request_prompts_seeds_tool_controls_sampling_and_native_budgets",
            "frozen_prompt_identity",
            "source_split_and_cohort",
            "contiguous_attempts",
            "stop_on_success",
            "ten_attempt_exhaustion",
            "raw_response_round_trip",
            "reward_and_tool_policy",
            "full_context_sft_selection",
            "outcome_population",
            "summary_totals",
            "published_file_checksums",
        ],
        "independent_render_then_encode_checks": len(sample),
        "independent_token_check_selection": "Longest returned response from every represented family",
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = audit(
        args.root, args.release or args.root / "release", args.output or args.root / "release-audit.json"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
