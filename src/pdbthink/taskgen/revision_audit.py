"""Compare a corrected release with its parent and replay the reported verifier failures."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import subprocess
import tarfile
import tempfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq

from ..prompts.library import PROMPT_VERSION, QUESTION_TEMPLATES
from ..util import REPO_ROOT, sha256_bytes, write_json
from .harbor import DOCKERFILE, gold_response
from .revisions import update_prompt

CASES = {
    "pdbthink-p03-8122893bee50bceae1e7",
    "pdbthink-p03-87b0ecaa6658baf7bf86",
    "pdbthink-g04-908a6b9e20e6ee9b26bd",
}


def archive_files(binary: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(binary), mode="r:gz") as archive:
        return {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}


def compare_shard(paths: tuple[Path, Path]) -> tuple[Counter, list[tuple[dict, dict]]]:
    old_path, new_path = paths
    counts = Counter()
    examples = []
    for old_batch, new_batch in zip(
        pq.ParquetFile(old_path).iter_batches(batch_size=16),
        pq.ParquetFile(new_path).iter_batches(batch_size=16),
        strict=True,
    ):
        for old, new in zip(old_batch.to_pylist(), new_batch.to_pylist(), strict=True):
            assert old.keys() == new.keys()
            changed_columns = {
                "dataset_version", "task_binary", "task_sha256", "prompt_sha256", "input_tokens", "provenance"
            }
            assert {k: v for k, v in old.items() if k not in changed_columns} == {
                k: v for k, v in new.items() if k not in changed_columns
            }
            assert old["dataset_version"] == "1.2.0" and new["dataset_version"] == "1.3.0"
            assert sha256_bytes(old["task_binary"]) == old["task_sha256"]
            assert sha256_bytes(new["task_binary"]) == new["task_sha256"]
            before, after = archive_files(old["task_binary"]), archive_files(new["task_binary"])
            assert before.keys() == after.keys()
            for name in before:
                if name not in {
                    "instruction.md", "prompt.json", "task.toml", "tests/gold.json", "tests/verifier.toml"
                } and not name.startswith("tests/coordinate_scoring/"):
                    assert before[name] == after[name], (old["path"], name)
            old_gold, new_gold = json.loads(before["tests/gold.json"]), json.loads(after["tests/gold.json"])
            assert old_gold["version"] == "1.0.0" and new_gold["version"] == "1.1.0"
            assert new_gold["scoring_version"] == "1.1.0"
            assert {k: v for k, v in old_gold.items() if k != "version"} == {
                k: v for k, v in new_gold.items() if k not in {"version", "scoring_version"}
            }
            previous, revised = json.loads(old["provenance"]), json.loads(new["provenance"])
            previous["render"].update(json.loads(before["prompt.json"]))
            revised["render"].update(json.loads(after["prompt.json"]))
            update_prompt(previous)
            assert previous == revised, old["path"]
            assert revised["render"]["prompt_version"] == PROMPT_VERSION
            assert revised["render"]["input_token_count"] == new["input_tokens"]
            assert sha256_bytes(
                (revised["render"]["system_prompt"] + "\n" + revised["render"]["user_prompt"]).encode()
            ) == new["prompt_sha256"]
            changed = old["prompt_sha256"] != new["prompt_sha256"]
            assert changed == (old["family"] == "G04")
            counts["tasks_compared"] += 1
            counts["changed_prompts"] += changed
            counts["unchanged_prompts"] += not changed
            counts["unchanged_gold_answers"] += 1
            counts["unchanged_splits_groups_and_solutions"] += 1
            if old["path"] in CASES:
                examples.append((old, new))
    return counts, examples


def native_verify(task: dict, answer: str) -> dict:
    """Execute the task's unchanged tests/test.sh inside its pinned, offline base image."""
    files = archive_files(task["task_binary"])
    image = DOCKERFILE.splitlines()[0].removeprefix("FROM ")
    with tempfile.TemporaryDirectory(prefix="pdbthink-verifier-") as temporary:
        root = Path(temporary)
        for name, content in files.items():
            if name.startswith("tests/"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        app, output = root / "app", root / "output"
        app.mkdir()
        output.mkdir()
        (app / "answer.txt").write_text(answer)
        result = subprocess.run(
            [
                "docker", "run", "--rm", "--network", "none", "--read-only",
                "--mount", f"type=bind,src={root / 'tests'},dst=/tests,readonly",
                "--mount", f"type=bind,src={app},dst=/app,readonly",
                "--mount", f"type=bind,src={output},dst=/logs/verifier",
                image, "sh", "/tests/test.sh",
            ],
            check=True, capture_output=True, text=True, timeout=60,
        )
        return {
            "details": json.loads((output / "details.json").read_text()),
            "reward": float((output / "reward.txt").read_text()),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "task_sha256": task["task_sha256"],
            "prompt_sha256": task["prompt_sha256"],
        }


def audit_revision(parent: Path, dataset: Path, output: Path, workers: int) -> dict:
    if not __debug__:
        raise RuntimeError("Run the audit with Python assertions enabled")
    old_manifest = json.loads((parent / "manifest.json").read_text())
    new_manifest = json.loads((dataset / "manifest.json").read_text())
    assert old_manifest["version"] == "1.2.0" and new_manifest["version"] == "1.3.0"
    assert set(old_manifest["data_hashes"]) == set(new_manifest["data_hashes"])
    for root, manifest in ((parent, old_manifest), (dataset, new_manifest)):
        for name, digest in manifest["data_hashes"].items():
            assert sha256_bytes((root / name).read_bytes()) == digest
    counts, examples = Counter(), []
    paths = [(parent / name, dataset / name) for name in sorted(old_manifest["data_hashes"])]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for shard_counts, shard_examples in executor.map(compare_shard, paths):
            counts.update(shard_counts)
            examples.extend(shard_examples)
    assert counts["tasks_compared"] == old_manifest["task_count"] == new_manifest["task_count"]
    assert counts["changed_prompts"] == old_manifest["family_counts"]["G04"]
    for field in (
        "family_counts", "split_counts", "source_entries", "source_groups", "exclusion_fingerprint"
    ):
        assert old_manifest[field] == new_manifest[field], field
    assert {old["path"] for old, _ in examples} == CASES
    saved = json.loads((REPO_ROOT / "docs/teacher-traces-glm53/tolerance-boundary-audit.json").read_text())
    numeric = {case["task_id"]: case["native_score"] for case in saved["cases"]}
    records = []
    for old, new in sorted(examples, key=lambda pair: pair[0]["path"]):
        if old["family"] == "P03":
            native = numeric[old["path"]]
            answer = gold_response("numeric_triple", {"value": native["predicted"]})
            before, after = native_verify(old, answer), native_verify(new, answer)
            assert before["reward"] == 0 and after["reward"] == 1
            outside = list(native["gold"])
            outside[0] = str(Decimal(str(outside[0])) + Decimal("0.0010001"))
            invalid = gold_response("numeric_triple", {"value": outside})
            control = native_verify(new, invalid)
            assert control["reward"] == 0
            records.append({
                "task_id": old["path"], "answer": answer, "previous": before, "revised": after,
                "outside_tolerance_answer": invalid, "outside_tolerance_control": control,
            })
        else:
            prompt = json.loads(archive_files(new["task_binary"])["prompt.json"])["user_prompt"]
            assert QUESTION_TEMPLATES["G04"] in prompt
            gold = json.loads(archive_files(new["task_binary"])["tests/gold.json"])["gold_answer"]
            answer = gold_response("residue_pair", gold)
            correct = native_verify(new, answer)
            excluded = native_verify(new, "FINAL: A:C11--A:C50\n")
            assert correct["reward"] == 1 and excluded["reward"] == 0
            records.append({
                "task_id": old["path"], "clarified_question": QUESTION_TEMPLATES["G04"],
                "correct_answer": answer, "correct_control": correct,
                "excluded_sulfur_pair_control": excluded,
            })
    report = {
        "passed": True,
        "parent_version": "1.2.0", "version": "1.3.0",
        "parent_task_set_fingerprint": old_manifest["task_set_fingerprint"],
        "task_set_fingerprint": new_manifest["task_set_fingerprint"],
        "comparison": dict(counts),
        "native_execution": (
            "Packaged tests/test.sh in the release's pinned Docker image; no network or model calls"
        ),
        "container_image": DOCKERFILE.splitlines()[0].removeprefix("FROM "),
        "regression_cases": records,
        "historical_teacher_scores_changed": False,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = audit_revision(args.parent, args.dataset, args.output, args.workers)
    print(json.dumps({"passed": report["passed"], "comparison": report["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
