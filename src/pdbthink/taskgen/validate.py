"""Audit packaged bytes, coordinate-derived answers and split isolation offline."""

from __future__ import annotations

import concurrent.futures
import importlib.metadata
import io
import json
import platform
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from ..config import Definitions
from ..prompts.library import PROMPT_VERSION, QUESTION_TEMPLATES, answer_format
from ..util import sha256_bytes, write_json
from .coordinates import recompute
from .harbor import contract_for, validate_reward


def _check_parquet(path: Path) -> dict:
    import pyarrow.parquet as pq

    definitions = Definitions.load()
    families, sources, prompts, semantics, groups = Counter(), set(), set(), set(), set()
    count = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            if sha256_bytes(row["task_binary"]) != row["task_sha256"]:
                raise ValueError("task archive checksum mismatch")
            with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:gz") as archive:
                for member in archive:
                    if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                        raise ValueError("unsafe task archive member")
                    if member.name.startswith("solution/"):
                        raise ValueError("oracle solution in task archive")
                prompt = json.load(archive.extractfile("prompt.json"))
                gold = json.load(archive.extractfile("tests/gold.json"))
                instruction = archive.extractfile("instruction.md").read().decode()
            if instruction != prompt["system_prompt"] + "\n\n" + prompt["user_prompt"]:
                raise ValueError("Harbor and structured prompts disagree")
            digest = sha256_bytes((prompt["system_prompt"] + "\n" + prompt["user_prompt"]).encode())
            if digest != row["prompt_sha256"]:
                raise ValueError("prompt checksum mismatch")
            task = json.loads(row["provenance"])
            task["render"].update(prompt)
            instance = task["instance"]
            if task["render"]["prompt_version"] == PROMPT_VERSION:
                expected = answer_format(instance["answer_schema"], row["family"])
                if not prompt["user_prompt"].endswith("\n\n" + expected + "\n"):
                    raise ValueError("incorrect family answer-format instructions")
                if row["family"] == "G04" and not prompt["user_prompt"].endswith(
                    QUESTION_TEMPLATES["G04"] + "\n\n" + expected + "\n"
                ):
                    raise ValueError("clash prompt omits the current operational rules")
            answer = recompute(
                prompt["user_prompt"], row["family"], instance["question_parameters"], definitions
            )
            if answer != gold["gold_answer"] or gold != contract_for(task):
                raise ValueError("packaged gold disagrees with displayed-coordinate oracle")
            validate_reward(task)
            with tarfile.open(fileobj=io.BytesIO(row["solution_binary"]), mode="r:gz") as archive:
                from ..scoring import score_response

                outcome = score_response(
                    archive.extractfile("solution/answer.txt").read().decode(),
                    gold["answer_schema"],
                    gold["gold_answer"],
                    parameters=gold["parameters"],
                )
                if not outcome["score"].get("correct"):
                    raise ValueError("packaged oracle solution rejected")
            count += 1
            families[row["family"]] += 1
            sources.update(row["source_entries"])
            prompts.add(digest)
            semantics.add(row["semantic_key"])
            groups.add(row["source_group"])
    return {
        "count": count,
        "families": families,
        "sources": sources,
        "prompts": prompts,
        "semantics": semantics,
        "groups": groups,
    }


def validate_release(directory: Path, *, workers: int = 8) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    files = sorted((directory / "data").glob("*.parquet"))
    for path in files:
        if sha256_bytes(path.read_bytes()) != manifest["data_hashes"][str(path.relative_to(directory))]:
            raise ValueError("Parquet checksum mismatch")
    count, families = 0, Counter()
    seen_prompts, seen_semantics = set(), set()
    split_sources, split_groups = defaultdict(set), defaultdict(set)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for path, result in zip(files, executor.map(_check_parquet, files)):
            count += result["count"]
            families.update(result["families"])
            if seen_prompts & result["prompts"] or seen_semantics & result["semantics"]:
                raise ValueError("duplicate tasks across shards")
            seen_prompts.update(result["prompts"])
            seen_semantics.update(result["semantics"])
            split = path.name.split("-", 1)[0]
            split_sources[split].update(result["sources"])
            split_groups[split].update(result["groups"])
            print("validated", count, "/", manifest["task_count"], flush=True)
    if count != manifest["task_count"] or dict(families) != manifest["family_counts"]:
        raise ValueError("release counts differ from manifest")
    if len(seen_prompts) != count or len(seen_semantics) != count:
        raise ValueError("duplicate tasks within a shard")
    exclusion = json.loads((directory / "audits/exclusion.json").read_text())
    audits = json.loads((directory / "audits/sources.json").read_text())
    split_sequences, split_clusters = defaultdict(set), defaultdict(set)
    for split, entries in split_sources.items():
        for entry in entries:
            if f"pdb:{entry}" in exclusion["sources"]:
                raise ValueError("benchmark source overlap")
            audit = audits[entry]
            split_sequences[split].update(audit["sequence_sha256s"])
            split_clusters[split].update(audit["rcsb30_clusters"])
        if split_sequences[split] & set(exclusion["sequence_sha256s"]):
            raise ValueError("benchmark sequence overlap")
        if split_clusters[split] & set(exclusion["rcsb30_clusters"]):
            raise ValueError("benchmark homology-cluster overlap")
    for sets in (split_sources, split_groups, split_sequences, split_clusters):
        for first in sets:
            for second in sets:
                if first < second and sets[first] & sets[second]:
                    raise ValueError("related structures cross split boundaries")
    report = {
        "task_count": count,
        "family_counts": dict(sorted(families.items())),
        "packaged_coordinate_oracle_checks": count,
        "packaged_solution_checks": count,
        "benchmark_overlap": 0,
        "split_overlap": 0,
        "duplicate_tasks": 0,
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "gemmi", "tiktoken", "pydantic", "pyarrow", "pyyaml")
            },
        },
    }
    write_json(directory / "validation.json", report)
    return report
