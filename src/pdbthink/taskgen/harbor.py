"""TaskTrove-compatible archives with a deterministic, isolated answer verifier."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

from ..evaluation.score import _scoring_parameters
from ..schemas import SemanticInstance
from ..scoring import score_response
from ..scoring.scorers import SCORING_VERSION
from ..util import sha256_bytes
from . import TASKGEN_VERSION

DOCKERFILE = (
    "FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e\n"
    "WORKDIR /app\nRUN mkdir -p /logs/verifier\n"
)

VERIFIER = '''"""Run only after the model response has been saved by the host harness."""
import json
import os
from pathlib import Path
from coordinate_scoring import score_response

def verify(root=Path("/tests"), answer=Path("/app/answer.txt"), output=Path("/logs/verifier")):
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward.txt").write_text("0\\n")
    contract = json.loads((root / "gold.json").read_text())
    raw = answer.read_text() if answer.exists() and answer.stat().st_size <= 8000000 else ""
    status_path = answer.with_name("answer_status.json")
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    outcome = score_response(raw, contract["answer_schema"], contract["gold_answer"],
        parameters=contract["parameters"], truncated=status.get("truncated", False),
        provider_refusal=status.get("provider_refusal", False))
    tool_violation = bool(status.get("tool_events", []))
    reward = float(outcome["score"].get("correct", False) and not tool_violation)
    details = {"reward": reward, "partial_credit": outcome["score"]["score"],
               "tool_violation": tool_violation, "outcome": outcome}
    (output / "details.json").write_text(json.dumps(details, indent=2))
    (output / "reward.txt").write_text(str(reward) + "\\n")
    return details

if __name__ == "__main__":
    verify(Path(os.environ.get("PDBTHINK_TEST_ROOT", "/tests")),
           Path(os.environ.get("PDBTHINK_ANSWER", "/app/answer.txt")),
           Path(os.environ.get("PDBTHINK_REWARD_DIR", "/logs/verifier")))
'''


def archive(files: dict[str, bytes | str]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, content in sorted(files.items()):
            data = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), 0
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(stream.getvalue(), mtime=0)


def gold_response(schema: str, gold: dict) -> str:
    if schema == "two_interaction_sets":
        return (
            "FINAL\n"
            + "\n".join(f"{key}: {', '.join(gold[key]) or 'none'}" for key in ("gained", "lost"))
            + "\n"
        )
    value = gold["value"]
    if isinstance(value, list):
        value = ", ".join(str(x) for x in value)
    return f"FINAL: {value}\n"


def contract_for(task: dict) -> dict:
    # Published provenance may include the additive track field from the
    # sequence-prediction branch; coordinate-only checkouts can still read it.
    provenance = dict(task["instance"])
    if provenance.pop("task_track", "coordinate_reasoning") != "coordinate_reasoning":
        raise ValueError("Harbor coordinate tasks cannot score a different task track")
    instance = SemanticInstance(**provenance)
    return {
        "version": "1.1.0",
        "scoring_version": SCORING_VERSION,
        "answer_schema": instance.answer_schema,
        "gold_answer": task["render"]["gold_answer"],
        "parameters": _scoring_parameters(instance),
        "reward": "exact_correct",
        "definition_version": instance.definition_version,
    }


def task_archives(task: dict, name: str, *, version: str = TASKGEN_VERSION) -> tuple[bytes, bytes]:
    render, instance = task["render"], task["instance"]
    contract = contract_for(task)
    final = gold_response(render["answer_schema"], render["gold_answer"])
    scored = score_response(
        final, contract["answer_schema"], contract["gold_answer"], parameters=contract["parameters"]
    )
    if not scored["score"].get("correct"):
        raise ValueError(f"oracle response rejected: {scored}")
    config = f'''schema_version = "1.4"
[task]
name = "open-athena/{name}"
version = "{version}"
description = "Tool-free protein coordinate interpretation: {instance["question_family"]}"
authors = [{{name = "Open Athena"}}]
keywords = ["protein", "coordinate-reasoning", "no-tools", "{instance["question_family"]}"]
[metadata]
family = "{instance["question_family"]}"
task_track = "coordinate_reasoning"
requires_tool_free_runner = true
answer_path = "/app/answer.txt"
reward = "exact_correct"
[agent]
timeout_sec = 7200
[verifier]
timeout_sec = 60
[environment]
cpus = 1
memory_mb = 512
storage_mb = 1024
network_mode = "no-network"
'''
    prompt_json = json.dumps(
        {"system_prompt": render["system_prompt"], "user_prompt": render["user_prompt"]}, sort_keys=True
    )
    files = {
        "instruction.md": render["system_prompt"] + "\n\n" + render["user_prompt"],
        "task.toml": config,
        "environment/Dockerfile": DOCKERFILE,
        "prompt.json": prompt_json,
        "tests/test.sh": "#!/bin/sh\nset -eu\npython /tests/verify.py\n",
        "tests/verifier.toml": (
            'version = "1.1.0"\nmode = "pdbthink-coordinate"\n'
            f'scoring_version = "{SCORING_VERSION}"\n'
            'answer_path = "/app/answer.txt"\ncontract = "/tests/gold.json"\n'
            'reward = "exact_correct"\n'
        ),
        "tests/gold.json": json.dumps(contract, sort_keys=True),
        "tests/verify.py": VERIFIER,
    }
    scoring = Path(__file__).resolve().parents[1] / "scoring"
    for module in ("__init__.py", "parse.py", "scorers.py"):
        files[f"tests/coordinate_scoring/{module}"] = (scoring / module).read_bytes()
    solution = {
        "solution/answer.txt": final,
        "solution/solve.sh": "#!/bin/sh\nset -eu\ncp /solution/answer.txt /app/answer.txt\n",
    }
    return archive(files), archive(solution)


def negative_response(task: dict) -> str:
    """A valid but wrong submission, preventing parser-only reward smoke tests."""
    schema, gold = task["render"]["answer_schema"], task["render"]["gold_answer"]
    if schema == "two_interaction_sets":
        return gold_response(schema, {"gained": gold["lost"], "lost": gold["gained"]})
    value = gold["value"]
    if schema == "category":
        values = contract_for(task)["parameters"].get("categories", [])
        alternative = next((v for v in values if v != value), "wrong")
        return f"FINAL: {alternative}\n"
    if isinstance(value, (int, float)):
        return f"FINAL: {value + 100}\n"
    if schema == "numeric_triple":
        return gold_response(schema, {"value": [v + 100 for v in value]})
    if isinstance(value, list):
        return gold_response(schema, {"value": value + ["Z:W9999" if schema != "string_set" else "Z"]})
    if schema == "atom":
        return "FINAL: Z:W9999:CA\n"
    if schema == "residue_pair":
        return "FINAL: Z:W9998--Z:W9999\n"
    return "FINAL: Z:W9999\n"


def validate_reward(task: dict) -> None:
    contract = contract_for(task)
    for raw, expected in [
        (gold_response(contract["answer_schema"], contract["gold_answer"]), True),
        ("", False),
        (negative_response(task), False),
    ]:
        outcome = score_response(
            raw, contract["answer_schema"], contract["gold_answer"], parameters=contract["parameters"]
        )
        if bool(outcome["score"].get("correct")) != expected:
            raise ValueError(f"reward probe failed: {outcome}")


def row_for(task: dict, split: str, cluster: str, *, version: str = TASKGEN_VERSION) -> dict:
    instance, render = task["instance"], task["render"]
    name = f"pdbthink-{instance['question_family'].lower()}-{task['semantic_key'][:20]}"
    task_binary, solution_binary = task_archives(task, name, version=version)
    validate_reward(task)
    return {
        "path": name,
        "source": "pdbthink-coordinate-v1",
        "dataset_version": version,
        "family": instance["question_family"],
        "template_id": instance["question_family"],
        "converter": "pdbthink-oracle-v1",
        "mode": "pdbthink-coordinate",
        "dockerfile_id": sha256_bytes(DOCKERFILE.encode())[:16],
        "language": "en",
        "tags": ["protein", "coordinate-reasoning", "no-tools", instance["question_family"]],
        "has_solution": True,
        "task_binary": task_binary,
        "solution_binary": solution_binary,
        "split": split,
        "source_group": cluster,
        "input_tokens": render["input_token_count"],
        "atom_count": render["atom_count"],
        "answer_schema": render["answer_schema"],
        "prompt_sha256": sha256_bytes((render["system_prompt"] + "\n" + render["user_prompt"]).encode()),
        "semantic_key": task["semantic_key"],
        "task_sha256": sha256_bytes(task_binary),
        "source_entries": instance["source_entries"],
        "provenance": json.dumps(
            {
                "instance": instance,
                "render": {k: v for k, v in render.items() if k not in ("user_prompt", "system_prompt")},
            },
            sort_keys=True,
        ),
    }
