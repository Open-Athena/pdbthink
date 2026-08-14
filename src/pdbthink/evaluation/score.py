"""Scoring of stored responses, with no model in the loop (section 12)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..dataset import load_dataset
from ..schemas import EvaluationResult
from ..scoring import score_response
from ..util import read_json, read_jsonl, write_json, write_jsonl


def score_run(
    dataset_dir: str | Path, run_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Score every completion in ``run_dir`` against the dataset gold labels."""
    dataset_dir, run_dir, output_dir = Path(dataset_dir), Path(run_dir), Path(output_dir)
    instances, renders = load_dataset(dataset_dir)
    by_instance = {i.semantic_instance_id: i for i in instances}
    by_render = {r.render_id: r for r in renders}

    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"no results.jsonl in {run_dir}")
    run_config = (
        read_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
    )

    # A failed attempt remains in the append-only audit log when --resume retries
    # it. Score only the latest attempt for each logical completion.
    latest_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in read_jsonl(results_path):
        key = (str(row.get("run_id")), row["render_id"], int(row["completion_index"]))
        latest_rows[key] = row

    scored: list[dict[str, Any]] = []
    unknown_renders: set[str] = set()
    for row in latest_rows.values():
        result = EvaluationResult(**row)
        render = by_render.get(result.render_id)
        if render is None:
            unknown_renders.add(result.render_id)
            continue
        instance = by_instance[render.semantic_instance_id]
        outcome = score_response(
            result.raw_response,
            render.answer_schema,
            render.gold_answer,
            parameters=_scoring_parameters(instance),
            truncated=result.truncated,
        )
        scored.append(
            {
                "run_id": result.run_id,
                "render_id": result.render_id,
                "semantic_instance_id": render.semantic_instance_id,
                "completion_index": result.completion_index,
                "question_family": render.question_family,
                "protein_group_id": render.protein_group_id,
                "cluster": (instance.gold_evidence or {}).get("cluster") or render.protein_group_id,
                "is_mechanistic": instance.is_mechanistic,
                "representation": render.representation,
                "rotation_seed": render.rotation_seed,
                "is_rotation_variant": render.is_rotation_variant,
                "state_order_seed": render.state_order_seed,
                "answer_schema": render.answer_schema,
                "model_id": result.model_id,
                "model_provider": result.model_provider,
                "model_revision": result.model_revision,
                "reasoning_effort": result.reasoning_effort,
                "input_token_count": render.input_token_count,
                "score": outcome["score"]["score"],
                "correct": bool(outcome["score"].get("correct")),
                "detail": outcome["score"],
                "parsed_answer": outcome["parsed"],
                "format_error": outcome["format_error"],
                "refusal": outcome["refusal"],
                "truncated": outcome["truncated"],
                "api_error": bool(result.error),
                "latency_seconds": result.latency_seconds,
                "usage": result.usage,
            }
        )

    if unknown_renders:
        raise ValueError(
            f"{len(unknown_renders)} responses reference renders absent from the dataset, "
            f"for example {sorted(unknown_renders)[:3]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "scores.jsonl", scored)

    summary = summarise(scored)
    summary["run_config"] = run_config
    summary["dataset_dir"] = str(dataset_dir.resolve())
    write_json(output_dir / "summary.json", summary)
    return summary


def _scoring_parameters(instance) -> dict[str, Any]:
    from ..generators import get_generator

    parameters = dict(instance.question_parameters)
    if instance.answer_schema == "multi_field":
        return parameters
    try:
        generator = get_generator(instance.question_family)
    except KeyError:
        return parameters
    return {**parameters, **generator.prompt_parameters(instance.question_parameters)}


def summarise(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro (per family) and micro (per completion) aggregates."""
    if not scored:
        return {
            "n_results": 0,
            "n_renders": 0,
            "macro_score": 0.0,
            "micro_score": 0.0,
            "format_errors": 0,
            "refusals": 0,
        }
    by_family: dict[str, list[float]] = defaultdict(list)
    by_instance: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        by_family[row["question_family"]].append(row["score"])
        by_instance[row["semantic_instance_id"]].append(row["score"])

    family_means = {f: sum(v) / len(v) for f, v in sorted(by_family.items())}
    micro = sum(r["score"] for r in scored) / len(scored)
    return {
        "n_results": len(scored),
        "n_renders": len({r["render_id"] for r in scored}),
        "n_instances": len(by_instance),
        "macro_score": sum(family_means.values()) / len(family_means),
        "micro_score": micro,
        "per_family": family_means,
        "per_family_counts": {f: len(v) for f, v in sorted(by_family.items())},
        "format_errors": sum(1 for r in scored if r["format_error"]),
        "refusals": sum(1 for r in scored if r["refusal"]),
        "truncated": sum(1 for r in scored if r["truncated"]),
        "api_errors": sum(1 for r in scored if r["api_error"]),
        "models": sorted({r["model_id"] for r in scored}),
    }
