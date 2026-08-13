"""Benchmark report generation (specification section 13)."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..util import read_json, read_jsonl, write_json
from .metrics import (
    agreement,
    clustered_bootstrap,
    context_only_baseline,
    failure_categories,
    group_by,
    instance_scores,
    macro_score,
    mechanistic_report,
    micro_score,
    paired_difference,
    summarise_group,
)


def build_report(
    score_dirs: Sequence[str | Path], output_dir: str | Path, *, bootstrap_samples: int = 2000
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for directory in score_dirs:
        directory = Path(directory)
        rows = list(read_jsonl(directory / "scores.jsonl"))
        if not rows:
            continue
        summary = (
            read_json(directory / "summary.json")
            if (directory / "summary.json").exists()
            else {}
        )
        runs.append(
            {
                "source": str(directory),
                "rows": rows,
                "run_config": summary.get("run_config", {}),
            }
        )
    if not runs:
        raise ValueError(f"no scores found in {list(score_dirs)}")

    report: dict[str, Any] = {
        "n_runs": len(runs),
        "bootstrap_samples": bootstrap_samples,
        "runs": [],
        "headline": [],
    }
    for run in runs:
        analysed = analyse_run(run["rows"], bootstrap_samples=bootstrap_samples)
        analysed["source"] = run["source"]
        analysed["run_config"] = run["run_config"]
        report["runs"].append(analysed)
        report["headline"].append(
            {
                "model_id": analysed["model_id"],
                "reasoning_effort": analysed["reasoning_effort"],
                "macro_score": analysed["macro"]["point"],
                "ci_low": analysed["macro"]["ci_low"],
                "ci_high": analysed["macro"]["ci_high"],
                "micro_score": analysed["micro_score"],
                "n_instances": analysed["n_instances"],
            }
        )
    report["headline"].sort(key=lambda r: -r["macro_score"])

    write_json(output / "report.json", report)
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (output / "report.html").write_text(render_html(report), encoding="utf-8")
    return report


def analyse_run(rows: list[dict[str, Any]], *, bootstrap_samples: int) -> dict[str, Any]:
    coordinate_rows = [r for r in rows if r["representation"] != "context_only"]
    primary = [r for r in coordinate_rows if not r["is_rotation_variant"]]

    pdb_rows = [r for r in rows if r["representation"] == "minimal_pdb" and not r["is_rotation_variant"]]
    table_rows = [r for r in rows if r["representation"] == "normalized_coordinates"]
    rotation_rows = [r for r in rows if r["is_rotation_variant"]]
    base_for_rotation = [
        r for r in rows if r["representation"] == "minimal_pdb" and not r["is_rotation_variant"]
    ]

    macro = clustered_bootstrap(
        primary or rows, samples=bootstrap_samples, seed="macro", statistic=macro_score
    )
    micro = clustered_bootstrap(
        primary or rows, samples=bootstrap_samples, seed="micro", statistic=micro_score
    )

    per_family = {
        family: {
            **summarise_group(group),
            **clustered_bootstrap(
                group, samples=max(200, bootstrap_samples // 4), seed=f"family:{family}",
                statistic=micro_score,
            ),
        }
        for family, group in group_by(primary or rows, "question_family").items()
    }
    per_protein = {
        protein: summarise_group(group)
        for protein, group in group_by(primary or rows, "protein_group_id").items()
    }
    per_representation = {
        representation: summarise_group(group)
        for representation, group in group_by(rows, "representation").items()
    }
    per_effort = {
        effort: summarise_group(group) for effort, group in group_by(rows, "reasoning_effort").items()
    }

    return {
        "model_id": _one(rows, "model_id"),
        "model_provider": _one(rows, "model_provider"),
        "model_revision": _one(rows, "model_revision"),
        "reasoning_effort": _one(rows, "reasoning_effort"),
        "n_results": len(rows),
        "n_instances": len({r["semantic_instance_id"] for r in rows}),
        "n_clusters": len({r.get("cluster") or r["protein_group_id"] for r in rows}),
        "macro": macro,
        "micro": micro,
        "micro_score": micro_score(primary or rows),
        "per_family": per_family,
        "per_protein": per_protein,
        "per_representation": per_representation,
        "per_reasoning_effort": per_effort,
        "context_only_baseline": context_only_baseline(rows),
        "representation_gap_pdb_minus_table": paired_difference(
            pdb_rows, table_rows, samples=bootstrap_samples, seed="representation"
        ),
        "rotation_gap_primary_minus_rotated": paired_difference(
            base_for_rotation, rotation_rows, samples=bootstrap_samples, seed="rotation"
        ),
        "consistency": agreement(rows),
        "mechanistic": mechanistic_report(rows),
        "failures": failure_categories(rows),
        "worst_instances": _worst(rows),
    }


def _one(rows: list[dict[str, Any]], key: str) -> Any:
    values = {r.get(key) for r in rows}
    return sorted(v for v in values if v is not None)[0] if any(values) else None


def _worst(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    scores = instance_scores(rows)
    families = {r["semantic_instance_id"]: r["question_family"] for r in rows}
    proteins = {r["semantic_instance_id"]: r["protein_group_id"] for r in rows}
    ordered = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))[:limit]
    return [
        {
            "semantic_instance_id": k,
            "question_family": families[k],
            "protein_group_id": proteins[k],
            "mean_score": v,
        }
        for k, v in ordered
    ]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Protein Structural Reasoning Benchmark report", ""]
    lines.append(f"Runs analysed: {report['n_runs']}  ·  bootstrap samples: {report['bootstrap_samples']}")
    lines.append("")
    lines.append("## Headline (macro score, clustered bootstrap 95% CI)")
    lines.append("")
    lines.append("| model | effort | macro | 95% CI | micro | instances |")
    lines.append("| --- | --- | ---: | :---: | ---: | ---: |")
    for row in report["headline"]:
        lines.append(
            f"| {row['model_id']} | {row['reasoning_effort'] or '-'} | {row['macro_score']:.3f} "
            f"| [{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['micro_score']:.3f} "
            f"| {row['n_instances']} |"
        )
    for run in report["runs"]:
        lines += ["", f"## {run['model_id']}", ""]
        lines.append(
            f"Provider `{run['model_provider']}`, {run['n_results']} completions over "
            f"{run['n_instances']} semantic instances in {run['n_clusters']} clusters."
        )
        lines += [
            "",
            "### Per family",
            "",
            "| family | n | mean | exact | 95% CI |",
            "| --- | ---: | ---: | ---: | :---: |",
        ]
        for family, stats in run["per_family"].items():
            lines.append(
                f"| {family} | {stats['n']} | {stats['mean_score']:.3f} | {stats['exact']:.3f} "
                f"| [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}] |"
            )
        lines += ["", "### Representation and rotation controls", ""]
        gap = run["representation_gap_pdb_minus_table"]
        lines.append(
            f"- minimal PDB minus normalized coordinates: {gap['mean_difference']:+.3f} "
            f"[{gap['ci_low']:+.3f}, {gap['ci_high']:+.3f}] over {gap['n_pairs']} paired instances"
        )
        rot = run["rotation_gap_primary_minus_rotated"]
        lines.append(
            f"- primary rotation minus alternate rotation: {rot['mean_difference']:+.3f} "
            f"[{rot['ci_low']:+.3f}, {rot['ci_high']:+.3f}] over {rot['n_pairs']} paired instances"
        )
        consistency = run["consistency"]
        lines.append(
            f"- repeated completions: exact agreement "
            f"{_fmt(consistency['pairwise_exact_agreement'])}, Jaccard "
            f"{_fmt(consistency['pairwise_jaccard_agreement'])} "
            f"({consistency['n_repeated_renders']} repeated prompts)"
        )
        if run["mechanistic"]:
            m = run["mechanistic"]
            lines += ["", "### Mechanistic episodes", ""]
            lines.append(f"- observation {_fmt(m['observation_score'])}, interaction "
                         f"{_fmt(m['interaction_score'])}, mechanism {_fmt(m['mechanism_score'])}")
            lines.append(f"- mechanism given a correct observation: "
                         f"{_fmt(m['mechanism_given_correct_observation'])}")
            lines.append(f"- gain over context-only input: {_fmt(m['gain_over_context_only'])}")
        if run.get("context_only_baseline"):
            lines += ["", "### What the coordinates are worth", ""]
            lines.append("| family | question only | with coordinates | gain |")
            lines.append("| --- | --- | --- | --- |")
            for family, b in sorted(run["context_only_baseline"].items()):
                lines.append(
                    f"| {family} | {_fmt(b['context_only_score'])} | "
                    f"{_fmt(b['with_coordinates_score'])} | {_fmt(b['gain_over_context_only'])} |"
                )
        lines += ["", "### Failure categories", ""]
        for key, value in run["failures"].items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_html(report: dict[str, Any]) -> str:
    body = html.escape(render_markdown(report))
    payload = html.escape(json.dumps(report["headline"], indent=2))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>pdbthink benchmark report</title>
<style>
 body {{ font: 15px/1.6 ui-sans-serif, system-ui, sans-serif; max-width: 62rem; margin: 2rem auto;
        padding: 0 1rem; color: #16181d; background: #fff; }}
 @media (prefers-color-scheme: dark) {{ body {{ color: #e7ebf0; background: #14171a; }} }}
 pre {{ white-space: pre-wrap; background: rgba(127,127,127,.12); padding: 1rem; border-radius: 8px;
        overflow-x: auto; }}
</style></head>
<body>
<h1>Protein Structural Reasoning Benchmark</h1>
<h2>Headline</h2>
<pre>{payload}</pre>
<h2>Full report</h2>
<pre>{body}</pre>
</body></html>
"""
