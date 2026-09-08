"""Collect every finished run into the single JSON the deck and writeup read.

Runs live outside the repository — they are large, they cost money, and they are
regenerable from the response cache — so this walks a directory of scored runs
and reduces them to the handful of numbers worth publishing.

Three things it computes that a plain score does not:

* **completion-conditioned scores.** Truncation and inability score identically,
  so a family whose responses all hit the output cap says nothing about
  capability. Every score is reported both as-is and restricted to responses
  that terminated.
* **the budget contrast.** Where higher-budget re-runs of the truncated prompts
  exist, they measure what the output cap was worth. There can be more than one
  tier: a prompt cut off at 32k may be re-run at 64k, and if it is cut off again
  at 262k. Each tier is folded in over the last, so a family is credited with
  the best answer the model gave when it had room to finish.
* **coverage.** A run cut short by a credit limit covers the alphabetically
  early families and no others, so a macro average over it is not comparable to
  a complete run. Coverage travels with every number.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CONTEXT_ONLY_FAMILIES = ("P01", "P02", "S03", "S04", "S05", "S08", "S09")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def macro(rows: list[dict]) -> float:
    per = defaultdict(list)
    for row in rows:
        per[row["question_family"]].append(float(row["score"]))
    return statistics.mean(statistics.mean(v) for v in per.values()) if per else 0.0


def output_budget(config_path: Path) -> str:
    """The re-run's output budget, which lives in its model config, not the scores."""
    if not config_path.exists():
        return ""
    for line in config_path.read_text().splitlines():
        if line.startswith("max_output_tokens:"):
            return f"{int(line.split(':')[1].strip()) // 1024}k"
    return ""


def summarise(label: str, rows: list[dict], *, tiers: list[tuple[str, list[dict]]] | None = None) -> dict:
    families = sorted({r["question_family"] for r in rows})
    finished = [r for r in rows if not r.get("truncated")]
    per_family = {}
    for family in families:
        group = [r for r in rows if r["question_family"] == family]
        ok = [r for r in group if not r.get("truncated")]
        per_family[family] = {
            "n": len(group),
            "score": statistics.mean(float(r["score"]) for r in group),
            "truncated": sum(1 for r in group if r.get("truncated")),
            "format_errors": sum(1 for r in group if r["format_error"]),
            "score_finished": (
                statistics.mean(float(r["score"]) for r in ok) if ok else None
            ),
        }

    # The context-only contrast, conditioned on completion. Comparing a ~200-token
    # blind prompt that never truncates against an 87,000-token one that does is
    # not like-for-like, and at high truncation it inverts the sign.
    baseline = {}
    for family in CONTEXT_ONLY_FAMILIES:
        blind = [
            r for r in rows
            if r["question_family"] == family and r["representation"] == "context_only"
        ]
        seen = [
            r for r in rows
            if r["question_family"] == family
            and r["representation"] == "minimal_pdb"
            and not r["is_rotation_variant"]
        ]
        if not blind or not seen:
            continue
        ok = [r for r in seen if not r.get("truncated")]
        blind_score = statistics.mean(float(r["score"]) for r in blind)
        naive = statistics.mean(float(r["score"]) for r in seen)
        conditioned = statistics.mean(float(r["score"]) for r in ok) if ok else None
        baseline[family] = {
            "floor": blind_score,
            "with_coordinates": naive,
            "gain_naive": naive - blind_score,
            "with_coordinates_finished": conditioned,
            "gain_conditioned": None if conditioned is None else conditioned - blind_score,
            "truncated_fraction": sum(1 for r in seen if r.get("truncated")) / len(seen),
        }

    out = {
        "label": label,
        "n_renders": len(rows),
        "n_families": len(families),
        "families": families,
        "complete": len(families) == 20,
        "macro": macro(rows),
        "macro_finished": macro(finished) if finished else None,
        "truncated": sum(1 for r in rows if r.get("truncated")),
        "format_errors": sum(1 for r in rows if r["format_error"]),
        "per_family": per_family,
        "context_only": baseline,
    }

    if tiers:
        # Each tier re-runs only what the previous one cut off, so the honest
        # comparison is the same prompts before and after, and each tier is
        # layered over the last rather than replacing it.
        merged = {r["render_id"]: r for r in rows}
        ladder = []
        for budget_label, tier_rows in tiers:
            if not tier_rows:
                continue
            ids = {r["render_id"] for r in tier_rows}
            before = [merged[i] for i in ids if i in merged]
            merged.update({r["render_id"]: r for r in tier_rows})
            ladder.append({
                "budget_label": budget_label,
                "n": len(tier_rows),
                "score_before": (
                    statistics.mean(float(r["score"]) for r in before) if before else None
                ),
                "score_after": statistics.mean(float(r["score"]) for r in tier_rows),
                "still_truncated": sum(1 for r in tier_rows if r.get("truncated")),
                "macro_after": macro(list(merged.values())),
                "truncated_after": sum(1 for r in merged.values() if r.get("truncated")),
            })
        if ladder:
            out["budget_ladder"] = ladder
            # The first tier keeps the old key so nothing downstream breaks.
            out["budget_rerun"] = ladder[0]
            out["macro_with_rerun"] = macro(list(merged.values()))
            out["n_truncated_after_rerun"] = sum(
                1 for r in merged.values() if r.get("truncated")
            )
    return out


def main(root: Path, output: Path) -> None:
    runs = []
    for scores_dir in sorted(root.glob("f3_scores_*")):
        label = scores_dir.name.replace("f3_scores_", "")
        path = scores_dir / "scores.jsonl"
        if not path.exists():
            continue
        tiers = []
        for suffix, prefix in (("hi", "hi_scores_"), ("max", "max_scores_")):
            tier_path = root / f"{prefix}{label}" / "scores.jsonl"
            if tier_path.exists():
                tiers.append((
                    output_budget(Path("configs/models") / f"together_{label}_{suffix}.yaml"),
                    load(tier_path),
                ))
        runs.append(summarise(label, load(path), tiers=tiers))
    runs.sort(key=lambda r: -r["macro"])
    output.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"{len(runs)} runs -> {output}")
    for run in runs:
        flag = "" if run["complete"] else f"  [{run['n_families']}/20 families]"
        print(
            f"  {run['label']:<20} macro {run['macro']:.3f}"
            f"  finished-only {run['macro_finished'] or 0:.3f}"
            f"  trunc {run['truncated']:>3}/{run['n_renders']}{flag}"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
