"""Collect every finished run into the single JSON the deck and writeup read.

Runs live outside the repository — they are large, they cost money, and they are
regenerable from the response cache — so this walks a directory of scored runs
and reduces them to the handful of numbers worth publishing.

Three things it computes that a plain score does not:

* **completion-conditioned scores.** Truncation and inability score identically,
  so a family whose responses all hit the output cap says nothing about
  capability. Every score is reported both as-is and restricted to responses
  that terminated.
* **the budget contrast.** Where a high-budget re-run of the truncated prompts
  exists, the pair of scores measures what the output cap was worth.
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


def summarise(label: str, rows: list[dict], *, hi_rows: list[dict] | None = None) -> dict:
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

    if hi_rows:
        # The high-budget re-run covers only the prompts that truncated, so the
        # comparable figure is the same prompts before and after.
        ids = {r["render_id"] for r in hi_rows}
        before = [r for r in rows if r["render_id"] in ids]
        out["budget_rerun"] = {
            "n": len(hi_rows),
            "score_before": statistics.mean(float(r["score"]) for r in before) if before else None,
            "score_after": statistics.mean(float(r["score"]) for r in hi_rows),
            "still_truncated": sum(1 for r in hi_rows if r.get("truncated")),
        }
        merged = {r["render_id"]: r for r in rows}
        merged.update({r["render_id"]: r for r in hi_rows})
        out["macro_with_rerun"] = macro(list(merged.values()))
    return out


def main(root: Path, output: Path) -> None:
    runs = []
    for scores_dir in sorted(root.glob("f3_scores_*")):
        label = scores_dir.name.replace("f3_scores_", "")
        path = scores_dir / "scores.jsonl"
        if not path.exists():
            continue
        hi_path = root / f"hi_scores_{label}" / "scores.jsonl"
        runs.append(
            summarise(
                label,
                load(path),
                hi_rows=load(hi_path) if hi_path.exists() else None,
            )
        )
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
