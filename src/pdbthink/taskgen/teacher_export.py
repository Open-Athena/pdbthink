"""Teacher traces, exact student context checks and descriptive plots."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .teacher_data import (
    MAX_ATTEMPTS,
    STUDENT_CONTEXT,
    STUDENT_MODEL,
    STUDENT_REVISION,
    connect,
    digest,
    load_native_scorer,
    messages,
    request_for,
    save_json,
    score_completion,
)

FAMILIES = {
    "P01": "Chain identifiers",
    "P02": "Residue counting",
    "P03": "Atom coordinates",
    "G01": "Atom distance",
    "G02": "Nearest eligible atom",
    "G03": "Nearest residue",
    "G04": "Steric clash",
    "S01": "Salt bridges",
    "S02": "Phosphorylation",
    "S03": "Solvent exposure",
    "S04": "Secondary structure",
    "S05": "Fold class",
    "S06": "Ligand contacts",
    "S07": "Metal coordination",
    "S08": "Disulfide partner",
    "S09": "Side-chain rotamer",
    "I01": "Interface contacts",
    "N01": "Shared contact",
    "T01": "Two-state contacts",
}
TOKENIZER = None


def init_tokenizer() -> None:
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(STUDENT_MODEL, revision=STUDENT_REVISION, local_files_only=True)


def assistant_text(reasoning: str, content: str) -> str:
    return f"{reasoning}\n\n{content}" if reasoning else content


def tolerance_boundary_rejection(result: dict) -> bool:
    """Flag decimal boundary disagreements without changing frozen rewards."""
    outcome = result["outcome"]
    score = outcome["score"]
    if (
        result["reward"]
        or result["tool_events"]
        or outcome["format_error"]
        or outcome["refusal"]
        or outcome["truncated"]
        or "tolerance" not in score
        or "predicted" not in score
        or "gold" not in score
    ):
        return False
    predicted, gold = score["predicted"], score["gold"]
    if not isinstance(predicted, list):
        predicted, gold = [predicted], [gold]
    tolerance = Decimal(str(score["tolerance"]))
    return len(predicted) == len(gold) and all(
        abs(Decimal(str(p)) - Decimal(str(g))) <= tolerance for p, g in zip(predicted, gold)
    )


def student_tokens(item: tuple[dict, dict]) -> dict:
    task, result = item
    conversation = messages(task) + [
        {"role": "assistant", "content": assistant_text(result["reasoning"], result["content"])}
    ]
    encoded = TOKENIZER.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        enable_thinking=True,
        tools=[],
        return_assistant_tokens_mask=True,
    )
    n = len(encoded["input_ids"])
    generated = sum(encoded["assistant_masks"])
    return {
        "path": task["path"],
        "attempt": result["attempt"],
        "student_total_tokens": n,
        "student_completion_tokens": generated,
        "fits_student_context": n <= STUDENT_CONTEXT,
        "completion_within_8k": generated <= 8192,
        "assistant_sha256": digest(assistant_text(result["reasoning"], result["content"]).encode()),
    }


def write_shards(rows: list[dict], directory: Path, *, size: int = 500) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.parquet"):
        path.unlink()
    # Infer one schema across shards without building a multi-gigabyte Arrow
    # string array: long reasoning traces can exceed 32-bit string offsets.
    schema = (
        pa.unify_schemas(
            [pa.Table.from_pylist(rows[index : index + size]).schema for index in range(0, len(rows), size)]
        )
        if rows
        else pa.schema([])
    )
    for index in range(0, len(rows), size):
        pq.write_table(
            pa.Table.from_pylist(rows[index : index + size], schema=schema),
            directory / f"train-{index // size:05d}.parquet",
            compression="zstd",
        )


def export(root: Path, output: Path, *, workers: int, allow_partial: bool = False) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    tasks = {r["path"]: r for r in pq.read_table(root / "tasks.parquet").to_pylist()}
    with connect(root) as db:
        db.execute("BEGIN")
        states = {r["path"]: dict(r) for r in db.execute("SELECT * FROM tasks")}
        attempt_refs = [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY path,attempt")]
    if set(states) != set(tasks) or len(tasks) != manifest["task_count"]:
        raise ValueError("Run state differs from the frozen cohort")
    if digest((root / "tasks.parquet").read_bytes()) != manifest["tasks_sha256"]:
        raise ValueError("Frozen task inputs changed")
    indexed = defaultdict(list)
    for attempt in attempt_refs:
        indexed[attempt["path"]].append(attempt)
    if set(indexed) - set(tasks):
        raise ValueError("Scored attempt is outside the cohort")
    for path, state in states.items():
        attempts = indexed[path]
        if [a["attempt"] for a in attempts] != list(range(1, state["attempts"] + 1)):
            raise ValueError("Missing or nonsequential scored attempt")
        correct = [a["attempt"] for a in attempts if a["reward"] == 1]
        if state["attempts"] > MAX_ATTEMPTS or correct != ([state["attempts"]] if state["solved"] else []):
            raise ValueError("Run violates stop-on-success or the attempt limit")
    complete = all(s["solved"] or s["attempts"] == MAX_ATTEMPTS for s in states.values())
    if not complete and not allow_partial:
        raise ValueError("Generation is incomplete; refusing a final release")
    scorer = load_native_scorer(root)
    results = [json.loads((root / r["result_path"]).read_text()) for r in attempt_refs]
    for result in results:
        rechecked = score_completion(tasks[result["path"]], result["raw_response"], scorer)
        if rechecked["reward"] != result["reward"] or rechecked["outcome"] != result["outcome"]:
            raise ValueError("Stored reward differs from released native verifier")
    cache_file = root / "student_lengths.parquet"
    cache = (
        {(r["path"], r["attempt"]): r for r in pq.read_table(cache_file).to_pylist()}
        if cache_file.exists()
        else {}
    )
    missing = [
        (tasks[r["path"]], r)
        for r in results
        if cache.get((r["path"], r["attempt"]), {}).get("assistant_sha256")
        != digest(assistant_text(r["reasoning"], r["content"]).encode())
    ]
    if missing:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=init_tokenizer) as pool:
            for index, counted in enumerate(pool.map(student_tokens, missing, chunksize=16), 1):
                cache[(counted["path"], counted["attempt"])] = counted
                if index % 500 == 0:
                    print("student token counts", index, "/", len(missing), flush=True)
        pq.write_table(pa.Table.from_pylist(list(cache.values())), cache_file, compression="zstd")
    output.mkdir(parents=True, exist_ok=True)
    all_rows, sft_rows, by_task = [], [], defaultdict(list)
    for result in results:
        task = tasks[result["path"]]
        lengths = cache[(task["path"], result["attempt"])]
        row = {
            "task_id": task["path"],
            "family": task["family"],
            "source_group": task["source_group"],
            "source_split": "train",
            "prompt_sha256": task["prompt_sha256"],
            "task_sha256": task["task_sha256"],
            "dataset_revision": manifest["dataset_revision"],
            "attempt": result["attempt"],
            "messages": messages(task)
            + [{"role": "assistant", "content": assistant_text(result["reasoning"], result["content"])}],
            "reasoning": result["reasoning"],
            "has_reasoning": bool(result["reasoning"].strip()),
            "answer": result["content"],
            "reward": result["reward"],
            "finish_reason": result["finish_reason"],
            "score_json": json.dumps(result["outcome"], sort_keys=True),
            "teacher_input_tokens": task["teacher_input_tokens"],
            "teacher_completion_tokens": result["usage"].get("completion_tokens"),
            "teacher_reasoning_tokens": (result["usage"].get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "student_input_tokens": task["student_input_tokens"],
            "student_total_tokens": lengths["student_total_tokens"],
            "student_completion_tokens": lengths["student_completion_tokens"],
            "fits_student_context": lengths["fits_student_context"],
            "completion_within_8k": lengths["completion_within_8k"],
            "tool_violation": bool(result["tool_events"]),
            "tolerance_boundary_rejection": tolerance_boundary_rejection(result),
            "request_max_tokens": result["request_max_tokens"],
            "seed": request_for(task, result["attempt"])["seed"],
            "raw_response_json": json.dumps(result["raw_response"], sort_keys=True),
        }
        all_rows.append(row)
        by_task[task["path"]].append(row)
        if row["reward"] == 1 and row["fits_student_context"]:
            contract = json.loads(task["gold_json"])
            joined_score = scorer(
                row["messages"][-1]["content"],
                contract["answer_schema"],
                contract["gold_answer"],
                parameters=contract["parameters"],
            )
            if not joined_score["score"]["correct"]:
                raise ValueError("Student serialization changed the verifiable final answer")
            sft_rows.append({k: v for k, v in row.items() if k not in {"raw_response_json", "score_json"}})
    family_counts, outcome_rows = {}, []
    for family in FAMILIES:
        members = [t for t in tasks.values() if t["family"] == family]
        first = Counter()
        usable = long_correct = no_reasoning = exhausted = pending = 0
        for task in members:
            attempts = by_task[task["path"]]
            correct = [r for r in attempts if r["reward"] == 1]
            state = states[task["path"]]
            if correct:
                first[correct[0]["attempt"]] += 1
                long_correct += not correct[0]["fits_student_context"]
                no_reasoning += not bool(correct[0]["reasoning"].strip())
                usable += bool(correct[0]["fits_student_context"])
            elif state["attempts"] >= MAX_ATTEMPTS:
                exhausted += 1
            else:
                pending += 1
            outcome_rows.append(
                {
                    "task_id": task["path"],
                    "family": family,
                    "source_group": task["source_group"],
                    "attempts": state["attempts"],
                    "first_correct_attempt": correct[0]["attempt"] if correct else None,
                    "status": "solved"
                    if correct
                    else "unsolved_after_10"
                    if state["attempts"] >= MAX_ATTEMPTS
                    else "pending",
                    "sft_eligible": bool(correct and correct[0]["fits_student_context"]),
                    "infrastructure_errors": state["infra_errors"],
                    "has_tolerance_boundary_rejection": any(
                        r["tolerance_boundary_rejection"] for r in attempts
                    ),
                }
            )
        family_counts[family] = {
            "label": FAMILIES[family],
            "total": len(members),
            "first_correct": dict(sorted(first.items())),
            "solved": sum(first.values()),
            "first_try": first[1],
            "sft_eligible": usable,
            "correct_over_context": long_correct,
            "correct_missing_reasoning": no_reasoning,
            "unsolved_after_10": exhausted,
            "pending": pending,
            "mean_first_correct_attempt": (
                sum(k * v for k, v in first.items()) / sum(first.values()) if first else None
            ),
            "scored_attempts": sum(states[t["path"]]["attempts"] for t in members),
            "native_context_exhaustions": sum(
                r["finish_reason"] == "length" for r in all_rows if r["family"] == family
            ),
        }
    summary = {
        "complete": complete,
        "task_count": len(tasks),
        "attempt_count": len(all_rows),
        "solved": sum(f["solved"] for f in family_counts.values()),
        "first_try": sum(f["first_try"] for f in family_counts.values()),
        "sft_traces": len(sft_rows),
        "sft_traces_with_reasoning": sum(r["has_reasoning"] for r in sft_rows),
        "sft_traces_completion_within_8k": sum(r["completion_within_8k"] for r in sft_rows),
        "correct_over_context": sum(f["correct_over_context"] for f in family_counts.values()),
        "unsolved_after_10": sum(f["unsolved_after_10"] for f in family_counts.values()),
        "pending": sum(f["pending"] for f in family_counts.values()),
        "families": family_counts,
        "unique_source_groups": len({t["source_group"] for t in tasks.values()}),
        "teacher_completion_tokens": sum(r["teacher_completion_tokens"] or 0 for r in all_rows),
        "teacher_input_tokens": sum(r["teacher_input_tokens"] for r in all_rows),
        "infrastructure_errors": sum(s["infra_errors"] for s in states.values()),
        "tool_violations": sum(r["tool_violation"] for r in all_rows),
        "tolerance_boundary_rejections": sum(r["tolerance_boundary_rejection"] for r in all_rows),
        "tasks_with_tolerance_boundary_rejection": sum(
            r["has_tolerance_boundary_rejection"] for r in outcome_rows
        ),
        "exhausted_tasks_with_tolerance_boundary_rejection": sum(
            r["has_tolerance_boundary_rejection"] and r["status"] == "unsolved_after_10"
            for r in outcome_rows
        ),
        "truncated_attempts": sum(r["finish_reason"] == "length" for r in all_rows),
        "failure_modes": dict(
            Counter(
                "tool_violation"
                if r["tool_events"]
                else "context_limit"
                if r["finish_reason"] == "length"
                else "refusal"
                if r["outcome"]["refusal"]
                else "format_error"
                if r["outcome"]["format_error"]
                else "incorrect_answer"
                for r in results
                if r["reward"] == 0
            )
        ),
    }
    write_shards(all_rows, output / "attempts")
    write_shards(sft_rows, output / "sft")
    write_shards(outcome_rows, output / "outcomes", size=5000)
    manifest["student_serialization"] = "snowball-assistant-text-v1"
    manifest["student_serialization_note"] = (
        "Join returned reasoning and final answer with a blank line in one assistant message. "
        "Use the pinned Snowball template with enable_thinking=True and assistant-only loss."
    )
    save_json(output / "manifest.json", manifest)
    save_json(output / "summary.json", summary)
    save_json(
        output / "tolerance-boundary-audit.json",
        {
            "policy": (
                "Diagnostic only: compare exact decimal values in native numeric scores. "
                "Frozen rewards, stopping decisions and SFT selection remain unchanged."
            ),
            "cases": [
                {
                    "task_id": r["task_id"],
                    "attempt": r["attempt"],
                    "native_score": json.loads(r["score_json"])["score"],
                }
                for r in all_rows
                if r["tolerance_boundary_rejection"]
            ],
        },
    )
    shutil.copytree(
        root / "native_verifier",
        output / "native_verifier",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in (
        "source-validation.json",
        "native-verifier-smoke.json",
        "persistent-failure-spot-checks.json",
    ):
        if (root / name).exists():
            shutil.copyfile(root / name, output / name)
    shutil.copyfile(Path(__file__).resolve().parents[3] / "LICENSE", output / "LICENSE")
    plots(summary, output / "plots")
    report(summary, manifest, output)
    save_json(
        output / "checksums.json",
        {
            str(p.relative_to(output)): digest(p.read_bytes())
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "checksums.json"
        },
    )
    return summary


def plots(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    families = [
        (k, {**v, "first_correct": {int(a): n for a, n in v["first_correct"].items()}})
        for k, v in summary["families"].items()
        if v["total"]
    ]
    families.sort(key=lambda kv: (kv[1]["solved"] / kv[1]["total"], kv[0]))
    names = [f"{k}  {v['label']}  (n={v['total']:,})" for k, v in families]
    y = np.arange(len(families))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 8))
    first = [v["first_try"] / v["total"] for _, v in families]
    later = [(v["solved"] - v["first_try"]) / v["total"] for _, v in families]
    ax.barh(y, first, color="#276a9f", label="Correct on first attempt")
    ax.barh(y, later, left=first, color="#73b6d6", label="Additional correct by attempt 10")
    ax.scatter(
        [v["sft_eligible"] / v["total"] for _, v in families],
        y,
        color="#222222",
        marker="|",
        s=120,
        label="Usable SFT fraction",
        zorder=3,
    )
    ax.set(yticks=y, yticklabels=names, xlim=(-0.01, 1.01), xlabel="Fraction of cohort tasks")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=9)
    if not summary["complete"]:
        ax.set_title(f"In progress: {summary['pending']:,} tasks still pending")
    fig.tight_layout()
    save_plot(fig, output, "family_solvability")
    fig, ax = plt.subplots(figsize=(10, 6))
    for family, values in sorted(families):
        counts = values["first_correct"]
        ys = [sum(counts.get(k, 0) for k in range(1, n + 1)) / values["total"] for n in range(1, 11)]
        ax.plot(range(1, 11), ys, marker=".", label=family, alpha=0.8)
    ax.set(
        xlabel="Maximum attempts",
        ylabel="Fraction with a correct answer",
        xticks=range(1, 11),
        ylim=(0, 1.02),
    )
    ax.legend(ncol=3, fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    if not summary["complete"]:
        ax.set_title("In progress: first attempts and retries are incomplete")
    fig.tight_layout()
    save_plot(fig, output, "cumulative_success")
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = [sum(v["first_correct"].get(k, 0) for _, v in families) for k in range(1, 11)]
    labels = [str(k) for k in range(1, 11)] + ["No success\nin 10"]
    values = counts + [summary["unsolved_after_10"]]
    ax.bar(labels, values, color=["#276a9f"] * 10 + ["#b6494d"])
    ax.set(xlabel="Attempt of first correct answer", ylabel="Tasks")
    for i, value in enumerate(values):
        ax.text(i, value + max(values, default=1) * 0.015, f"{value:,}", ha="center", fontsize=9)
    if not summary["complete"]:
        ax.set_title(f"Incomplete run: {summary['pending']:,} tasks still pending")
    fig.tight_layout()
    save_plot(fig, output, "first_success")
    ordered = sorted(families)
    counts = np.array(
        [[v["first_correct"].get(k, 0) for k in range(1, 11)] + [v["unsolved_after_10"]] for _, v in ordered]
    )
    fractions = counts / np.array([v["total"] for _, v in ordered])[:, None]
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set(
        xticks=range(11),
        xticklabels=[str(k) for k in range(1, 11)] + ["Unsolved"],
        yticks=range(len(ordered)),
        yticklabels=[f"{k}  {v['label']}" for k, v in ordered],
        xlabel="Attempt of first correct answer (cell labels are task counts)",
    )
    for i, row in enumerate(counts):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if fractions[i, j] > 0.5 else "#182b3a",
            )
    fig.colorbar(heatmap, ax=ax, label="Fraction of family cohort", shrink=0.75)
    if not summary["complete"]:
        ax.set_title("In progress: counts exclude pending tasks")
    fig.tight_layout()
    save_plot(fig, output, "attempts_by_family")


def save_plot(fig, output: Path, name: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(s: dict, manifest: dict, output: Path) -> None:
    rows = []
    markdown_rows = []
    for family, value in s["families"].items():
        n = value["total"]
        if not n:
            markdown_rows.append(f"| {family} — {value['label']} | 0 | — | — | — | — |")
            rows.append(
                f"<tr><td>{family} — {value['label']}</td><td>0</td>"
                "<td colspan='4'>Excluded by context filter</td></tr>"
            )
            continue
        rows.append(
            f"<tr><td>{family} — {value['label']}</td><td>{n:,}</td>"
            f"<td>{value['first_try']:,} ({value['first_try'] / n:.1%})</td>"
            f"<td>{value['solved']:,} ({value['solved'] / n:.1%})</td>"
            f"<td>{value['unsolved_after_10']:,}</td><td>{value['sft_eligible']:,}</td></tr>"
        )
        markdown_rows.append(
            f"| {family} — {value['label']} | {n:,} | {value['first_try'] / n:.1%} | "
            f"{value['solved'] / n:.1%} | {value['unsolved_after_10']:,} | {value['sft_eligible']:,} |"
        )
    state = "Complete" if s["complete"] else f"IN PROGRESS — {s['pending']:,} tasks pending"
    narrative = f"""
    <p><strong>{state}.</strong> GLM-5.3 produced {s["attempt_count"]:,} scored attempts for a
    cohort of {s["task_count"]:,} coordinate-interpretation training tasks. It answered
    {s["first_try"]:,} correctly on the first attempt and {s["solved"]:,} correctly within
    ten attempts. {s["sft_traces"]:,} complete correct teacher traces fit Snowball's
    32,768-token training context.</p>
    <p>The cohort is exactly the training split of PDBThink Coordinate Tasks v1.2.0 whose
    prompts leave at least 8,192 Snowball tokens. It covers 18 families and
    {s["unique_source_groups"]:,} source groups. T01 has no eligible examples; I01 has only
    12. Tasks sharing a structure are correlated; these are task-level descriptive counts.</p>
    <p>Each attempt presents the unchanged system and user prompts, without tools, gold
    answers, verifier feedback, or previous attempts. Sampling uses temperature 1.0,
    top-p 0.95 and high reasoning effort. Stable task/attempt seeds are recorded in the
    generation code. Each task stops at the first correct answer or ten scored attempts.
    Infrastructure failures are retained separately and do not consume a scored attempt.</p>
    <p>The teacher receives its full remaining served 262,144-token context for reasoning
    and output. The 8K reserve selects inputs; it is not an imposed teacher output cap.
    Scores come from the exact verifier code and gold contract bundled in each released
    Harbor task. Only the final answer is scored, with the task's existing numerical
    tolerances and exact-set rules. Correct answers do not certify every step of the
    teacher's reasoning. No generated tool calls are executed.</p>
    <p>Every retained response includes raw returned reasoning and final text. SFT uses
    an assistant message containing the reasoning, a blank line, and the final answer,
    rendered with Snowball's pinned thinking-enabled template. Exact full-sequence counts
    include chat delimiters and the assistant end token. No trace is truncated.
    {s["correct_over_context"]:,} correct traces exceed Snowball's context; they remain
    in the attempt archive. {s["sft_traces_completion_within_8k"]:,} SFT traces have an
    assistant completion of at most 8,192 tokens. Only training-split tasks were sent to
    the teacher; the original validation and test splits remain untouched.</p>
    <p>{s["sft_traces_with_reasoning"]:,} SFT traces contain separately emitted reasoning.
    Responses without separate reasoning retain their entire answer message;
    no reasoning is invented for them. The has_reasoning column supports filtering.</p>
    <p>There were {s["infrastructure_errors"]:,} infrastructure failures,
    {s["truncated_attempts"]:,} responses that exhausted the teacher's native context,
    and {s["tool_violations"]:,} tool
    violations. Scored attempts consumed {s["teacher_completion_tokens"]:,} reported
    completion tokens. The served model reports GLM-5.3 with FP8 weights; no immutable
    deployed weight revision is exposed. Tokenizer revisions and verifier hashes are
    pinned in manifest.json.</p>
    <p>The curves show observed cumulative success under stop-on-success sampling.
    We do not apply the standard fixed-sample pass@k estimator, because successful tasks
    were not sampled ten times. Failure after ten attempts
    does not establish impossibility. Rejection-sampled SFT data favours easier tasks and
    shorter successful traces; family counts should guide later training mixtures.
    In small categorical answer spaces, retries can also find the correct label by
    chance. Compare first-attempt accuracy alongside cumulative success.</p>
    """
    boundary_note = ""
    if s["tolerance_boundary_rejections"]:
        boundary_note = (
            f"The frozen verifier rejected {s['tolerance_boundary_rejections']:,} numeric responses "
            f"across {s['tasks_with_tolerance_boundary_rejection']:,} tasks at a floating-point "
            "tolerance boundary, although exact decimal comparison would accept them. "
            f"{s['exhausted_tasks_with_tolerance_boundary_rejection']:,} of these tasks exhausted "
            "ten attempts without a native-verifier success. The primary results, retry counts "
            "and SFT selection retain the original verifier's decisions; affected attempts and "
            "outcomes carry explicit flags. These cases should not be interpreted as evidence "
            "that GLM cannot meet the mathematical tolerance. A scorer correction requires a "
            "separate versioned task release."
        )
        narrative += (
            f"<h2>Verifier boundary sensitivity</h2><p>{boundary_note} "
            '<a href="tolerance-boundary-audit.json">Affected responses and native scores</a>.</p>'
        )
    audit_note = ""
    audit_path = output / "persistent-failure-spot-checks.json"
    if audit_path.exists():
        checks = json.loads(audit_path.read_text())["checks"]
        audit_note = (
            f"A targeted audit of {len(checks)} persistent distance failures found repeated "
            "coordinate-reading errors: GLM dropped a negative sign, then calculated a "
            "distance using the altered coordinate. Independent fixed-width PDB extraction "
            "and math.dist reproduced every audited gold answer. These deliberately selected "
            "examples illustrate a failure mechanism, rather than estimating its prevalence."
        )
        narrative += (
            f"<h2>Persistent-error spot checks</h2><p>{audit_note} "
            '<a href="persistent-failure-spot-checks.json">Coordinates, calculations and teacher text</a>.'
            "</p>"
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>GLM coordinate teacher traces</title>
    <style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#182b3a}}
    img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}
    td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}</style></head>
    <body><h1>GLM-5.3 teacher traces for Snowball</h1>{narrative}
    <h2>Results by family</h2><table><tr><th>Family</th><th>Tasks</th><th>First try</th>
    <th>By ten tries</th><th>Unsolved after ten</th><th>SFT traces</th></tr>
    {"".join(rows)}</table>
    <img src="plots/family_solvability.png" alt="Success rates by family">
    <img src="plots/cumulative_success.png" alt="Cumulative success by attempt">
    <img src="plots/first_success.png" alt="Distribution of first successful attempts">
    <img src="plots/attempts_by_family.png" alt="First successful attempt counts for each family">
    <p>Source dataset revision: {manifest["dataset_revision"]}. Machine-readable results:
    <a href="summary.json">summary.json</a>; task outcomes and all attempts are supplied as Parquet.
    </p></body></html>"""
    (output / "report.html").write_text(document)
    card = f"""---
pretty_name: PDBThink GLM-5.3 Teacher Traces
license: apache-2.0
language: [en]
task_categories: [text-generation]
tags: [protein, coordinate-reasoning, reasoning, sft, no-tools]
configs:
  - config_name: sft
    data_files:
      - split: train
        path: sft/*.parquet
  - config_name: attempts
    data_files:
      - split: train
        path: attempts/*.parquet
  - config_name: outcomes
    data_files:
      - split: train
        path: outcomes/*.parquet
---

# PDBThink GLM-5.3 Teacher Traces

**{state}.** {s["sft_traces"]:,} verified correct teacher traces fit Snowball's 32K
context, from {s["task_count"]:,} training tasks and {s["attempt_count"]:,} scored attempts.
GLM solved {s["first_try"]:,} on the first attempt and {s["solved"]:,} within ten attempts.

The source is [PDBThink Coordinate Tasks v1.2.0](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/tree/{manifest["dataset_revision"]}).
The cohort contains only training tasks with at least 8,192 tokens of output headroom
under the pinned Snowball tokenizer. All original validation/test tasks are excluded.

~~~python
from datasets import load_dataset
data = load_dataset("open-athena/pdbthink-glm53-teacher-traces", "sft", split="train")
# Train on messages with the pinned Snowball chat template, enable_thinking=True.
# Mask system/user tokens; supervise the assistant reasoning and final answer.
~~~

- **sft:** first correct response per solved task with exact full-sequence length
  <=32,768. Contains messages, reasoning, answer and token counts.
  has_reasoning indicates whether the API emitted a separate reasoning field.
- **attempts:** every scored attempt, including incorrect and over-context correct
  traces, raw API response JSON, native verifier outcomes and exact student counts.
- **outcomes:** one row per cohort task, including first success, exhaustion and
  infrastructure-error counts. Use this population for denominators.

Teacher generation uses high reasoning effort, temperature 1.0 and top-p 0.95.
Each retry samples the unchanged prompt independently, without answers or feedback.
Stop at first correct answer or ten total scored attempts. API failures do not count
as wrong answers. The teacher uses its full remaining 262,144-token served context.
8K is a cohort reserve, not a teacher output cap; use completion_within_8k for the
stricter completion-length subset. No successful traces are truncated.

Reasoning and the final answer are joined with a blank line in one assistant message,
matching Snowball's plain assistant-text format. Their original text is also preserved
in separate columns. Exact student lengths use
{STUDENT_MODEL} at {STUDENT_REVISION}. Prompt and assistant end tokens are included.

See [the full report](report.html), [summary](summary.json), and the plots below.
Counts are descriptive: tasks share structures. The curves show observed success
by attempt; we do not apply a fixed-sample pass@k estimator to this adaptive run.
A correct final answer does not prove correct reasoning.
Retries can find correct labels by chance in small categorical answer spaces;
compare first-attempt accuracy alongside cumulative success.
Training selection will need to account for family and success-selection imbalance.
T01 has no context-eligible examples; I01 has only 12.

| Family | Tasks | First try | Correct by ten | Unsolved after ten | SFT examples |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(markdown_rows)}

![Success by family](plots/family_solvability.png)
![Cumulative success](plots/cumulative_success.png)
![First success](plots/first_success.png)
![Attempts by family](plots/attempts_by_family.png)

Reproduction settings and native verifier hashes are in manifest.json. The deployed
teacher's immutable weight revision is unavailable; its tokenizer is pinned separately.
The source data's benchmark exclusions are preserved; they do not establish absence
from the teacher's pretraining. Coordinate gold is derived only from displayed coordinates.
Code is Apache-2.0; source coordinates originate in the public Protein Data Bank.
"""
    if audit_note:
        card += (
            "\n## Persistent-error spot checks\n\n"
            + audit_note
            + " See [the audit records](persistent-failure-spot-checks.json).\n"
        )
    if boundary_note:
        card += (
            "\n## Verifier boundary sensitivity\n\n"
            + boundary_note
            + " See [the diagnostic audit](tolerance-boundary-audit.json).\n"
        )
    (output / "README.md").write_text(card)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.root, args.output, workers=args.workers, allow_partial=args.allow_partial), indent=2
        )
    )


if __name__ == "__main__":
    main()
