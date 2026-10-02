"""Teacher traces, exact student context checks and descriptive plots."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
from collections import Counter, defaultdict
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
    return f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content


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
    }


def write_shards(rows: list[dict], directory: Path, *, size: int = 500) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.parquet"):
        path.unlink()
    for index in range(0, len(rows), size):
        pq.write_table(
            pa.Table.from_pylist(rows[index : index + size]),
            directory / f"train-{index // size:05d}.parquet",
            compression="zstd",
        )


def export(root: Path, output: Path, *, workers: int, allow_partial: bool = False) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    tasks = {r["path"]: r for r in pq.read_table(root / "tasks.parquet").to_pylist()}
    with connect(root) as db:
        states = {r["path"]: dict(r) for r in db.execute("SELECT * FROM tasks")}
        attempt_refs = [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY path,attempt")]
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
    missing = [(tasks[r["path"]], r) for r in results if (r["path"], r["attempt"]) not in cache]
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
            "request_max_tokens": result["request_max_tokens"],
            "raw_response_json": json.dumps(result["raw_response"], sort_keys=True),
        }
        all_rows.append(row)
        by_task[task["path"]].append(row)
        if row["reward"] == 1 and row["fits_student_context"]:
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
        "truncated_attempts": sum(r["finish_reason"] == "length" for r in all_rows),
    }
    write_shards(all_rows, output / "attempts")
    write_shards(sft_rows, output / "sft")
    write_shards(outcome_rows, output / "outcomes", size=5000)
    save_json(output / "manifest.json", manifest)
    save_json(output / "summary.json", summary)
    shutil.copytree(root / "native_verifier", output / "native_verifier", dirs_exist_ok=True)
    for name in ("source-validation.json", "native-verifier-smoke.json"):
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
    families = [(k, v) for k, v in summary["families"].items() if v["total"]]
    families.sort(key=lambda kv: kv[1]["solved"] / kv[1]["total"])
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
    ax.set(yticks=y, yticklabels=names, xlim=(0, 1), xlabel="Fraction of cohort tasks")
    ax.legend(loc="lower right")
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
    an assistant message containing &lt;think&gt; reasoning &lt;/think&gt; followed by the answer,
    rendered with Snowball's pinned thinking-enabled template. Exact full-sequence counts
    include chat delimiters and the assistant end token. No trace is truncated.
    {s["correct_over_context"]:,} correct traces exceed Snowball's context; they remain
    in the attempt archive. {s["sft_traces_completion_within_8k"]:,} SFT traces have an
    assistant completion of at most 8,192 tokens. Only training-split tasks were sent to
    the teacher; the original validation and test splits remain untouched.</p>
    <p>{s["sft_traces_with_reasoning"]:,} SFT traces contain separately emitted reasoning.
    Correct answers without emitted reasoning are retained as answer-only examples;
    no reasoning is invented for them. The has_reasoning column supports filtering.</p>
    <p>There were {s["infrastructure_errors"]:,} infrastructure failures,
    {s["truncated_attempts"]:,} length-limited responses and {s["tool_violations"]:,} tool
    violations. Scored attempts consumed {s["teacher_completion_tokens"]:,} reported
    completion tokens. The served model reports GLM-5.3 with FP8 weights; no immutable
    deployed weight revision is exposed. Tokenizer revisions and verifier hashes are
    pinned in manifest.json.</p>
    <p>The curves show observed cumulative success under stop-on-success sampling, not an
    unbiased pass@k estimate from ten samples of every task. Failure after ten attempts
    does not establish impossibility. Rejection-sampled SFT data favours easier tasks and
    shorter successful traces; family counts should guide later training mixtures.</p>
    """
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
  has_reasoning distinguishes reasoning traces from correct answer-only examples.
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

Reasoning is wrapped in &lt;think&gt;...&lt;/think&gt; in the assistant message. The raw reasoning
and answer are also separate columns. Exact student lengths use
{STUDENT_MODEL} at {STUDENT_REVISION}. Prompt and assistant end tokens are included.

See [the full report](report.html), [summary](summary.json), and the plots below.
Counts are descriptive: tasks share structures, and stop-on-success curves are not
unbiased pass@k estimates. A correct final answer does not prove correct reasoning.
Training selection will need to account for family and success-selection imbalance.
T01 has no context-eligible examples; I01 has only 12.

| Family | Tasks | First try | Correct by ten | Unsolved after ten | SFT examples |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(markdown_rows)}

![Success by family](plots/family_solvability.png)
![Cumulative success](plots/cumulative_success.png)
![First success](plots/first_success.png)

Reproduction settings and native verifier hashes are in manifest.json. The deployed
teacher's immutable weight revision is unavailable; its tokenizer is pinned separately.
The source data's benchmark exclusions are preserved; they do not establish absence
from the teacher's pretraining. Coordinate gold is derived only from displayed coordinates.
Code is Apache-2.0; source coordinates originate in the public Protein Data Bank.
"""
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
