"""Export frozen prompts and complete model messages, bound to the report's selected attempts."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from pdbthink.dataset import load_dataset
from pdbthink.evaluation.response_protocol import answer_content
from pdbthink.util import read_json, sha256_bytes, stable_hash, write_json

OUTCOMES = ("correct", "context_exhausted", "output_limit", "refusal", "format_error", "wrong_answer")


def outcome(row: dict) -> str:
    """Make categories exclusive without changing the scorer's overlapping flags."""
    if row["score"]["correct"]:
        return "correct"
    if row["truncated"]:
        return "context_exhausted" if row["budget_limit"] == "context" else "output_limit"
    if row["refusal"]:
        return "refusal"
    return "format_error" if row["format_error"] else "wrong_answer"


def public_response(response: dict) -> dict:
    # Keep model output verbatim; provider errors, account data and HTTP headers are not model output.
    return {
        "choices": [
            {
                key: choice[key]
                for key in ("index", "message", "finish_reason", "native_finish_reason")
                if key in choice
            }
            for choice in response.get("choices", [])
        ]
    }


def partial_response(path: Path) -> dict:
    """Recover all textual deltas from an interrupted stream, never a provider error message."""
    fragments: dict[int, dict] = {}
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                try:
                    event = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue  # An interrupted final SSE line can be incomplete.
                for choice in event.get("choices", []):
                    index = choice.get("index", 0)
                    record = fragments.setdefault(
                        index, {"index": index, "message": {}, "finish_reason": None}
                    )
                    for key in ("content", "reasoning", "reasoning_content", "refusal"):
                        value = choice.get("delta", {}).get(key)
                        if value is not None:
                            if not isinstance(value, str):
                                raise ValueError("Unexpected non-text stream fragment")
                            record["message"].setdefault(key, []).append(value)
                    if choice.get("finish_reason"):
                        record["finish_reason"] = choice["finish_reason"]
    for record in fragments.values():
        record["message"] = {key: "".join(parts) for key, parts in record["message"].items()}
    return {"choices": [fragments[key] for key in sorted(fragments)]}


def verify_prompt(request: dict, render) -> None:
    expected = [
        {"role": "system", "content": render.system_prompt},
        {"role": "user", "content": render.user_prompt},
    ]
    if request["messages"] != expected:
        raise ValueError("Drill-down request differs from the frozen prompt")


def build_drilldown(dataset: Path, runs: Path, output: Path, reports: list[dict], family_names: dict) -> dict:
    _, renders = load_dataset(dataset)
    by_id = {render.render_id: render for render in renders}
    destination = output / "questions"
    destination.mkdir(exist_ok=True)
    assets = []

    def emit(name: str, value: dict) -> str:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
        assets.append("questions/" + name)
        return name

    variants = {
        name: {"label": row["label"], "cohort": report["cohort"], **report["models"][name]}
        for report in reports
        for row in report["metrics"]
        for name in [row["model"]]
    }
    questions = {
        rid: {
            "id": stable_hash(rid)[:16],
            "render_id": rid,
            "family": r.question_family,
            "family_name": family_names[r.question_family],
            "messages": [
                {"role": "system", "content": r.system_prompt},
                {"role": "user", "content": r.user_prompt},
            ],
            "answer_schema": r.answer_schema,
            "gold_answer": r.gold_answer,
            "models": {name: {"status": "missing", "attempts": []} for name in variants},
        }
        for rid, r in by_id.items()
    }
    selected_rows = {}
    messages_by_case = {}
    for report in reports:
        for name, attempts in report["attempts"].items():
            last = {attempt["render_id"]: attempt["path"] for attempt in attempts}
            selected_rows[name] = []
            for attempt in attempts:
                folder = (runs / attempt["path"]).parent
                row = read_json(folder / "result.json")
                if sha256_bytes((folder / "result.json").read_bytes()) != attempt["sha256"]:
                    raise ValueError("Attempt changed after report audit")
                request = read_json(folder / "request.json")
                response_path = folder / "response.json"
                if sha256_bytes(response_path.read_bytes()) != attempt["response_sha256"]:
                    raise ValueError("Response changed after report audit")
                if stable_hash(request) != attempt["request_sha256"]:
                    raise ValueError("Request changed after report audit")
                rid = row["render_id"]
                verify_prompt(request, by_id[rid])
                response = read_json(response_path)
                chosen = attempt["path"] == last[rid]
                record = {
                    "model": name,
                    "selected": chosen,
                    "status": outcome(row),
                    "stage": attempt["stage"],
                    "audit": attempt,
                    "parsed": row["parsed"],
                    "score": row["score"],
                    "flags": {key: row[key] for key in ("truncated", "format_error", "refusal")},
                    "scored_answer": answer_content(
                        response["choices"][0]["message"], row.get("response_protocol", "plain")
                    ),
                    **public_response(response),
                }
                link = emit(f"outputs/{questions[rid]['id']}/{name}/{attempt['stage']}.json", record)
                entry = questions[rid]["models"][name]
                entry["attempts"].append(
                    {"path": link, "selected": chosen, "status": record["status"], "stage": attempt["stage"]}
                )
                if chosen:
                    entry.update(status=record["status"], selected=link)
                    selected_rows[name].append(row)
                    messages_by_case[(name, questions[rid]["id"])] = record

        # Include terminal collection failures from this exact snapshot, including partial text.
        for failure in report["collection_failures"]:
            name, rid = failure["model"], failure["render_id"]
            question = questions[rid]
            folder = runs / variants[name].get("run_namespace", "") / failure["stage"] / name / question["id"]
            if sha256_bytes((folder / "failure.json").read_bytes()) != failure["record_sha256"]:
                raise ValueError("Collection failure changed after report audit")
            request_path = folder / "request.json"
            request = read_json(request_path) if request_path.exists() else None
            if request:
                verify_prompt(request, by_id[rid])
                if stable_hash(request) != failure["request_sha256"]:
                    raise ValueError("Failed request changed after report audit")
            response_path = folder / "response.json"
            if response_path.exists():
                if sha256_bytes(response_path.read_bytes()) != failure["response_sha256"]:
                    raise ValueError("Failed response changed after report audit")
                response = public_response(read_json(response_path))
            else:
                response = partial_response(folder / "response-http-body.txt")
            record = {
                "model": name,
                "selected": False,
                "status": "collection_failure",
                "stage": failure["stage"],
                "audit": failure,
                "request_controls": {
                    key: value for key, value in (request or {}).items() if key != "messages"
                },
                "partial": not response_path.exists(),
                **response,
            }
            link = emit(f"outputs/{question['id']}/{name}/{failure['stage']}-failure.json", record)
            question["models"][name]["attempts"].append(
                {"path": link, "selected": False, "status": failure["category"], "stage": failure["stage"]}
            )

    analysis = {"categories": list(OUTCOMES), "rows": [], "families": {}, "contact_sets": {}, "examples": []}
    for name in variants:
        if name.removeprefix("standardized-") not in ("snowball", "mochi", "qwen"):
            continue
        rows = selected_rows[name]
        if len(rows) != len(questions):
            raise ValueError("Failure analysis requires a complete model panel")
        counts = Counter(outcome(row) for row in rows)
        analysis["rows"].append({"model": name, "total": len(rows), **{key: counts[key] for key in OUTCOMES}})
        analysis["families"][name] = {
            family: dict(Counter(outcome(row) for row in rows if row["family"] == family))
            for family in family_names
        }
        analysis["contact_sets"][name] = {}
        for family in ("I01", "S06"):
            totals = Counter()
            for row in rows:
                if row["family"] != family or outcome(row) not in ("correct", "wrong_answer"):
                    continue
                gold = set(by_id[row["render_id"]].gold_answer["value"])
                predicted = set(row["parsed"]["value"])
                totals.update(
                    {
                        "parseable_answers": 1,
                        "true_positive": len(gold & predicted),
                        "false_negative": len(gold - predicted),
                        "false_positive": len(predicted - gold),
                    }
                )
            analysis["contact_sets"][name][family] = dict(totals)
    examples_path = Path(__file__).parents[1] / "configs/report_failure_examples_v1.json"
    if examples_path.exists():
        for example in read_json(examples_path):
            case = messages_by_case.get((example["model"], example["question"]))
            if case is None:
                continue
            message = case["choices"][0]["message"]
            if example["quote"] not in (message.get(example["field"]) or ""):
                raise ValueError("Failure example quotation differs from saved output")
            if case["status"] != example["status"]:
                raise ValueError("Failure example category differs from frozen scoring")
            if example.get("prompt_quote"):
                question = next(q for q in questions.values() if q["id"] == example["question"])
                if example["prompt_quote"] not in question["messages"][1]["content"]:
                    raise ValueError("Failure example evidence differs from the frozen prompt")
            analysis["examples"].append(
                {
                    **example,
                    "response_sha256": case["audit"]["response_sha256"],
                    "link": f"questions/?question={example['question']}&model={example['model']}",
                }
            )
    emit("failure-analysis.json", analysis)
    plot_outcomes(analysis["rows"], output)
    assets.extend(f"failure-outcomes.{extension}" for extension in ("svg", "pdf", "png"))
    # Keep the existing slide download complete when the diagnostic figure is present.
    import zipfile

    with zipfile.ZipFile(output / "figures.zip", "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for extension in ("svg", "pdf", "png"):
            path = output / f"failure-outcomes.{extension}"
            archive.write(path, path.name)
    index = []
    for rid, question in sorted(questions.items(), key=lambda item: (item[1]["family"], item[1]["id"])):
        link = emit(f"prompts/{question['id']}.json", question)
        index.append(
            {key: question[key] for key in ("id", "family", "family_name", "render_id")}
            | {
                "path": link,
                "statuses": {name: entry["status"] for name, entry in question["models"].items()},
            }
        )
    emit(
        "index.json",
        {
            "questions": index,
            "models": variants,
            "generated_at": reports[0]["generated_at"],
            "dataset_fingerprint": reports[0]["dataset_fingerprint"],
        },
    )
    for name in ("index.html", "explorer.js", "explorer.css"):
        shutil.copy2(Path(__file__).parent / "templates" / "question_browser" / name, destination / name)
        assets.append("questions/" + name)
    return {
        "path": "questions/",
        "question_count": len(questions),
        "model_count": len(variants),
        "analysis": analysis,
        "artifacts": sorted(assets),
    }


def plot_outcomes(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(
        rows,
        key=lambda row: (
            ("snowball", "mochi", "qwen").index(row["model"].removeprefix("standardized-")),
            row["model"].startswith("standardized-"),
        ),
    )
    colours = {
        "correct": "#367a68",
        "wrong_answer": "#526d91",
        "format_error": "#c3a258",
        "refusal": "#a68bad",
        "context_exhausted": "#b76550",
        "output_limit": "#d6aaa0",
    }
    categories = [key for key in colours if any(row[key] for row in ordered)]
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    left = [0] * len(ordered)
    for key in categories:
        values = [row[key] for row in ordered]
        ax.barh(
            range(len(ordered)),
            values,
            left=left,
            height=0.64,
            color=colours[key],
            label=key.replace("_", " ").capitalize(),
        )
        left = [a + b for a, b in zip(left, values, strict=True)]
    ax.set_yticks(range(len(ordered)), [row["model"] for row in ordered])
    ax.invert_yaxis()
    ax.set_xlim(0, 88)
    ax.set_xticks([0, 22, 44, 66, 88])
    ax.set_xlabel("Questions")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.02), ncol=3, frameon=False)
    for extension in ("svg", "pdf", "png"):
        path = output / f"failure-outcomes.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        if extension == "svg":
            metadata = {"Date": None}
        fig.savefig(path, dpi=300, bbox_inches="tight", metadata=metadata)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
