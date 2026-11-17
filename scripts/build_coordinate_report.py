"""Build a public aggregate report without publishing fresh prompts or gold answers."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup

from pdbthink.chem import chemistry_fingerprint
from pdbthink.config import Definitions
from pdbthink.dataset import load_dataset
from pdbthink.evaluation.response_limits import provider_clamped_output
from pdbthink.evaluation.response_protocol import answer_content, response_tool_events
from pdbthink.evaluation.score import _scoring_parameters
from pdbthink.evaluation.standardized_conditions import require_standardized_condition
from pdbthink.evaluation.tool_policy import require_tool_free
from pdbthink.prompts.library import PROMPT_VERSION, prompt_fingerprint
from pdbthink.scoring import SCORING_VERSION, score_response
from pdbthink.taskgen.protocol import request_payload
from pdbthink.util import derive_seed, read_json, sha256_bytes, stable_hash, write_json

LABELS = {
    "astra": "Astra",
    "opus": "Opus 5",
    "kimi": "Kimi K3",
    "glm": "GLM 5.3 (max)",
    "qwen": "Qwen 3.6 27B FP8",
    "snowball": "Marin Snowball v0.1.1 (256K)",
    "deepseek": "DeepSeek V4.1 Flash 1017",
    "mochi": "Mochi Hero SFT 400407 (BF16)",
}
LABELS.update({"standardized-" + name: "standardized-" + name for name in (*LABELS, "nemotron")})
RUN_STAGES = (
    "api",
    "api_parallel",
    "stream_recovery",
    "parallel_recovery",
    "context_recovery",
    "context_recovery_parallel",
    "context_retry_1",
    "context_retry_2",
    "context_retry_3",
    "context_retry_4",
    "context_retry_5",
    "context_retry_6",
    "context_retry_7",
    "mochi_v1",
    "mochi_recovery_1",
    "mochi_recovery_2",
    "standardized_v1",
    *(f"standardized_recovery_{n}" for n in range(1, 17)),
)

FAMILY_NAMES = {
    "P01": "Chain inventory",
    "P02": "Residue count",
    "P03": "Atom coordinates",
    "G01": "Atom distance",
    "G02": "Nearest nonbonded atom",
    "G03": "Closest candidate",
    "G04": "Severe clash",
    "S01": "Salt-bridge partner",
    "S02": "Phosphorylated residue",
    "S03": "Residue burial",
    "S04": "Secondary structure",
    "S05": "Fold class",
    "S06": "Ligand contacts",
    "S07": "Metal coordination",
    "S08": "Disulfide partner",
    "S09": "Chi1 rotamer",
    "I01": "Interface residues",
    "N01": "Bridging residue",
    "T01": "Contact changes",
}


def audit_result(folder: Path, row: dict, question: dict, protocol: str = "plain") -> None:
    """Bind every published score to the frozen question and its full budget."""
    request = read_json(folder / "request.json")
    response = read_json(folder / "response.json")
    budget = read_json(folder / "budget.json")
    if stable_hash(request) != row["request_sha256"]:
        raise ValueError("Request hash mismatch")
    if sha256_bytes((folder / "response.json").read_bytes()) != row["response_sha256"]:
        raise ValueError("Response hash mismatch")
    require_tool_free(request, "openai_chat", "https://openrouter.ai/api/v1")
    for message, role in zip(request["messages"], ["system", "user"], strict=True):
        if message["role"] != role or sha256_bytes(message["content"].encode()) != question[role + "_sha256"]:
            raise ValueError("Request differs from the frozen question")
    maximum = request_payload(request["model"], request["messages"], budget)["max_tokens"]
    if request["max_tokens"] != maximum or row["max_output_tokens"] != maximum:
        raise ValueError("Result did not receive the full endpoint budget")
    if row["usage"] != response["usage"] or row["usage"]["prompt_tokens"] != budget["input_tokens"]:
        raise ValueError("Recorded token usage differs from the provider response")
    if row["model_id"] != request["model"] or row["family"] != question["family"]:
        raise ValueError("Result condition differs from the frozen request")
    if row.get("response_protocol", "plain") != protocol:
        raise ValueError("Response protocol differs from the declared model condition")
    if response_tool_events(response, protocol) or row["tool_events"]:
        raise ValueError("Tool event in a purported tool-free result")
    if "answer_content_sha256" in row:
        answer = answer_content(response["choices"][0]["message"], protocol)
        if sha256_bytes(answer.encode()) != row["answer_content_sha256"]:
            raise ValueError("Extracted answer hash differs from the recorded response protocol")
    if len(response["choices"]) != 1 or response["choices"][0]["finish_reason"] not in (
        "stop",
        "length",
        "content_filter",
    ):
        raise ValueError("Unexpected completion count or termination")
    if row["finish_reason"] != response["choices"][0]["finish_reason"]:
        raise ValueError("Recorded termination differs from the provider response")
    if provider_clamped_output(response, maximum):
        raise ValueError("Unresolved provider output clamp")


def token_usage(usage: dict) -> dict:
    """Publish the token audit while keeping account billing fields in the private archive."""
    fields = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    }
    return {key: value for key, value in usage.items() if key in fields}


def response_cost_usd(usage: dict) -> float | None:
    """Only recorded non-BYOK inference charges have comparable monetary units here."""
    value = usage.get("cost")
    if value is None or usage.get("is_byok"):
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("Invalid recorded response cost")
    return float(value)


def cost_metrics(selected: dict, planned: int, pending: set[str]) -> dict:
    """Never turn unmetered compute or a partially collected panel into a cheap model."""
    costs = [response_cost_usd(row["usage"]) for row in selected.values()]
    known = [cost for cost in costs if cost is not None]
    complete = len(selected) == planned and not pending and len(known) == planned
    total = math.fsum(known) if known else None
    return {
        "costed_responses": len(known),
        "known_response_cost_usd": total,
        "panel_response_cost_usd": total if complete else None,
        "mean_response_cost_usd": total / planned if complete else None,
    }


def collection_failure(path: Path, stage: str, model: str) -> dict:
    """Expose collection limits without publishing account details from provider errors."""
    failure = read_json(path)
    reason = failure.get("error", "")
    if "clamped output" in reason:
        category = "provider_output_clamp"
    elif "finish_reason=error" in reason or "Unsupported finish reason" in reason:
        category = "provider_error_termination"
    elif "Incomplete provider stream" in reason:
        category = "incomplete_stream"
    elif "Local collection process interrupted" in reason:
        category = "client_interruption"
    elif failure.get("http_status"):
        category = "http_error"
    else:
        category = "collection_error"
    record = {
        "stage": stage,
        "model": model,
        "render_id": failure["render_id"],
        "category": category,
        "http_status": failure.get("http_status"),
        "record_sha256": sha256_bytes(path.read_bytes()),
    }
    request_path = path.with_name("request.json")
    if request_path.exists():
        request = read_json(request_path)
        record.update(request_sha256=stable_hash(request), max_output_tokens=request["max_tokens"])
    response_path = path.with_name("response.json")
    if response_path.exists():
        response = read_json(response_path)
        record.update(
            response_sha256=sha256_bytes(response_path.read_bytes()),
            usage=token_usage(response.get("usage", {})),
            finish_reasons=[choice.get("finish_reason") for choice in response.get("choices", [])],
        )
    return record


def audit_model_condition(request: dict, endpoint: dict, render_id: str) -> None:
    """Bind the development-model results to sampling settings chosen before collection."""
    expected = {"model": endpoint["model_id"], **endpoint.get("extra", {})}
    if "seed_base" in endpoint:
        expected["seed"] = derive_seed(endpoint["seed_base"], render_id)
    mismatches = [name for name, value in expected.items() if request.get(name) != value]
    if endpoint.get("benchmark_condition") == "standardized-v1":
        controlled = {
            "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
            "repetition_penalty", "seed", "reasoning_effort", "chat_template_kwargs",
        }
        mismatches.extend(sorted((request.keys() & controlled) - expected.keys()))
    if mismatches:
        raise ValueError(f"Request differs from the declared model condition: {', '.join(mismatches)}")


def collect(
    root: Path, model: str, admitted: dict, protocol: str = "plain", condition: dict | None = None,
) -> tuple[dict, list]:
    selected = {}
    attempts = []
    # Recovery waves are explicit, chronological transport/budget retries, never score selection.
    for stage in RUN_STAGES:
        for path in sorted((root / stage / model).glob("*/result.json")):
            row = read_json(path)
            if row["render_id"] not in admitted:
                raise ValueError("Response does not belong to the frozen panel")
            audit_result(path.parent, row, admitted[row["render_id"]], protocol)
            request = read_json(path.parent / "request.json")
            if condition is not None:
                audit_model_condition(request, condition, row["render_id"])
            budget = read_json(path.parent / "budget.json")
            if condition and condition.get("benchmark_condition") == "standardized-v1":
                for key in ("context_window", "native_output_limit"):
                    if budget[key] != condition[key]:
                        raise ValueError(f"Budget differs from the declared standardized {key}")
            response = read_json(path.parent / "response.json")
            raw_path = path.parent / "response-http-body.txt"
            raw_format = "http_body"
            if not raw_path.exists():
                # Initial non-streaming probes preserved the complete provider JSON object.
                raw_path = path.parent / "response.json"
                raw_format = "provider_json_object"
            record = {
                "stage": stage,
                "path": str(path.relative_to(root)),
                "sha256": sha256_bytes(path.read_bytes()),
                "request_sha256": row["request_sha256"],
                "response_sha256": row["response_sha256"],
                "raw_response_sha256": sha256_bytes(raw_path.read_bytes()),
                "raw_response_format": raw_format,
                "context_window": budget["context_window"],
                "declared_output_limit": budget["native_output_limit"],
                "request_controls": {k: v for k, v in request.items() if k != "messages"},
                "served_model": response.get("model"),
                "provider": response.get("provider"),
                "created": response.get("created"),
                "render_id": row["render_id"],
                "correct": row["score"]["correct"],
                "max_output_tokens": row["max_output_tokens"],
                "usage": token_usage(row["usage"]),
                "response_cost_usd": response_cost_usd(row["usage"]),
                "truncated": row["truncated"],
                "finish_reason": row["finish_reason"],
                "budget_limit": row["budget_limit"],
                "response_protocol": row.get("response_protocol", "plain"),
                "answer_content_sha256": row.get("answer_content_sha256"),
            }
            if previous := selected.get(row["render_id"]):
                if not previous["truncated"]:
                    raise ValueError(
                        "A completed uncapped answer was rerun; review selection before publication"
                    )
            selected[row["render_id"]] = row
            attempts.append(record)
    return selected, attempts


def preliminary_glm(runs: Path) -> dict:
    """Keep the replaced condition visible without pooling its scores into the headline."""
    root = runs / "api" / "glm-high-preliminary"
    if not root.exists():
        return {}
    attempts = []
    for path in sorted(root.glob("*/request.json")):
        request = read_json(path)
        row = {"request_sha256": stable_hash(request), "max_output_tokens": request["max_tokens"]}
        response_path = path.parent / "response.json"
        if response_path.exists():
            response = read_json(response_path)
            row.update(
                usage=token_usage(response["usage"]),
                finish_reason=response["choices"][0]["finish_reason"],
                response_sha256=sha256_bytes(response_path.read_bytes()),
            )
            row["provider_clamp"] = (
                row["finish_reason"] == "length" and row["usage"]["completion_tokens"] < request["max_tokens"]
            )
        else:
            row["uncollected"] = True
        result_path = path.parent / "result.json"
        if result_path.exists():
            result = read_json(result_path)
            row.update(render_id=result["render_id"], correct=result["score"]["correct"])
        attempts.append(row)
    return {
        "condition": "GLM 5.3 / Parasail FP8 / high reasoning effort",
        "replacement": "Uniform maximum-effort rerun on all 88 questions; original responses preserved",
        "attempts": attempts,
        "requested_questions": len(attempts),
        "uncapped_results": sum("correct" in r for r in attempts),
        "uncapped_correct": sum(r.get("correct", False) for r in attempts),
        "provider_clamps": sum(r.get("provider_clamp", False) for r in attempts),
        "uncollected": sum(r.get("uncollected", False) for r in attempts),
    }


def plot_accuracy(rows: list[dict], families: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 13, "font.family": "DejaVu Sans", "svg.fonttype": "none",
        "svg.hashsalt": "pdbthink", "pdf.fonttype": 42,
    })
    names = {
        "astra": "Astra", "opus": "Opus 5", "kimi": "Kimi K3", "glm": "GLM 5.3",
        "qwen": "Qwen 3.6 27B", "snowball": "Snowball v0.1.1",
        "deepseek": "DeepSeek V4.1 Flash", "mochi": "Mochi SFT 400407",
        "nemotron": "Nemotron 3 Super",
    }
    names.update({"standardized-" + name: label for name, label in list(names.items())})
    ordered = sorted(rows, key=lambda row: -row["macro_lower"])
    artifacts = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ("svg", "pdf", "png"):
            path = output / f"{name}.{extension}"
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
            if extension == "svg":
                metadata = {"Date": None}
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", metadata=metadata)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            artifacts.append(path)
        plt.close(fig)

    for name, metric, xlabel in (
        ("accuracy", "macro", "Mean family accuracy (%)"),
        ("question-accuracy", "micro", "Question accuracy (%)"),
    ):
        fig, ax = plt.subplots(figsize=(10, max(5.2, 0.5 * len(rows) + 1.2)))
        for y, row in enumerate(ordered):
            lower, upper = 100 * row[f"{metric}_lower"], 100 * row[f"{metric}_upper"]
            ax.barh(y, lower, color="#32677c", height=0.62)
            label = f"{lower:.1f}"
            if upper > lower:
                ax.plot([lower, upper], [y, y], color="#20313a", linewidth=2)
                ax.plot([upper, upper], [y - 0.13, y + 0.13], color="#20313a", linewidth=1.3)
                label = f"{lower:.1f}–{upper:.1f}"
            inside = lower > 88
            ax.text(lower - 1.5 if inside else upper + 1.5, y, label, va="center",
                    ha="right" if inside else "left", color="white" if inside else "#20313a", fontsize=12)
        ax.set_yticks(range(len(ordered)), [names.get(row["model"], row["label"]) for row in ordered])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(xlabel, labelpad=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=10)
        save(fig, name)

    family_ids = [family for family in FAMILY_NAMES if family in families[ordered[0]["model"]]]
    values = np.full((len(family_ids), len(ordered)), np.nan)
    fig, ax = plt.subplots(figsize=(11.5, 9))
    for x, row in enumerate(ordered):
        for y, family in enumerate(family_ids):
            cell = families[row["model"]][family]
            if cell["returned"] == cell["planned"]:
                value = cell["correct"] / cell["planned"]
                values[y, x] = value
                ax.text(x, y, f"{cell['correct']}/{cell['planned']}", ha="center", va="center",
                        color="white" if value > 0.6 else "#20313a", fontsize=11)
    colours = plt.get_cmap("Blues").copy()
    colours.set_bad("#dedede")
    ax.set_facecolor("#dedede")
    heatmap = ax.pcolormesh(
        np.arange(len(ordered) + 1) - 0.5, np.arange(len(family_ids) + 1) - 0.5,
        np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=colours, shading="flat",
    )
    ax.set_ylim(len(family_ids) - 0.5, -0.5)
    ax.set_xlim(-0.5, len(ordered) - 0.5)
    ax.set_xticks(range(len(ordered)), [names.get(row["model"], row["label"]) for row in ordered],
                  rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(family_ids)), [f"{family}  {FAMILY_NAMES[family]}" for family in family_ids])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    bar = fig.colorbar(heatmap, ax=ax, fraction=0.025, pad=0.03, ticks=[0, 0.5, 1])
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
    bar.ax.set_yticklabels(["0", "50", "100"])
    bar.set_label("Accuracy (%)")
    save(fig, "family-accuracy")

    fig, ax = plt.subplots(figsize=(9, 5.6))
    costed = [row for row in ordered if row["mean_response_cost_usd"] is not None]
    for row in costed:
        x, y = row["mean_response_cost_usd"], 100 * row["macro_lower"]
        ax.scatter(x, y, s=65, color="#32677c", zorder=3)
        ax.annotate(names.get(row["model"], row["label"]), (x, y), xytext=(9, -15),
                    textcoords="offset points", fontsize=12)
    maximum = max((row["mean_response_cost_usd"] for row in costed), default=1)
    ax.set_xlim(0, maximum * 1.35 if maximum else 1)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Mean response cost (USD / question)", labelpad=10)
    ax.set_ylabel("Mean family accuracy (%)", labelpad=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "accuracy-cost")
    with zipfile.ZipFile(output / "figures.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in artifacts:
            info = zipfile.ZipInfo(path.name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


def check_contract(manifest: dict) -> None:
    """An older panel must not acquire a current-main label during report regeneration."""
    current = {
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint(),
        "chemistry_fingerprint": chemistry_fingerprint(),
        "scoring_version": SCORING_VERSION,
        "definitions_sha256": Definitions.load().content_sha256,
    }
    changed = [key for key, value in current.items() if manifest.get(key) != value]
    if changed:
        raise ValueError(
            f"Frozen panel no longer matches this checkout ({', '.join(changed)}); rebuild and rerun"
        )


def pending_context_recoveries(selected: dict, attempts: list[dict], serving: dict) -> set[str]:
    """Full native-context answers remain provisional when a documented extension is available."""
    extended = serving.get("extended_context_recovery", {}).get("max_model_len", 0)
    contexts = {a["render_id"]: a["context_window"] for a in attempts}
    return {
        render_id
        for render_id, row in selected.items()
        if row["truncated"] and contexts[render_id] < extended
    }


def build(
    dataset: Path, runs: Path, config_path: Path, output: Path, *, standardized_report: dict | None = None,
) -> dict:
    manifest = read_json(dataset / "build_report.json")
    check_contract(manifest)
    models = read_json(config_path)
    names = [name for name in LABELS if name in models]
    if models.keys() - LABELS.keys():
        raise ValueError("Unknown model in report configuration")
    standardized = all(name.startswith("standardized-") for name in names)
    if not names or (any(name.startswith("standardized-") for name in names) and not standardized):
        raise ValueError("Original and standardized conditions must have separate result panels")
    serving = read_json(config_path.with_name(config_path.name.replace("models", "serving")))
    admitted = {r["render_id"]: r for r in manifest["changes"] if r["status"] == "admitted"}
    counts = Counter(q["family"] for q in admitted.values())
    instances, renders = load_dataset(dataset)
    by_instance = {i.semantic_instance_id: i for i in instances}
    by_render = {r.render_id: r for r in renders}
    output.mkdir(parents=True, exist_ok=True)
    rows, families, audits = [], {}, {}
    for name in names:
        require_standardized_condition(models[name])
        if standardized and models[name].get("benchmark_condition") != "standardized-v1":
            raise ValueError("Standardized results require a versioned standardized condition")
        protocol = models.get(name, {}).get("response_protocol", "plain")
        condition = models[name] if protocol != "plain" or standardized else None
        selected, attempts = collect(runs, name, admitted, protocol, condition)
        for attempt in attempts:
            folder = (runs / attempt["path"]).parent
            row = read_json(folder / "result.json")
            response = read_json(folder / "response.json")
            choice = response["choices"][0]
            message = choice["message"]
            render = by_render[row["render_id"]]
            rescored = score_response(
                answer_content(message, protocol),
                render.answer_schema,
                render.gold_answer,
                parameters=_scoring_parameters(by_instance[render.semantic_instance_id]),
                truncated=choice.get("finish_reason") == "length",
                provider_refusal=bool(message.get("refusal"))
                or choice.get("finish_reason") == "content_filter",
            )
            if any(row[key] != value for key, value in rescored.items()):
                raise ValueError("Saved score differs from current frozen scoring contract")
        audits[name] = attempts
        correct = Counter(r["family"] for r in selected.values() if r["score"]["correct"])
        returned = Counter(r["family"] for r in selected.values())
        pending = pending_context_recoveries(selected, attempts, serving.get(name, {}))
        recovering = Counter(selected[render_id]["family"] for render_id in pending)
        low = sum(correct[f] / n for f, n in counts.items()) / len(counts)
        high = sum(
            (correct[f] + n - returned[f] + recovering[f]) / n for f, n in counts.items()
        ) / len(counts)
        rows.append(
            {
                "model": name,
                "label": LABELS[name],
                "returned": len(selected),
                "planned": len(admitted),
                "pending_capacity_recoveries": len(pending),
                "correct": sum(correct.values()),
                "macro_lower": low,
                "macro_upper": high,
                "micro_lower": sum(correct.values()) / len(admitted),
                "micro_upper": (sum(correct.values()) + len(admitted) - len(selected) + len(pending))
                / len(admitted),
                "micro": (
                    sum(correct.values()) / len(admitted)
                    if len(selected) == len(admitted) and not pending
                    else None
                ),
                "truncated": sum(r["truncated"] for r in selected.values()),
                "native_output_truncations": sum(
                    r["truncated"] and r["budget_limit"] == "native_output" for r in selected.values()
                ),
                "context_truncations": sum(
                    r["truncated"] and r["budget_limit"] == "context" for r in selected.values()
                ),
                "format_errors": sum(r["format_error"] for r in selected.values()),
                "tool_events": sum(len(r["tool_events"]) for r in selected.values()),
                "completion_tokens": sum(r["usage"]["completion_tokens"] for r in selected.values()),
                "longest_completion": max(
                    (r["usage"]["completion_tokens"] for r in selected.values()), default=None
                ),
                "output_budget_min": min((r["max_output_tokens"] for r in selected.values()), default=None),
                "output_budget_max": max((r["max_output_tokens"] for r in selected.values()), default=None),
                **cost_metrics(selected, len(admitted), pending),
            }
        )
        families[name] = {
            f: {"correct": correct[f], "returned": returned[f], "planned": n} for f, n in counts.items()
        }
    failures = []
    for stage in RUN_STAGES:
        for name in names:
            for path in sorted((runs / stage / name).glob("*/failure.json")):
                failures.append(collection_failure(path, stage, name))
    evidence = {
        "report_version": 7 if standardized or standardized_report else 6 if "mochi" in names else 5,
        "cohort": "standardized-v1" if standardized else "original",
        "version": manifest["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_commit": manifest["scientific_commit"],
        "prompt_version": manifest["prompt_version"],
        "prompt_fingerprint": manifest["prompt_fingerprint"],
        "chemistry_fingerprint": manifest["chemistry_fingerprint"],
        "definitions_sha256": manifest["definitions_sha256"],
        "parent_archive_sha256": manifest["parent_archive_sha256"],
        "scoring_version": manifest["scoring_version"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "question_count": len(admitted),
        "family_count": len(counts),
        "metrics": rows,
        "cost_methodology": {
            "currency": "USD",
            "source": "Recorded OpenRouter usage.cost on non-BYOK selected responses",
            "source_documentation": "https://openrouter.ai/docs/cookbook/administration/usage-accounting",
            "scope": "Prompt, reasoning and answer charges for the selected response to each question",
            "plot_eligibility": (
                "Complete panel, no pending capacity recoveries, every selected response costed"
            ),
            "excluded": ["Probes", "Failed and superseded attempts", "Credit-purchase fees",
                         "Unmetered self-hosted infrastructure"],
            "missing_cost": "Unknown, never zero; no estimates or extrapolation",
        },
        "families": families,
        "models": {
            n: {
                k: v
                for k, v in models.get(n, {}).items()
                if k
                in [
                    "model_id",
                    "model_revision",
                    "context_window",
                    "native_output_limit",
                    "limit_source",
                    "extra",
                    "response_protocol",
                    "seed_base",
                    "benchmark_condition",
                    "sampling_exception",
                ]
            }
            for n in names
        },
        "attempts": audits,
        "collection_failures": failures,
        "budget_comparisons": {"glm_preliminary": {} if standardized else preliminary_glm(runs)},
        "serving": serving,
        "context_preflight": {
            name: read_json(runs / "discovery" / f"{name}-context-preflight.json")
            for name in names
            if models[name].get("tokenize")
        },
        "complete": all(r["returned"] == r["planned"] and not r["pending_capacity_recoveries"] for r in rows),
        "prompt_changes": {
            "system": sum(x.get("system_changed", False) for x in admitted.values()),
            "user": sum(x.get("user_changed", False) for x in admitted.values()),
            "gold": sum(x.get("gold_changed", False) for x in admitted.values()),
        },
    }
    if standardized_report is not None:
        if standardized_report["dataset_fingerprint"] != evidence["dataset_fingerprint"]:
            raise ValueError("Standardized comparison uses a different question panel")
        evidence["standardized_comparison"] = {
            "path": "standardized/", "complete": standardized_report["complete"],
            "metrics": standardized_report["metrics"], "models": standardized_report["models"],
        }
    plot_accuracy(rows, families, output)
    write_json(output / "evidence.json", evidence)
    with (output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metric_rows = []
    for r in rows:
        accuracy = (
            f"{100 * r['macro_lower']:.1f}%"
            if r["returned"] == r["planned"] and not r["pending_capacity_recoveries"]
            else f"{100 * r['macro_lower']:.1f}–{100 * r['macro_upper']:.1f}% (incomplete)"
        )
        micro = (
            f"{100 * r['micro']:.1f}%" if r["micro"] is not None
            else f"{100 * r['micro_lower']:.1f}–{100 * r['micro_upper']:.1f}%"
        )
        metric_rows.append(
            f"<tr><th>{html.escape(r['label'])}</th>"
            f"<td>{r['correct']} / {r['returned']} / {r['planned']}</td>"
            f"<td>{accuracy}</td><td>{micro}</td>"
            f"<td>{r['native_output_truncations']}</td><td>{r['context_truncations']}</td>"
            f"<td>{r['format_errors']}</td></tr>"
        )
    family_rows = []
    for family in (family for family in FAMILY_NAMES if family in counts):
        cells = "".join(
            f"<td>{families[n][family]['correct']} / {families[n][family]['returned']}</td>" for n in names
        )
        family_rows.append(
            f"<tr><th>{family} · {FAMILY_NAMES[family]} ({counts[family]} questions)</th>{cells}</tr>"
        )
    model_rows = []
    for n in names:
        model = models.get(n, {})
        revision = html.escape(model.get("model_revision") or "Provider does not expose immutable weights")
        contexts = sorted({a["context_window"] for a in audits[n]} or {model.get("context_window", 0)})
        ceilings = sorted(
            {a["declared_output_limit"] for a in audits[n]} or {model.get("native_output_limit", 0)}
        )
        context_text = " / ".join(f"{value:,}" for value in contexts)
        ceiling_text = " / ".join(f"{value:,}" for value in ceilings)
        model_rows.append(
            f"<tr><th>{html.escape(LABELS[n])}</th>"
            f"<td><code>{html.escape(model.get('model_id', 'Pending'))}</code></td>"
            f"<td>{context_text}</td><td>{ceiling_text}</td>"
            f"<td>{revision}</td>"
            "</tr>"
            if model
            else f'<tr><th>{LABELS[n]}</th><td colspan="4">Pending server verification</td></tr>'
        )
    budget_rows = []
    for row in rows:
        values = [row[key] for key in ("output_budget_min", "output_budget_max", "longest_completion")]
        cells = "".join(
            f"<td>{value:,}</td>" if value is not None else "<td>Pending</td>" for value in values
        )
        budget_rows.append(f"<tr><th>{html.escape(row['label'])}</th>{cells}</tr>")
    status = (
        "Complete head-to-head comparison"
        if evidence["complete"]
        else "Current results — incomplete response coverage or required capacity recoveries remain"
    )
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=True,
        undefined=StrictUndefined,
    )
    preflight = evidence["context_preflight"].get("snowball", {})
    template = "standardized_report.html.j2" if standardized else "coordinate_report.html.j2"
    page = environment.get_template(template).render(
        question_count=len(admitted),
        family_count=len(counts),
        generated_at=evidence["generated_at"],
        scientific_commit=manifest["scientific_commit"],
        scientific_short_commit=manifest["scientific_commit"][:7],
        prompt_version=manifest["prompt_version"],
        scoring_version=manifest["scoring_version"],
        status=status,
        has_mochi="mochi" in names,
        report_version=evidence["report_version"],
        standardized_report=standardized_report,
        models=models,
        serving=serving,
        metrics=rows,
        metric_rows=Markup("".join(metric_rows)),
        model_headers=Markup("".join(f"<th>{html.escape(LABELS[name])}</th>" for name in names)),
        family_rows=Markup("".join(family_rows)),
        model_rows=Markup("".join(model_rows)),
        budget_rows=Markup("".join(budget_rows)),
        snowball_min_input=f"{preflight.get('min_input_tokens', 0):,}",
        snowball_max_input=f"{preflight.get('max_input_tokens', 0):,}",
        dataset_fingerprint=manifest["dataset_fingerprint"],
        comparison=evidence["budget_comparisons"]["glm_preliminary"],
    )
    (output / "index.html").write_text(page)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = build(args.dataset, args.runs, args.config, args.output)
    print(
        json.dumps(
            {"complete": data["complete"], "coverage": {r["model"]: r["returned"] for r in data["metrics"]}}
        )
    )
