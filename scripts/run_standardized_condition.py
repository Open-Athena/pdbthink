"""Collect one frozen comparison condition, retrying only transport failures."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_coordinate_report import api, frozen, run

from pdbthink.dataset import load_dataset
from pdbthink.evaluation.standardized_conditions import require_standardized_condition
from pdbthink.util import read_json, sha256_bytes, stable_hash, write_json

STAGES = ("standardized_v1", *(f"standardized_recovery_{n}" for n in range(1, 4)))


def collect_condition(
    dataset: Path, config: Path, name: str, runs: Path, workers: int, *, shard: int = 0, shards: int = 1,
) -> None:
    os.umask(0o077)
    if not name.startswith("standardized-") or not 0 <= shard < shards:
        raise ValueError("Expected a standardized variant and valid disjoint shard")
    endpoint = read_json(config)[name]
    require_standardized_condition(endpoint)
    runs = runs / endpoint.get("run_namespace", "")
    _, panel = load_dataset(dataset)
    renders = [render for i, render in enumerate(sorted(panel, key=lambda r: r.render_id))
               if i % shards == shard]
    ids = {render.render_id for render in renders}
    state_path = runs / "states" / f"{name}-{shard}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    def state(status: str, **values) -> None:
        payload = {"status": status, "model": name, "shard": shard, "shards": shards,
                   "planned": len(ids), "updated_at": time.time(), **values}
        temporary = state_path.with_suffix(".tmp")
        write_json(temporary, payload)
        temporary.replace(state_path)
        print(payload, flush=True)

    def selected() -> dict:
        found = {}
        for stage in STAGES:
            for path in (runs / stage / name).glob("*/result.json"):
                row = read_json(path)
                if row["render_id"] not in ids:
                    continue
                if row["render_id"] in found:
                    raise ValueError("A returned standardized answer was evaluated twice")
                found[row["render_id"]] = row
        return found

    def preflight(render) -> dict:
        folder = runs / STAGES[0] / name / stable_hash(render.render_id)[:16]
        folder.mkdir(parents=True, exist_ok=True)
        request = {
            "model": endpoint["model_id"],
            "messages": [{"role": "system", "content": render.system_prompt},
                         {"role": "user", "content": render.user_prompt}],
            "add_generation_prompt": True,
            "chat_template_kwargs": endpoint["extra"]["chat_template_kwargs"],
            "tools": None if endpoint.get("tools_null") else [], "tool_choice": "none",
        }
        frozen(folder / "tokenize-request.json", request)
        response_path = folder / "tokenize-response.json"
        if not response_path.exists():
            response = api({**endpoint, "base_url": endpoint["base_url"].removesuffix("/v1")},
                           "/tokenize", request)
            write_json(response_path, response)
        count = read_json(response_path)["count"]
        if not 0 < count < endpoint["context_window"]:
            raise ValueError("Frozen prompt does not fit the comparison context; record an exclusion")
        return {"render_id": render.render_id, "input_tokens": count,
                "remaining_output_capacity": endpoint["context_window"] - count,
                "tokenize_request_sha256": stable_hash(request),
                "tokenize_response_sha256": sha256_bytes(response_path.read_bytes())}

    try:
        state("preflight", workers=workers)
        frozen(runs / STAGES[0] / name / "endpoint.json", endpoint)
        discovery = runs / "discovery"
        discovery.mkdir(parents=True, exist_ok=True)
        if endpoint.get("tokenize"):
            registry = api(endpoint, "/models")
            model = next(row for row in registry["data"] if row["id"] == endpoint["model_id"])
            if model.get("max_model_len") != endpoint["context_window"]:
                raise ValueError("Live model registry differs from the standardized context")
            if name == "standardized-mochi" and model.get("root") != endpoint["model_revision"]:
                raise ValueError("Mochi deployment no longer serves the frozen checkpoint")
            write_json(discovery / f"{name}-registry-{shard}.json", registry)
            with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
                rows = list(pool.map(preflight, renders))
            record = {"model": name, "context_window": 262144, "questions": len(rows), "rows": rows,
                      "min_input_tokens": min(row["input_tokens"] for row in rows),
                      "max_input_tokens": max(row["input_tokens"] for row in rows)}
            suffix = "" if shards == 1 else f"-shard{shard}"
            write_json(discovery / f"{name}-context-preflight{suffix}.json", record)
        for wave, stage in enumerate(STAGES):
            remaining = ids - selected().keys()
            if not remaining:
                break
            for previous in STAGES[:wave]:
                for render_id in remaining:
                    folder = runs / previous / name / stable_hash(render_id)[:16]
                    failure = folder / "failure.json"
                    if (folder / "submission-intent.json").exists() and not failure.exists():
                        raise ValueError("Unreconciled submission requires transport review before retrying")
                    if failure.exists() and any(word in read_json(failure).get("error", "").lower()
                                                for word in ("tool event", "clamped", "token count")):
                        raise ValueError("Protocol or provider-capacity failure requires review")
            # Re-entering the controller must not resubmit an uncertain earlier request.
            todo = set()
            for render_id in remaining:
                folder = runs / stage / name / stable_hash(render_id)[:16]
                submitted = (folder / "submission-intent.json").exists()
                if not submitted and not (folder / "failure.json").exists():
                    todo.add(render_id)
            state("collecting" if wave == 0 else "transport_recovery", completed=len(selected()),
                  remaining=len(remaining), wave=wave, stage=stage, workers=workers)
            if todo:
                run(dataset, {name: endpoint}, runs / stage, workers, todo)
            if wave < len(STAGES) - 1 and ids - selected().keys():
                # Avoid immediately amplifying a transient provider outage.
                time.sleep(30)
        found = selected()
        state("complete" if len(found) == len(ids) else "needs_attention", completed=len(found),
              correct=sum(row["score"]["correct"] for row in found.values()),
              truncated=sum(row["truncated"] for row in found.values()))
        if len(found) != len(ids):
            raise RuntimeError("Collection remains incomplete after recorded transport retries")
    except Exception as error:
        state("needs_attention", completed=len(selected()), error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    collect_condition(args.dataset, args.config, args.model, args.runs, args.workers,
                      shard=args.shard, shards=args.shards)
