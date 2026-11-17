"""Resume rejected or interrupted requests without repeating completed answers.

Use after the initial collectors have stopped. Recovery shares one small worker
pool across providers and honours their cooldowns, so a depleted in-flight credit
budget does not turn the rest of the panel into a burst of rejected requests.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from evaluate_coordinate_report import frozen, run_one

from pdbthink.dataset import load_dataset
from pdbthink.evaluation.standardized_conditions import require_standardized_condition
from pdbthink.util import read_json, stable_hash, write_json

STAGES = ("standardized_v1", *(f"standardized_recovery_{n}" for n in range(1, 17)))


def missing_requests(runs: Path, name: str, ids: set[str]) -> set[str]:
    completed: set[str] = set()
    review: dict[str, str] = {}
    for stage in STAGES:
        for folder in (runs / stage / name).glob("*"):
            if not folder.is_dir():
                continue
            result = folder / "result.json"
            failure = folder / "failure.json"
            if result.exists():
                row = read_json(result)
                render_id = row["render_id"]
                if render_id not in ids or render_id in completed:
                    raise ValueError("Unknown or duplicate returned answer")
                completed.add(render_id)
            elif failure.exists():
                row = read_json(failure)
                error = row.get("error", "").lower()
                if any(word in error for word in ("clamped", "tool event", "token count")):
                    review[row["render_id"]] = error
            elif (folder / "submission-intent.json").exists():
                raise ValueError(f"Unreconciled submission: {folder}")
    if review.keys() - completed:
        raise ValueError("Protocol or capacity failures need separate review: " + str(review))
    return ids - completed


def retry_delay(result: dict) -> float:
    if result.get("status") != "provider_error":
        return 0
    status = result.get("http_status")
    error = result.get("error", "")
    if status not in (402, 429, 503) and not any(str(code) in error for code in (402, 429, 503)):
        return 0
    delay = 120.0
    try:
        body = json.loads(result.get("provider_body", "{}"))
        headers = body.get("error", {}).get("metadata", {}).get("headers", {})
        delay = max(delay, float(headers.get("Retry-After", delay)))
    except (ValueError, TypeError):
        pass
    return delay


def recover(dataset: Path, config: Path, runs: Path, names: list[str], workers: int = 2) -> None:
    os.umask(0o077)
    models = read_json(config)
    instances, renders = load_dataset(dataset)
    by_id = {instance.semantic_instance_id: instance for instance in instances}
    ids = {render.render_id for render in renders}
    for name in names:
        require_standardized_condition(models[name])
        if not models[name]["base_url"].startswith("https://openrouter.ai/"):
            raise ValueError("This recovery pool is for the hosted conditions")
        for stage in STAGES:
            path = runs / stage / name / "endpoint.json"
            if path.exists() and read_json(path) != models[name]:
                raise ValueError("Recovery would change a frozen model condition")
    state_path = runs / "states/standardized-hosted-recovery.json"

    def state(status: str, **values) -> None:
        temporary = state_path.with_suffix(".tmp")
        write_json(temporary, {"status": status, "updated_at": time.time(), **values})
        temporary.replace(state_path)

    try:
        for stage in STAGES[4:]:
            missing = {name: missing_requests(runs, name, ids) for name in names}
            if not any(missing.values()):
                state("complete", models=names)
                return
            jobs = []
            for render in sorted(renders, key=lambda row: row.render_id):
                for name in names:
                    if render.render_id not in missing[name]:
                        continue
                    folder = runs / stage / name / stable_hash(render.render_id)[:16]
                    if (folder / "submission-intent.json").exists() or (folder / "failure.json").exists():
                        continue
                    jobs.append((name, render))
            if not jobs:
                continue
            for name in names:
                frozen(runs / stage / name / "endpoint.json", models[name])
            state("collecting", stage=stage, pending=len(jobs), workers=workers)
            cooldown_until = 0.0
            iterator = iter(jobs)
            exhausted = False
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {}
                while pending or not exhausted:
                    while len(pending) < workers and not exhausted and time.time() >= cooldown_until:
                        try:
                            name, render = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                        future = pool.submit(run_one, models[name], by_id[render.semantic_instance_id],
                                             render, runs / stage / name)
                        pending[future] = (name, render.render_id)
                    if not pending:
                        time.sleep(min(30, max(0.1, cooldown_until - time.time())))
                        continue
                    done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        name, render_id = pending.pop(future)
                        result = future.result()
                        delay = retry_delay(result)
                        cooldown_until = max(cooldown_until, time.time() + delay)
                        print(json.dumps({"model": name, "render_id": render_id,
                                          "status": result["status"], "cooldown_seconds": delay}), flush=True)
                        if delay:
                            state("provider_cooldown", stage=stage, retry_at=cooldown_until)
            time.sleep(120)
        missing = {name: sorted(missing_requests(runs, name, ids)) for name in names}
        if any(missing.values()):
            raise RuntimeError(f"Transport recovery remains incomplete: {missing}")
        state("complete", models=names)
    except Exception as error:
        state("needs_attention", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    recover(args.dataset, args.config, args.runs, args.models, args.workers)
