"""Resumable, server-checkpointed teacher sampling with native task rewards."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fcntl
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .teacher_data import (
    CONTEXT,
    MAX_ATTEMPTS,
    connect,
    digest,
    load_native_scorer,
    request_for,
    save_json,
    score_completion,
)

TERMINAL = {"completed", "failed", "expired", "cancelled", "canceled"}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Client:
    def __init__(self) -> None:
        self.base_url = os.environ["TEACHER_BASE_URL"].rstrip("/")
        self.token = os.environ["TEACHER_API_KEY"]

    def call(self, method: str, path: str, payload=None, *, raw: bool = False):
        headers = {"Authorization": "Bearer " + self.token, "x-priority": "bulk"}
        data = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload).encode()
            if payload is not None
            else None
        )
        if data is not None:
            headers["Content-Type"] = (
                "application/jsonl" if isinstance(payload, bytes) else "application/json"
            )
        request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            content = response.read()
        return content if raw else json.loads(content)


def register_job(root: Path, directory: Path, job: dict) -> None:
    selection = json.loads((directory / "selection.json").read_text())
    with connect(root) as db:
        db.execute(
            "INSERT OR IGNORE INTO batches VALUES (?,?,?,?)",
            (job["id"], str(directory.relative_to(root)), "submitted", now()),
        )
        for item in selection:
            row = db.execute(
                "SELECT attempts,active_batch FROM tasks WHERE path=?", (item["path"],)
            ).fetchone()
            if row["attempts"] >= item["attempt"]:
                continue
            if row["active_batch"] not in (None, job["id"]):
                raise ValueError("Task already assigned to another batch")
            db.execute("UPDATE tasks SET active_batch=? WHERE path=?", (job["id"], item["path"]))


def reconcile(root: Path, client: Client) -> None:
    """Reconcile a lost create response by input file identity, never by resubmission."""
    for directory in sorted((root / "batches").glob("*")):
        if (directory / "job.json").exists():
            job = json.loads((directory / "job.json").read_text())
            with connect(root) as db:
                registered = db.execute("SELECT 1 FROM batches WHERE id=?", (job["id"],)).fetchone()
            if not registered:
                register_job(root, directory, job)
        elif (directory / "submission-intent.json").exists():
            file_id = json.loads((directory / "input-file.json").read_text())["id"]
            listing = client.call("GET", "/batches?limit=1000")
            matches = [j for j in listing.get("data", []) if j.get("input_file_id") == file_id]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Unreconciled batch submission in {directory}; refusing duplicate execution"
                )
            save_json(directory / "job.json", matches[0])
            register_job(root, directory, matches[0])


def submit(root: Path, client: Client, tasks: dict, selected: list[dict]) -> None:
    selection = [{"path": r["path"], "attempt": r["attempts"] + 1} for r in selected]
    identity = digest(json.dumps(selection, sort_keys=True).encode())[:16]
    # Infrastructure retries retain semantic attempt numbers but have separate evidence.
    folders = list((root / "batches").glob(identity + "-*"))
    directory = root / "batches" / f"{identity}-{len(folders):03d}"
    directory.mkdir(parents=True)
    save_json(directory / "selection.json", selection)
    lines = []
    for index, item in enumerate(selection):
        lines.append(
            {
                "custom_id": str(index),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_for(tasks[item["path"]], item["attempt"]),
            }
        )
    data = ("\n".join(json.dumps(r) for r in lines) + "\n").encode()
    (directory / "requests.jsonl").write_bytes(data)
    upload = client.call("POST", f"/files?purpose=batch&filename=teacher-{directory.name}.jsonl", data)
    save_json(directory / "input-file.json", upload)
    payload = {"input_file_id": upload["id"], "endpoint": "/v1/chat/completions", "priority": "bulk"}
    save_json(directory / "submission-intent.json", {"time": now(), "payload": payload})
    job = client.call("POST", "/batches", payload)
    if not job.get("id"):
        raise RuntimeError("Batch create returned no job ID; inspect submission intent")
    save_json(directory / "job.json", job)
    register_job(root, directory, job)
    print(json.dumps({"submitted": len(selection), "batch": directory.name, "time": now()}), flush=True)


def process_line(root: Path, directory: Path, item: dict, line: dict, task: dict, scorer) -> bool:
    with connect(root) as db:
        existing = db.execute(
            "SELECT result_path FROM attempts WHERE path=? AND attempt=?", (task["path"], item["attempt"])
        ).fetchone()
    if existing:
        previous = json.loads((root / existing["result_path"]).read_text())
        if previous["raw_response"] != (line.get("response") or {}).get("body"):
            raise ValueError("Conflicting replay of an already scored attempt")
        return True
    response = line.get("response") or {}
    body = response.get("body") or {}
    if response.get("status_code") != 200 or not body.get("choices"):
        return False
    if body.get("model") != "glm-5.3":
        raise ValueError("Unexpected teacher model identity")
    observed = body.get("usage", {}).get("prompt_tokens")
    if observed != task["teacher_input_tokens"]:
        save_json(
            directory / "tokenizer-mismatch.json",
            {"path": task["path"], "observed": observed, "expected": task["teacher_input_tokens"]},
        )
        raise ValueError("Teacher prompt tokenisation differs from frozen request budget")
    result = score_completion(task, body, scorer)
    allowed = CONTEXT - task["teacher_input_tokens"]
    if result["finish_reason"] == "length" and result["usage"].get("completion_tokens", 0) < allowed:
        save_json(
            directory / "budget-investigation.json", {"path": task["path"], "allowed": allowed, **result}
        )
        raise ValueError("Provider output clamp requires investigation before counting a failure")
    result.update(
        {
            "path": task["path"],
            "family": task["family"],
            "attempt": item["attempt"],
            "prompt_sha256": task["prompt_sha256"],
            "request_max_tokens": allowed,
            "raw_response": body,
            "batch_directory": directory.name,
            "recorded_at": now(),
        }
    )
    target = root / "attempts" / digest(task["path"].encode()) / f"{item['attempt']:02d}.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous["raw_response"] != body:
            raise ValueError("Different responses claim the same attempt identity")
    else:
        save_json(target, result)
    with connect(root) as db:
        db.execute(
            "INSERT OR IGNORE INTO attempts VALUES (?,?,?,?)",
            (task["path"], item["attempt"], result["reward"], str(target.relative_to(root))),
        )
        db.execute(
            "UPDATE tasks SET attempts=?, solved=?, active_batch=NULL WHERE path=?",
            (item["attempt"], int(result["reward"] == 1), task["path"]),
        )
    return True


def collect(root: Path, client: Client, tasks: dict, scorer, batch: dict) -> None:
    directory = root / batch["directory"]
    job = client.call("GET", "/batches/" + batch["id"])
    save_json(directory / "status.json", job)
    if job["status"] not in TERMINAL:
        return
    selection = json.loads((directory / "selection.json").read_text())
    returned, seen = set(), set()
    for field in ("output_file_id", "error_file_id"):
        if not job.get(field):
            continue
        data = client.call("GET", "/files/" + job[field] + "/content", raw=True)
        (directory / (field + ".jsonl")).write_bytes(data)
        for raw in data.splitlines():
            if not raw.strip():
                continue
            line = json.loads(raw)
            index = int(line["custom_id"])
            if index < 0 or index >= len(selection) or index in seen:
                raise ValueError("Unknown or duplicate batch response identity")
            seen.add(index)
            item = selection[index]
            if process_line(root, directory, item, line, tasks[item["path"]], scorer):
                returned.add(index)
    with connect(root) as db:
        for index, item in enumerate(selection):
            if index not in returned:
                db.execute(
                    "UPDATE tasks SET active_batch=NULL, infra_errors=infra_errors+1 "
                    "WHERE path=? AND active_batch=?",
                    (item["path"], batch["id"]),
                )
        db.execute("UPDATE batches SET status=? WHERE id=?", (job["status"], batch["id"]))
    print(
        json.dumps(
            {
                "collected": len(returned),
                "infrastructure_failures": len(selection) - len(returned),
                "batch": directory.name,
                "time": now(),
            }
        ),
        flush=True,
    )


def status(root: Path) -> dict:
    with connect(root) as db:
        rows = db.execute(
            "SELECT family,COUNT(*) total,SUM(solved) solved,SUM(attempts) attempts, "
            "SUM(CASE WHEN attempts>=? AND solved=0 THEN 1 ELSE 0 END) exhausted, "
            "SUM(CASE WHEN active_batch IS NOT NULL THEN 1 ELSE 0 END) active, "
            "SUM(infra_errors) infrastructure_errors FROM tasks GROUP BY family",
            (MAX_ATTEMPTS,),
        ).fetchall()
        families = {r["family"]: {k: r[k] for k in r.keys() if k != "family"} for r in rows}
        totals = {k: sum(v[k] for v in families.values()) for k in next(iter(families.values()))}
    result = {
        "time": now(),
        "totals": totals,
        "families": families,
        "complete": totals["solved"] + totals["exhausted"] == totals["total"],
    }
    save_json(root / "progress.json", result)
    return result


def remaining_requests(root: Path, batch: dict) -> int:
    directory = root / batch["directory"]
    status_file = directory / "status.json"
    if status_file.exists():
        counts = json.loads(status_file.read_text()).get("request_counts", {})
        if counts.get("total", 0) > 0:
            remaining = counts["total"] - counts["completed"] - counts["failed"]
            if remaining < 0:
                raise ValueError("Provider batch counts are inconsistent")
            return remaining
    return len(json.loads((directory / "selection.json").read_text()))


def run(root: Path, *, batch_size: int, active_jobs: int, pilot: bool, max_inflight: int = 0) -> None:
    import pyarrow.parquet as pq

    tasks = {r["path"]: r for r in pq.read_table(root / "tasks.parquet").to_pylist()}
    scorer, client = load_native_scorer(root), Client()
    pilot_paths = set()
    for family in sorted({t["family"] for t in tasks.values()}):
        eligible = [t for t in tasks.values() if t["family"] == family]
        eligible.sort(key=lambda t: digest((t["path"] + ":pilot").encode()))
        pilot_paths.add(eligible[0]["path"])
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reconcile(root, client)
        while True:
            with connect(root) as db:
                active = [
                    dict(r)
                    for r in db.execute(
                        "SELECT * FROM batches WHERE status NOT IN "
                        "('completed','failed','expired','cancelled','canceled')"
                    )
                ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(collect, root, client, tasks, scorer, batch) for batch in active]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except (urllib.error.URLError, TimeoutError, OSError) as error:
                        print(
                            json.dumps({"collection_deferred": type(error).__name__, "time": now()}),
                            flush=True,
                        )
            progress = status(root)
            print(json.dumps({"progress": progress["totals"], "time": now()}), flush=True)
            if progress["complete"]:
                return
            with connect(root) as db:
                live = [
                    dict(r)
                    for r in db.execute(
                        "SELECT * FROM batches WHERE status NOT IN "
                        "('completed','failed','expired','cancelled','canceled')"
                    )
                ]
                count = len(live)
                pending = [
                    dict(r)
                    for r in db.execute(
                        "SELECT * FROM tasks WHERE solved=0 AND attempts<? "
                        "AND active_batch IS NULL ORDER BY attempts DESC,path",
                        (MAX_ATTEMPTS,),
                    )
                ]
            if pilot:
                pending = [r for r in pending if r["path"] in pilot_paths and r["attempts"] == 0]
            if not pending and count == 0:
                return
            capacity = (active_jobs - count) * batch_size
            if max_inflight:
                capacity = min(capacity, max_inflight - sum(remaining_requests(root, b) for b in live))
            available = pending[: max(0, capacity)]
            selections = [
                available[start : start + batch_size] for start in range(0, len(available), batch_size)
            ]
            if any(r["infra_errors"] >= 20 for chosen in selections for r in chosen):
                raise RuntimeError("Repeated infrastructure failures require inspection")
            # Each disjoint selection has independent request and submission evidence.
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(submit, root, client, tasks, chosen) for chosen in selections]
                for future in futures:
                    future.result()
            time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--active-jobs", type=int, default=12)
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=0,
        help="Bound unfinished server requests, excluding finished lines",
    )
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(args.root), indent=2))
    else:
        run(
            args.root,
            batch_size=args.batch_size,
            active_jobs=args.active_jobs,
            pilot=args.pilot,
            max_inflight=args.max_inflight,
        )


if __name__ == "__main__":
    main()
