"""Teacher data must never confuse answer correctness, context fit or retries."""

import json
import sqlite3

import pytest

from pdbthink.taskgen.teacher_data import request_for, save_json, score_completion
from pdbthink.taskgen.teacher_run import process_line, remaining_requests


def task():
    return {
        "path": "test-task",
        "family": "G01",
        "prompt_sha256": "prompt-digest",
        "system_prompt": "No tools.",
        "user_prompt": "Compute the distance.",
        "teacher_input_tokens": 100,
        "gold_json": json.dumps(
            {"answer_schema": "distance", "gold_answer": {"value": 2.0}, "parameters": {"tolerance": 0.02}}
        ),
    }


def test_retries_are_fresh_and_contain_neither_gold_nor_failed_answers():
    first, second = request_for(task(), 1), request_for(task(), 2)
    assert (
        first["messages"]
        == second["messages"]
        == [
            {"role": "system", "content": "No tools."},
            {"role": "user", "content": "Compute the distance."},
        ]
    )
    assert first["seed"] != second["seed"]
    assert first["max_tokens"] == 262044
    assert first["tools"] is None and first["tool_choice"] == "none"
    assert first["prompt_cache_key"] == second["prompt_cache_key"]


def test_native_reward_rejects_tool_use_even_with_a_correct_final_answer():
    from pdbthink.scoring import score_response

    body = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "FINAL: 2.0",
                    "reasoning_content": "Distance calculation.",
                    "tool_calls": [{"function": {"name": "python"}}],
                },
            }
        ]
    }
    scored = score_completion(task(), body, score_response)
    assert scored["outcome"]["score"]["correct"]
    assert scored["reward"] == 0 and scored["tool_events"]


def test_replaying_an_old_batch_cannot_overwrite_a_later_success(tmp_path):
    db = sqlite3.connect(tmp_path / "state.sqlite")
    db.executescript("""
        CREATE TABLE tasks(path TEXT PRIMARY KEY, attempts INTEGER, solved INTEGER, active_batch TEXT);
        CREATE TABLE attempts(path TEXT, attempt INTEGER, reward REAL, result_path TEXT,
                              PRIMARY KEY(path,attempt));
        INSERT INTO tasks VALUES ('test-task', 2, 1, NULL);
        INSERT INTO attempts VALUES ('test-task',1,0,'first.json');
    """)
    db.commit()
    old = {"choices": [{"message": {"content": "FINAL: 9"}}]}
    (tmp_path / "first.json").write_text(json.dumps({"raw_response": old}))
    assert process_line(tmp_path, tmp_path, {"attempt": 1}, {"response": {"body": old}}, task(), None)
    assert db.execute("SELECT attempts,solved FROM tasks").fetchone() == (2, 1)
    with pytest.raises(ValueError, match="Conflicting replay"):
        process_line(
            tmp_path, tmp_path, {"attempt": 1}, {"response": {"body": {"choices": []}}}, task(), None
        )


def test_finished_lines_release_capacity_before_the_last_batch_line_finishes(tmp_path):
    directory = tmp_path / "batch"
    directory.mkdir()
    (directory / "selection.json").write_text(json.dumps([{"path": str(i)} for i in range(32)]))
    batch = {"directory": "batch"}
    assert remaining_requests(tmp_path, batch) == 32
    (directory / "status.json").write_text(
        json.dumps(
            {
                "status": "in_progress",
                "request_counts": {"total": 32, "completed": 30, "failed": 1},
            }
        )
    )
    assert remaining_requests(tmp_path, batch) == 1


def test_export_shards_share_types_when_early_responses_omit_reasoning_usage(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("matplotlib")
    pytest.importorskip("transformers")
    import pyarrow.parquet as pq

    from pdbthink.taskgen.teacher_export import write_shards

    rows = [{"reasoning_tokens": None, "answer": "A"}, {"reasoning_tokens": 12, "answer": "B"}]
    write_shards(rows, tmp_path, size=1)
    shards = sorted(tmp_path.glob("*.parquet"))
    assert pq.read_schema(shards[0]) == pq.read_schema(shards[1])
    assert pq.read_table(tmp_path).to_pylist() == rows


def test_concurrent_progress_writers_leave_one_complete_document(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    target = tmp_path / "progress.json"
    documents = [{"writer": i, "values": [i] * 100} for i in range(64)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: save_json(target, value), documents))
    assert json.loads(target.read_text()) in documents
    assert list(tmp_path.iterdir()) == [target]
