"""Teacher data must never confuse answer correctness, context fit or retries."""

import json
import sqlite3

import pytest

from pdbthink.taskgen.teacher_data import request_for, score_completion
from pdbthink.taskgen.teacher_run import process_line


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
