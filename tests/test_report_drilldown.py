"""Full trace exports must preserve evidence without confusing missing output with failure."""

import importlib.util
import json
from pathlib import Path

import pytest


def script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outcomes_separate_exhaustion_refusal_and_format():
    module = script("report_drilldown")
    row = {
        "score": {"correct": False},
        "truncated": True,
        "budget_limit": "context",
        "refusal": False,
        "format_error": True,
    }
    assert module.outcome(row) == "context_exhausted"
    row.update(truncated=False, refusal=True)
    assert module.outcome(row) == "refusal"
    row.update(refusal=False)
    assert module.outcome(row) == "format_error"
    row.update(format_error=False)
    assert module.outcome(row) == "wrong_answer"


def test_complete_output_preserves_every_message_field_without_account_metadata():
    module = script("report_drilldown")
    message = {
        "role": "assistant",
        "content": "<|start_think|>\n" + "α <script>" * 100000 + "<|end_think|>FINAL: A",
        "reasoning": "verbatim\n",
        "reasoning_details": [{"text": "detail"}],
    }
    response = {
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"account": "private"},
        "error": {"message": "private provider error"},
    }
    exported = json.loads(json.dumps(module.public_response(response)))
    assert exported["choices"][0]["message"] == message
    assert "usage" not in exported and "error" not in exported


def test_partial_stream_preserves_text_but_not_provider_errors(tmp_path):
    module = script("report_drilldown")
    path = tmp_path / "stream.txt"
    events = [
        {"choices": [{"index": 0, "delta": {"reasoning": "First\n"}}]},
        {"choices": [{"index": 0, "delta": {"content": "<|start_think|>"}}]},
        {"choices": [{"index": 0, "delta": {"reasoning": "Second", "content": "full text"}}]},
        {"error": {"message": "account secret"}},
    ]
    path.write_text("".join("data: " + json.dumps(event) + "\n\n" for event in events) + 'data: {"choices":[')
    output = module.partial_response(path)
    assert output["choices"][0] == {
        "index": 0,
        "finish_reason": None,
        "message": {"reasoning": "First\nSecond", "content": "<|start_think|>full text"},
    }
    assert "secret" not in json.dumps(output)


def test_export_rejects_prompt_substitution():
    from types import SimpleNamespace

    module = script("report_drilldown")
    render = SimpleNamespace(system_prompt="frozen system", user_prompt="exact coordinates\n")
    request = {
        "messages": [
            {"role": "system", "content": render.system_prompt},
            {"role": "user", "content": render.user_prompt},
        ]
    }
    module.verify_prompt(request, render)
    request["messages"][1]["content"] = request["messages"][1]["content"].strip()
    with pytest.raises(ValueError, match="frozen prompt"):
        module.verify_prompt(request, render)


@pytest.mark.parametrize("mutation", ["hash", "path"])
def test_publication_validates_question_manifest(tmp_path, mutation):
    from pdbthink.util import sha256_bytes, write_json

    module = script("prepare_report_publication")
    write_json(tmp_path / "evidence.json", {"drilldown": {"manifest": "question-artifacts.json"}})
    (tmp_path / "questions").mkdir()
    path = tmp_path / "questions/index.json"
    path.write_text("full output")
    name = "../outside.json" if mutation == "path" else "questions/index.json"
    digest = sha256_bytes(path.read_bytes()) if mutation == "path" else "incorrect hash"
    write_json(tmp_path / "question-artifacts.json", {"artifacts": {name: digest}})
    with pytest.raises(ValueError, match="Unsafe|hash mismatch"):
        module.artifact_paths(tmp_path)
