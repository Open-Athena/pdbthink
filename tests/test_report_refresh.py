"""A refreshed report must keep historical requests separate and preserve its budget contract."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest


def script(name):
    path = Path(__file__).parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transport_rejects_tools_before_contacting_provider(monkeypatch):
    module = script("evaluate_coordinate_report")

    def unexpected(*args, **kwargs):
        pytest.fail("Unsafe request reached network transport")

    monkeypatch.setattr(module.urllib.request, "urlopen", unexpected)
    with pytest.raises(ValueError, match="explicit tools"):
        module.api(
            {"base_url": "https://openrouter.ai/api/v1"},
            "/chat/completions",
            {"model": "example", "messages": []},
        )


def test_iap_refreshes_credentials_and_omits_openai_authorization(monkeypatch):
    from types import SimpleNamespace

    module = script("evaluate_coordinate_report")
    tokens = iter(["first", "refreshed"])
    requests = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-sent")
    monkeypatch.setattr(
        module, "iap_credentials", lambda cluster: SimpleNamespace(get_token=lambda: next(tokens))
    )

    def opener(*handlers):
        assert module.NoRedirect in handlers

        def open_request(request, **kwargs):
            requests.append({key.lower(): value for key, value in request.header_items()})
            return io.BytesIO(b'{}')

        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(module.urllib.request, "build_opener", opener)
    endpoint = {"base_url": "https://iris.oa.dev/proxy/example/v1", "iap_cluster": "marin"}
    module.api(endpoint, "/models")
    module.api(endpoint, "/models")
    assert [headers["proxy-authorization"] for headers in requests] == ["Bearer first", "Bearer refreshed"]
    assert all("authorization" not in headers for headers in requests)
    assert module.NoRedirect().redirect_request(None, None, None, None, None, None) is None


@pytest.mark.parametrize("changed", ["temperature", "seed"])
def test_model_condition_audit_rejects_changed_sampling(changed):
    module = script("build_coordinate_report")
    endpoint = {"model_id": "mochi", "extra": {"temperature": 0.7}, "seed_base": 123}
    request = {"model": "mochi", "temperature": 0.7, "seed": module.derive_seed(123, "question")}
    module.audit_model_condition(request, endpoint, "question")
    request[changed] = 0
    with pytest.raises(ValueError, match=changed):
        module.audit_model_condition(request, endpoint, "question")


def test_stream_preserves_reasoning_usage_and_completion(tmp_path, monkeypatch):
    module = script("evaluate_coordinate_report")
    chunks = [
        {"id": "completion-1", "choices": [{"index": 0, "delta": {"reasoning": "Calculate. "}}]},
        {"choices": [{"index": 0, "delta": {"content": "FINAL: A"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
    ]
    data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(data.encode()))
    payload = {"model": "test", "messages": [], "tools": [], "tool_choice": "none", "stream": True}
    raw_path = tmp_path / "stream.txt"
    response = module.api({"base_url": "https://example.test/v1"}, "/chat/completions", payload, raw_path)
    assert response["choices"][0]["message"]["content"] == "FINAL: A"
    assert response["choices"][0]["message"]["reasoning"] == "Calculate. "
    assert response["usage"]["completion_tokens"] == 20
    assert raw_path.read_text() == data.rstrip() + "\n"


def test_incomplete_stream_is_retained_but_not_scored(tmp_path, monkeypatch):
    module = script("evaluate_coordinate_report")
    data = b'data: {"choices":[{"delta":{"content":"FINAL: A"}}]}\n\n'
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(data))
    path = tmp_path / "stream.txt"
    with pytest.raises(ValueError, match="Incomplete provider stream"):
        module.api(
            {"base_url": "https://example.test/v1"},
            "/chat/completions",
            {"model": "test", "messages": [], "tools": [], "tool_choice": "none", "stream": True},
            path,
        )
    assert path.read_bytes() == data


@pytest.mark.parametrize(
    "mutation, message", [("prompt", "frozen question"), ("budget", "full endpoint budget")]
)
def test_publication_rejects_wrong_prompt_or_shortened_budget(tmp_path, mutation, message):
    from pdbthink.util import sha256_bytes, stable_hash, write_json

    module = script("build_coordinate_report")
    messages = [{"role": "system", "content": "Frozen rules"}, {"role": "user", "content": "Frozen question"}]
    question = {
        "family": "P01",
        **{m["role"] + "_sha256": sha256_bytes(m["content"].encode()) for m in messages},
    }
    budget = {
        "input_tokens": 20,
        "context_window": 1000,
        "native_output_limit": 500,
        "tokenizer_revision": "exact server",
        "endpoint_limit_source": "registry",
    }
    request = module.request_payload("test-model", messages, budget)
    if mutation == "prompt":
        request["messages"][1]["content"] = "A different question"
    else:
        request["max_tokens"] = 100
    response = {
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        "choices": [{"message": {"content": "FINAL: A"}, "finish_reason": "stop"}],
    }
    write_json(tmp_path / "request.json", request)
    write_json(tmp_path / "response.json", response)
    write_json(tmp_path / "budget.json", budget)
    row = {
        "request_sha256": stable_hash(request),
        "response_sha256": sha256_bytes((tmp_path / "response.json").read_bytes()),
        "max_output_tokens": request["max_tokens"],
    }
    with pytest.raises(ValueError, match=message):
        module.audit_result(tmp_path, row, question)


@pytest.mark.parametrize("complete", [True, False])
def test_publication_preserves_previous_report_and_requires_explicit_partial_publication(tmp_path, complete):
    import subprocess

    from pdbthink.util import read_json, write_json

    module = script("prepare_report_publication")
    source, site = tmp_path / "source", tmp_path / "site"
    for directory in (source, site):
        directory.mkdir()
        subprocess.run(["git", "init", "-q", str(directory)], check=True)

    def commit(directory):
        subprocess.run(["git", "-C", str(directory), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(directory),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )

    (site / "index.html").write_text("Previous report")
    (site / "exact-records.zip").write_bytes(b"Published historical records")
    (site / "historical/v3").mkdir(parents=True)
    (site / "historical/v3/index.html").write_text("Earlier report")
    commit(site)
    for name in module.ARTIFACTS:
        (source / name).write_text("New artifact")
    evidence = {"complete": False, "scientific_commit": "frozen", "dataset_fingerprint": "panel"}
    write_json(source / "evidence.json", evidence)
    commit(source)
    with pytest.raises(ValueError, match="incomplete"):
        module.prepare(source, site, "v4")
    assert module.git(site, "status", "--porcelain") == ""
    if complete:
        evidence["complete"] = True
        write_json(source / "evidence.json", evidence)
        commit(source)
    module.prepare(source, site, "v4", allow_incomplete=not complete, snapshot_name="current-results")
    assert (site / "historical/v4/index.html").read_text() == "Previous report"
    assert (site / "exact-records.zip").read_bytes() == b"Published historical records"
    assert (site / "historical/v4/exact-records.zip").read_bytes() == b"Published historical records"
    assert (site / "historical/v3/index.html").read_text() == "Earlier report"
    for name in module.ARTIFACTS:
        assert (site / name).read_bytes() == (source / name).read_bytes()
        assert (site / "reports/current-results" / name).read_bytes() == (source / name).read_bytes()
    assert read_json(site / "publication.json")["source_commit"] == module.git(source, "rev-parse", "HEAD")
    assert read_json(site / "publication.json")["complete"] is complete


def test_http_success_with_provider_error_is_not_a_scored_answer(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = script("evaluate_coordinate_report")
    endpoint = {
        "model_id": "test",
        "base_url": "https://example.test/v1",
        "context_window": 1000,
        "native_output_limit": 500,
        "limit_source": "registry",
    }
    render = SimpleNamespace(
        render_id="frozen-question", system_prompt="Rules", user_prompt="Question", question_family="P01"
    )

    def provider(endpoint, path, body, raw_path=None):
        if body["max_tokens"] == 1:
            return {"usage": {"prompt_tokens": 20, "completion_tokens": 1}}
        return {
            "choices": [{"finish_reason": "error", "message": {"content": ""}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30},
        }

    monkeypatch.setattr(module, "api", provider)
    result = module.run_one(endpoint, None, render, tmp_path)
    assert result["status"] == "provider_error"
    assert not list(tmp_path.glob("*/result.json"))
    assert len(list(tmp_path.glob("*/response.json"))) == 1


def test_report_refuses_a_panel_from_an_older_scientific_contract():
    module = script("build_coordinate_report")
    manifest = {
        "prompt_version": module.PROMPT_VERSION,
        "prompt_fingerprint": module.prompt_fingerprint(),
        "chemistry_fingerprint": module.chemistry_fingerprint(),
        "scoring_version": module.SCORING_VERSION,
        "definitions_sha256": module.Definitions.load().content_sha256,
    }
    module.check_contract(manifest)
    manifest["prompt_fingerprint"] = "older-question-wording"
    with pytest.raises(ValueError, match="no longer matches this checkout"):
        module.check_contract(manifest)


def test_native_cap_requires_available_extension_but_completed_answers_are_not_rerun():
    module = script("build_coordinate_report")
    serving = {"extended_context_recovery": {"max_model_len": 1010000}}
    selected = {"capped": {"truncated": True}, "answered": {"truncated": False}}
    attempts = [{"render_id": name, "context_window": 262144} for name in selected]
    assert module.pending_context_recoveries(selected, attempts, serving) == {"capped"}
    attempts.append({"render_id": "capped", "context_window": 1010000})
    assert module.pending_context_recoveries(selected, attempts, serving) == set()
    assert module.pending_context_recoveries(selected, attempts, {}) == set()


def test_public_failure_evidence_distinguishes_provider_clamps_and_omits_billing(tmp_path):
    from pdbthink.util import write_json

    module = script("build_coordinate_report")
    write_json(
        tmp_path / "failure.json",
        {"render_id": "question", "error": "Provider clamped output below requested budget"},
    )
    write_json(tmp_path / "request.json", {"max_tokens": 943718})
    write_json(
        tmp_path / "response.json",
        {
            "choices": [{"finish_reason": "length"}],
            "usage": {"prompt_tokens": 50000, "completion_tokens": 131072, "cost": 123},
        },
    )
    record = module.collection_failure(tmp_path / "failure.json", "api", "glm")
    assert record["category"] == "provider_output_clamp"
    assert record["max_output_tokens"] > record["usage"]["completion_tokens"]
    assert "cost" not in record["usage"]


@pytest.mark.parametrize("provider", ["GMICloud", None])
def test_provider_stop_does_not_hide_the_observed_glm_reasoning_clamp(provider):
    from pdbthink.evaluation.response_limits import provider_clamped_output

    response = {
        "provider": provider,
        "model": "z-ai/glm-5.3",
        "choices": [{"finish_reason": "stop", "message": {"content": "", "reasoning": "Unfinished"}}],
        "usage": {"completion_tokens": 131072, "completion_tokens_details": {"reasoning_tokens": 131072}},
    }
    assert provider_clamped_output(response, 943718)
    response["choices"][0]["message"]["content"] = "FINAL: A"
    assert not provider_clamped_output(response, 943718)
