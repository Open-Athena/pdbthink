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
@pytest.mark.parametrize("comparison", [True, False])
def test_publication_preserves_previous_report_and_requires_explicit_partial_publication(
    tmp_path, complete, comparison,
):
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
    if comparison:
        (source / "standardized").mkdir()
        for name in module.ARTIFACTS:
            (source / "standardized" / name).write_text("Standardized artifact")
        write_json(source / "standardized/evidence.json", {
            "complete": True, "dataset_fingerprint": "panel", "metrics": [{"model": "standardized-qwen"}],
        })
        evidence["standardized_comparison"] = {
            "path": "standardized/", "complete": True, "metrics": [{"model": "standardized-qwen"}],
        }
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
    for name in module.artifact_paths(source):
        assert (site / name).read_bytes() == (source / name).read_bytes()
        assert (site / "reports/current-results" / name).read_bytes() == (source / name).read_bytes()
    assert read_json(site / "publication.json")["source_commit"] == module.git(source, "rev-parse", "HEAD")
    assert read_json(site / "publication.json")["complete"] is complete


def test_standardized_conditions_reject_sampling_drift_and_false_exceptions():
    from copy import deepcopy

    from pdbthink.evaluation.standardized_conditions import require_standardized_condition

    models = json.loads((Path(__file__).parents[1] / "configs/report_models_v7.json").read_text())
    for endpoint in models.values():
        require_standardized_condition(endpoint)
    for name, key, value in (
        ("standardized-qwen", "temperature", 1.0),
        ("standardized-kimi", "top_k", 20),
        ("standardized-astra", "temperature", 0.7),
        ("standardized-mochi", "seed", 123),
    ):
        changed = deepcopy(models[name])
        changed["extra"][key] = value
        with pytest.raises(ValueError):
            require_standardized_condition(changed)
    changed = deepcopy(models["standardized-deepseek"])
    changed["sampling_exception"] = "Silently omitted temperature"
    with pytest.raises(ValueError, match="Only Astra and Opus"):
        require_standardized_condition(changed)


def test_standardized_audit_rejects_a_hidden_sampling_control():
    module = script("build_coordinate_report")
    endpoint = {"model_id": "astra", "benchmark_condition": "standardized-v1", "extra": {}}
    module.audit_model_condition({"model": "astra"}, endpoint, "question")
    with pytest.raises(ValueError, match="temperature"):
        module.audit_model_condition({"model": "astra", "temperature": 0.7}, endpoint, "question")


def test_nemotron_reasoning_budget_requires_a_new_condition_and_is_audited():
    from copy import deepcopy

    from pdbthink.evaluation.standardized_conditions import (
        reasoning_controls,
        require_standardized_condition,
    )

    models = json.loads((Path(__file__).parents[1] / "configs/report_models_v8.json").read_text())
    for endpoint in models.values():
        require_standardized_condition(endpoint)
    endpoint = models["standardized-nemotron"]
    request = {"model": endpoint["model_id"], "max_tokens": 180502, **endpoint["extra"],
               **reasoning_controls(endpoint, 180502)}
    module = script("build_coordinate_report")
    module.audit_model_condition(request, endpoint, "question")
    assert request["reasoning"] == {"max_tokens": 180501}
    request["reasoning"] = {"max_tokens": 65536}
    with pytest.raises(ValueError, match="reasoning"):
        module.audit_model_condition(request, endpoint, "question")
    changed = deepcopy(endpoint)
    changed.pop("run_namespace")
    with pytest.raises(ValueError, match="versioned Nemotron"):
        require_standardized_condition(changed)
    changed = deepcopy(endpoint)
    changed["extra"]["reasoning_effort"] = "high"
    with pytest.raises(ValueError, match="versioned Nemotron"):
        require_standardized_condition(changed)


@pytest.mark.parametrize("prompt_tokens", [15000, 100000])
def test_nemotron_request_uses_the_full_reasoning_and_output_allowances(tmp_path, monkeypatch, prompt_tokens):
    from types import SimpleNamespace

    module = script("evaluate_coordinate_report")
    endpoint = json.loads((Path(__file__).parents[1] / "configs/report_models_v8.json").read_text())[
        "standardized-nemotron"
    ]
    render = SimpleNamespace(render_id="question", system_prompt="Rules", user_prompt="Question",
                             question_family="P01")
    requests = []

    def provider(endpoint, path, body, raw_path=None):
        requests.append(body)
        if body["max_tokens"] == 1:
            assert body["reasoning"] == {"enabled": True}
            return {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1}}
        allowance = min(235929, 262144 - prompt_tokens)
        assert body["max_tokens"] == allowance
        assert body["reasoning"] == {"max_tokens": allowance - 1}
        assert "reasoning_effort" not in body
        assert body["tools"] == [] and body["tool_choice"] == "none"
        assert body["temperature"] == 0.7
        raise ValueError("Stopped after request audit")

    monkeypatch.setattr(module, "api", provider)
    result = module.run_one(endpoint, None, render, tmp_path)
    assert len(requests) == 2
    assert result["error"] == "Stopped after request audit"


def test_standardized_retries_only_failed_transport_not_wrong_answers(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pdbthink.util import stable_hash, write_json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = script("run_standardized_condition")
    renders = [SimpleNamespace(render_id=name) for name in ("wrong-answer", "interrupted")]
    monkeypatch.setattr(module, "load_dataset", lambda path: ([], renders))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    calls = []

    def run(dataset, config, root, workers, ids):
        calls.append(ids)
        for render_id in ids:
            folder = root / "standardized-deepseek" / stable_hash(render_id)[:16]
            write_json(folder / "submission-intent.json", {})
            if render_id == "interrupted" and len(calls) == 1:
                write_json(folder / "failure.json", {"error": "Incomplete provider stream"})
            else:
                write_json(folder / "result.json", {
                    "render_id": render_id, "score": {"correct": False}, "truncated": False,
                })

    monkeypatch.setattr(module, "run", run)
    config = Path(__file__).parents[1] / "configs/report_models_v7.json"
    module.collect_condition(tmp_path, config, "standardized-deepseek", tmp_path, 2)
    assert calls == [{"wrong-answer", "interrupted"}, {"interrupted"}]


def test_nemotron_new_namespace_does_not_reuse_earlier_completed_answers(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pdbthink.util import stable_hash, write_json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = script("run_standardized_condition")
    monkeypatch.setattr(module, "load_dataset", lambda path: ([], [SimpleNamespace(render_id="question")]))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    old = tmp_path / "standardized_v1/standardized-nemotron" / stable_hash("question")[:16]
    write_json(old / "result.json", {"render_id": "question", "score": {"correct": True}})
    calls = []

    def run(dataset, config, root, workers, ids):
        calls.append(ids)
        assert root == tmp_path / "nemotron-budget-v2/standardized_v1"
        write_json(root / "standardized-nemotron" / stable_hash("question")[:16] / "result.json",
                   {"render_id": "question", "score": {"correct": False}, "truncated": False})

    monkeypatch.setattr(module, "run", run)
    config = Path(__file__).parents[1] / "configs/report_models_v8.json"
    module.collect_condition(tmp_path, config, "standardized-nemotron", tmp_path, 1)
    assert calls == [{"question"}]
    assert json.loads((old / "result.json").read_text())["score"]["correct"] is True


def test_standardized_will_not_retry_an_unreconciled_submission(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pdbthink.util import stable_hash, write_json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = script("run_standardized_condition")
    monkeypatch.setattr(module, "load_dataset", lambda path: ([], [SimpleNamespace(render_id="unknown")]))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(module, "run", lambda *args: pytest.fail("Uncertain request was resubmitted"))
    write_json(tmp_path / "standardized_v1/standardized-deepseek" / stable_hash("unknown")[:16]
               / "submission-intent.json", {})
    config = Path(__file__).parents[1] / "configs/report_models_v7.json"
    with pytest.raises(ValueError, match="Unreconciled"):
        module.collect_condition(tmp_path, config, "standardized-deepseek", tmp_path, 1)


def test_hosted_recovery_keeps_wrong_answers_and_requires_capacity_review(tmp_path, monkeypatch):
    from pdbthink.util import write_json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = script("recover_standardized_transports")
    root = tmp_path / "standardized_v1/standardized-nemotron"
    write_json(root / "wrong/result.json", {
        "render_id": "wrong", "score": {"correct": False},
    })
    write_json(root / "rejected/failure.json", {
        "render_id": "rejected", "http_status": 402, "error": "Payment Required",
    })
    ids = {"wrong", "rejected", "unstarted"}
    assert module.missing_requests(tmp_path, "standardized-nemotron", ids) == {"rejected", "unstarted"}
    write_json(root / "clamped/failure.json", {
        "render_id": "clamped", "error": "Provider clamped output below requested budget",
    })
    with pytest.raises(ValueError, match="separate review"):
        module.missing_requests(tmp_path, "standardized-nemotron", ids | {"clamped"})


def test_hosted_recovery_rejects_unreconciled_inflight_request(tmp_path, monkeypatch):
    from pdbthink.util import write_json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = script("recover_standardized_transports")
    write_json(tmp_path / "standardized_v1/standardized-astra/inflight/submission-intent.json", {})
    with pytest.raises(ValueError, match="Unreconciled"):
        module.missing_requests(tmp_path, "standardized-astra", {"inflight"})


def test_hosted_recovery_honours_credit_reservation_retry_after(monkeypatch):
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = script("recover_standardized_transports")
    result = {
        "status": "provider_error", "http_status": 402,
        "provider_body": json.dumps({"error": {"metadata": {"headers": {"Retry-After": "240"}}}}),
    }
    assert module.retry_delay(result) == 240
    assert module.retry_delay({"status": "returned"}) == 0
    assert module.retry_delay({"status": "provider_error", "http_status": 429}) == 120


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


def test_cost_comparison_requires_full_coverage_and_never_imputes_missing_cost():
    module = script("build_coordinate_report")
    selected = {
        "wrong": {"usage": {"cost": 0.2}, "score": {"correct": False}},
        "truncated": {"usage": {"cost": 0.4}, "truncated": True},
    }
    costs = module.cost_metrics(selected, 2, set())
    assert costs["panel_response_cost_usd"] == pytest.approx(0.6)
    assert costs["mean_response_cost_usd"] == pytest.approx(0.3)
    assert module.cost_metrics(selected, 3, set())["mean_response_cost_usd"] is None
    assert module.cost_metrics(selected, 2, {"truncated"})["mean_response_cost_usd"] is None
    selected["truncated"]["usage"] = {}
    costs = module.cost_metrics(selected, 2, set())
    assert costs["costed_responses"] == 1
    assert costs["known_response_cost_usd"] == 0.2
    assert costs["panel_response_cost_usd"] is None
    assert costs["mean_response_cost_usd"] is None


def test_cost_accounting_distinguishes_free_unmetered_and_byok_responses():
    module = script("build_coordinate_report")
    assert module.response_cost_usd({"cost": 0}) == 0
    assert module.response_cost_usd({}) is None
    assert module.response_cost_usd({"cost": 0.05, "is_byok": True}) is None
    for value in (True, "0.1", -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="Invalid recorded response cost"):
            module.response_cost_usd({"cost": value})


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
