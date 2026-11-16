"""Run the frozen report panel with explicit tool controls and full endpoint budgets.

Requests, metering probes and raw responses have separate durable records. An
uncertain submission is never silently retried or counted as a model failure.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from pdbthink.dataset import load_dataset
from pdbthink.evaluation.response_limits import provider_clamped_output
from pdbthink.evaluation.response_protocol import answer_content, response_tool_events
from pdbthink.evaluation.score import _scoring_parameters
from pdbthink.evaluation.tool_policy import require_tool_free
from pdbthink.schemas import RenderedVariant, SemanticInstance
from pdbthink.scoring import score_response
from pdbthink.taskgen.protocol import request_payload, tool_events
from pdbthink.util import derive_seed, read_json, sha256_bytes, stable_hash, write_json


@lru_cache
def iap_credentials(cluster: str):
    from rigging.credentials import iap_provider_for

    return iap_provider_for(cluster)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def api(endpoint: dict, path: str, body: dict | None = None, raw_path: Path | None = None) -> dict:
    if path == "/chat/completions":
        require_tool_free(body, "openai_chat", endpoint["base_url"])
    headers = {"Content-Type": "application/json", "User-Agent": "pdbthink/0.1"}
    iap = endpoint.get("iap_cluster")
    if not iap and (key := os.environ.get(endpoint.get("api_key_env", "OPENROUTER_API_KEY"))):
        headers["Authorization"] = "Bearer " + key
    headers.update(endpoint.get("headers", {}))
    if iap:
        if urllib.parse.urlsplit(endpoint["base_url"]).hostname != "iris.oa.dev":
            raise ValueError("Marin IAP credentials may only be sent to the Iris proxy")
        if any(name.lower() == "authorization" for name in headers):
            raise ValueError("Marin IAP requests must not supply an Authorization header")
        headers["Proxy-Authorization"] = "Bearer " + iap_credentials(iap).get_token()
    request = urllib.request.Request(
        endpoint["base_url"] + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
    )
    # An expired IAP credential must never be forwarded to an external login redirect.
    open_request = urllib.request.build_opener(NoRedirect).open if iap else urllib.request.urlopen
    with open_request(request, timeout=endpoint.get("read_timeout", 172800)) as response:
        if body and body.get("stream"):
            result = {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "", "reasoning": ""},
                        "finish_reason": None,
                    }
                ]
            }
            done = False
            fragments = {}
            with raw_path.open("wb") as log:
                for line in response:
                    log.write(line)
                    log.flush()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise ValueError(str(event["error"]))
                    if tool_events(event):
                        raise ValueError("Tool event in streamed response; quarantined")
                    if event.get("usage"):
                        result["usage"] = event["usage"]
                    for key in ("id", "model", "provider", "created"):
                        if key in event:
                            result[key] = event[key]
                    for choice in event.get("choices", []):
                        if choice.get("index", 0) != 0:
                            raise ValueError("Unexpected additional completion")
                        delta = choice.get("delta", {})
                        for key in ("content", "reasoning", "reasoning_content", "refusal"):
                            if delta.get(key):
                                fragments.setdefault(key, []).append(delta[key])
                        if choice.get("finish_reason"):
                            result["choices"][0]["finish_reason"] = choice["finish_reason"]
                if not done or not result["choices"][0]["finish_reason"] or "usage" not in result:
                    raise ValueError("Incomplete provider stream; partial response retained")
            result["choices"][0]["message"].update({key: "".join(parts) for key, parts in fragments.items()})
            return result
        data = response.read()
        if raw_path is not None:
            raw_path.write_bytes(data)
        return json.loads(data)


def frozen(path: Path, data: dict) -> None:
    if path.exists():
        if read_json(path) != data:
            raise ValueError(f"Frozen request changed: {path}")
    else:
        write_json(path, data)


def run_one(endpoint: dict, instance: SemanticInstance, render: RenderedVariant, root: Path) -> dict:
    folder = root / stable_hash(render.render_id)[:16]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        return read_json(folder / "result.json")
    messages = [
        {"role": "system", "content": render.system_prompt},
        {"role": "user", "content": render.user_prompt},
    ]
    extra = dict(endpoint.get("extra", {}))
    if "seed_base" in endpoint:
        extra["seed"] = derive_seed(endpoint["seed_base"], render.render_id)
    # Meter exactly the same messages and template controls, separately from evaluation.
    probe = {
        "model": endpoint["model_id"],
        "messages": messages,
        "max_tokens": 1,
        **extra,
        "tools": [],
        "tool_choice": "none",
        "plugins": [{"id": "web", "enabled": False}],
    }
    if endpoint.get("tools_null"):
        probe["tools"] = None
    frozen(folder / "meter-request.json", probe)
    try:
        if endpoint.get("tokenize"):
            token_request = {
                "model": endpoint["model_id"],
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": extra.get("chat_template_kwargs", {}),
                "tools": None if endpoint.get("tools_null") else [],
                "tool_choice": "none",
            }
            frozen(folder / "tokenize-request.json", token_request)
            if not (folder / "tokenize-response.json").exists():
                token_endpoint = {**endpoint, "base_url": endpoint["base_url"].removesuffix("/v1")}
                write_json(folder / "tokenize-response.json", api(token_endpoint, "/tokenize", token_request))
            tokens = read_json(folder / "tokenize-response.json")["count"]
        else:
            if not (folder / "meter-response.json").exists():
                write_json(
                    folder / "meter-response.json",
                    api(endpoint, "/chat/completions", probe, folder / "meter-http-body.txt"),
                )
            metered = read_json(folder / "meter-response.json")
            if tool_events(metered):
                raise ValueError("Tool event during metering")
            tokens = metered["usage"]["prompt_tokens"]
        budget = {
            "input_tokens": tokens,
            "context_window": endpoint["context_window"],
            "native_output_limit": endpoint["native_output_limit"],
            "tokenizer_revision": endpoint.get("tokenizer_revision", "provider metering of exact request"),
            "endpoint_limit_source": endpoint["limit_source"],
        }
        body = request_payload(endpoint["model_id"], messages, budget, extra=extra)
        if endpoint.get("tools_null"):
            body["tools"] = None
        if endpoint.get("stream"):
            body.update(stream=True, stream_options={"include_usage": True})
        frozen(folder / "budget.json", budget)
        frozen(folder / "request.json", body)
        if not (folder / "response.json").exists():
            if (folder / "submission-intent.json").exists():
                raise ValueError("Unreconciled previous submission; inspect before retrying")
            write_json(
                folder / "submission-intent.json",
                {"request_sha256": stable_hash(body), "started_at": time.time()},
            )
            raw = api(endpoint, "/chat/completions", body, folder / "response-http-body.txt")
            write_json(folder / "response.json", raw)
        raw = read_json(folder / "response.json")
        protocol = endpoint.get("response_protocol", "plain")
        if response_tool_events(raw, protocol):
            raise ValueError("Provider returned tool events; quarantined")
        if raw.get("error"):
            raise ValueError(str(raw["error"]))
        if len(raw["choices"]) != 1:
            raise ValueError("Expected exactly one completion")
        choice = raw["choices"][0]
        if choice.get("finish_reason") not in ("stop", "length", "content_filter"):
            raise ValueError("Unsupported finish reason; response requires investigation")
        message = choice["message"]
        answer = answer_content(message, protocol)
        usage = raw["usage"]
        if usage["prompt_tokens"] != tokens:
            raise ValueError("Input token count changed between metering and evaluation")
        truncated = choice.get("finish_reason") == "length"
        if provider_clamped_output(raw, body["max_tokens"]):
            raise ValueError("Provider clamped output below requested budget; investigation required")
        result = {
            "render_id": render.render_id,
            "family": render.question_family,
            "status": "returned",
            "model_id": endpoint["model_id"],
            "request_sha256": stable_hash(body),
            "response_sha256": sha256_bytes((folder / "response.json").read_bytes()),
            "usage": usage,
            "max_output_tokens": body["max_tokens"],
            "finish_reason": choice.get("finish_reason"),
            "tool_events": [],
            "response_protocol": protocol,
            "answer_content_sha256": sha256_bytes(answer.encode()),
            "budget_limit": (
                "native_output" if body["max_tokens"] == endpoint["native_output_limit"] else "context"
            ),
            **score_response(
                answer,
                render.answer_schema,
                render.gold_answer,
                parameters=_scoring_parameters(instance),
                truncated=truncated,
                provider_refusal=bool(message.get("refusal"))
                or choice.get("finish_reason") == "content_filter",
            ),
        }
        write_json(folder / "result.json", result)
        return result
    except Exception as exc:
        error = {
            "render_id": render.render_id,
            "family": render.question_family,
            "status": "provider_error",
            "error": str(exc),
        }
        if isinstance(exc, urllib.error.HTTPError):
            error["http_status"] = exc.code
            error["provider_body"] = exc.read().decode(errors="replace")
        write_json(folder / "failure.json", error)
        return error


def run(dataset: Path, config: dict, root: Path, workers: int, render_ids: set[str] | None = None) -> None:
    instances, renders = load_dataset(dataset)
    if render_ids is not None:
        unknown = render_ids - {r.render_id for r in renders}
        if unknown:
            raise ValueError(f"Unknown render IDs: {sorted(unknown)}")
        renders = [r for r in renders if r.render_id in render_ids]
    by_id = {i.semantic_instance_id: i for i in instances}
    for name, endpoint in config.items():
        frozen(root / name / "endpoint.json", endpoint)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(run_one, endpoint, by_id[r.semantic_instance_id], r, root / name): (name, r)
            for r in renders
            for name, endpoint in config.items()
        }
        for future in as_completed(pending):
            name, render = pending[future]
            result = future.result()
            print(
                json.dumps(
                    {
                        "model": name,
                        "render": render.render_id,
                        "status": result["status"],
                        "correct": result.get("score", {}).get("correct"),
                        "tokens": result.get("usage", {}).get("completion_tokens"),
                        "error": result.get("error"),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--render-ids", type=Path, help="JSON list for a recorded recovery wave")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.models:
        config = {name: config[name] for name in args.models}
    render_ids = set(read_json(args.render_ids)) if args.render_ids else None
    run(args.dataset, config, args.output, args.workers, render_ids)
