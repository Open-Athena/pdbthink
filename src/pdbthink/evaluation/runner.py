"""Resumable model evaluation (specification section 6, deliverable 6).

Supported providers:

``openai_chat``
    Any OpenAI-compatible ``/v1/chat/completions`` endpoint, which covers vLLM,
    llama.cpp servers, OpenAI itself and most gateways.
``anthropic_messages``
    The Anthropic Messages API.
``ollama_chat``
    Ollama's native ``/api/chat``. Ollama also exposes an OpenAI-compatible
    endpoint, but that shim silently drops the ``think`` parameter, so a hybrid
    reasoning model spends its whole output budget on a trace and returns empty
    content. The native endpoint honours it.
``mock``
    A deterministic offline provider used by the tests and by the no-model
    validation step of the cost-controlled workflow.

Only the standard library is used for transport, so evaluation has no extra
dependencies and works behind a plain HTTP proxy.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import ConfigError
from ..dataset import load_dataset
from ..schemas import EvaluationResult, RenderedVariant
from ..util import append_jsonl, gold_hash, read_jsonl, sha256_text, stable_hash, write_json
from .cache import (
    CachedResponse,
    CacheKey,
    ResponseCache,
    extract_reasoning,
    openai_response_error,
)

DEFAULT_TIMEOUT = 900
OUTPUT_TOKEN_PARAMETERS = ("max_tokens", "max_completion_tokens")
THINKING_MODES = ("manual", "adaptive")
PROVIDERS = ("mock", "openai_chat", "anthropic_messages", "ollama_chat")
#: Some providers sit behind a WAF that rejects the default `Python-urllib`
#: agent outright (Together returns Cloudflare error 1010), so identify
#: ourselves properly on every request.
USER_AGENT = "pdbthink/0.1 (structural reasoning benchmark)"


@dataclass
class ModelConfig:
    """A versioned description of one model under evaluation."""

    model_id: str
    provider: str = "openai_chat"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model_revision: str | None = None
    reasoning_effort: str | None = None
    thinking_mode: str | None = None
    max_output_tokens: int = 8192
    output_token_parameter: str = "max_tokens"
    temperature: float | None = 0.0
    top_p: float | None = None
    completions: int = 1
    concurrency: int = 4
    request_timeout: int = DEFAULT_TIMEOUT
    max_retries: int = 4
    extra_body: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("model_id", "provider", "base_url", "api_key_env"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}, got {self.provider!r}")
        for name in ("model_revision", "reasoning_effort", "thinking_mode"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null")
        if not isinstance(self.label, str):
            raise ValueError("label must be a string")
        if self.output_token_parameter not in OUTPUT_TOKEN_PARAMETERS:
            raise ValueError(
                f"output_token_parameter must be one of {OUTPUT_TOKEN_PARAMETERS}, "
                f"got {self.output_token_parameter!r}"
            )
        if self.thinking_mode is not None and self.thinking_mode not in THINKING_MODES:
            raise ValueError(
                f"thinking_mode must be one of {THINKING_MODES}, got {self.thinking_mode!r}"
            )
        if self.thinking_mode is not None and self.provider != "anthropic_messages":
            raise ValueError("thinking_mode is only valid for provider: anthropic_messages")
        for name in (
            "max_output_tokens",
            "completions",
            "concurrency",
            "request_timeout",
            "max_retries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1")
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number or null")
        if not isinstance(self.extra_body, dict) or not all(
            isinstance(key, str) for key in self.extra_body
        ):
            raise ValueError("extra_body must be a mapping with string keys")

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        try:
            loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            raw = {} if loaded is None else loaded
            if not isinstance(raw, dict):
                raise ValueError("top level must be a mapping")
            known = set(cls.__dataclass_fields__)
            unknown = set(raw) - known
            if unknown:
                raise ValueError(f"unknown keys {sorted(unknown)}")
            return cls(**raw)
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            raise ConfigError(f"invalid model config {path}: {exc}") from exc

    @property
    def sampling_parameters(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "max_output_tokens": self.max_output_tokens,
        }
        if self.provider == "openai_chat":
            out["output_token_parameter"] = self.output_token_parameter
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.top_p is not None:
            out["top_p"] = self.top_p
        if self.reasoning_effort:
            out["reasoning_effort"] = self.reasoning_effort
        if self.thinking_mode:
            out["thinking_mode"] = self.thinking_mode
        out.update(self.extra_body)
        return out

    def sampling_parameters_for(self, completion_index: int) -> dict[str, Any]:
        """Parameters determining one request, independent of total repeat count."""
        out = dict(self.sampling_parameters)
        seed = self.completion_seed(completion_index)
        if seed is not None and self.provider == "openai_chat":
            out.setdefault("seed", seed)
        elif (
            seed is not None
            and self.provider == "ollama_chat"
            and "options" not in self.extra_body
        ):
            out["ollama_options_seed"] = seed
        return out

    @property
    def endpoint_identity(self) -> str:
        """The non-secret provider endpoint used for run and cache identity."""
        return self.base_url.rstrip("/")

    def completion_seed(self, completion_index: int) -> int | None:
        """The deterministic seed added by providers that accept one."""
        supports_seed = self.provider in ("openai_chat", "ollama_chat")
        return 1000 + completion_index if supports_seed and self.completions > 1 else None

    def generated_seed_is_sent(self, completion_index: int) -> bool:
        if self.completion_seed(completion_index) is None:
            return False
        if self.provider == "openai_chat":
            return "seed" not in self.extra_body
        if self.provider == "ollama_chat":
            return "options" not in self.extra_body
        return False

    def run_id(self, dataset_fingerprint: str) -> str:
        digest = stable_hash(
            self.model_id,
            self.provider,
            self.endpoint_identity,
            self.model_revision,
            self.reasoning_effort,
            self.completions,
            self.sampling_parameters,
            dataset_fingerprint,
        )[:10]
        base = self.label or self.model_id.replace("/", "_")
        return f"{base}-{digest}"


class ProviderError(RuntimeError):
    pass


class ResumeError(RuntimeError):
    """An output directory cannot safely be resumed for this model run."""


def dataset_fingerprint(renders: Iterable[RenderedVariant]) -> str:
    """Hash the exact prompts and labels that define an evaluation dataset."""
    rows = [
        (
            render.render_id,
            render.prompt_version,
            sha256_text(render.system_prompt),
            sha256_text(render.user_prompt),
            render.answer_schema,
            render.displayed_coordinates_sha256,
            gold_hash(render.gold_answer),
        )
        for render in sorted(renders, key=lambda item: item.render_id)
    ]
    return stable_hash(rows)


class EvaluationRunner:
    """Runs every rendered variant, appending results as they complete."""

    def __init__(
        self,
        dataset_dir: str | Path,
        model: ModelConfig,
        output_dir: str | Path,
        *,
        resume: bool = False,
        cache: ResponseCache | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.model = model
        self.output_dir = Path(output_dir)
        self.resume = resume
        # Shared across runs and across dataset rebuilds: keyed on the prompt
        # text, not on the render identifier, so questions can come and go. The
        # mock providers are free and deterministic, so caching them would spend
        # disk to avoid no cost at all.
        self.cache = cache if cache is not None else ResponseCache()
        if self.model.provider == "mock":
            self.cache.enabled = False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "results.jsonl"
        self.dataset_fingerprint = ""
        self.run_id = ""

    def run(
        self,
        *,
        limit: int | None = None,
        families: Iterable[str] | None = None,
        max_input_tokens: int | None = None,
    ) -> dict[str, Any]:
        instances, renders = load_dataset(self.dataset_dir)
        by_instance = {i.semantic_instance_id: i for i in instances}
        accepted = {
            i.semantic_instance_id for i in instances if i.curation_status != "rejected"
        }
        renders = [r for r in renders if r.semantic_instance_id in accepted]
        self.dataset_fingerprint = dataset_fingerprint(renders)
        self.run_id = self.model.run_id(self.dataset_fingerprint)
        if families:
            wanted = set(families)
            renders = [r for r in renders if r.question_family in wanted]
        # A model with a short context window cannot ingest the longer prompts at
        # all. Skipping them explicitly keeps the run honest: the count is stored
        # in the run configuration rather than surfacing as opaque API errors, and
        # the report shows which families lost coverage.
        skipped_too_long = []
        if max_input_tokens is not None:
            keep = []
            for render in renders:
                if (render.input_token_count or 0) > max_input_tokens:
                    skipped_too_long.append(render.render_id)
                else:
                    keep.append(render)
            renders = keep
        renders.sort(key=lambda r: r.render_id)
        if limit:
            renders = renders[:limit]

        selected_keys = {
            (render.render_id, index)
            for render in renders
            for index in range(self.model.completions)
        }
        done = self._completed_keys(selected_keys) if self.resume else set()
        if not self.resume and self.results_path.exists():
            self.results_path.unlink()

        jobs = [
            (render, index)
            for render in renders
            for index in range(self.model.completions)
            if (render.render_id, index) not in done
        ]

        write_json(
            self.output_dir / "run_config.json",
            {
                "run_id": self.run_id,
                "dataset_dir": str(self.dataset_dir.resolve()),
                "dataset_fingerprint": self.dataset_fingerprint,
                "model": self.model.__dict__,
                "sampling_parameters": self.model.sampling_parameters,
                "n_renders": len(renders),
                "n_jobs": len(jobs),
                "n_reused": len(done),
                "max_input_tokens": max_input_tokens,
                "response_cache": self.cache.statistics(),
                "n_skipped_over_input_limit": len(skipped_too_long),
                "skipped_over_input_limit": skipped_too_long,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )

        errors = 0
        cache_errors = 0
        completed = 0
        if jobs:
            with ThreadPoolExecutor(max_workers=max(1, self.model.concurrency)) as pool:
                for result in pool.map(lambda job: self._one(*job), jobs):
                    append_jsonl(self.results_path, result.model_dump())
                    completed += int(not result.error)
                    errors += int(bool(result.error))
                    cache_errors += int(bool(result.cache_error))
        return {
            "run_id": self.run_id,
            "completed": completed,
            "attempted": completed + errors,
            "skipped": len(done),
            "errors": errors,
            "cache_errors": cache_errors,
            "cache": self.cache.statistics(),
            "n_renders": len(renders),
            "skipped_over_input_limit": len(skipped_too_long),
            "output_dir": str(self.output_dir),
            "n_instances": len({r.semantic_instance_id for r in renders}),
            "unused_instances": len(by_instance) - len({r.semantic_instance_id for r in renders}),
        }

    # ------------------------------------------------------------------ #
    def _completed_keys(
        self, selected_keys: set[tuple[str, int]]
    ) -> set[tuple[str, int]]:
        keys: set[tuple[str, int]] = set()
        existing_keys: set[tuple[str, int]] = set()
        incompatible: set[str] = set()
        for row in read_jsonl(self.results_path):
            row_run_id = row.get("run_id")
            if row_run_id != self.run_id:
                incompatible.add(str(row_run_id or "<missing>"))
                continue
            key = (row["render_id"], int(row["completion_index"]))
            existing_keys.add(key)
            if not row.get("error"):
                keys.add(key)
        if incompatible:
            found = ", ".join(sorted(incompatible))
            raise ResumeError(
                f"cannot resume {self.run_id!r} in {self.output_dir}: existing results "
                f"belong to {found}. Use a separate --output directory for each "
                "model configuration."
            )
        outside = existing_keys - selected_keys
        if outside:
            raise ResumeError(
                f"cannot resume {self.run_id!r} with a narrower or disjoint selection: "
                f"{len(outside)} existing completion(s) are outside this invocation. "
                "Use the same or a broader selection, or a separate --output directory."
            )
        return keys

    def _cache_key(self, render: RenderedVariant, completion_index: int) -> CacheKey:
        return CacheKey(
            provider=self.model.provider,
            endpoint=self.model.endpoint_identity,
            model_id=self.model.model_id,
            model_revision=self.model.model_revision,
            reasoning_effort=self.model.reasoning_effort,
            max_output_tokens=self.model.max_output_tokens,
            sampling_parameters=self.model.sampling_parameters_for(completion_index),
            system_prompt=render.system_prompt,
            user_prompt=render.user_prompt,
            completion_index=completion_index,
            legacy_v2_completions=self.model.completions,
            legacy_v2_generated_seed=self.model.generated_seed_is_sent(completion_index),
        )

    def _one(self, render: RenderedVariant, completion_index: int) -> EvaluationResult:
        started = time.time()
        error: str | None = None
        cache_error: str | None = None
        key = self._cache_key(render, completion_index)
        response = CachedResponse(text="")
        from_cache = False
        try:
            with self.cache.request_lock(key):
                cached = self.cache.get(key)
                from_cache = cached is not None
                if cached is not None:
                    response = cached
                else:
                    response = call_model(
                        self.model,
                        render.system_prompt,
                        render.user_prompt,
                        completion_index,
                        render=render,
                    )
                    try:
                        self.cache.put(
                            key,
                            response,
                            provenance={
                                "render_id": render.render_id,
                                "semantic_instance_id": render.semantic_instance_id,
                                "question_family": render.question_family,
                                "protein_group_id": render.protein_group_id,
                                "representation": render.representation,
                                "input_token_count": render.input_token_count,
                                "dataset_dir": str(self.dataset_dir.resolve()),
                                "run_id": self.run_id,
                            },
                        )
                    except Exception as exc:  # noqa: BLE001 - response is still usable
                        cache_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            error = f"{type(exc).__name__}: {exc}"
        text, usage, truncated = response.text, response.usage, response.truncated
        return EvaluationResult(
            run_id=self.run_id,
            render_id=render.render_id,
            completion_index=completion_index,
            model_provider=self.model.provider,
            model_id=self.model.model_id,
            model_revision=self.model.model_revision,
            reasoning_effort=self.model.reasoning_effort,
            sampling_parameters=self.model.sampling_parameters_for(completion_index),
            max_output_tokens=self.model.max_output_tokens,
            raw_response=text,
            usage=usage,
            truncated=truncated,
            latency_seconds=round(time.time() - started, 3),
            error=error,
            cache_error=cache_error,
            cache_key=key.digest,
            from_cache=from_cache,
            reasoning_characters=len(response.reasoning),
        )


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #

def call_model(
    model: ModelConfig,
    system_prompt: str,
    user_prompt: str,
    completion_index: int,
    *,
    render: RenderedVariant | None = None,
) -> CachedResponse:
    """Call a provider and return the full response, reasoning trace included."""
    if model.provider == "mock":
        return _mock(model, system_prompt, user_prompt, completion_index, render)
    if model.provider == "openai_chat":
        return _openai_chat(model, system_prompt, user_prompt, completion_index)
    if model.provider == "anthropic_messages":
        return _anthropic(model, system_prompt, user_prompt, completion_index)
    if model.provider == "ollama_chat":
        return _ollama(model, system_prompt, user_prompt, completion_index)
    raise ProviderError(f"unknown provider {model.provider!r}")


def _post(
    url: str, payload: dict[str, Any], headers: dict[str, str], model: ModelConfig
) -> Any:
    body = json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **headers}
    last: Exception | None = None
    for attempt in range(model.max_retries):
        retry_after: float | None = None
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=model.request_timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (400, 401, 402, 403, 404, 413, 422):
                raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
            last = ProviderError(f"HTTP {exc.code}: {detail}")
            value = exc.headers.get("Retry-After") if exc.headers else None
            if value:
                try:
                    retry_after = min(300.0, max(0.0, float(value)))
                except ValueError:
                    pass
        except Exception as exc:  # noqa: BLE001 - retried
            last = exc
        if attempt + 1 < model.max_retries:
            time.sleep(retry_after if retry_after is not None else min(30.0, 2.0 ** attempt))
    raise ProviderError(str(last))


def _provider_payload(data: Any, provider: str) -> dict[str, Any]:
    """Require a successful JSON object before provider-specific parsing."""
    if not isinstance(data, dict):
        raise ProviderError(f"{provider} response was not an object")
    if "error" in data and data["error"] is not None:
        error = data["error"]
        if isinstance(error, dict):
            error = error.get("message") or error.get("type") or error
        raise ProviderError(f"{provider} error: {error}")
    return data


def _openai_chat(
    model: ModelConfig, system_prompt: str, user_prompt: str, completion_index: int
) -> CachedResponse:
    payload: dict[str, Any] = {
        "model": model.model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model.output_token_parameter: model.max_output_tokens,
    }
    if model.temperature is not None:
        payload["temperature"] = model.temperature
    if model.top_p is not None:
        payload["top_p"] = model.top_p
    if model.reasoning_effort:
        payload["reasoning_effort"] = model.reasoning_effort
    seed = model.completion_seed(completion_index)
    if seed is not None:
        payload["seed"] = seed
    payload.update(model.extra_body)

    headers = {"Content-Type": "application/json"}
    key = os.environ.get(model.api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    data = _post(f"{model.base_url.rstrip('/')}/chat/completions", payload, headers, model)
    api_error = openai_response_error(data)
    if api_error:
        raise ProviderError(api_error)
    choice = data["choices"][0]
    message = choice["message"]
    return CachedResponse(
        text=message.get("content") or message.get("refusal") or "",
        usage=data.get("usage") or {},
        truncated=choice.get("finish_reason") == "length",
        reasoning=extract_reasoning(choice),
        raw=data,
    )


def _anthropic(
    model: ModelConfig, system_prompt: str, user_prompt: str, completion_index: int
) -> CachedResponse:
    payload: dict[str, Any] = {
        "model": model.model_id,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": model.max_output_tokens,
    }
    if model.temperature is not None:
        payload["temperature"] = model.temperature
    if model.reasoning_effort:
        if model.thinking_mode == "adaptive":
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": model.reasoning_effort}
        else:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": model.max_output_tokens // 2,
            }
        payload.pop("temperature", None)
    payload.update(model.extra_body)

    key = os.environ.get(model.api_key_env, "")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    data = _provider_payload(
        _post(f"{model.base_url.rstrip('/')}/messages", payload, headers, model),
        "Anthropic",
    )
    blocks = data.get("content")
    usage = data.get("usage", {})
    if not isinstance(blocks, list):
        raise ProviderError("Anthropic response content was not a list")
    if not isinstance(usage, dict):
        raise ProviderError("Anthropic response usage was not an object")
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise ProviderError("Anthropic response content block was not an object")
        if block.get("type") == "text":
            value = block.get("text")
            if not isinstance(value, str):
                raise ProviderError("Anthropic text block did not contain text")
            text_parts.append(value)
        elif block.get("type") == "thinking":
            value = block.get("thinking")
            if not isinstance(value, str):
                raise ProviderError("Anthropic thinking block did not contain text")
            reasoning_parts.append(value)
    return CachedResponse(
        text="".join(text_parts),
        usage=usage,
        truncated=data.get("stop_reason") == "max_tokens",
        reasoning="".join(reasoning_parts),
        raw=data,
    )


def _ollama(
    model: ModelConfig, system_prompt: str, user_prompt: str, completion_index: int
) -> CachedResponse:
    options: dict[str, Any] = {"num_predict": model.max_output_tokens}
    if model.temperature is not None:
        options["temperature"] = model.temperature
    if model.top_p is not None:
        options["top_p"] = model.top_p
    if model.completions > 1:
        options["seed"] = 1000 + completion_index

    payload: dict[str, Any] = {
        "model": model.model_id,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": options,
    }
    # Default to no reasoning trace: the benchmark scores the FINAL line only, and
    # a small model that reasons for its entire budget never reaches one.
    payload["think"] = bool(model.extra_body.get("think", False))
    payload.update({k: v for k, v in model.extra_body.items() if k != "think"})

    data = _provider_payload(
        _post(
            f"{model.base_url.rstrip('/')}/api/chat",
            payload,
            {"Content-Type": "application/json"},
            model,
        ),
        "Ollama",
    )
    message = data.get("message")
    if not isinstance(message, dict):
        raise ProviderError("Ollama response message was not an object")
    content = message.get("content")
    thinking = message.get("thinking", "")
    if not isinstance(content, str):
        raise ProviderError("Ollama response content was not text")
    if thinking is None:
        thinking = ""
    if not isinstance(thinking, str):
        raise ProviderError("Ollama response thinking was not text")
    usage = {
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
        "total_duration_ns": data.get("total_duration"),
    }
    if any(value is not None and type(value) is not int for value in usage.values()):
        raise ProviderError("Ollama response usage contained a non-integer value")
    return CachedResponse(
        text=content,
        usage=usage,
        truncated=data.get("done_reason") == "length",
        reasoning=thinking,
        raw=data,
    )


def _mock(
    model: ModelConfig,
    system_prompt: str,
    user_prompt: str,
    completion_index: int,
    render: RenderedVariant | None = None,
) -> CachedResponse:
    """Deterministic offline provider.

    Three behaviours, selected through ``extra_body``:

    ``answer_gold: true``
        Reply with the correctly formatted gold answer. A run of this against a
        dataset must score exactly 1.0, which is the no-model validation step of
        the cost-controlled workflow (section 14.1).
    ``mock_answers: {substring: reply}``
        Reply with a fixed answer whenever the prompt contains the substring.
    default
        Echo the format example printed in the prompt, exercising parsing,
        scoring and reporting without a model.
    """
    if model.extra_body.get("answer_gold") and render is not None:
        from ..validate import format_gold_answer

        text = format_gold_answer(render.answer_schema, render.gold_answer)
        return CachedResponse(
            text=f"Mock provider answering from the gold label (completion {completion_index}).\n{text}",
            usage={"prompt_tokens": len(user_prompt) // 4, "completion_tokens": 16},
        )
    answers: dict[str, str] = model.extra_body.get("mock_answers", {})
    for needle, reply in answers.items():
        if needle in user_prompt:
            return CachedResponse(text=reply, usage={"prompt_tokens": len(user_prompt) // 4})
    example = ""
    for line in user_prompt.splitlines():
        if line.strip().startswith("Example: FINAL:"):
            example = line.split("Example:", 1)[1].strip()
    reply = example or "FINAL: unknown"
    return CachedResponse(
        text=f"Reasoning omitted by the mock provider (completion {completion_index}).\n{reply}",
        usage={"prompt_tokens": len(user_prompt) // 4, "completion_tokens": 12},
    )
