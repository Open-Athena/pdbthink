"""Batch inference through OpenRouter, at half the synchronous price.

OpenRouter's batch API is a different shape from Together's and from OpenAI's,
so it gets its own runner rather than being forced through
:class:`~pdbthink.evaluation.batch.BatchRun`, whose recovery machinery is built
around uploading a file and downloading an output file. Here there is neither:

* requests are sent **inline** in the create call, not uploaded;
* ``endpoint`` and ``model`` must appear *before* ``requests`` in the JSON body,
  or the API rejects it -- so the body is assembled in order deliberately;
* ``endpoint`` is ``/v1/chat/completions`` even though the path actually called
  is ``/api/v1/chat/completions``;
* results come back **inline** on the batch object once it completes.

None of that is documented publicly at the time of writing; it was established
by probing the API, and the probe is preserved in the tests.

The contract with the rest of the package is the same one the Together path
keeps: a batch run does exactly one thing, **fill the response cache**, and then
the ordinary evaluator finds every completion already there. Scoring and
reporting never learn that batching happened.

The model recorded in the cache key is the *plain* model id, not the ``:batch``
variant that the wire request carries. Batch is a delivery mechanism, not a
different model, and a batched answer must be interchangeable with a
synchronous one for the same prompt.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..util import read_json, write_json
from .cache import (
    CachedResponse,
    CacheKey,
    ResponseCache,
    extract_reasoning,
    openai_response_error,
)

USER_AGENT = "pdbthink/0.1 (structural reasoning benchmark)"
BATCHES_PATH = "/api/beta/batches"
#: The value the API wants in the `endpoint` field, which is not the path called.
BATCH_ENDPOINT = "/v1/chat/completions"
#: Statuses meaning the provider is finished, successfully or not.
TERMINAL = ("completed", "failed", "expired", "cancelled", "error")
#: Requests travel inline, so a batch is bounded by payload size rather than by
#: an upload limit. Chosen well under any plausible body cap: the benchmark's
#: largest prompt is ~350KB of JSON, so a chunk holds a few dozen of those.
MAX_BATCH_BYTES = 6 * 1024 * 1024
MAX_BATCH_REQUESTS = 200


class OpenRouterBatchError(RuntimeError):
    pass


@dataclass
class OpenRouterBatchJob:
    """One submitted chunk and the mapping back to cache keys."""

    batch_id: str
    n_requests: int
    #: custom_id -> the cache key digest it stands for.
    custom_ids: dict[str, str] = field(default_factory=dict)
    status: str = "submitted"
    fetched: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "n_requests": self.n_requests,
            "custom_ids": self.custom_ids,
            "status": self.status,
            "fetched": self.fetched,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> OpenRouterBatchJob:
        job = cls(**row)
        if not isinstance(job.batch_id, str) or not job.batch_id:
            raise OpenRouterBatchError("batch job id must be a non-empty string")
        if job.n_requests != len(job.custom_ids):
            raise OpenRouterBatchError("n_requests must equal the custom_ids cardinality")
        return job


class OpenRouterClient:
    """The three calls the batch API needs, over urllib."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 300.0) -> None:
        # Model configs point at /api/v1 for chat; the batch API sits beside it.
        self.root = base_url.rstrip("/").removesuffix("/api/v1").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, body: bytes | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.root}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if body else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise OpenRouterBatchError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc

    def create(self, model: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
        # Key order is load-bearing: the API rejects a body whose `requests`
        # array appears before `endpoint` and `model`.
        body = json.dumps(
            {"endpoint": BATCH_ENDPOINT, "model": model, "requests": requests}
        ).encode("utf-8")
        return self._request("POST", BATCHES_PATH, body)

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        return self._request("GET", f"{BATCHES_PATH}/{batch_id}")


def batch_model_id(model_id: str) -> str:
    """The wire model id, which selects half-price batch capacity."""
    return model_id if model_id.endswith(":batch") else f"{model_id}:batch"


class OpenRouterBatchRun:
    """Submit a dataset's uncached prompts, then fold results into the cache."""

    def __init__(
        self,
        model,
        cache: ResponseCache,
        state_dir: str | Path,
        *,
        client: OpenRouterClient | None = None,
    ) -> None:
        self.model = model
        self.cache = cache
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "openrouter_batch_state.json"
        if client is not None:
            self.client = client
        else:
            key = os.environ.get(model.api_key_env, "")
            if not key:
                raise OpenRouterBatchError(f"{model.api_key_env} is not set")
            self.client = OpenRouterClient(model.base_url, key)

    # ------------------------------------------------------------------ #
    def key_for(self, render, completion_index: int) -> CacheKey:
        """The cache key, which names the plain model rather than `:batch`."""
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
        )

    def request_body(self, render) -> dict[str, Any]:
        """One request's body. The model is set on the batch, not per request."""
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": render.system_prompt},
                {"role": "user", "content": render.user_prompt},
            ],
            "max_tokens": self.model.max_output_tokens,
        }
        if self.model.temperature is not None:
            payload["temperature"] = self.model.temperature
        if self.model.top_p is not None:
            payload["top_p"] = self.model.top_p
        if self.model.reasoning_effort:
            payload["reasoning_effort"] = self.model.reasoning_effort
        payload.update(self.model.extra_body)
        return payload

    def pending(self, renders) -> list[tuple[Any, int, CacheKey]]:
        """Renders with no cached completion -- the only ones worth money."""
        out = []
        for render in renders:
            for index in range(self.model.completions):
                key = self.key_for(render, index)
                if self.cache.get(key) is None:
                    out.append((render, index, key))
        return out

    def preflight(self) -> None:
        """Spend one token confirming the model is reachable before submitting.

        A batch is only discovered to be unservable after its completion window,
        so an unusable model has to fail here instead.
        """
        from dataclasses import replace

        from .runner import ProviderError, _openai_chat

        probe = replace(self.model, max_output_tokens=1, reasoning_effort=None, extra_body={})
        try:
            _openai_chat(probe, "", "hi", 0)
        except ProviderError as exc:
            raise OpenRouterBatchError(
                f"{self.model.model_id} is not usable through this account: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    def submit(self, renders) -> list[OpenRouterBatchJob]:
        jobs = self._load_jobs()
        if jobs:
            return jobs
        work = self.pending(renders)
        if not work:
            return []

        chunks: list[list[tuple[Any, int, CacheKey]]] = [[]]
        size = 0
        for item in work:
            encoded = len(json.dumps(self.request_body(item[0])))
            if chunks[-1] and (
                len(chunks[-1]) >= MAX_BATCH_REQUESTS or size + encoded > MAX_BATCH_BYTES
            ):
                chunks.append([])
                size = 0
            chunks[-1].append(item)
            size += encoded

        wire_model = batch_model_id(self.model.model_id)
        for number, chunk in enumerate(chunks):
            custom_ids: dict[str, str] = {}
            requests = []
            for position, (render, _index, key) in enumerate(chunk):
                custom_id = f"r{number:02d}-{position:05d}"
                custom_ids[custom_id] = key.digest
                requests.append({"custom_id": custom_id, "body": self.request_body(render)})
            created = self.client.create(wire_model, requests)
            batch_id = created.get("id")
            if not batch_id:
                raise OpenRouterBatchError(f"batch creation returned no id: {created}")
            jobs.append(
                OpenRouterBatchJob(
                    batch_id=str(batch_id),
                    n_requests=len(chunk),
                    custom_ids=custom_ids,
                    status=str(created.get("status", "submitted")).lower(),
                )
            )
            # Persist after every create: a chunk that is submitted but not
            # recorded would be paid for and never collected.
            self._save_jobs(jobs)
        return jobs

    def poll(self) -> list[OpenRouterBatchJob]:
        jobs = self._load_jobs()
        for job in jobs:
            if job.fetched:
                continue
            job.status = str(self.client.retrieve(job.batch_id).get("status", "")).lower()
        self._save_jobs(jobs)
        return jobs

    def fetch(self, renders) -> dict[str, Any]:
        """Collect finished batches and write every completion into the cache."""
        by_digest = {
            self.key_for(r, i).digest: (r, i, self.key_for(r, i))
            for r in renders
            for i in range(self.model.completions)
        }
        stored = failed = unknown = 0
        messages: dict[str, int] = {}
        jobs = self._load_jobs()
        for job in jobs:
            payload = self.client.retrieve(job.batch_id)
            job.status = str(payload.get("status", "")).lower()
            results = payload.get("results")
            if not isinstance(results, list):
                continue
            for row in results:
                digest = job.custom_ids.get(row.get("custom_id", ""))
                if digest is None or digest not in by_digest:
                    unknown += 1
                    continue
                render, _index, key = by_digest[digest]
                response = row.get("response") or {}
                body = response.get("body") or {}
                problem = (
                    f"HTTP {response.get('status_code')}"
                    if response.get("status_code") not in (200, None)
                    else openai_response_error(body)
                )
                if problem:
                    failed += 1
                    messages[str(problem)[:200]] = messages.get(str(problem)[:200], 0) + 1
                    continue
                choice = (body.get("choices") or [{}])[0]
                self.cache.put(
                    key,
                    CachedResponse(
                        text=(choice.get("message") or {}).get("content") or "",
                        usage=body.get("usage") or {},
                        truncated=choice.get("finish_reason") == "length",
                        reasoning=extract_reasoning(choice),
                        raw=body,
                    ),
                    provenance={
                        "render_id": render.render_id,
                        "semantic_instance_id": render.semantic_instance_id,
                        "question_family": render.question_family,
                        "protein_group_id": render.protein_group_id,
                        "representation": render.representation,
                        "input_token_count": render.input_token_count,
                        "batch_id": job.batch_id,
                        "delivery": "openrouter_batch",
                    },
                )
                stored += 1
            if job.status in TERMINAL:
                job.fetched = True
        self._save_jobs(jobs)
        return {"stored": stored, "failed": failed, "unknown": unknown, "errors": messages}

    def wait(self, *, interval: float = 60.0, timeout: float = 24 * 3600):
        deadline = time.time() + timeout
        while time.time() < deadline:
            jobs = self.poll()
            if all(job.status in TERMINAL for job in jobs):
                return jobs
            time.sleep(interval)
        raise OpenRouterBatchError("batch did not finish inside the completion window")

    # ------------------------------------------------------------------ #
    def _load_jobs(self) -> list[OpenRouterBatchJob]:
        if not self.state_path.exists():
            return []
        state = read_json(self.state_path)
        if state.get("model_id") != self.model.model_id:
            raise OpenRouterBatchError(
                f"{self.state_path} holds batches for {state.get('model_id')!r}, "
                f"not {self.model.model_id!r}; use a different state directory"
            )
        if int(state.get("max_output_tokens", -1)) != self.model.max_output_tokens:
            raise OpenRouterBatchError(
                f"{self.state_path} was submitted at a different output budget "
                f"({state.get('max_output_tokens')}); its answers belong to other cache keys"
            )
        return [OpenRouterBatchJob.from_dict(row) for row in state.get("jobs", [])]

    def _save_jobs(self, jobs: list[OpenRouterBatchJob]) -> None:
        write_json(
            self.state_path,
            {
                "model_id": self.model.model_id,
                "wire_model_id": batch_model_id(self.model.model_id),
                "provider": self.model.provider,
                "max_output_tokens": self.model.max_output_tokens,
                "jobs": [job.as_dict() for job in jobs],
            },
        )
