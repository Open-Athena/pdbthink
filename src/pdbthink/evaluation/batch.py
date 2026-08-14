"""Batch inference against an OpenAI-compatible Batch API (Together).

Batch pricing is roughly half of synchronous pricing for the same model, at the
cost of a completion window measured in hours rather than seconds. That trade is
right for a benchmark: nothing here is interactive.

The design keeps batching strictly out of the scoring path. A batch run does one
thing — **fill the response cache** — and then the ordinary
:class:`~pdbthink.evaluation.runner.EvaluationRunner` runs as usual and finds
every completion already there. Scoring, reporting and the results schema are
untouched, a partially returned batch simply leaves some prompts uncached, and a
batch that fails outright costs nothing but time.

Prompts already in the cache are never submitted, so re-running after adding a
few questions to the benchmark submits only those questions.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..util import read_json
from .cache import (
    CACHE_FORMAT,
    CachedResponse,
    CacheKey,
    ResponseCache,
    extract_reasoning,
    openai_response_error,
)

#: Statuses that mean the provider is done with a batch, successfully or not.
TERMINAL = ("completed", "failed", "expired", "cancelled", "error")
#: Together rejects very large uploads; split the work rather than discover it.
MAX_REQUESTS_PER_BATCH = 3000
MAX_UPLOAD_BYTES = 90 * 1024 * 1024
TOGETHER_ENDPOINT = "https://api.together.xyz/v1"
BATCH_STATE_FORMAT = 2


class BatchError(RuntimeError):
    pass


@dataclass
class BatchJob:
    """One submitted batch and the mapping back to cache keys."""

    batch_id: str
    input_file_id: str
    n_requests: int
    #: custom_id -> the cache key digest it stands for.
    custom_ids: dict[str, str] = field(default_factory=dict)
    status: str = "submitted"
    output_file_id: str | None = None
    error_file_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "input_file_id": self.input_file_id,
            "n_requests": self.n_requests,
            "custom_ids": self.custom_ids,
            "status": self.status,
            "output_file_id": self.output_file_id,
            "error_file_id": self.error_file_id,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> BatchJob:
        return cls(**row)


class TogetherBatchClient:
    """Thin wrapper over the Together SDK.

    Together's file upload is a three-step signed-redirect protocol — a
    metadata POST that answers 302, a PUT of the bytes to the returned URL, and
    a preprocess callback — none of it documented as a stable contract. Their
    client is the right place for that to live, so batching is an optional
    extra rather than something this package reimplements and then has to chase.
    """

    def __init__(self, api_key: str, *, timeout: float = 600.0) -> None:
        try:
            import together
        except ImportError as exc:  # pragma: no cover - exercised by the CLI
            raise BatchError(
                "batch inference needs the Together client: "
                "pip install -e '.[batch]'"
            ) from exc
        self._client = together.Together(api_key=api_key, timeout=timeout)

    def upload(self, path: Path, purpose: str = "batch-api") -> str:
        response = self._client.files.upload(file=path, purpose=purpose)
        file_id = getattr(response, "id", None)
        if not file_id:
            raise BatchError(f"upload returned no file id: {response}")
        return str(file_id)

    def create(self, input_file_id: str, *, endpoint: str = "/v1/chat/completions",
               completion_window: str = "24h") -> dict[str, Any]:
        created = self._client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
        )
        return _as_dict(created)

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        return _as_dict(self._client.batches.retrieve(batch_id))

    def content(self, file_id: str) -> bytes:
        payload = self._client.files.content(file_id)
        for attribute in ("content", "text", "read"):
            value = getattr(payload, attribute, None)
            if callable(value):
                value = value()
            if isinstance(value, bytes):
                return value
            if isinstance(value, str):
                return value.encode("utf-8")
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        raise BatchError(f"could not read file {file_id}: {type(payload).__name__}")


def _as_dict(payload: Any) -> dict[str, Any]:
    """Normalise an SDK model to a plain dict without depending on its type.

    Together wraps a created batch in a ``job`` envelope while ``retrieve``
    returns it bare, so unwrap when the envelope is present.
    """
    for attribute in ("model_dump", "dict", "to_dict"):
        method = getattr(payload, attribute, None)
        if callable(method):
            try:
                payload = dict(method())
                break
            except TypeError:
                continue
    else:
        payload = (
            payload if isinstance(payload, dict)
            else {k: v for k, v in vars(payload).items() if not k.startswith("_")}
        )
    inner = payload.get("job")
    return dict(inner) if isinstance(inner, dict) else payload


def _status_of(payload: dict[str, Any]) -> str:
    return str(payload.get("status") or payload.get("state") or "unknown").lower()


def _output_file(payload: dict[str, Any]) -> str | None:
    for field_name in ("output_file_id", "file_id", "output_file"):
        value = payload.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


class BatchRun:
    """Submit the uncached prompts of a dataset, then fold results into the cache."""

    def __init__(
        self,
        model,
        cache: ResponseCache,
        state_dir: str | Path,
        *,
        client: TogetherBatchClient | None = None,
    ) -> None:
        self.model = model
        self.cache = cache
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "batch_state.json"
        self._custom_id_cache_format = CACHE_FORMAT
        self._legacy_unidentified_state = False
        if model.provider != "openai_chat" or model.endpoint_identity != TOGETHER_ENDPOINT:
            raise BatchError(
                "batch inference uses Together's Batch API and requires "
                f"provider: openai_chat and base_url: {TOGETHER_ENDPOINT}"
            )
        if client is not None:
            self.client = client
        else:
            key = os.environ.get(model.api_key_env, "")
            if not key:
                raise BatchError(f"{model.api_key_env} is not set")
            self.client = TogetherBatchClient(key)

    # ------------------------------------------------------------------ #
    def key_for(self, render, completion_index: int) -> CacheKey:
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

    def request_body(self, render, completion_index: int = 0) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model.model_id,
            "messages": [
                {"role": "system", "content": render.system_prompt},
                {"role": "user", "content": render.user_prompt},
            ],
            self.model.output_token_parameter: self.model.max_output_tokens,
        }
        if self.model.temperature is not None:
            payload["temperature"] = self.model.temperature
        if self.model.top_p is not None:
            payload["top_p"] = self.model.top_p
        if self.model.reasoning_effort:
            payload["reasoning_effort"] = self.model.reasoning_effort
        seed = self.model.completion_seed(completion_index)
        if seed is not None:
            payload["seed"] = seed
        payload.update(self.model.extra_body)
        return payload

    def pending(self, renders) -> list[tuple[Any, int, CacheKey]]:
        """Renders with no cached completion, which are the only ones worth money."""
        out = []
        for render in renders:
            for index in range(self.model.completions):
                key = self.key_for(render, index)
                if self.cache.path_for(key).exists():
                    continue
                out.append((render, index, key))
        return out

    def preflight(self) -> None:
        """Spend one token to find out whether the model is reachable at all.

        Together lists models it will not serve. ``pricing`` and ``running`` do
        not distinguish them — DeepSeek V4 Flash reports ``running: false`` and
        works, while GLM-4.7 carries real per-token pricing and answers
        "model is disabled". A rejected batch is only discovered after the
        completion window, and a batch that is accepted but unservable comes
        back hours later with every request in the error file. One synchronous
        call up front costs a token and turns both into an immediate failure.
        """
        from .runner import ProviderError, _openai_chat

        probe = replace(
            self.model, max_output_tokens=1, reasoning_effort=None, extra_body={}
        )
        try:
            _openai_chat(probe, "", "hi", 0)
        except ProviderError as exc:
            raise BatchError(
                f"{self.model.model_id} is not usable through this account: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    def submit(self, renders) -> list[BatchJob]:
        with self._state_lock():
            jobs = self._load_jobs()
            submitted = {
                digest
                for job in jobs
                for digest in job.custom_ids.values()
            }
            work = [
                item for item in self.pending(renders)
                if item[2].digest not in submitted
                and item[2].legacy_v1_digest not in submitted
                and item[2].legacy_v2_digest(self.model.completions) not in submitted
            ]
            if not work:
                return jobs
            if self._legacy_unidentified_state:
                raise BatchError(
                    "legacy batch state can be polled and fetched but does not identify "
                    "enough request parameters for new submissions; use a new --state-dir"
                )

            chunks: list[list[tuple[Any, int, CacheKey]]] = [[]]
            size = 0
            for item in work:
                line_size = len(json.dumps(self.request_body(item[0], item[1])))
                if chunks[-1] and (
                    len(chunks[-1]) >= MAX_REQUESTS_PER_BATCH
                    or size + line_size > MAX_UPLOAD_BYTES
                ):
                    chunks.append([])
                    size = 0
                chunks[-1].append(item)
                size += line_size

            first_number = len(jobs)
            for offset, chunk in enumerate(chunks):
                number = first_number + offset
                path = self.state_dir / f"input_{number:02d}.jsonl"
                custom_ids: dict[str, str] = {}
                with path.open("w", encoding="utf-8") as handle:
                    for position, (render, index, key) in enumerate(chunk):
                        # Short positional ids: provider limits on custom_id length are
                        # not uniform, and the mapping back to the digest lives here.
                        custom_id = f"r{number:02d}-{position:05d}"
                        custom_ids[custom_id] = key.digest
                        handle.write(json.dumps({
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": self.request_body(render, index),
                        }) + "\n")
                file_id = self.client.upload(path)
                created = self.client.create(file_id)
                batch_id = created.get("id") or created.get("batch_id")
                if not batch_id:
                    raise BatchError(f"batch creation returned no id: {created}")
                jobs.append(BatchJob(
                    batch_id=batch_id,
                    input_file_id=file_id,
                    n_requests=len(chunk),
                    custom_ids=custom_ids,
                    status=_status_of(created),
                ))
                # Persist each provider-created id immediately. A later chunk can
                # fail without making the successful paid submission disappear.
                self._save_jobs(jobs)
            return jobs

    def poll(self) -> list[BatchJob]:
        with self._state_lock():
            jobs = self._load_jobs()
            for job in jobs:
                if job.status in TERMINAL and job.output_file_id:
                    continue
                payload = self.client.retrieve(job.batch_id)
                job.status = _status_of(payload)
                job.output_file_id = _output_file(payload) or job.output_file_id
                job.error_file_id = payload.get("error_file_id") or job.error_file_id
            self._save_jobs(jobs)
            return jobs

    def errors(self) -> dict[str, int]:
        """Messages from any error file, so a silent zero-result fetch explains itself."""
        counts: dict[str, int] = {}
        for job in self._load_jobs():
            if not job.error_file_id:
                continue
            raw = self.client.content(job.error_file_id)
            for line in raw.decode("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                error = row.get("error") or (row.get("response") or {}).get("body", {}).get("error")
                message = error.get("message") if isinstance(error, dict) else str(error)
                counts[str(message)[:200]] = counts.get(str(message)[:200], 0) + 1
        return counts

    def fetch(self, renders) -> dict[str, Any]:
        """Download finished batches and write every completion into the cache."""
        jobs = self._load_jobs()
        by_digest = {}
        for render in renders:
            for index in range(self.model.completions):
                key = self.key_for(render, index)
                target = (render, index, key)
                by_digest[key.digest] = target
                if self._custom_id_cache_format == 1:
                    # Legacy state can reach this point only after its Together-
                    # specific request assumptions have been validated.
                    by_digest[key.legacy_v1_digest] = target
                elif self._custom_id_cache_format == 2:
                    by_digest[key.legacy_v2_digest(self.model.completions)] = target
        stored = failed = unknown = 0
        for job in jobs:
            if not job.output_file_id:
                continue
            raw = self.client.content(job.output_file_id)
            for line in raw.decode("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                digest = job.custom_ids.get(row.get("custom_id", ""))
                if digest is None or digest not in by_digest:
                    unknown += 1
                    continue
                render, index, key = by_digest[digest]
                body = (row.get("response") or {}).get("body") or row.get("response") or {}
                if row.get("error") or openai_response_error(body):
                    failed += 1
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
                        "delivery": "batch",
                    },
                )
                stored += 1
        return {"stored": stored, "failed": failed, "unknown": unknown}

    def wait(self, *, interval: float = 60.0, timeout: float = 24 * 3600) -> list[BatchJob]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            jobs = self.poll()
            if all(job.status in TERMINAL for job in jobs):
                return jobs
            time.sleep(interval)
        raise BatchError("batch did not finish inside the completion window")

    # ------------------------------------------------------------------ #
    def _load_jobs(self) -> list[BatchJob]:
        if not self.state_path.exists():
            return []
        state = read_json(self.state_path)
        identity = state.get("request_identity")
        if identity is None:
            self._validate_legacy_state(state)
            self._custom_id_cache_format = 1
            self._legacy_unidentified_state = True
        elif self._normalise_request_identity(identity) != self._request_identity():
            raise BatchError(
                f"{self.state_dir} belongs to another batch model configuration; "
                "use a separate --state-dir"
            )
        else:
            self._legacy_unidentified_state = False
            self._custom_id_cache_format = int(
                state.get("custom_id_cache_format", CACHE_FORMAT)
            )
        return [BatchJob.from_dict(row) for row in state.get("jobs", [])]

    @staticmethod
    def _normalise_request_identity(identity: dict[str, Any]) -> dict[str, Any]:
        """Upgrade cache-format-2 state identity without changing its request."""
        identity = dict(identity)
        sampling = dict(identity.get("sampling_parameters") or {})
        identity.setdefault("completions", int(sampling.pop("completions", 1)))
        identity["sampling_parameters"] = sampling
        return identity

    def _request_identity(self) -> dict[str, Any]:
        return {
            "provider": self.model.provider,
            "endpoint": self.model.endpoint_identity,
            "model_id": self.model.model_id,
            "model_revision": self.model.model_revision,
            "completions": self.model.completions,
            "sampling_parameters": self.model.sampling_parameters,
        }

    def _validate_legacy_state(self, state: dict[str, Any]) -> None:
        expected = {
            "model_id": self.model.model_id,
            "provider": self.model.provider,
            "max_output_tokens": self.model.max_output_tokens,
        }
        mismatched = [key for key, value in expected.items() if state.get(key) != value]
        if (
            mismatched
            or self.model.output_token_parameter != "max_tokens"
        ):
            raise BatchError(
                f"legacy batch state in {self.state_dir} does not identify this exact "
                "request configuration; fetch it with its original Together config"
            )

    def _save_jobs(self, jobs: list[BatchJob]) -> None:
        state = {
            "custom_id_cache_format": self._custom_id_cache_format,
            "model_id": self.model.model_id,
            "provider": self.model.provider,
            "max_output_tokens": self.model.max_output_tokens,
            "jobs": [job.as_dict() for job in jobs],
        }
        # An old state file did not record enough information to identify all
        # sampling parameters. Polling it must not invent that identity and lock
        # out the original config; the v1 request digests validate it at fetch.
        if not self._legacy_unidentified_state:
            state["batch_state_format"] = BATCH_STATE_FORMAT
            state["request_identity"] = self._request_identity()
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.state_dir,
                prefix=".batch_state.",
                suffix=".partial",
                delete=False,
            ) as handle:
                json.dump(state, handle, indent=2, sort_keys=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            temporary.replace(self.state_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @contextlib.contextmanager
    def _state_lock(self):
        """Prevent two processes from creating or overwriting the same batch state."""
        lock_path = self.state_dir / ".batch_state.lock"
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
