"""A content-addressed store of model responses.

The cache is keyed on everything that determines a completion — the provider
endpoint, model, sampling parameters and output budget, and the exact prompt
text — and on nothing else. In particular it is *not* keyed on ``render_id`` or
on the instance identifier, both of which move when the dataset is rebuilt with
a new seed or a different protein pool. Adding a question to the benchmark
therefore costs exactly the calls for that question; removing one costs nothing
at all, and its responses stay on disk for inspection.

Each entry holds the provider's full response body, so reasoning traces,
finish reasons and token accounting survive for later analysis rather than
being reduced to the answer text at call time. Prompts are stored by hash
only: they are large, they are already in the dataset's ``prompts/`` directory,
and duplicating them here would multiply the cache size by an order of
magnitude for no gain.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..util import sha256_text, stable_hash
from .locking import file_lock

#: Where responses land unless a run says otherwise.
DEFAULT_CACHE_DIR = Path("data/response_cache")
CACHE_DIR_ENV = "PDBTHINK_RESPONSE_CACHE"
#: Bumped only if the stored layout changes in a way older entries cannot satisfy.
CACHE_FORMAT = 3
#: Recheck shard mtimes periodically so an older concurrent writer is noticed.
LEGACY_V2_INDEX_REFRESH_SECONDS = 5.0
LEGACY_V2_HEADER = re.compile(
    rb'^\s*\{\s*"cache_format"\s*:\s*2(?:\s*[,}])'
)


def default_cache_dir() -> Path:
    return Path(os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR)


@dataclass(frozen=True)
class CacheKey:
    """Everything that determines a completion."""

    provider: str
    endpoint: str
    model_id: str
    model_revision: str | None
    reasoning_effort: str | None
    max_output_tokens: int
    sampling_parameters: dict[str, Any]
    system_prompt: str
    user_prompt: str
    completion_index: int
    #: Format 2 stored the run repeat count even though it is not part of the
    #: current per-request identity. Retain it only as migration context.
    legacy_v2_completions: int | None = None
    #: Distinguishes an automatically added repeat seed from an explicit seed
    #: with the same numeric value when reconstructing a format-2 key.
    legacy_v2_generated_seed: bool | None = None

    @property
    def digest(self) -> str:
        return stable_hash(
            CACHE_FORMAT,
            self.provider,
            self.endpoint,
            self.model_id,
            self.model_revision or "",
            self.reasoning_effort or "",
            self.max_output_tokens,
            sorted(self.sampling_parameters.items()),
            sha256_text(self.system_prompt),
            sha256_text(self.user_prompt),
            self.completion_index,
        )

    @property
    def legacy_v1_digest(self) -> str:
        """The pre-endpoint key, used only to fetch already submitted batches."""
        sampling_parameters = {
            key: value
            for key, value in self.sampling_parameters.items()
            if key not in ("completions", "output_token_parameter", "seed")
        }
        return stable_hash(
            1,
            self.provider,
            self.model_id,
            self.model_revision or "",
            self.reasoning_effort or "",
            self.max_output_tokens,
            sorted(sampling_parameters.items()),
            sha256_text(self.system_prompt),
            sha256_text(self.user_prompt),
            self.completion_index,
        )

    def legacy_v2_digest(self, completions: int) -> str:
        """The endpoint-scoped key used by batches submitted with cache format 2."""
        sampling_parameters = self._legacy_v2_sampling_parameters(completions)
        return stable_hash(
            2,
            self.provider,
            self.endpoint,
            self.model_id,
            self.model_revision or "",
            self.reasoning_effort or "",
            self.max_output_tokens,
            sorted(sampling_parameters.items()),
            sha256_text(self.system_prompt),
            sha256_text(self.user_prompt),
            self.completion_index,
        )

    def _legacy_v2_sampling_parameters(self, completions: int) -> dict[str, Any]:
        sampling = dict(self.sampling_parameters)
        generated_seed = 1000 + self.completion_index if completions > 1 else None
        generated = self.legacy_v2_generated_seed
        if generated is None:
            generated = (
                generated_seed is not None
                and self.provider in ("openai_chat", "ollama_chat")
            )
        if generated and self.provider == "openai_chat":
            sampling.pop("seed", None)
        if generated and self.provider == "ollama_chat":
            sampling.pop("ollama_options_seed", None)
        sampling["completions"] = completions
        return sampling

    def request_fingerprint(self) -> dict[str, Any]:
        """The human-readable half of the key, stored alongside the response."""
        return {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "sampling_parameters": self.sampling_parameters,
            "completion_index": self.completion_index,
            "system_prompt_sha256": sha256_text(self.system_prompt),
            "user_prompt_sha256": sha256_text(self.user_prompt),
        }


@dataclass
class CachedResponse:
    """One stored completion."""

    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    #: Absent on a fresh call, set when the entry came off disk.
    cached_at: str | None = None


class ResponseCache:
    """A directory of JSON entries, one per completion, sharded by digest."""

    def __init__(self, directory: str | Path | None = None, *, enabled: bool = True) -> None:
        self.directory = Path(directory) if directory else default_cache_dir()
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._locks_guard = threading.Lock()
        self._request_locks: dict[str, threading.Lock] = {}
        self._legacy_v2_index_guard = threading.Lock()
        self._legacy_v2_indexes: dict[str, dict[str, list[Path]]] = {}
        self._legacy_v2_shards: dict[
            str,
            dict[Path, tuple[int, dict[str, list[Path]]]],
        ] = {}
        self._legacy_v2_index_checked_at: dict[str, float] = {}

    def path_for(self, key: CacheKey) -> Path:
        digest = key.digest
        return self.directory / key.provider / digest[:2] / f"{digest}.json"

    def get(self, key: CacheKey) -> CachedResponse | None:
        if not self.enabled:
            return None
        path = self.path_for(key)
        entry = self._read_entry(path)
        legacy_completions = key.legacy_v2_completions
        if legacy_completions is None and "seed" not in key.sampling_parameters:
            legacy_completions = 1
        if entry is None and legacy_completions is not None:
            for legacy_path in self._legacy_v2_candidates(key, legacy_completions):
                legacy_entry = self._read_entry(legacy_path)
                if not self._valid_legacy_v2_entry(legacy_entry, key, legacy_path):
                    continue
                legacy_digest = legacy_path.stem
                response = self._cached_response(legacy_entry)
                provenance = dict(legacy_entry.get("provenance") or {})
                provenance.update({
                    "migrated_from_cache_format": 2,
                    "migrated_from_cache_key": legacy_digest,
                    "legacy_created_at": legacy_entry.get("created_at"),
                })
                self.put(key, response, provenance=provenance)
                entry = self._read_entry(path)
                break
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return self._cached_response(entry)

    @staticmethod
    def _cached_response(entry: dict[str, Any]) -> CachedResponse:
        derived = entry.get("derived") or {}
        return CachedResponse(
            text=derived.get("text", ""),
            usage=derived.get("usage") or {},
            truncated=bool(derived.get("truncated")),
            reasoning=derived.get("reasoning", ""),
            raw=entry.get("response") or {},
            cached_at=entry.get("created_at"),
        )

    @staticmethod
    def _read_entry(path: Path) -> dict[str, Any] | None:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A half-written entry is a miss, not a crash: the call is repeatable.
            return None
        # Valid JSON with the wrong top-level type is corrupt cache data too.
        return entry if isinstance(entry, dict) else None

    def _valid_legacy_v2_entry(
        self,
        entry: dict[str, Any] | None,
        key: CacheKey,
        legacy_path: Path,
    ) -> bool:
        if not entry or entry.get("cache_format") != 2:
            return False
        request = entry.get("request")
        current_token = self._legacy_v2_request_token(key.request_fingerprint())
        legacy_token = self._legacy_v2_request_token(request)
        legacy_digest = self._legacy_v2_digest_from_request(request)
        if (
            current_token is None
            or legacy_token is None
            or legacy_token[1] is None
            or legacy_digest is None
        ):
            return False
        expected_path = (
            self.directory
            / key.provider
            / legacy_digest[:2]
            / f"{legacy_digest}.json"
        )
        response = entry.get("response")
        return (
            entry.get("cache_key") == legacy_digest
            and legacy_path == expected_path
            and legacy_token[0] == current_token[0]
            and (
                key.provider != "openai_chat"
                or (
                    isinstance(response, dict)
                    and openai_response_error(response) is None
                )
            )
        )

    @staticmethod
    def _legacy_v2_digest_from_request(request: Any) -> str | None:
        """Recompute the format-2 key from the exact stored request fields."""
        if not isinstance(request, dict):
            return None
        sampling = request.get("sampling_parameters")
        if not isinstance(sampling, dict) or any(
            not isinstance(name, str) for name in sampling
        ):
            return None
        try:
            return stable_hash(
                2,
                request.get("provider"),
                request.get("endpoint"),
                request.get("model_id"),
                request.get("model_revision") or "",
                request.get("reasoning_effort") or "",
                request.get("max_output_tokens"),
                sorted(sampling.items()),
                request.get("system_prompt_sha256"),
                request.get("user_prompt_sha256"),
                request.get("completion_index"),
            )
        except (TypeError, ValueError):
            return None

    def _legacy_v2_candidates(
        self, key: CacheKey, direct_completions: int
    ) -> Iterator[Path]:
        """Find old keys even when only the run's total repeat count changed.

        The direct path keeps the common case cheap. Fallback discovery reads
        only format headers in changed digest shards and reuses their index.
        """
        direct_digest = key.legacy_v2_digest(direct_completions)
        direct_path = (
            self.directory
            / key.provider
            / direct_digest[:2]
            / f"{direct_digest}.json"
        )
        yield direct_path
        token = self._legacy_v2_request_token(key.request_fingerprint())
        if token is None:
            return
        index = self._legacy_v2_index(key.provider)
        for candidate in index.get(token[0], []):
            if candidate != direct_path:
                yield candidate

    def _legacy_v2_index(self, provider: str) -> dict[str, list[Path]]:
        now = time.monotonic()
        with self._legacy_v2_index_guard:
            index = self._legacy_v2_indexes.get(provider)
            checked_at = self._legacy_v2_index_checked_at.get(provider, 0.0)
            if (
                index is not None
                and now - checked_at < LEGACY_V2_INDEX_REFRESH_SECONDS
            ):
                return index
            index = self._refresh_legacy_v2_index(provider)
            self._legacy_v2_indexes[provider] = index
            self._legacy_v2_index_checked_at[provider] = now
            return index

    def _refresh_legacy_v2_index(self, provider: str) -> dict[str, list[Path]]:
        root = self.directory / provider
        previous = self._legacy_v2_shards.get(provider, {})
        current: dict[Path, tuple[int, dict[str, list[Path]]]] = {}
        try:
            children = list(root.iterdir())
        except OSError:
            children = []
        for shard in children:
            if (
                len(shard.name) != 2
                or any(character not in "0123456789abcdef" for character in shard.name)
                or shard.is_symlink()
            ):
                continue
            try:
                if not shard.is_dir():
                    continue
                signature = shard.stat().st_mtime_ns
            except OSError:
                continue
            known = previous.get(shard)
            if known is not None and known[0] == signature:
                current[shard] = known
            else:
                current[shard] = (
                    signature,
                    self._scan_legacy_v2_shard(shard),
                )
        self._legacy_v2_shards[provider] = current

        index: dict[str, list[Path]] = {}
        for _, shard_index in current.values():
            for token, paths in shard_index.items():
                index.setdefault(token, []).extend(paths)
        for candidates in index.values():
            candidates.sort(key=str)
        return index

    def _scan_legacy_v2_shard(self, shard: Path) -> dict[str, list[Path]]:
        index: dict[str, list[Path]] = {}
        try:
            paths = list(shard.glob("*.json"))
        except OSError:
            return index
        for path in paths:
            if not self._looks_like_legacy_v2(path):
                continue
            entry = self._read_entry(path)
            if not entry or entry.get("cache_format") != 2:
                continue
            token = self._legacy_v2_request_token(entry.get("request"))
            if token is None or token[1] is None:
                continue
            index.setdefault(token[0], []).append(path)
        return index

    @staticmethod
    def _looks_like_legacy_v2(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                header = handle.read(256)
        except OSError:
            return False
        return LEGACY_V2_HEADER.match(header) is not None

    @staticmethod
    def _legacy_v2_request_token(
        request: Any,
    ) -> tuple[str, int | None] | None:
        """Map a format-2 request to its current per-completion identity.

        Format 2 omitted automatically generated repeat seeds from its stored
        sampling parameters, so restore them before comparing with format 3.
        """
        if not isinstance(request, dict):
            return None
        normalised = dict(request)
        raw_sampling = normalised.get("sampling_parameters")
        if not isinstance(raw_sampling, dict):
            return None
        sampling = dict(raw_sampling)
        completions = sampling.pop("completions", None)
        if completions is not None and (
            type(completions) is not int or completions < 1
        ):
            return None
        completion_index = normalised.get("completion_index")
        provider = normalised.get("provider")
        if (
            completions is not None
            and completions > 1
            and type(completion_index) is int
        ):
            if provider == "openai_chat" and "seed" not in sampling:
                sampling["seed"] = 1000 + completion_index
            elif (
                provider == "ollama_chat"
                and "ollama_options_seed" not in sampling
                and "options" not in sampling
            ):
                sampling["ollama_options_seed"] = 1000 + completion_index
        normalised["sampling_parameters"] = sampling
        return stable_hash("legacy-v2-request", normalised), completions

    @contextlib.contextmanager
    def request_lock(self, key: CacheKey):
        """Serialize one paid request across threads and evaluator processes."""
        if not self.enabled:
            yield
            return
        digest = key.digest
        with self._locks_guard:
            thread_lock = self._request_locks.setdefault(digest, threading.Lock())
        with thread_lock:
            lock_path = self.directory / ".locks" / digest[:2] / f"{digest}.lock"
            with file_lock(lock_path):
                yield

    def put(
        self,
        key: CacheKey,
        response: CachedResponse,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> Path | None:
        """Store a completion. Errors are recorded as a failure to cache, never raised."""
        if not self.enabled:
            return None
        path = self.path_for(key)
        entry = {
            "cache_format": CACHE_FORMAT,
            "cache_key": key.digest,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "request": key.request_fingerprint(),
            "provenance": provenance or {},
            "derived": {
                "text": response.text,
                "reasoning": response.reasoning,
                "usage": response.usage,
                "truncated": response.truncated,
            },
            "response": response.raw,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a killed run never leaves a half-parsed entry that
        # a later run would treat as a legitimate response.
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".partial",
                delete=False,
            ) as handle:
                json.dump(entry, handle, indent=2, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.writes += 1
        return path

    def statistics(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }

    # ------------------------------------------------------------------ #
    def entries(self, provider: str | None = None):
        """Every stored entry, for inspection and export."""
        root = self.directory / provider if provider else self.directory
        if not root.exists():
            return
        for path in sorted(root.rglob("*.json")):
            try:
                yield path, json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue


#: Provider fields that carry a reasoning trace, in the order they are tried.
REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")


def extract_reasoning(choice: dict[str, Any]) -> str:
    """Pull a reasoning trace out of an OpenAI-shaped choice.

    Providers disagree: DeepSeek and Together use ``reasoning_content``, some
    gateways use ``reasoning``, and vLLM's parser emits either depending on
    version. Take the first that carries text rather than guessing from the
    model name.
    """
    message = choice.get("message") or {}
    for field_name in REASONING_FIELDS:
        value = message.get(field_name) or choice.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            joined = "\n".join(
                part.get("text", "") for part in value if isinstance(part, dict)
            ).strip()
            if joined:
                return joined
    return ""


def openai_response_error(payload: dict[str, Any]) -> str | None:
    """Return an OpenAI-shaped in-band API error, including gateway choice errors."""
    choices = payload.get("choices") or []
    error = payload.get("error")
    if error:
        return _error_text(error)
    if not choices:
        return "response contained no choices"
    choice = choices[0]
    error = choice.get("error")
    if error or choice.get("finish_reason") == "error":
        return _error_text(error or "generation failed")
    return None


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or error)[:400]
    return str(error)[:400]
