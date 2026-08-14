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
from .locking import durable_mkdir, file_lock, fsync_directory

#: Where responses land unless a run says otherwise.
DEFAULT_CACHE_DIR = Path("data/response_cache")
CACHE_DIR_ENV = "PDBTHINK_RESPONSE_CACHE"
#: Bumped only if the stored layout changes in a way older entries cannot satisfy.
CACHE_FORMAT = 3
# Format-2 fallback is a process-lifetime snapshot. Old-format writers must be
# stopped before a current evaluator starts; current writers coordinate by v3 key.
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


class CacheDiscoveryError(RuntimeError):
    """The cache could not establish whether a paid request was already stored."""


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
        self._legacy_v2_snapshots: set[str] = set()

    def path_for(self, key: CacheKey) -> Path:
        digest = key.digest
        return self.directory / key.provider / digest[:2] / f"{digest}.json"

    def get(self, key: CacheKey) -> CachedResponse | None:
        if not self.enabled:
            return None
        path = self.path_for(key)
        entry, read_complete, present = self._read_entry_with_status(path)
        if not read_complete:
            raise CacheDiscoveryError(
                f"could not read cache entry {path}; refusing a provider request"
            )
        cached = None
        if present:
            if not self._valid_current_entry(entry, key):
                raise CacheDiscoveryError(
                    f"cache entry {path} does not match its key; move or remove it "
                    "explicitly before retrying"
                )
            cached = self._cached_response(entry)
            if cached is None:
                raise CacheDiscoveryError(
                    f"cache entry {path} is corrupt; move or remove it explicitly "
                    "before retrying"
                )
        legacy_completions = key.legacy_v2_completions
        if legacy_completions is None and "seed" not in key.sampling_parameters:
            legacy_completions = 1
        if cached is None and legacy_completions is not None:
            for legacy_path in self._legacy_v2_candidates(key, legacy_completions):
                legacy_entry, read_complete, _ = self._read_entry_with_status(legacy_path)
                if not read_complete:
                    raise CacheDiscoveryError(
                        f"could not read legacy cache entry {legacy_path}; "
                        "refusing a provider request"
                    )
                if not self._valid_legacy_v2_entry(legacy_entry, key, legacy_path):
                    continue
                legacy_digest = legacy_path.stem
                response = self._cached_response(legacy_entry)
                if response is None:
                    continue
                stored_provenance = legacy_entry.get("provenance")
                provenance = (
                    dict(stored_provenance)
                    if isinstance(stored_provenance, dict)
                    else {}
                )
                provenance.update({
                    "migrated_from_cache_format": 2,
                    "migrated_from_cache_key": legacy_digest,
                    "legacy_created_at": legacy_entry.get("created_at"),
                })
                self.put(key, response, provenance=provenance)
                cached = response
                break
        if cached is None:
            self.misses += 1
            return None
        self.hits += 1
        return cached

    @staticmethod
    def _cached_response(entry: dict[str, Any] | None) -> CachedResponse | None:
        if not isinstance(entry, dict):
            return None
        derived = entry.get("derived")
        raw = entry.get("response")
        if not isinstance(derived, dict) or not isinstance(raw, dict):
            return None
        text = derived.get("text", "")
        usage = derived.get("usage", {})
        reasoning = derived.get("reasoning", "")
        truncated = derived.get("truncated", False)
        cached_at = entry.get("created_at")
        if (
            not isinstance(text, str)
            or not isinstance(usage, dict)
            or not isinstance(reasoning, str)
            or type(truncated) is not bool
            or (cached_at is not None and not isinstance(cached_at, str))
        ):
            return None
        return CachedResponse(
            text=text,
            usage=usage,
            truncated=truncated,
            reasoning=reasoning,
            raw=raw,
            cached_at=cached_at,
        )

    @staticmethod
    def _valid_current_entry(entry: dict[str, Any] | None, key: CacheKey) -> bool:
        return bool(
            entry
            and entry.get("cache_format") == CACHE_FORMAT
            and entry.get("cache_key") == key.digest
            and entry.get("request") == key.request_fingerprint()
        )

    @staticmethod
    def _read_entry_with_status(
        path: Path,
    ) -> tuple[dict[str, Any] | None, bool, bool]:
        """Return an entry plus whether the read completed and the path was present."""
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None, True, False
        except OSError:
            # A file that may exist but cannot be read is not an authoritative
            # miss: allowing a provider call here could pay for a duplicate.
            return None, False, True
        except json.JSONDecodeError:
            # Entries are atomically renamed into place, so malformed JSON is
            # corrupt data rather than a partial write that repeated scans fix.
            return None, True, True
        # Valid JSON with the wrong top-level type is corrupt cache data, but the
        # read itself completed and need not be retried.
        return (entry if isinstance(entry, dict) else None), True, True

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
        index, complete, refreshed = self._legacy_v2_index(key.provider)
        candidates = index.get(token[0], [])
        for candidate in candidates:
            if candidate != direct_path:
                yield candidate

        # A complete snapshot proves a miss for a quiescent format-2 cache. This
        # compatibility path deliberately does not coordinate with active old-code
        # writers; stop them before evaluating with the current cache format.
        if complete and (refreshed or not candidates):
            return
        index, complete, _ = self._legacy_v2_index(key.provider, force=True)
        for candidate in index.get(token[0], []):
            if candidate != direct_path and candidate not in candidates:
                yield candidate
        if not complete:
            raise CacheDiscoveryError(
                f"could not completely inspect the legacy {key.provider} cache; "
                "refusing a provider request"
            )

    def _legacy_v2_index(
        self, provider: str, *, force: bool = False
    ) -> tuple[dict[str, list[Path]], bool, bool]:
        with self._legacy_v2_index_guard:
            index = self._legacy_v2_indexes.get(provider)
            if not force and index is not None and provider in self._legacy_v2_snapshots:
                return index, True, False
            index, complete = self._refresh_legacy_v2_index(provider)
            self._legacy_v2_indexes[provider] = index
            if complete:
                self._legacy_v2_snapshots.add(provider)
            else:
                self._legacy_v2_snapshots.discard(provider)
            return index, complete, True

    def _refresh_legacy_v2_index(
        self, provider: str
    ) -> tuple[dict[str, list[Path]], bool]:
        root = self.directory / provider
        previous = self._legacy_v2_shards.get(provider, {})
        current: dict[Path, tuple[int, dict[str, list[Path]]]] = {}
        complete = True
        try:
            children = list(root.iterdir())
        except FileNotFoundError:
            if previous:
                return self._combine_legacy_v2_shards(previous), False
            self._legacy_v2_shards[provider] = {}
            return {}, True
        except OSError:
            # A transient root error must not erase a previously valid index.
            return self._combine_legacy_v2_shards(previous), False
        for shard in children:
            if (
                len(shard.name) != 2
                or any(character not in "0123456789abcdef" for character in shard.name)
                or shard.is_symlink()
            ):
                continue
            known = previous.get(shard)
            try:
                if not shard.is_dir():
                    continue
                signature = shard.stat().st_mtime_ns
            except OSError:
                complete = False
                if known is not None:
                    current[shard] = known
                continue
            if known is not None and known[0] == signature:
                current[shard] = known
                continue
            shard_index, shard_complete = self._scan_legacy_v2_shard(shard)
            if shard_complete:
                current[shard] = (signature, shard_index)
            else:
                complete = False
                if known is not None:
                    # Retain the last complete view with its old signature so
                    # the changed shard is retried on the next lookup.
                    current[shard] = known
        self._legacy_v2_shards[provider] = current
        return self._combine_legacy_v2_shards(current), complete

    @staticmethod
    def _combine_legacy_v2_shards(
        shards: dict[Path, tuple[int, dict[str, list[Path]]]],
    ) -> dict[str, list[Path]]:
        index: dict[str, list[Path]] = {}
        for _, shard_index in shards.values():
            for token, paths in shard_index.items():
                index.setdefault(token, []).extend(paths)
        for candidates in index.values():
            candidates.sort(key=str)
        return index

    def _scan_legacy_v2_shard(
        self, shard: Path
    ) -> tuple[dict[str, list[Path]], bool]:
        index: dict[str, list[Path]] = {}
        complete = True
        try:
            paths = list(shard.glob("*.json"))
        except OSError:
            return index, False
        for path in paths:
            looks_legacy = self._looks_like_legacy_v2(path)
            if looks_legacy is None:
                complete = False
                continue
            if not looks_legacy:
                continue
            entry, read_complete, _ = self._read_entry_with_status(path)
            if not read_complete:
                complete = False
                continue
            if not entry or entry.get("cache_format") != 2:
                continue
            token = self._legacy_v2_request_token(entry.get("request"))
            if token is None or token[1] is None:
                continue
            index.setdefault(token[0], []).append(path)
        return index, complete

    @staticmethod
    def _looks_like_legacy_v2(path: Path) -> bool | None:
        try:
            with path.open("rb") as handle:
                header = handle.read(256)
        except OSError:
            return None
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
        durable_mkdir(path.parent)
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
            fsync_directory(path.parent)
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
    if not isinstance(choice, dict):
        return ""
    raw_message = choice.get("message")
    message = raw_message if isinstance(raw_message, dict) else {}
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


def openai_response_error(payload: Any) -> str | None:
    """Return an OpenAI-shaped in-band API error, including malformed responses."""
    if not isinstance(payload, dict):
        return "response was not an object"
    error = payload.get("error")
    if error:
        return _error_text(error)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "response contained no choices"
    choice = choices[0]
    if not isinstance(choice, dict):
        return "response choice was not an object"
    error = choice.get("error")
    if error or choice.get("finish_reason") == "error":
        return _error_text(error or "generation failed")
    message = choice.get("message")
    if not isinstance(message, dict):
        return "response choice contained no message object"
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        return "response message content was not text"
    usage = payload.get("usage")
    if usage is not None and not isinstance(usage, dict):
        return "response usage was not an object"
    return None


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or error)[:400]
    return str(error)[:400]
