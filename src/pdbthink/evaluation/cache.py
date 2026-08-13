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

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..util import sha256_text, stable_hash

#: Where responses land unless a run says otherwise.
DEFAULT_CACHE_DIR = Path("data/response_cache")
CACHE_DIR_ENV = "PDBTHINK_RESPONSE_CACHE"
#: Bumped only if the stored layout changes in a way older entries cannot satisfy.
CACHE_FORMAT = 2


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
        return stable_hash(
            1,
            self.provider,
            self.model_id,
            self.model_revision or "",
            self.reasoning_effort or "",
            self.max_output_tokens,
            sorted(self.sampling_parameters.items()),
            sha256_text(self.system_prompt),
            sha256_text(self.user_prompt),
            self.completion_index,
        )

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

    def path_for(self, key: CacheKey) -> Path:
        digest = key.digest
        return self.directory / key.provider / digest[:2] / f"{digest}.json"

    def get(self, key: CacheKey) -> CachedResponse | None:
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A half-written entry is a miss, not a crash: the call is repeatable.
            self.misses += 1
            return None
        derived = entry.get("derived") or {}
        self.hits += 1
        return CachedResponse(
            text=derived.get("text", ""),
            usage=derived.get("usage") or {},
            truncated=bool(derived.get("truncated")),
            reasoning=derived.get("reasoning", ""),
            raw=entry.get("response") or {},
            cached_at=entry.get("created_at"),
        )

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
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(entry, indent=2, sort_keys=False), encoding="utf-8")
        temporary.replace(path)
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
