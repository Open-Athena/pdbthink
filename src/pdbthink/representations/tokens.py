"""Reference-tokenizer token counting for the input budget (section 6).

The dataset configuration names a reference tokenizer whose counts gate the
input budget. Actual per-model token counts are recorded separately at
evaluation time from provider usage fields.
"""

from __future__ import annotations

from functools import lru_cache

HEURISTIC = "heuristic_chars_v1"
#: Characters per token used when no reference tokenizer is installed. Measured
#: on rendered minimal-PDB text, which is unusually token-dense.
HEURISTIC_CHARS_PER_TOKEN = 2.9


@lru_cache(maxsize=4)
def _encoder(name: str):
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - optional dependency
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:  # pragma: no cover - unknown encoding name
        return None


def count_tokens(text: str, tokenizer: str = "cl100k_base") -> tuple[int, str]:
    """Return ``(token_count, tokenizer_actually_used)``."""
    encoder = _encoder(tokenizer)
    if encoder is None:
        return (int(len(text) / HEURISTIC_CHARS_PER_TOKEN) + 1, HEURISTIC)
    return (len(encoder.encode(text)), tokenizer)


def tokenizer_available(tokenizer: str = "cl100k_base") -> bool:
    return _encoder(tokenizer) is not None
