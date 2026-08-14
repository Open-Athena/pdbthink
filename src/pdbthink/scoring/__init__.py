"""Deterministic answer parsing and scoring."""

from __future__ import annotations

from typing import Any

from .parse import (
    AnswerFormatError,
    ParsedAnswer,
    canonical_atom,
    canonical_pair,
    canonical_residue,
    extract_final,
    looks_like_refusal,
    parse_answer,
)
from .scorers import score_answer, set_scores

__all__ = [
    "AnswerFormatError",
    "ParsedAnswer",
    "canonical_atom",
    "canonical_pair",
    "canonical_residue",
    "extract_final",
    "looks_like_refusal",
    "parse_answer",
    "score_answer",
    "score_response",
    "set_scores",
]


def score_response(
    raw_response: str,
    answer_schema: str,
    gold_answer: dict[str, Any],
    *,
    parameters: dict[str, Any] | None = None,
    truncated: bool = False,
    provider_refusal: bool = False,
) -> dict[str, Any]:
    """Parse and score one raw model response end to end.

    Returns ``{"parsed": ..., "score": ..., "format_error": ..., "refusal": ...}``.
    Malformed, refused and truncated answers score zero and are reported under
    their own failure category (section 12). ``provider_refusal`` covers an
    explicit terminal API signal even when partial text happens to parse.
    """
    parameters = parameters or {}
    parsed = parse_answer(raw_response, answer_schema, parameters)
    refusal = provider_refusal or (
        parsed.format_error and looks_like_refusal(raw_response)
    )
    result = score_answer(answer_schema, gold_answer, parsed, parameters)
    if truncated or provider_refusal:
        result = {**result, "score": 0.0, "correct": False}
    if truncated:
        result["truncated"] = True
    return {
        "parsed": parsed.as_dict(),
        "score": result,
        "format_error": bool(parsed.format_error),
        "refusal": bool(refusal),
        "truncated": bool(truncated),
    }
