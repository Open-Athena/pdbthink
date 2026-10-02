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
from .scorers import SCORING_VERSION, score_answer, set_scores

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

    ``format_error`` and ``refusal`` are mutually exclusive. Both score zero, but
    they are opposite events: a refusal is a model saying the question cannot be
    answered from what it was given, which on a context-only control is the
    correct response; a format error is a model failing to say anything the
    scorer can read. Counting a refusal under both made a model that declines
    to guess look like one that cannot write an answer.
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
        "scoring_version": SCORING_VERSION,
        "parsed": parsed.as_dict(),
        "score": result,
        "format_error": bool(parsed.format_error and not refusal),
        "refusal": bool(refusal),
        "truncated": bool(truncated),
    }
