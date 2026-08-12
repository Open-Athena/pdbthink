"""Deterministic scorers (specification section 12).

No scorer consults a language model. Every scorer returns a dictionary with a
primary ``score`` in [0, 1] plus the secondary metrics the report needs.
"""

from __future__ import annotations

from typing import Any

from .parse import ParsedAnswer, canonical_pair, canonical_residue

DEFAULT_DISTANCE_TOLERANCE = 0.02      # A.4
DEFAULT_COORDINATE_TOLERANCE = 0.001   # section 8, P03


def set_scores(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    """Set F1 (primary) and exact-set accuracy (strict secondary)."""
    gold_set, pred_set = set(gold), set(predicted)
    true_positive = len(gold_set & pred_set)
    precision = true_positive / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
    recall = true_positive / len(gold_set) if gold_set else (1.0 if not pred_set else 0.0)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    if not gold_set and not pred_set:
        precision = recall = f1 = 1.0
    return {
        "score": f1,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "exact_set": float(gold_set == pred_set),
        "correct": bool(gold_set == pred_set),
        "n_gold": len(gold_set),
        "n_predicted": len(pred_set),
        "missing": sorted(gold_set - pred_set),
        "spurious": sorted(pred_set - gold_set),
    }


def score_answer(
    answer_schema: str,
    gold: dict[str, Any],
    parsed: ParsedAnswer,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one parsed answer against its gold label."""
    parameters = parameters or {}
    if parsed.format_error or parsed.value is None:
        return {
            "score": 0.0,
            "correct": False,
            "format_error": True,
            "error": parsed.error or "unparseable answer",
        }

    predicted = parsed.value
    if answer_schema in ("residue", "atom", "category", "multiple_choice"):
        expected = gold["value"]
        correct = str(predicted) == str(expected)
        return {"score": float(correct), "correct": correct, "predicted": predicted, "gold": expected}

    if answer_schema == "boolean":
        expected = bool(gold["value"])
        correct = bool(predicted) == expected
        return {"score": float(correct), "correct": correct, "predicted": predicted, "gold": expected}

    if answer_schema == "integer":
        expected = int(gold["value"])
        correct = int(predicted) == expected
        return {"score": float(correct), "correct": correct, "predicted": predicted, "gold": expected}

    if answer_schema == "distance":
        tolerance = float(parameters.get("tolerance", DEFAULT_DISTANCE_TOLERANCE))
        expected = float(gold["value"])
        error = abs(float(predicted) - expected)
        correct = error <= tolerance
        return {
            "score": float(correct),
            "correct": correct,
            "absolute_error": error,
            "tolerance": tolerance,
            "predicted": predicted,
            "gold": expected,
        }

    if answer_schema == "numeric_triple":
        tolerance = float(parameters.get("tolerance", DEFAULT_COORDINATE_TOLERANCE))
        expected = [float(v) for v in gold["value"]]
        errors = [abs(float(p) - e) for p, e in zip(predicted, expected)]
        correct = all(e <= tolerance for e in errors)
        return {
            "score": float(correct),
            "correct": correct,
            "component_errors": errors,
            "components_within_tolerance": sum(1 for e in errors if e <= tolerance),
            "tolerance": tolerance,
            "predicted": predicted,
            "gold": expected,
        }

    if answer_schema == "residue_pair":
        expected = canonical_pair(gold["value"])
        correct = predicted == expected
        return {"score": float(correct), "correct": correct, "predicted": predicted, "gold": expected}

    if answer_schema in ("string_set", "residue_set", "residue_pair_set"):
        expected = list(gold["value"])
        out = set_scores(expected, list(predicted))
        out.update({"predicted": sorted(predicted), "gold": sorted(expected)})
        return out

    if answer_schema == "ordered_path":
        expected = [canonical_residue(v) for v in gold["value"]]
        exact = list(predicted) == expected
        positions = sum(
            1 for a, b in zip(predicted, expected) if a == b
        ) / max(len(expected), 1)
        return {
            "score": float(exact),
            "correct": exact,
            "per_position": positions,
            "predicted": list(predicted),
            "gold": expected,
        }

    if answer_schema == "two_interaction_sets":
        gained = set_scores(list(gold.get("gained", [])), list(predicted.get("gained", [])))
        lost = set_scores(list(gold.get("lost", [])), list(predicted.get("lost", [])))
        mean = (gained["score"] + lost["score"]) / 2.0
        return {
            "score": mean,
            "correct": bool(gained["correct"] and lost["correct"]),
            "gained": gained,
            "lost": lost,
            "exact_set": float(gained["correct"] and lost["correct"]),
        }

    if answer_schema == "multi_field":
        return _score_multi_field(gold, predicted, parameters)

    raise ValueError(f"no scorer for answer schema {answer_schema!r}")


def _score_multi_field(
    gold: dict[str, Any], predicted: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    fields = gold.get("fields") or {}
    per_field: dict[str, Any] = {}
    for name, spec in fields.items():
        sub_schema = spec["schema"]
        sub_gold = {k: v for k, v in spec.items() if k != "schema"}
        value = predicted.get(name)
        sub_parsed = ParsedAnswer(value=value, format_error=value is None)
        per_field[name] = score_answer(
            sub_schema, sub_gold, sub_parsed, parameters.get(name, {})
        )
    scores = [v["score"] for v in per_field.values()]
    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "correct": all(v["correct"] for v in per_field.values()) if per_field else False,
        "fields": per_field,
    }
