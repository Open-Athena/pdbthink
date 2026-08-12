"""Parsing and canonicalisation of model answers (specification section 7).

Only the ``FINAL`` field is scored. The parser is deliberately forgiving about
harmless surface variation (whitespace, markdown emphasis, trailing full stops,
case) and deliberately strict about content: an answer that names extra items,
uses the wrong separator or omits the field scores zero and is reported as a
format error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Only horizontal whitespace is allowed around the marker: a class containing
# `\s` would swallow the newline of a bare ``FINAL`` line and mistake the first
# field of a multi-field answer for a single-line answer.
FINAL_PATTERN = re.compile(
    r"(?im)^[^\S\n]*[*_`#>]*[^\S\n]*final\b[*_`]*[^\S\n]*(:)?[*_`]*[^\S\n]*(.*)$"
)
FIELD_PATTERN = re.compile(r"^[^\S\n]*[*_`]*([A-Za-z][A-Za-z0-9_ \-]*?)[*_`]*[^\S\n]*:[^\S\n]*(.*)$")
RESIDUE_PATTERN = re.compile(r"^([A-Za-z0-9])[:\s]+([A-Za-z])[\s]*(-?\d+)$")
ATOM_PATTERN = re.compile(r"^([A-Za-z0-9])[:\s]+([A-Za-z])[\s]*(-?\d+)[:\s]+([A-Za-z0-9']{1,4})$")
# Non-greedy component name so ``M:ZN501`` parses as ZN/501 and ``L:L2401`` as
# L/2401 -- either way the parse is deterministic and round-trips the renderer.
ENTITY_PATTERN = re.compile(r"^([A-Za-z0-9])[:\s]+([A-Za-z][A-Za-z0-9]{0,2}?)[\s]*(-?\d+)$")
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

PAIR_SEPARATOR = "--"
PATH_SEPARATOR = "->"

REFUSAL_MARKERS = (
    "i cannot", "i can't", "i am unable", "i'm unable", "cannot answer",
    "as an ai", "i do not have access", "i don't have access", "unable to determine",
    "insufficient information to answer",
)


class AnswerFormatError(ValueError):
    """The model output did not contain a parseable FINAL answer."""


@dataclass
class ParsedAnswer:
    """Result of parsing one model response."""

    raw_final: str = ""
    fields: dict[str, str] = field(default_factory=dict)
    value: Any = None
    format_error: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_final": self.raw_final,
            "fields": self.fields,
            "value": self.value,
            "format_error": self.format_error,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Locating the FINAL block
# --------------------------------------------------------------------------- #

def strip_markup(text: str) -> str:
    out = text.strip()
    out = out.strip("`")
    out = re.sub(r"\*\*(.*?)\*\*", r"\1", out)
    out = re.sub(r"^[*_`\s]+|[*_`\s]+$", "", out)
    out = out.rstrip(".")
    return out.strip()


def extract_final(text: str) -> tuple[str, dict[str, str]]:
    """Return ``(inline_value, fields)`` for the last FINAL block in ``text``.

    A single-line answer yields a non-empty ``inline_value``. A multi-field
    answer yields ``fields`` keyed by lowercase field name.
    """
    if not text:
        raise AnswerFormatError("empty response")
    matches = list(FINAL_PATTERN.finditer(text))
    if not matches:
        raise AnswerFormatError("no FINAL field found")
    match = matches[-1]
    inline = strip_markup(match.group(2) or "")
    lines = text[match.end():].splitlines()

    # Continuation lines for long single-line lists that wrapped. The split
    # leaves an empty first element for the newline itself, so skip it.
    if inline and inline.endswith(","):
        continuation = list(lines)
        while continuation and not continuation[0].strip():
            continuation.pop(0)
        for line in continuation:
            candidate = strip_markup(line)
            if not candidate:
                break
            inline = f"{inline} {candidate}".strip()
            if not inline.endswith(","):
                break

    fields: dict[str, str] = {}
    if not inline:
        for line in lines:
            if not line.strip():
                if fields:
                    break
                continue
            field_match = FIELD_PATTERN.match(line)
            if not field_match:
                break
            key = field_match.group(1).strip().lower().replace(" ", "_")
            fields[key] = strip_markup(field_match.group(2))
    if not inline and not fields:
        raise AnswerFormatError("FINAL field is empty")
    return inline, fields


def looks_like_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


# --------------------------------------------------------------------------- #
# Canonicalisation of individual tokens
# --------------------------------------------------------------------------- #

def canonical_residue(token: str) -> str:
    """``a: v 22`` -> ``A:V22``; raises for anything that is not a residue id.

    Chain identifiers are upper-cased. The dataset builder rejects any rendering
    whose chains differ only by case, so this cannot merge distinct chains.
    """
    text = strip_markup(token).replace(" ", "")
    match = RESIDUE_PATTERN.match(text)
    if match:
        return f"{match.group(1).upper()}:{match.group(2).upper()}{int(match.group(3))}"
    entity = ENTITY_PATTERN.match(text)
    if entity:
        return f"{entity.group(1).upper()}:{entity.group(2).upper()}{int(entity.group(3))}"
    raise AnswerFormatError(f"not a residue identifier: {token!r}")


def canonical_atom(token: str) -> str:
    text = strip_markup(token).replace(" ", "")
    match = ATOM_PATTERN.match(text)
    if not match:
        raise AnswerFormatError(f"not an atom identifier: {token!r}")
    return (
        f"{match.group(1).upper()}:{match.group(2).upper()}{int(match.group(3))}"
        f":{match.group(4).upper()}"
    )


def canonical_pair(token: str) -> str:
    """Unordered residue pair, canonicalised to sorted ``A--B`` form."""
    text = strip_markup(token)
    parts = [p for p in re.split(r"--+|\s+--\s+|—|–", text) if p.strip()]
    if len(parts) != 2:
        raise AnswerFormatError(f"not a residue pair: {token!r}")
    a, b = (canonical_residue(parts[0]), canonical_residue(parts[1]))
    first, second = sorted((a, b))
    return f"{first}{PAIR_SEPARATOR}{second}"


def split_list(text: str) -> list[str]:
    """Split a comma or semicolon separated list, tolerating ``and``/newlines."""
    cleaned = strip_markup(text)
    if not cleaned or cleaned.lower() in ("none", "no residues", "empty", "n/a", "-"):
        return []
    cleaned = re.sub(r"\band\b", ",", cleaned, flags=re.IGNORECASE)
    parts = re.split(r"[,;\n]+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def canonical_residue_set(text: str) -> list[str]:
    return sorted({canonical_residue(p) for p in split_list(text)})


def canonical_pair_set(text: str) -> list[str]:
    return sorted({canonical_pair(p) for p in split_list(text)})


def canonical_string_set(text: str) -> list[str]:
    return sorted({strip_markup(p).upper() for p in split_list(text) if strip_markup(p)})


def canonical_category(text: str, allowed: list[str] | None = None) -> str:
    cleaned = re.sub(r"\s+", " ", strip_markup(text)).strip().lower()
    if allowed is None:
        return cleaned
    lookup = {a.lower(): a for a in allowed}
    if cleaned in lookup:
        return lookup[cleaned]
    # Tolerate a model that answers with the option in a longer sentence.
    for option in sorted(allowed, key=len, reverse=True):
        if re.search(rf"\b{re.escape(option.lower())}\b", cleaned):
            return option
    raise AnswerFormatError(f"category {text!r} is not one of {allowed}")


def canonical_boolean(text: str) -> bool:
    cleaned = strip_markup(text).lower()
    if cleaned in ("yes", "y", "true", "t"):
        return True
    if cleaned in ("no", "n", "false", "f"):
        return False
    raise AnswerFormatError(f"not a boolean: {text!r}")


def canonical_choice(text: str, options: list[str] | None = None) -> str:
    cleaned = strip_markup(text).upper()
    match = re.match(r"^\(?([A-Z])\)?\b", cleaned)
    if not match:
        raise AnswerFormatError(f"not a multiple-choice letter: {text!r}")
    letter = match.group(1)
    if options and letter not in [o.upper() for o in options]:
        raise AnswerFormatError(f"option {letter!r} is not among {options}")
    return letter


def canonical_number(text: str) -> float:
    cleaned = strip_markup(text).replace("Å", "").replace("angstrom", "").strip()
    match = NUMBER_PATTERN.search(cleaned)
    if not match:
        raise AnswerFormatError(f"no number found in {text!r}")
    return float(match.group(0))


def canonical_integer(text: str) -> int:
    value = canonical_number(text)
    if abs(value - round(value)) > 1e-9:
        raise AnswerFormatError(f"not an integer: {text!r}")
    return int(round(value))


def canonical_triple(text: str) -> list[float]:
    cleaned = strip_markup(text).replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    numbers = NUMBER_PATTERN.findall(cleaned)
    if len(numbers) != 3:
        raise AnswerFormatError(f"expected three numbers, found {len(numbers)} in {text!r}")
    return [float(n) for n in numbers]


def canonical_path(text: str) -> list[str]:
    cleaned = strip_markup(text)
    parts = [p for p in re.split(r"->|→", cleaned) if p.strip()]
    if len(parts) < 2:
        raise AnswerFormatError(f"not an ordered path: {text!r}")
    return [canonical_residue(p) for p in parts]


# --------------------------------------------------------------------------- #
# Schema dispatch
# --------------------------------------------------------------------------- #

def parse_answer(text: str, answer_schema: str, parameters: dict[str, Any] | None = None) -> ParsedAnswer:
    """Parse a raw model response into the canonical form for ``answer_schema``."""
    parameters = parameters or {}
    try:
        inline, fields = extract_final(text)
    except AnswerFormatError as exc:
        return ParsedAnswer(format_error=True, error=str(exc))

    parsed = ParsedAnswer(raw_final=inline or _fields_repr(fields), fields=fields)
    try:
        parsed.value = _dispatch(answer_schema, inline, fields, parameters)
    except AnswerFormatError as exc:
        parsed.format_error = True
        parsed.error = str(exc)
    return parsed


def _fields_repr(fields: dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in fields.items())


def _dispatch(
    schema: str, inline: str, fields: dict[str, str], parameters: dict[str, Any]
) -> Any:
    if schema == "string_set":
        return canonical_string_set(inline)
    if schema == "integer":
        return canonical_integer(inline)
    if schema == "numeric_triple":
        return canonical_triple(inline)
    if schema == "distance":
        return canonical_number(inline)
    if schema == "atom":
        return canonical_atom(inline)
    if schema == "residue":
        return canonical_residue(inline)
    if schema == "residue_set":
        return canonical_residue_set(inline)
    if schema == "residue_pair":
        return canonical_pair(inline)
    if schema == "residue_pair_set":
        return canonical_pair_set(inline)
    if schema == "category":
        return canonical_category(inline, parameters.get("categories"))
    if schema == "boolean":
        return canonical_boolean(inline)
    if schema == "multiple_choice":
        return canonical_choice(inline, parameters.get("options"))
    if schema == "ordered_path":
        return canonical_path(inline)
    if schema == "two_interaction_sets":
        return _parse_two_sets(inline, fields)
    if schema == "multi_field":
        return _parse_multi_field(fields, parameters)
    raise AnswerFormatError(f"unknown answer schema {schema!r}")


def _parse_two_sets(inline: str, fields: dict[str, str]) -> dict[str, list[str]]:
    source = dict(fields)
    if inline and not source:
        # Tolerate "gained: ...; lost: ..." on one line.
        for chunk in re.split(r";|\|", inline):
            m = FIELD_PATTERN.match(chunk.strip())
            if m:
                source[m.group(1).strip().lower()] = m.group(2)
    missing = [k for k in ("gained", "lost") if k not in source]
    if missing:
        raise AnswerFormatError(f"missing required field(s) {missing}")
    return {
        "gained": canonical_pair_set(source["gained"]),
        "lost": canonical_pair_set(source["lost"]),
    }


def _parse_multi_field(fields: dict[str, str], parameters: dict[str, Any]) -> dict[str, Any]:
    spec: dict[str, str] = parameters.get("field_schemas") or {}
    if not spec:
        raise AnswerFormatError("multi_field answers require a field_schemas parameter")
    missing = [name for name in spec if name not in fields]
    if missing:
        raise AnswerFormatError(f"missing required field(s) {missing}")
    out: dict[str, Any] = {}
    errors: list[str] = []
    for name, sub_schema in spec.items():
        try:
            out[name] = _dispatch(sub_schema, fields[name], {}, parameters.get(name, {}))
        except AnswerFormatError as exc:
            out[name] = None
            errors.append(f"{name}: {exc}")
    if errors and len(errors) == len(spec):
        raise AnswerFormatError("; ".join(errors))
    return out
