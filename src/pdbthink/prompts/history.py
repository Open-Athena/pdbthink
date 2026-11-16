"""Resolve historical system messages without silently adopting newer instructions."""

from __future__ import annotations

LEGACY_SYSTEM_PROMPT = """You will be given one or more molecular structures and a question about them.
Answer using only the information supplied in the prompt. Do not use tools or
external resources.

Protein residues are identified as <chain>:<one-letter amino-acid code><author
residue number>. For example, A:V22 is valine 22 in chain A. When asked for a
residue, use exactly this format. When asked for a list, return a comma-separated
list. Do not include additional residues.

Place the machine-readable answer after FINAL: using the format requested by
the question. You may reason before the final answer, but only the FINAL field
will be scored."""


def system_prompt_for_version(version: str) -> str:
    from .library import PROMPT_VERSION, SYSTEM_PROMPT

    if version == PROMPT_VERSION:
        return SYSTEM_PROMPT
    if version in {"v1", "v2", "v3", "v4", "v5"}:
        return LEGACY_SYSTEM_PROMPT
    raise ValueError(f"Unknown historical prompt version {version!r}; supply its frozen system_prompt")
