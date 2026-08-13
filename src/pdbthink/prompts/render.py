"""Assemble model-visible prompts from structures, questions and formats."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..preprocessing.model import Structure
from ..representations.minimal_pdb import render_entity_legend, render_minimal_pdb
from ..representations.table import render_table
from .library import (
    CONTEXT_ONLY_SUBSTITUTIONS,
    CROP_NOTICE,
    FORMAT_INSTRUCTIONS,
    PROMPT_VERSION,
    REPRESENTATION_NOTICE,
    ROTATION_NOTICE,
    SYSTEM_PROMPT,
)


@dataclass
class StructureBlock:
    """One structure as it appears in a prompt."""

    label: str                 # "Structure" or "Structure 1"
    structure: Structure
    cropped: bool = False


@dataclass
class PromptParts:
    system_prompt: str
    user_prompt: str
    prompt_version: str = PROMPT_VERSION
    structure_text: dict[str, str] = field(default_factory=dict)


def render_structure_text(structure: Structure, representation: str) -> str:
    if representation == "minimal_pdb":
        return render_minimal_pdb(structure)
    if representation == "normalized_coordinates":
        return render_table(structure)
    raise ValueError(f"cannot render coordinates for representation {representation!r}")


def build_prompt(
    *,
    representation: str,
    blocks: list[StructureBlock],
    context: str,
    question: str,
    answer_schema: str,
    extra_instructions: str = "",
    format_instructions: str | None = None,
) -> PromptParts:
    """Compose the user prompt for one rendered variant."""
    sections: list[str] = [context.strip()]

    if representation == "context_only":
        sections[0] = CONTEXT_ONLY_SUBSTITUTIONS.get(sections[0], sections[0])
        sections.append(REPRESENTATION_NOTICE["context_only"])
    else:
        notice = REPRESENTATION_NOTICE[representation]
        if any(b.cropped for b in blocks):
            notice = f"{notice}\n{CROP_NOTICE}"
        notice = f"{notice}\n{ROTATION_NOTICE}"
        sections.append(notice)

    structure_text: dict[str, str] = {}
    if representation != "context_only":
        for block in blocks:
            text = render_structure_text(block.structure, representation)
            structure_text[block.label] = text
            legend = render_entity_legend(block.structure)
            header = f"{block.label}:" if block.label else "Structure:"
            piece = f"{header}\n{text.rstrip()}"
            if legend:
                piece += f"\n\nNon-polymer entities in {block.label.lower()}:\n{legend}"
            sections.append(piece)

    sections.append(f"Question: {question.strip()}")
    if extra_instructions.strip():
        sections.append(extra_instructions.strip())
    instructions = (
        format_instructions if format_instructions is not None else FORMAT_INSTRUCTIONS[answer_schema]
    )
    sections.append(instructions.strip())

    return PromptParts(
        system_prompt=SYSTEM_PROMPT,
        user_prompt="\n\n".join(s for s in sections if s.strip()) + "\n",
        structure_text=structure_text,
    )
