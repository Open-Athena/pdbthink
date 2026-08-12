"""On-disk record schemas (specification section 5).

Every record written by the pipeline is validated through these models, so a
schema violation is a hard error with a nonzero exit code rather than a silently
malformed dataset.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SourceType = Literal["pdb", "afdb"]
Representation = Literal["minimal_pdb", "normalized_coordinates", "context_only"]
CurationStatus = Literal["proposed", "accepted", "rejected"]

#: Answer schemas understood by the scorer. Each maps to one scorer in
#: :mod:`pdbthink.scoring.scorers`.
ANSWER_SCHEMAS = (
    "string_set",          # unordered set of bare strings (chain IDs)
    "integer",             # exact integer
    "numeric_triple",      # x, y, z with per-component tolerance
    "distance",            # single number with absolute tolerance
    "atom",                # A:H57:NE2
    "residue",             # A:V22
    "residue_set",         # unordered set of residues, scored by set F1
    "residue_pair",        # A:C24--B:C81 (unordered within the pair)
    "residue_pair_set",    # unordered set of unordered pairs
    "category",            # controlled vocabulary token
    "boolean",             # yes / no
    "multiple_choice",     # A / B / C / D
    "ordered_path",        # A:R10 -> A:F42 -> B:E77
    "two_interaction_sets",  # gained: ... / lost: ...
    "multi_field",         # mechanistic episode: named fields, scored separately
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SemanticInstance(StrictModel):
    """One underlying scientific prompt (specification section 5)."""

    semantic_instance_id: str
    question_family: str
    question_version: str
    protein_group_id: str
    source_type: SourceType
    source_entries: list[str]
    source_file_sha256s: list[str]
    release_dates: list[str] = Field(default_factory=list)
    source_publications: list[str] = Field(default_factory=list)
    biological_assembly_ids: list[str] = Field(default_factory=list)
    selected_chains: list[str] = Field(default_factory=list)
    question_parameters: dict[str, Any] = Field(default_factory=dict)
    answer_schema: str
    gold_answer: dict[str, Any]
    gold_evidence: dict[str, Any] = Field(default_factory=dict)
    selection_margins: dict[str, Any] = Field(default_factory=dict)
    definition_version: str
    curation_status: CurationStatus = "proposed"
    curator_notes: str = ""

    # Bookkeeping beyond the minimum required by the specification.
    experimental_method: str | None = None
    resolution: float | None = None
    criteria_passed: list[str] = Field(default_factory=list)
    criteria_failed: list[str] = Field(default_factory=list)
    acceptance_reasons: list[str] = Field(default_factory=list)
    is_mechanistic: bool = False
    curator_override: dict[str, Any] | None = None

    @field_validator("answer_schema")
    @classmethod
    def _known_schema(cls, v: str) -> str:
        if v not in ANSWER_SCHEMAS:
            raise ValueError(f"unknown answer_schema {v!r}; expected one of {ANSWER_SCHEMAS}")
        return v

    @model_validator(mode="after")
    def _gold_not_empty(self) -> SemanticInstance:
        if not self.gold_answer:
            raise ValueError(f"{self.semantic_instance_id}: gold_answer must not be empty")
        return self


class RenderedVariant(StrictModel):
    """One model-visible rendering of a semantic instance (section 5)."""

    render_id: str
    semantic_instance_id: str
    representation: Representation
    rotation_seed: int
    state_order_seed: int | None = None
    prompt_version: str
    system_prompt: str
    user_prompt: str
    rotation_matrix: list[list[float]]
    translation_vector: list[float]
    displayed_coordinates_sha256: str
    input_token_count: int | None = None

    # Bookkeeping.
    question_family: str
    protein_group_id: str
    answer_schema: str
    gold_answer: dict[str, Any]
    is_rotation_variant: bool = False
    state_order: list[str] = Field(default_factory=list)
    crop: dict[str, Any] | None = None
    atom_count: int = 0
    tokenizer: str = ""

    @field_validator("rotation_matrix")
    @classmethod
    def _shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(r) != 3 for r in v):
            raise ValueError("rotation_matrix must be 3x3")
        return v

    @field_validator("translation_vector")
    @classmethod
    def _vec3(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("translation_vector must have length 3")
        return v


class EvaluationResult(StrictModel):
    """One model completion for one rendered variant (section 5)."""

    run_id: str
    render_id: str
    completion_index: int
    model_provider: str
    model_id: str
    model_revision: str | None = None
    reasoning_effort: str | None = None
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)
    max_output_tokens: int
    raw_response: str
    parsed_answer: dict[str, Any] | None = None
    score: dict[str, Any] = Field(default_factory=dict)
    format_error: bool = False
    refusal: bool = False
    truncated: bool = False
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_seconds: float | None = None
    error: str | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.render_id, self.completion_index)


class ReviewDecision(StrictModel):
    """A curator decision exported by the review interface (section 10)."""

    semantic_instance_id: str
    decision: Literal["accept", "reject"]
    reason: str = ""
    notes: str = ""
    label_override: dict[str, Any] | None = None
    curator: str = ""
    timestamp: str = ""

    @model_validator(mode="after")
    def _override_requires_reason(self) -> ReviewDecision:
        if self.label_override is not None and not self.reason.strip():
            raise ValueError(
                f"{self.semantic_instance_id}: a label override requires a recorded reason"
            )
        return self


class RejectionRecord(StrictModel):
    """Machine-readable record of why a candidate was not produced (A.34)."""

    candidate_id: str
    question_family: str
    protein_group_id: str
    reason: str
    detail: dict[str, Any] = Field(default_factory=dict)
    criteria_failed: list[str] = Field(default_factory=list)
