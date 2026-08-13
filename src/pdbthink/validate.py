"""Dataset validation (specification section 16, "Definition of done").

``pdbthink validate`` fails with a nonzero exit code on any schema violation or
inconsistent gold label. Every check below corresponds to a bullet in the
definition of done.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canary import CANARY_GUID, is_canary
from .config import DatasetConfig
from .dataset import load_dataset
from .representations.table import HEADER
from .schemas import RenderedVariant, SemanticInstance
from .scoring import score_response
from .util import gold_hash, read_jsonl

COORDINATE_DEPENDENT_SCHEMAS = ("numeric_triple",)
FINAL_RANGE = (90, 110)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def validate_dataset(
    directory: str | Path,
    *,
    config: DatasetConfig | None = None,
    require_final_size: bool = False,
    require_reviewed: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    instances, renders = load_dataset(directory)
    by_id = {i.semantic_instance_id: i for i in instances}

    report.stats["n_instances"] = len(instances)
    report.stats["n_renders"] = len(renders)
    report.stats["realised_counts"] = dict(
        sorted(Counter(i.question_family for i in instances).items())
    )
    report.stats["instances_per_protein"] = dict(
        sorted(Counter(i.protein_group_id for i in instances).items())
    )

    if not instances:
        report.error("dataset contains no semantic instances")
        return report

    _check_canary_and_gold_hashes(directory, instances, renders, report)
    _check_provenance(instances, report, require_reviewed=require_reviewed)
    _check_composition(instances, report, config, require_final_size=require_final_size)
    _check_renders(instances, by_id, renders, report, config)
    _check_gold_scores_itself(instances, by_id, renders, report)
    _check_no_provenance_leak(instances, by_id, renders, report)
    return report


# --------------------------------------------------------------------------- #

def _check_canary_and_gold_hashes(directory, instances, renders, report) -> None:
    """Every manifest carries the canary, and every gold answer matches its hash.

    The hash is what makes a withheld answer verifiable: a local rebuild can be
    proven identical to the canonical one without the answer ever being
    published (docs/contamination.md).
    """
    directory = Path(directory)
    for name in ("instances.jsonl", "renders.jsonl", "rejections.jsonl"):
        rows = list(read_jsonl(directory / name))
        if not rows:
            continue
        if not is_canary(rows[0]):
            report.error(f"{name}: missing the canary header record")
        elif rows[0]["_canary"] != CANARY_GUID:
            report.error(f"{name}: canary is {rows[0]['_canary']!r}, expected {CANARY_GUID!r}")

    committed = {
        row["semantic_instance_id"]: row.get("gold_sha256", "")
        for row in read_jsonl(directory / "instances.jsonl")
        if not is_canary(row)
    }
    mismatched = 0
    for instance in instances:
        expected = committed.get(instance.semantic_instance_id)
        if not expected:
            continue
        actual = gold_hash(instance.gold_answer)
        if actual != expected:
            mismatched += 1
            report.error(
                f"{instance.semantic_instance_id}: gold answer does not match the committed "
                f"hash ({actual[:12]} vs {expected[:12]}); this build differs from the "
                "canonical one"
            )
    report.stats["gold_hash_mismatches"] = mismatched
    report.stats["gold_hashes_checked"] = len(committed)


def _check_provenance(
    instances: list[SemanticInstance], report: ValidationReport, *, require_reviewed: bool
) -> None:
    for instance in instances:
        tag = instance.semantic_instance_id
        if not instance.source_entries or not instance.source_file_sha256s:
            report.error(f"{tag}: missing source provenance")
        if len(instance.source_entries) != len(instance.source_file_sha256s):
            report.error(f"{tag}: source entries and file hashes disagree in length")
        if not instance.gold_evidence:
            report.error(f"{tag}: missing hidden gold evidence")
        if not instance.selection_margins and not instance.is_mechanistic:
            report.error(f"{tag}: missing selection margins")
        if not instance.definition_version:
            report.error(f"{tag}: missing definition_version")
        if require_reviewed and instance.curation_status != "accepted":
            report.error(f"{tag}: curation_status is {instance.curation_status}, expected accepted")
        if instance.curator_override is not None and not instance.curator_notes.strip():
            report.warn(f"{tag}: label override without curator notes")


def _check_composition(
    instances: list[SemanticInstance],
    report: ValidationReport,
    config: DatasetConfig | None,
    *,
    require_final_size: bool,
) -> None:
    counts = Counter(i.question_family for i in instances)
    if config is not None:
        for family, target in config.family_targets.items():
            realised = counts.get(family, 0)
            if realised < target:
                report.warn(
                    f"family {family}: {realised} instances realised against a target of {target}"
                )
        per_protein = Counter(i.protein_group_id for i in instances)
        automatic = Counter(
            i.protein_group_id for i in instances if not i.is_mechanistic
        )
        for protein, n in automatic.items():
            if n > config.max_instances_per_protein:
                report.error(
                    f"protein {protein} contributes {n} automatic instances, "
                    f"above the cap of {config.max_instances_per_protein}"
                )
        report.stats["max_instances_per_protein_seen"] = max(per_protein.values(), default=0)

    if require_final_size:
        low, high = FINAL_RANGE
        if not low <= len(instances) <= high:
            report.error(
                f"final manifest has {len(instances)} semantic instances, outside {low}-{high}"
            )


def _check_renders(
    instances: list[SemanticInstance],
    by_id: dict[str, SemanticInstance],
    renders: list[RenderedVariant],
    report: ValidationReport,
    config: DatasetConfig | None,
) -> None:
    grouped: dict[str, list[RenderedVariant]] = defaultdict(list)
    for render in renders:
        if render.semantic_instance_id not in by_id:
            report.error(f"{render.render_id}: references unknown semantic instance")
            continue
        grouped[render.semantic_instance_id].append(render)

    for instance in instances:
        variants = grouped.get(instance.semantic_instance_id, [])
        if not variants:
            report.error(f"{instance.semantic_instance_id}: has no rendered variant")
            continue

        budget = (
            (config.token_budget_mechanistic if instance.is_mechanistic else config.token_budget_automatic)
            if config
            else None
        )
        for render in variants:
            if not render.user_prompt.strip():
                report.error(f"{render.render_id}: empty user prompt")
            if budget and (render.input_token_count or 0) > budget:
                report.error(
                    f"{render.render_id}: {render.input_token_count} tokens exceeds the "
                    f"{budget}-token budget"
                )
            if render.representation == "normalized_coordinates" and HEADER not in render.user_prompt:
                report.error(f"{render.render_id}: normalized table is missing its header")
            if render.representation == "minimal_pdb" and "\nEND" not in render.user_prompt:
                report.error(f"{render.render_id}: minimal PDB rendering is missing END")
            _check_forbidden_records(render, report)

        # Paired representations must agree on gold and on the displayed atoms.
        by_key: dict[tuple[int, int | None], list[RenderedVariant]] = defaultdict(list)
        for render in variants:
            by_key[(render.rotation_seed, render.state_order_seed)].append(render)
        for key, group in by_key.items():
            coordinate_variants = [r for r in group if r.representation != "context_only"]
            golds = {_gold_key(r.gold_answer) for r in coordinate_variants}
            if len(golds) > 1:
                report.error(
                    f"{instance.semantic_instance_id}: paired representations at seed {key} "
                    "have different gold answers"
                )
            atom_counts = {r.atom_count for r in coordinate_variants}
            if len(atom_counts) > 1:
                report.error(
                    f"{instance.semantic_instance_id}: paired representations at seed {key} "
                    f"show different atom counts {sorted(atom_counts)}"
                )

        if instance.answer_schema not in COORDINATE_DEPENDENT_SCHEMAS:
            golds = {
                _gold_key(r.gold_answer)
                for r in variants
                if r.representation != "context_only" and r.state_order_seed in (None, 0)
            }
            if len(golds) > 1:
                report.error(
                    f"{instance.semantic_instance_id}: gold answer changes under rotation"
                )


FORBIDDEN_RECORDS = ("HELIX ", "SHEET ", "SSBOND", "LINK  ", "ANISOU", "HEADER", "TITLE ", "REMARK")


def _check_forbidden_records(render: RenderedVariant, report: ValidationReport) -> None:
    for record in FORBIDDEN_RECORDS:
        if record in render.user_prompt:
            report.error(f"{render.render_id}: model-visible text contains a {record.strip()} record")


def _gold_key(gold: dict[str, Any]) -> str:
    import json

    return json.dumps(gold, sort_keys=True, default=str)


def _check_gold_scores_itself(
    instances: list[SemanticInstance],
    by_id: dict[str, SemanticInstance],
    renders: list[RenderedVariant],
    report: ValidationReport,
) -> None:
    """A perfectly formatted gold answer must score 1.0 under the real scorer."""
    for render in renders:
        instance = by_id.get(render.semantic_instance_id)
        if instance is None:
            continue
        text = format_gold_answer(render.answer_schema, render.gold_answer)
        outcome = score_response(
            text,
            render.answer_schema,
            render.gold_answer,
            parameters=_scoring_parameters(instance),
        )
        if outcome["format_error"]:
            report.error(
                f"{render.render_id}: the gold answer does not parse "
                f"({outcome['parsed'].get('error')})"
            )
        elif outcome["score"]["score"] < 1.0:
            report.error(
                f"{render.render_id}: gold answer scores {outcome['score']['score']:.3f}, not 1.0"
            )


def _scoring_parameters(instance: SemanticInstance) -> dict[str, Any]:
    from .generators import get_generator

    parameters = dict(instance.question_parameters)
    if instance.answer_schema == "multi_field":
        return parameters
    try:
        generator = get_generator(instance.question_family)
    except KeyError:
        return parameters
    return {**parameters, **generator.prompt_parameters(instance.question_parameters)}


def format_gold_answer(answer_schema: str, gold: dict[str, Any]) -> str:
    """Render a gold label exactly as a compliant model would report it."""
    if answer_schema == "two_interaction_sets":
        gained = ", ".join(gold.get("gained", [])) or "none"
        lost = ", ".join(gold.get("lost", [])) or "none"
        return f"FINAL\ngained: {gained}\nlost: {lost}\n"
    if answer_schema == "multi_field":
        lines = ["FINAL"]
        for name, spec in gold["fields"].items():
            lines.append(f"{name}: {_format_value(spec['schema'], spec)}")
        return "\n".join(lines) + "\n"
    return f"FINAL: {_format_value(answer_schema, gold)}\n"


def _format_value(schema: str, gold: dict[str, Any]) -> str:
    value = gold["value"]
    if schema in ("string_set", "residue_set", "residue_pair_set"):
        return ", ".join(value) if value else "none"
    if schema == "numeric_triple":
        return ", ".join(f"{v:.3f}" for v in value)
    if schema == "distance":
        return f"{float(value):.3f}"
    if schema == "boolean":
        return "yes" if value else "no"
    if schema == "ordered_path":
        return " -> ".join(value)
    return str(value)


ENTRY_PATTERN = re.compile(r"\b[1-9][A-Za-z0-9]{3}\b")


def _check_no_provenance_leak(
    instances: list[SemanticInstance],
    by_id: dict[str, SemanticInstance],
    renders: list[RenderedVariant],
    report: ValidationReport,
) -> None:
    """Source identifiers and annotations must never reach a model-visible prompt."""
    for render in renders:
        instance = by_id.get(render.semantic_instance_id)
        if instance is None:
            continue
        prompt = render.user_prompt
        for entry in instance.source_entries:
            if re.search(rf"\b{re.escape(entry)}\b", prompt, flags=re.IGNORECASE):
                report.error(f"{render.render_id}: prompt mentions source entry {entry}")
        for publication in instance.source_publications:
            token = publication.split(":", 1)[-1]
            if len(token) > 4 and token in prompt:
                report.error(f"{render.render_id}: prompt mentions publication {publication}")
        original_components = (instance.gold_evidence or {}).get("ligand_map") or {}
        for _label, original in original_components.items():
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(original)}(?![A-Za-z0-9])", prompt):
                report.error(
                    f"{render.render_id}: prompt exposes original component code {original}"
                )
