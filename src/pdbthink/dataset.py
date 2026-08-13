"""Dataset construction: proposals -> semantic instances -> rendered variants.

The builder is deterministic given the dataset configuration and its seed. It
enforces the composition rules of section 9 (per-family targets, per-protein
caps, one protein per family where possible), the input budget and cropping
rules of section 6, and the curator gate of section 10.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .acquisition.cache import AcquisitionError, StructureCache
from .canary import canary_record, is_canary
from .canary import check as check_canary
from .config import DatasetConfig, Definitions, ProteinSpec
from .generators import (
    T01,
    V1_FAMILIES,
    Analysis,
    GenerationContext,
    Generator,
    Proposal,
    Rejection,
    build_two_state_context,
    get_generator,
)
from .preprocessing.crop import CropError, CropInfo, crop_around, crop_like
from .preprocessing.loader import ProcessedStructure, StructureRejected, load_processed
from .preprocessing.model import Structure
from .preprocessing.transform import (
    DisplayedStructure,
    TransformError,
    build_transform,
    display,
)
from .prompts.library import PROMPT_VERSION, prompt_fingerprint
from .prompts.render import StructureBlock, build_prompt
from .representations.tokens import count_tokens
from .schemas import RejectionRecord, RenderedVariant, SemanticInstance
from .util import (
    derive_seed,
    gold_hash,
    read_jsonl,
    rng_for,
    sha256_text,
    stable_hash,
    write_json,
    write_jsonl,
)

REPRESENTATIONS = ("minimal_pdb", "normalized_coordinates")
COORDINATE_DEPENDENT_SCHEMAS = ("numeric_triple",)


@dataclass
class Candidate:
    """A proposal promoted to a concrete, renderable instance."""

    family: str
    generator: Any
    spec: ProteinSpec
    processed: ProcessedStructure
    proposal: Proposal
    base_structure: Structure
    crop: CropInfo | None = None
    second_structure: Structure | None = None
    second_processed: ProcessedStructure | None = None
    state_labels: tuple[str, str] = ("Structure 1", "Structure 2")
    notes: list[str] = field(default_factory=list)

    @property
    def is_two_state(self) -> bool:
        return self.second_structure is not None


@dataclass
class BuildResult:
    instances: list[SemanticInstance] = field(default_factory=list)
    renders: list[RenderedVariant] = field(default_factory=list)
    rejections: list[RejectionRecord] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def realised_counts(self) -> dict[str, int]:
        return dict(Counter(i.question_family for i in self.instances))


class DatasetBuilder:
    """Builds a candidate or final dataset from a versioned configuration."""

    def __init__(
        self,
        config: DatasetConfig,
        definitions: Definitions,
        cache: StructureCache,
        *,
        decisions: dict[str, Any] | None = None,
        accepted_only: bool = False,
        max_rejection_examples: int = 8,
    ) -> None:
        self.config = config
        self.definitions = definitions
        self.cache = cache
        self.decisions = decisions or {}
        self.accepted_only = accepted_only
        self.max_rejection_examples = max_rejection_examples
        self._processed: dict[tuple[str, str], ProcessedStructure] = {}
        self._frames: dict[str, tuple[DisplayedStructure, Analysis]] = {}
        self._rejection_counts: Counter = Counter()
        self._afdb_min_plddt = float(definitions.get("afdb.min_plddt"))
        self._afdb_radius = float(definitions.get("afdb.neighbourhood_radius"))
        self._afdb_ineligible = set(definitions.get("afdb.ineligible_families"))

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def build(self, families: Iterable[str] | None = None) -> BuildResult:
        result = BuildResult()
        families = list(families) if families is not None else list(V1_FAMILIES)
        per_protein: Counter = Counter()

        for family in families:
            if family == "T01":
                candidates = self._collect_two_state(family, result)
            else:
                candidates = self._collect_single(family, result)
            target = self.config.family_targets.get(family, 0)
            # Over-select so that a candidate which fails to render (token budget,
            # crop, or an oracle that no longer finds a unique answer) is replaced
            # rather than leaving the family short of its target.
            selected = self._select(candidates, target)
            produced = 0
            for candidate in selected:
                if produced >= target:
                    break
                cluster = candidate.spec.cluster or candidate.spec.id
                if per_protein[cluster] >= self.config.max_instances_per_protein:
                    continue
                try:
                    instance, renders = self._materialise(candidate)
                except (BuildRejection, TransformError, CropError, ValueError) as exc:
                    self._record_rejection(
                        result,
                        candidate.family,
                        candidate.spec.id,
                        reason=getattr(exc, "reason", "materialisation_failed"),
                        detail={"error": str(exc), "parameters": candidate.proposal.parameters},
                        criteria=getattr(exc, "criteria", []),
                    )
                    continue
                per_protein[cluster] += 1
                produced += 1
                result.instances.append(instance)
                result.renders.extend(renders)

        if self.config.episodes:
            self._build_episodes(result)

        result.instances = [i for i in result.instances if self._keep(i)]
        keep_ids = {i.semantic_instance_id for i in result.instances}
        result.renders = [r for r in result.renders if r.semantic_instance_id in keep_ids]
        result.stats = self._summarise(result, per_protein)
        return result

    # ------------------------------------------------------------------ #
    # collection
    # ------------------------------------------------------------------ #
    def _collect_single(self, family: str, result: BuildResult) -> list[Candidate]:
        generator = get_generator(family)
        out: list[Candidate] = []
        for spec in self.config.proteins_for(family):
            try:
                processed = self._load(spec)
            except (StructureRejected, AcquisitionError) as exc:
                self._record_rejection(
                    result,
                    family,
                    spec.id,
                    reason=getattr(exc, "reason", "acquisition_failed"),
                    detail=getattr(exc, "detail", {"error": str(exc)}),
                    criteria=["missing_required_atoms"],
                )
                continue
            if spec.source_type == "afdb" and family in self._afdb_ineligible:
                self._record_rejection(
                    result,
                    family,
                    spec.id,
                    "afdb_ineligible_for_family",
                    {"family": family},
                    ["uncertain_biological_assembly"],
                )
                continue
            displayed, analysis = self._proposal_frame(spec, processed)
            ctx = GenerationContext(
                spec=spec,
                processed=processed,
                displayed=displayed,
                definitions=self.definitions,
                rng=rng_for(self.config.seed, family, spec.id),
                analysis=analysis,
            )
            proposals: list[Proposal] = []
            for item in generator.propose(ctx):
                if isinstance(item, Rejection):
                    self._record_rejection(
                        result, family, spec.id, item.reason, item.detail, item.criteria_failed
                    )
                    continue
                if spec.source_type == "afdb":
                    low = self._low_confidence_labels(processed, displayed.structure, item)
                    if low:
                        self._record_rejection(
                            result,
                            family,
                            spec.id,
                            "afdb_low_plddt_near_query",
                            {"residues": low[:8], "min_plddt": self._afdb_min_plddt},
                            ["inadequate_local_quality"],
                        )
                        continue
                proposals.append(item)
            proposals.sort(key=lambda p: (p.rank, p.key()))
            for proposal in self._diversify(proposals):
                out.append(
                    Candidate(
                        family=family,
                        generator=generator,
                        spec=spec,
                        processed=processed,
                        proposal=proposal,
                        base_structure=processed.structure,
                    )
                )
        return out

    def _collect_two_state(self, family: str, result: BuildResult) -> list[Candidate]:
        out: list[Candidate] = []
        for pair in self.config.state_pairs:
            try:
                processed1 = self._load(pair.state1)
                processed2 = self._load(pair.state2)
                ctx = build_two_state_context(
                    pair, processed1, processed2, self.definitions
                )
            except (StructureRejected, AcquisitionError, ValueError) as exc:
                self._record_rejection(
                    result,
                    family,
                    pair.id,
                    reason=getattr(exc, "reason", "state_pair_failed"),
                    detail=getattr(exc, "detail", {"error": str(exc)}),
                    criteria=["uncertain_state_mapping"],
                )
                continue
            for item in T01.propose(ctx):
                if isinstance(item, Rejection):
                    self._record_rejection(
                        result, family, pair.id, item.reason, item.detail, item.criteria_failed
                    )
                    continue
                spec = ProteinSpec(
                    id=pair.id,
                    source_type=pair.state1.source_type,
                    entry=pair.state1.entry,
                    assembly_id=pair.state1.assembly_id,
                    chains=pair.state1.chains,
                    cluster=pair.cluster or pair.id,
                    notes=pair.notes,
                )
                out.append(
                    Candidate(
                        family=family,
                        generator=T01,
                        spec=spec,
                        processed=processed1,
                        proposal=item,
                        base_structure=ctx.structure1,
                        second_structure=ctx.structure2,
                        second_processed=processed2,
                        notes=ctx.notes,
                    )
                )
        return out

    def _diversify(self, proposals: list[Proposal]) -> list[Proposal]:
        """At most a few proposals per protein, spread across the family's tags."""
        by_tag: dict[str, list[Proposal]] = defaultdict(list)
        for proposal in proposals:
            by_tag[proposal.tag].append(proposal)
        out: list[Proposal] = []
        for round_index in range(3):
            for tag in sorted(by_tag):
                bucket = by_tag[tag]
                if round_index < len(bucket):
                    out.append(bucket[round_index])
        return out[:6]

    def _select(
        self, candidates: list[Candidate], target: int, *, oversample: int = 4
    ) -> list[Candidate]:
        """Order candidates round-robin over proteins, one per protein per round.

        Round-robin is what implements "each generated instance must use a
        different source protein within its type where feasible": every protein
        contributes its best candidate before any protein contributes a second.
        """
        if target <= 0:
            return []
        grouped: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.spec.cluster or candidate.spec.id].append(candidate)
        order = sorted(grouped)
        selected: list[Candidate] = []
        limit = target * oversample
        for round_index in range(max((len(v) for v in grouped.values()), default=0)):
            for protein_id in order:
                bucket = grouped[protein_id]
                if round_index < len(bucket):
                    selected.append(bucket[round_index])
                if len(selected) >= limit:
                    return selected
        return selected

    # ------------------------------------------------------------------ #
    # materialisation
    # ------------------------------------------------------------------ #
    def _materialise(self, candidate: Candidate) -> tuple[SemanticInstance, list[RenderedVariant]]:
        generator = candidate.generator
        instance_id = self._instance_id(candidate)
        budget = self.config.token_budget_automatic

        base = candidate.base_structure
        second = candidate.second_structure
        crop_info: CropInfo | None = None

        primary_seed = derive_seed(self.config.seed, instance_id, "rotation", 0)
        displayed = self._display(candidate, base, second, primary_seed)
        tokens = self._token_count(candidate, generator, displayed, instance_id)

        if tokens > budget:
            if not getattr(generator, "croppable", False):
                raise BuildRejection(
                    "over_token_budget_and_not_croppable",
                    {"tokens": tokens, "budget": budget},
                    ["crop_removes_required_atoms"],
                )
            original, original_second = base, second
            for shrink in (1.0, 0.75, 0.55, 0.4):
                base, crop_info = self._crop(candidate, original, shrink)
                second = (
                    crop_like(original_second, crop_info) if original_second is not None else None
                )
                displayed = self._display(candidate, base, second, primary_seed)
                tokens = self._token_count(candidate, generator, displayed, instance_id)
                if tokens <= budget:
                    break
            else:
                raise BuildRejection(
                    "over_token_budget_after_crop",
                    {"tokens": tokens, "budget": budget, "smallest_radius_factor": 0.4},
                )

        gold, evidence = self._oracle(candidate, displayed)
        instance = self._instance(candidate, instance_id, gold, evidence, crop_info)

        renders: list[RenderedVariant] = []
        renders.append(
            self._render(candidate, instance, displayed, "minimal_pdb", primary_seed, False, crop_info)
        )
        if self._wants_representation_variant(instance_id):
            renders.append(
                self._render(
                    candidate,
                    instance,
                    displayed,
                    "normalized_coordinates",
                    primary_seed,
                    False,
                    crop_info,
                )
            )
        if self._wants_rotation_variant(instance_id):
            alt_seed = derive_seed(self.config.seed, instance_id, "rotation", 1)
            alt = self._display(candidate, base, second, alt_seed)
            alt_gold, _ = self._oracle(candidate, alt)
            if instance.answer_schema not in COORDINATE_DEPENDENT_SCHEMAS and alt_gold != gold:
                raise BuildRejection(
                    "gold_answer_changed_under_rotation",
                    {"primary": gold, "rotated": alt_gold},
                    ["multiple_valid_answers"],
                )
            renders.append(
                self._render(
                    candidate, instance, alt, "minimal_pdb", alt_seed, True, crop_info, gold=alt_gold
                )
            )
        return instance, renders

    def _display(
        self,
        candidate: Candidate,
        base: Structure,
        second: Structure | None,
        seed: int,
    ) -> list[DisplayedStructure]:
        structures = [base] if second is None else [base, second]
        transform = build_transform(structures, seed, self.definitions)
        labels = ("Structure",) if second is None else candidate.state_labels
        return [
            display(s, transform, self.definitions, label=label)
            for s, label in zip(structures, labels)
        ]

    def _crop(
        self, candidate: Candidate, base: Structure, shrink: float = 1.0
    ) -> tuple[Structure, CropInfo]:
        centers = [
            i
            for label in candidate.proposal.crop_centers
            if (i := base.find_index(label)) is not None
        ]
        if not centers:
            raise BuildRejection("no_crop_centers", {"family": candidate.family})
        return crop_around(
            base,
            centers,
            candidate.generator.crop_radius * shrink,
            required_labels=candidate.proposal.required_labels,
        )

    def _oracle(
        self, candidate: Candidate, displayed: list[DisplayedStructure]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        parameters = candidate.proposal.parameters
        if candidate.is_two_state:
            out = candidate.generator.oracle(
                (displayed[0].structure, displayed[1].structure), parameters, self.definitions
            )
            return out["gold_answer"], out["evidence"]
        structure = displayed[0].structure
        analysis = Analysis(structure, self.definitions)
        result = candidate.generator.oracle(structure, parameters, self.definitions, analysis)
        return result.gold_answer, result.evidence

    def _token_count(
        self,
        candidate: Candidate,
        generator: Generator,
        displayed: list[DisplayedStructure],
        instance_id: str,
    ) -> int:
        prompt = self._prompt(candidate, displayed, "minimal_pdb")
        tokens, _ = count_tokens(
            prompt.system_prompt + "\n" + prompt.user_prompt, self.config.tokenizer
        )
        return tokens

    def _prompt(
        self,
        candidate: Candidate,
        displayed: list[DisplayedStructure],
        representation: str,
        *,
        crop: CropInfo | None = None,
    ):
        generator = candidate.generator
        parameters = candidate.proposal.parameters
        blocks = [
            StructureBlock(
                label=d.label,
                structure=d.structure,
                cropped=crop is not None,
            )
            for d in displayed
        ]
        question = generator.question(parameters, displayed[0].structure)
        return build_prompt(
            representation=representation,
            blocks=blocks,
            context=generator.context(parameters),
            question=question,
            answer_schema=generator.answer_schema,
        )

    # ------------------------------------------------------------------ #
    # record construction
    # ------------------------------------------------------------------ #
    def _instance(
        self,
        candidate: Candidate,
        instance_id: str,
        gold: dict[str, Any],
        evidence: dict[str, Any],
        crop: CropInfo | None,
    ) -> SemanticInstance:
        processed = candidate.processed
        record = processed.record
        entries = [record.entry]
        sha = [record.sha256]
        dates = [record.release_date or ""]
        publications = list(record.publications)
        assemblies = [processed.assembly_id or ""]
        if candidate.second_processed is not None:
            other = candidate.second_processed
            entries.append(other.record.entry)
            sha.append(other.record.sha256)
            dates.append(other.record.release_date or "")
            publications.extend(other.record.publications)
            assemblies.append(other.assembly_id or "")

        decision = self.decisions.get(instance_id)
        status = "proposed"
        notes = ""
        override = None
        if decision:
            status = "accepted" if decision.get("decision") == "accept" else "rejected"
            notes = decision.get("notes", "")
            override = decision.get("label_override")
            if override:
                gold = override

        evidence = dict(evidence)
        evidence["cluster"] = candidate.spec.cluster or candidate.spec.id
        evidence["ligand_map"] = processed.ligand_map
        if crop is not None:
            evidence["crop"] = crop.as_dict()
        if candidate.notes:
            evidence["alignment_notes"] = candidate.notes

        return SemanticInstance(
            semantic_instance_id=instance_id,
            question_family=candidate.family,
            question_version=candidate.generator.version,
            protein_group_id=candidate.spec.id,
            source_type=candidate.spec.source_type,
            source_entries=entries,
            source_file_sha256s=sha,
            release_dates=[d for d in dates if d],
            source_publications=publications,
            biological_assembly_ids=[a for a in assemblies if a],
            selected_chains=candidate.base_structure.chains,
            question_parameters=candidate.proposal.parameters,
            answer_schema=candidate.generator.answer_schema,
            gold_answer=gold,
            gold_evidence=evidence,
            selection_margins=candidate.proposal.margins,
            definition_version=self.definitions.version,
            curation_status=status,
            curator_notes=notes,
            experimental_method=record.experimental_method,
            resolution=record.resolution,
            criteria_passed=candidate.proposal.criteria_passed,
            acceptance_reasons=candidate.proposal.reasons,
            curator_override=override,
        )

    def _render(
        self,
        candidate: Candidate,
        instance: SemanticInstance,
        displayed: list[DisplayedStructure],
        representation: str,
        rotation_seed: int,
        is_rotation_variant: bool,
        crop: CropInfo | None,
        *,
        gold: dict[str, Any] | None = None,
    ) -> RenderedVariant:
        prompt = self._prompt(candidate, displayed, representation, crop=crop)
        coordinate_text = "".join(prompt.structure_text.get(d.label, "") for d in displayed)
        tokens, tokenizer = count_tokens(
            prompt.system_prompt + "\n" + prompt.user_prompt, self.config.tokenizer
        )
        transform = displayed[0].transform
        render_id = f"{instance.semantic_instance_id}::{representation}::{rotation_seed % 10**8}"
        return RenderedVariant(
            render_id=render_id,
            semantic_instance_id=instance.semantic_instance_id,
            representation=representation,
            rotation_seed=int(rotation_seed),
            state_order_seed=None,
            prompt_version=PROMPT_VERSION,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            rotation_matrix=[[float(v) for v in row] for row in transform.rotation],
            translation_vector=[float(v) for v in transform.translation],
            displayed_coordinates_sha256=sha256_text(coordinate_text),
            input_token_count=tokens,
            question_family=instance.question_family,
            protein_group_id=instance.protein_group_id,
            answer_schema=instance.answer_schema,
            gold_answer=gold if gold is not None else instance.gold_answer,
            is_rotation_variant=is_rotation_variant,
            state_order=[d.label for d in displayed],
            crop=crop.as_dict() if crop else None,
            atom_count=sum(d.atom_count for d in displayed),
            tokenizer=tokenizer,
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _load(self, spec: ProteinSpec) -> ProcessedStructure:
        key = (spec.id, ",".join(sorted(spec.keep_components)))
        if key not in self._processed:
            record = self.cache.get(spec.source_type, spec.entry)
            self._processed[key] = load_processed(record, spec, self.definitions)
        return self._processed[key]

    def _proposal_frame(
        self, spec: ProteinSpec, processed: ProcessedStructure
    ) -> tuple[DisplayedStructure, Analysis]:
        """A stable displayed frame used only for proposing and margin checks.

        Cached per protein so DSSP, SASA, the contact graph and the clash list
        are computed once rather than once per question family.
        """
        if spec.id not in self._frames:
            seed = derive_seed(self.config.seed, "proposal_frame", spec.id)
            transform = build_transform([processed.structure], seed, self.definitions)
            displayed = display(processed.structure, transform, self.definitions)
            self._frames[spec.id] = (displayed, Analysis(displayed.structure, self.definitions))
        return self._frames[spec.id]

    def _low_confidence_labels(
        self, processed: ProcessedStructure, structure: Structure, proposal: Proposal
    ) -> list[str]:
        """AlphaFold residues near the query whose pLDDT is below the threshold."""
        if not processed.plddt:
            return []
        queried = set(proposal.required_labels) | set(proposal.crop_centers)
        for value in proposal.parameters.values():
            if isinstance(value, str):
                queried.add(value)
            elif isinstance(value, list):
                queried.update(v for v in value if isinstance(v, str))

        neighbourhood: set[str] = set()
        index = structure.index
        for label in queried:
            residue = structure.find(label)
            if residue is None:
                continue
            neighbourhood.add(label)
            for atom in residue.atoms:
                for ai in index.within(atom.pos, self._afdb_radius):
                    neighbourhood.add(structure.residues[int(index.residue_of[ai])].label)
        return sorted(
            label
            for label in neighbourhood
            if processed.plddt.get(label, 100.0) < self._afdb_min_plddt
        )

    def _instance_id(self, candidate: Candidate) -> str:
        digest = stable_hash(
            candidate.family, candidate.spec.id, candidate.proposal.parameters
        )[:8]
        return f"{candidate.family}-{candidate.spec.id}-{digest}"

    def _wants_rotation_variant(self, instance_id: str) -> bool:
        fraction = self.config.rotation_variant_fraction
        return (derive_seed(self.config.seed, "rotation_variant", instance_id) % 1000) < fraction * 1000

    def _wants_representation_variant(self, instance_id: str) -> bool:
        fraction = self.config.representation_variant_fraction
        return (
            derive_seed(self.config.seed, "representation_variant", instance_id) % 1000
        ) < fraction * 1000

    def _keep(self, instance: SemanticInstance) -> bool:
        if instance.curation_status == "rejected":
            return False
        if self.accepted_only and instance.curation_status != "accepted":
            return False
        return True

    def _record_rejection(
        self,
        result: BuildResult,
        family: str,
        protein_id: str,
        reason: str,
        detail: dict[str, Any],
        criteria: list[str],
    ) -> None:
        key = (family, protein_id, reason)
        self._rejection_counts[key] += 1
        if self._rejection_counts[key] > self.max_rejection_examples:
            return
        result.rejections.append(
            RejectionRecord(
                candidate_id=f"{family}-{protein_id}-{self._rejection_counts[key]:03d}",
                question_family=family,
                protein_group_id=protein_id,
                reason=reason,
                detail=detail,
                criteria_failed=criteria,
            )
        )

    def _build_episodes(self, result: BuildResult) -> None:
        from .mechanistic.pipeline import build_episodes

        build_episodes(self, result)

    def _summarise(self, result: BuildResult, per_protein: Counter) -> dict[str, Any]:
        by_family = Counter(i.question_family for i in result.instances)
        return {
            "dataset": self.config.name,
            "dataset_version": self.config.version,
            "definition_version": self.definitions.version,
            "definitions_sha256": self.definitions.content_sha256,
            "dataset_config_sha256": self.config.content_sha256,
            "prompt_version": PROMPT_VERSION,
            "prompt_fingerprint": prompt_fingerprint(),
            "seed": self.config.seed,
            "tokenizer": self.config.tokenizer,
            "n_instances": len(result.instances),
            "n_renders": len(result.renders),
            "n_rejection_records": len(result.rejections),
            "rejection_totals": {
                f"{f}:{r}": n for (f, _p, r), n in sorted(self._rejection_counts.items())
            },
            "realised_counts": dict(sorted(by_family.items())),
            "target_counts": dict(sorted(self.config.family_targets.items())),
            "instances_per_protein": dict(
                sorted(Counter(i.protein_group_id for i in result.instances).items())
            ),
            "representation_counts": dict(
                sorted(Counter(r.representation for r in result.renders).items())
            ),
            "rotation_variants": sum(1 for r in result.renders if r.is_rotation_variant),
            "token_counts": {
                "max": max((r.input_token_count or 0) for r in result.renders) if result.renders else 0,
                "mean": (
                    sum((r.input_token_count or 0) for r in result.renders) / len(result.renders)
                    if result.renders
                    else 0
                ),
            },
        }


class BuildRejection(RuntimeError):
    """A candidate that could not be turned into a valid instance."""

    def __init__(self, reason: str, detail: dict[str, Any] | None = None, criteria: list[str] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}
        self.criteria = criteria or []


GOLD_FILE = "gold.jsonl"
WITHHELD_NOTICE = (
    "Gold answers are withheld from this manifest and regenerated by "
    "`structural-reasoning build`; see docs/contamination.md."
)


def write_dataset(
    result: BuildResult, output_dir: str | Path, *, split_gold: bool = True
) -> dict[str, Any]:
    """Write a dataset directory: instances, renders, prompts and provenance.

    With ``split_gold`` the manifests carry ``gold_sha256`` instead of the answer
    itself and the answers go to ``gold.jsonl``, which is not committed. A build
    is byte-identical across machines, so the hash is enough to prove that a
    local rebuild matches the canonical one -- without the answers ever
    appearing in a crawlable file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    instances, renders, gold_rows = [], [], []
    for instance in result.instances:
        row = instance.model_dump()
        row["gold_sha256"] = gold_hash(instance.gold_answer)
        if split_gold:
            gold_rows.append(
                {
                    "semantic_instance_id": instance.semantic_instance_id,
                    "gold_answer": instance.gold_answer,
                    "gold_evidence": instance.gold_evidence,
                }
            )
            row["gold_answer"], row["gold_evidence"] = {}, {"withheld": WITHHELD_NOTICE}
        instances.append(row)
    for render in result.renders:
        row = {
            k: v
            for k, v in render.model_dump().items()
            if k not in ("system_prompt", "user_prompt")
        }
        row["gold_sha256"] = gold_hash(render.gold_answer)
        if split_gold:
            gold_rows.append({"render_id": render.render_id, "gold_answer": render.gold_answer})
            row["gold_answer"] = {}
        renders.append(row)

    write_jsonl(out / "instances.jsonl", [canary_record(), *instances])
    write_jsonl(out / "renders.jsonl", [canary_record(), *renders])
    if split_gold:
        write_jsonl(out / GOLD_FILE, [canary_record(), *gold_rows])
    elif (out / GOLD_FILE).exists():
        (out / GOLD_FILE).unlink()
    # Rebuild from scratch: a stale prompt left over from an earlier build with
    # different configuration would silently survive and break reproducibility.
    prompts_dir = out / "prompts"
    if prompts_dir.exists():
        for stale in prompts_dir.glob("*.txt"):
            stale.unlink()
    prompts_dir.mkdir(exist_ok=True)
    for render in result.renders:
        (prompts_dir / f"{render.render_id.replace('::', '__')}.txt").write_text(
            render.user_prompt, encoding="utf-8"
        )
    write_jsonl(
        out / "rejections.jsonl", [canary_record(), *[r.model_dump() for r in result.rejections]]
    )
    stats = {**result.stats, "canary": canary_record()["_canary"], "gold_split": split_gold}
    write_json(out / "build_report.json", stats)
    return stats


def load_dataset(directory: str | Path) -> tuple[list[SemanticInstance], list[RenderedVariant]]:
    """Read back a dataset directory, re-attaching prompt text to each render."""
    out = Path(directory)
    gold_by_instance, gold_by_render = _load_gold(out)

    instance_rows = list(read_jsonl(out / "instances.jsonl"))
    if instance_rows:
        check_canary(instance_rows, str(out / "instances.jsonl"))
    instances: list[SemanticInstance] = []
    for row in instance_rows:
        if is_canary(row):
            continue
        gold = gold_by_instance.get(row["semantic_instance_id"])
        if gold is not None:
            row["gold_answer"] = gold["gold_answer"]
            row["gold_evidence"] = gold.get("gold_evidence", {})
        elif not row.get("gold_answer"):
            raise MissingGold(out)
        instances.append(SemanticInstance(**row))

    renders: list[RenderedVariant] = []
    from .prompts.library import SYSTEM_PROMPT

    for row in read_jsonl(out / "renders.jsonl"):
        if is_canary(row):
            continue
        gold = gold_by_render.get(row["render_id"])
        if gold is not None:
            row["gold_answer"] = gold["gold_answer"]
        elif not row.get("gold_answer"):
            raise MissingGold(out)
        path = out / "prompts" / f"{row['render_id'].replace('::', '__')}.txt"
        row["user_prompt"] = path.read_text(encoding="utf-8") if path.exists() else ""
        row["system_prompt"] = SYSTEM_PROMPT
        renders.append(RenderedVariant(**row))
    return instances, renders


class MissingGold(RuntimeError):
    """The manifest withholds its gold answers and no local gold.jsonl exists."""

    def __init__(self, directory: Path) -> None:
        super().__init__(
            f"{directory} withholds its gold answers. Regenerate them with\n"
            f"    structural-reasoning build --config <dataset.yaml> --output {directory}\n"
            "The build is deterministic, so the regenerated answers are identical to the "
            "canonical ones; `validate` checks that against the committed gold_sha256."
        )


def _load_gold(out: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    by_instance: dict[str, dict] = {}
    by_render: dict[str, dict] = {}
    for row in read_jsonl(out / GOLD_FILE):
        if is_canary(row):
            continue
        if "semantic_instance_id" in row:
            by_instance[row["semantic_instance_id"]] = row
        else:
            by_render[row["render_id"]] = row
    return by_instance, by_render
