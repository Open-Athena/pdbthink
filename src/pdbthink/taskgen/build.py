"""Deterministically extend all active coordinate families on the disjoint pool.

Each cached shard is one source view, making interrupted builds resumable.
The release selector counts semantic questions, never rotated render variants.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..acquisition.cache import StructureCache
from ..config import DatasetConfig, Definitions, ProteinSpec
from ..dataset import Candidate, DatasetBuilder
from ..generators import V1_FAMILIES, GenerationContext, Rejection, get_generator
from ..util import derive_seed, stable_hash, write_json
from .coordinates import recompute


def configuration(seed: int, token_budget: int = 64000) -> DatasetConfig:
    return DatasetConfig(
        name="pdbthink-coordinate-tasks",
        version="1.0.0",
        seed=seed,
        token_budget_automatic=token_budget,
        token_budget_mechanistic=0,
        tokenizer="cl100k_base",
        rotation_variant_fraction=0.0,
        representation_variant_fraction=0.0,
        max_instances_per_protein=1000,
        family_targets={},
        proteins=[],
        state_pairs=[],
        family_proteins={},
        episodes=[],
        crop={},
        source_path="taskgen-v1",
        content_sha256=stable_hash(seed, token_budget),
        raw={"strict_provenance_selection": True},
    )


def _source_worker(args):
    repo, directory, data, seed, per_family = args
    repo, directory = Path(repo), Path(directory)
    path = directory / "shards" / f"{data['id']}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as stream:
            old = json.load(stream)
        if old["seed"] != seed or old["per_family"] != per_family:
            raise ValueError("shard build settings changed; use a new build directory")
        return old["counts"]
    definitions = Definitions.load()
    cache = StructureCache(repo / "data/cache", offline=True)
    builder = DatasetBuilder(configuration(seed), definitions, cache)
    spec = ProteinSpec(**{k: v for k, v in data.items() if k in ProteinSpec.__dataclass_fields__})
    rejections, tasks = [], []
    try:
        processed = builder._load(spec)
        # All released tasks use protein-only polymers; nucleic acid mixtures
        # need their own serialisation/reconstruction contract before inclusion.
        if any(r.polymer_kind and not r.is_protein for r in processed.structure.residues):
            raise ValueError("non-protein polymer present")
        if "assembly" in data["roles"] and len(processed.structure.protein_chains) < 2:
            raise ValueError("assembly does not contain two protein chains")
        displayed, analysis = builder._proposal_frame(spec, processed)
        from ..util import rng_for

        families = (
            ["P01", "P02", "I01"]
            if "assembly" in data["roles"]
            else [f for f in V1_FAMILIES if f not in ("I01", "T01")]
        )
        for family in families:
            generator = get_generator(family)
            ctx = GenerationContext(
                spec, processed, displayed, definitions, rng_for(seed, family, spec.id), analysis
            )
            proposals = []
            for proposal in generator.propose(ctx):
                if isinstance(proposal, Rejection):
                    rejections.append(
                        {"family": family, "reason": proposal.reason, "detail": proposal.detail}
                    )
                else:
                    proposals.append(proposal)
            proposals.sort(key=lambda p: (p.rank, p.key()))
            proposals = builder._diversify(builder._seeded_choice(proposals, family, spec.id))
            if proposals:
                offset = derive_seed(seed, "tag-start", family, spec.id) % len(proposals)
                proposals = proposals[offset:] + proposals[:offset]
            produced = 0
            for proposal in proposals:
                if produced >= per_family:
                    break
                candidate = Candidate(family, generator, spec, processed, proposal, processed.structure)
                try:
                    instance, renders = builder._materialise(candidate)
                    render = next(r for r in renders if r.representation == "minimal_pdb")
                    recovered = recompute(render.user_prompt, family, proposal.parameters, definitions)
                    if recovered != render.gold_answer:
                        raise ValueError("serialised-coordinate oracle disagrees with materialised oracle")
                    tasks.append(
                        {
                            "instance": instance.model_dump(),
                            "render": render.model_dump(),
                            "source_view": data["id"],
                            "source_entities": [data["entity"]],
                        }
                    )
                    produced += 1
                except (ValueError, RuntimeError) as exc:
                    rejections.append(
                        {
                            "family": family,
                            "reason": getattr(exc, "reason", "oracle_or_render_rejected"),
                            "detail": {"error": str(exc), "parameters": proposal.parameters},
                        }
                    )
    except Exception as exc:
        rejections.append(
            {
                "family": "*",
                "reason": getattr(exc, "reason", "source_processing_failed"),
                "detail": {"error": str(exc)},
            }
        )
    counts = dict(Counter(t["instance"]["question_family"] for t in tasks))
    result = {
        "seed": seed,
        "per_family": per_family,
        "source_view": data,
        "counts": counts,
        "tasks": tasks,
        "rejections": rejections,
    }
    raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(gzip.compress(raw, mtime=0))
    return counts


def build_pool(
    repo: Path, directory: Path, *, workers: int = 16, per_family: int = 2, limit: int | None = None
) -> None:
    source = json.loads((directory / "sources.json").read_text())
    specs = sorted(source["specs"], key=lambda x: (derive_seed(source["seed"], "build", x["id"]), x["id"]))
    if limit:
        specs = specs[:limit]
    (directory / "shards").mkdir(exist_ok=True)
    counts = Counter()
    args = [(str(repo), str(directory), spec, source["seed"], per_family) for spec in specs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(_source_worker, args), 1):
            counts.update(result)
            if index % 20 == 0:
                print(
                    "built",
                    index,
                    "/",
                    len(specs),
                    "tasks",
                    sum(counts.values()),
                    dict(sorted(counts.items())),
                    flush=True,
                )
    write_json(
        directory / "pool_summary.json", {"source_views": len(specs), "counts": dict(sorted(counts.items()))}
    )
    print("pool complete", dict(sorted(counts.items())), flush=True)


def select_tasks(directory: Path, total: int = 10000) -> list[dict]:
    """Balance families and sources without treating alternate renderings as tasks."""
    by_family = defaultdict(list)
    seen = set()
    for path in sorted((directory / "shards").glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            shard = json.load(stream)
        for task in shard["tasks"]:
            i = task["instance"]
            # Source entries and view identity belong to the semantics. Rotation
            # and build seed do not create an additional semantic question.
            key = stable_hash(
                i["question_family"],
                sorted(i["source_entries"]),
                i["selected_chains"],
                i["biological_assembly_ids"],
                i["question_parameters"],
            )
            if key in seen:
                continue
            seen.add(key)
            task["semantic_key"] = key
            by_family[i["question_family"]].append(task)
    missing = sorted(set(V1_FAMILIES) - set(by_family))
    if missing:
        raise ValueError(f"no releasable tasks for families {missing}")
    if sum(map(len, by_family.values())) < total:
        raise ValueError(f"insufficient unique tasks: {sum(map(len, by_family.values()))} < {total}")
    for family, tasks in by_family.items():
        groups = defaultdict(list)
        for task in tasks:
            groups[task["instance"]["gold_evidence"]["cluster"]].append(task)
        ordered = []
        for group in groups.values():
            group.sort(key=lambda t: stable_hash("select", t["semantic_key"]))
        for index in range(max(map(len, groups.values()))):
            for key in sorted(groups, key=lambda key: stable_hash("source-order", key)):
                if index < len(groups[key]):
                    ordered.append(groups[key][index])
        by_family[family] = ordered
    selected = []
    position = 0
    while len(selected) < total:
        for family in V1_FAMILIES:
            if position < len(by_family[family]):
                selected.append(by_family[family][position])
                if len(selected) == total:
                    break
        position += 1
    return selected
