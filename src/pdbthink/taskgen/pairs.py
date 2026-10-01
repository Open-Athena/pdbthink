"""Acquire and compare fresh experimental states for the T01 contact-change family."""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import gemmi

from ..acquisition.cache import StructureCache
from ..config import Definitions, ProteinSpec, StatePairSpec
from ..dataset import Candidate, DatasetBuilder
from ..generators import T01, Rejection, build_two_state_context
from ..util import derive_seed, stable_hash, write_json
from .build import configuration
from .coordinates import recompute
from .exclusion import audit_source, clusters


def acquire_pairs(repo: Path, directory: Path, *, limit: int = 350, workers: int = 8) -> None:
    source = json.loads((directory / "sources.json").read_text())
    exclusion = json.loads((directory / "exclusion.json").read_text())
    mapping30 = clusters(directory / "clusters-by-entity-30.txt")
    cluster_file = directory / "clusters-by-entity-90.txt"
    if not cluster_file.exists():
        url = "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-90.txt"
        with urllib.request.urlopen(url, timeout=120) as response:
            cluster_file.write_bytes(response.read())
    groups90 = [line.split() for line in cluster_file.read_text().splitlines()]
    group_for = {entity: group for group in groups90 for entity in group}
    blocked = set(exclusion["sources"])
    blocked_clusters = set(exclusion["rcsb30_clusters"])
    bases = [s for s in source["specs"] if "assembly" not in s["roles"] and "small" in s["roles"]]
    bases.sort(key=lambda s: (derive_seed(source["seed"], "pairs", s["id"]), s["id"]))
    candidates = []
    for base in bases:
        partners = [
            e
            for e in group_for.get(base["entity"], [])
            if re.fullmatch(r"[0-9][A-Za-z0-9]{3}_[0-9]+", e)
            and e.split("_")[0] != base["entry"]
            and f"pdb:{e.split('_')[0]}" not in blocked
            and e in mapping30
            and mapping30[e] not in blocked_clusters
        ]
        partners.sort(key=lambda e: (derive_seed(source["seed"], "pair-partner", base["entity"], e), e))
        for partner in partners[:2]:
            candidates.append((base, partner))
        if len(candidates) >= limit:
            break
    cache = StructureCache(repo / "data/cache")

    def acquire(item):
        base, partner = item
        entry, entity = partner.split("_")
        try:
            record = cache.get_pdb(entry)
            audit = audit_source(record, exclusion, mapping30)
            write_json(directory / "audits" / f"{entry}.json", audit)
            if not audit["allowed"]:
                raise ValueError("benchmark overlap: " + ", ".join(audit["reasons"]))
            block = gemmi.cif.read_file(record.path).sole_block()
            ids = list(block.find_values("_entity_poly.entity_id"))
            chain = list(block.find_values("_entity_poly.pdbx_strand_id"))[ids.index(entity)]
            chain = chain.strip("'\"").split(",")[0]
            second = {
                "id": f"train_{entry.lower()}_{entity}",
                "entry": entry,
                "source_type": "pdb",
                "chains": [chain],
                "cluster": mapping30[partner],
                "entity": partner,
            }
            return {"first": base, "second": second}, None
        except Exception as exc:
            return None, {"entry": entry, "reason": "pair_source_rejected", "error": str(exc)}

    pairs, rejected = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (pair, error) in enumerate(executor.map(acquire, candidates), 1):
            if pair:
                pairs.append(pair)
            else:
                rejected.append(error)
            if index % 25 == 0:
                print("pair sources", index, "/", len(candidates), "clean", len(pairs), flush=True)
    write_json(directory / "pairs.json", {"seed": source["seed"], "pairs": pairs, "rejections": rejected})


def _pair_worker(args):
    repo, directory, data, seed = args
    directory = Path(directory)
    key = "pair_" + stable_hash(data["first"]["entity"], data["second"]["entity"])[:20]
    path = directory / "shards" / f"{key}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as stream:
            old = json.load(stream)
        if old["seed"] != seed or old["source_view"] != data:
            raise ValueError("pair build settings changed; use a new build directory")
        return old["counts"]
    tasks, rejected = [], []
    try:
        definitions = Definitions.load()
        builder = DatasetBuilder(
            configuration(seed), definitions, StructureCache(Path(repo) / "data/cache", offline=True)
        )
        specs = [
            ProteinSpec(**{k: v for k, v in data[which].items() if k in ProteinSpec.__dataclass_fields__})
            for which in ("first", "second")
        ]
        first, second = [builder._load(s) for s in specs]
        if any(r.polymer_kind and not r.is_protein for p in (first, second) for r in p.structure.residues):
            raise ValueError("non-protein polymer present")
        pair = StatePairSpec(key, *specs, cluster=specs[0].cluster)
        ctx = build_two_state_context(pair, first, second, definitions, seed=seed)
        seen = set()
        for variant in range(16):
            ctx.seed = derive_seed(seed, key, "candidate-set", variant)
            for proposal in T01.propose(ctx):
                if isinstance(proposal, Rejection):
                    rejected.append({"family": "T01", "reason": proposal.reason, "detail": proposal.detail})
                    continue
                if proposal.key() in seen:
                    continue
                seen.add(proposal.key())
                spec = ProteinSpec(
                    id=key,
                    source_type="pdb",
                    entry=specs[0].entry,
                    chains=specs[0].chains,
                    cluster=specs[0].cluster,
                )
                candidate = Candidate(
                    "T01",
                    T01,
                    spec,
                    first,
                    proposal,
                    ctx.structure1,
                    second_structure=ctx.structure2,
                    second_processed=second,
                    notes=ctx.notes,
                )
                try:
                    instance, renders = builder._materialise(candidate)
                    render = renders[0]
                    if (
                        recompute(render.user_prompt, "T01", proposal.parameters, definitions)
                        != render.gold_answer
                    ):
                        raise ValueError("serialised two-state oracle mismatch")
                    tasks.append(
                        {
                            "instance": instance.model_dump(),
                            "render": render.model_dump(),
                            "source_view": key,
                            "source_entities": [data["first"]["entity"], data["second"]["entity"]],
                        }
                    )
                except (ValueError, RuntimeError) as exc:
                    rejected.append(
                        {
                            "family": "T01",
                            "reason": "pair_materialisation_rejected",
                            "detail": {"error": str(exc)},
                        }
                    )
    except Exception as exc:
        rejected.append({"family": "T01", "reason": "state_pair_rejected", "detail": {"error": str(exc)}})
    counts = {"T01": len(tasks)}
    path.write_bytes(
        gzip.compress(
            json.dumps(
                {
                    "seed": seed,
                    "per_family": 16,
                    "counts": counts,
                    "tasks": tasks,
                    "rejections": rejected,
                    "source_view": data,
                },
                sort_keys=True,
            ).encode(),
            mtime=0,
        )
    )
    return counts


def build_pairs(repo: Path, directory: Path, *, workers: int = 16) -> None:
    source = json.loads((directory / "pairs.json").read_text())
    counts = Counter()
    args = [(str(repo), str(directory), pair, source["seed"]) for pair in source["pairs"]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(_pair_worker, args), 1):
            counts.update(result)
            if index % 20 == 0:
                print("built pairs", index, "/", len(args), dict(counts), flush=True)
    print("pair pool complete", dict(counts), flush=True)
