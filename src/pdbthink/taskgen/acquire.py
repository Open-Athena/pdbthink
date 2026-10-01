"""Network-only acquisition of a frozen, benchmark-disjoint source pool."""

from __future__ import annotations

import concurrent.futures
import json
import urllib.parse
import urllib.request
from pathlib import Path

import gemmi

from ..acquisition.cache import StructureCache
from ..util import derive_seed, sha256_bytes, write_json
from .exclusion import audit_source, clusters, freeze_exclusions


def terminal(attribute, operator, value):
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": attribute,
            "operator": operator,
            "value": value,
        },
    }


def query_phosphoproteins(directory: Path) -> list[str]:
    destination = directory / "phosphoproteins.json"
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                terminal("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein"),
                terminal(
                    "rcsb_polymer_entity_container_identifiers.chem_comp_monomers",
                    "in",
                    ["SEP", "TPO", "PTR"],
                ),
                terminal("entity_poly.rcsb_sample_sequence_length", "range", {"from": 10, "to": 195}),
                terminal("rcsb_entry_info.resolution_combined", "less_or_equal", 3.0),
                terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {"paginate": {"start": 0, "rows": 10000}},
    }
    write_json(directory / "phosphoproteins.query.json", query)
    if not destination.exists():
        url = "https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(json.dumps(query))
        with urllib.request.urlopen(url, timeout=120) as response:
            destination.write_bytes(response.read())
    return [row["identifier"] for row in json.loads(destination.read_text()).get("result_set", [])]


def acquire_pool(repo: Path, directory: Path, *, seed: int, limit: int = 1600, workers: int = 8) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cluster_file = directory / "clusters-by-entity-30.txt"
    if not cluster_file.exists():
        cluster_file.write_bytes((repo / "data/expansion_sources/clusters-by-entity-30.txt").read_bytes())
    exclusion_file = directory / "exclusion.json"
    exclusion = (
        json.loads(exclusion_file.read_text())
        if exclusion_file.exists()
        else freeze_exclusions(repo, exclusion_file, cluster_file)
    )
    mapping = clusters(cluster_file)
    blocked_sources = set(exclusion["sources"])
    blocked_clusters = set(exclusion["rcsb30_clusters"])
    pools = {}
    for name in ("small", "interfaces", "ligands"):
        frozen_pool = directory / f"{name}.json"
        raw = (
            frozen_pool.read_bytes()
            if frozen_pool.exists()
            else (repo / f"data/expansion_sources/{name}_v2.json").read_bytes()
        )
        frozen_pool.write_bytes(raw)
        pools[name] = [r["identifier"] for r in json.loads(raw)["result_set"]]
    pools["phosphorylation"] = query_phosphoproteins(directory)
    choices = {}
    for name, entities in sorted(pools.items()):
        eligible = [
            e
            for e in entities
            if f"pdb:{e.split('_')[0]}" not in blocked_sources
            and e in mapping
            and mapping[e] not in blocked_clusters
        ]
        eligible.sort(key=lambda e: (derive_seed(seed, "source", name, e), e))
        quota = {"small": limit, "interfaces": limit // 2, "ligands": limit // 2, "phosphorylation": limit}[
            name
        ]
        choices[name] = eligible[:quota]
        print(name, "eligible", len(eligible), "selected", len(choices[name]), flush=True)
    entity_roles = {}
    for name, entities in choices.items():
        for entity in entities:
            entity_roles.setdefault(entity, []).append(name)
    entries = sorted({e.split("_")[0] for e in entity_roles})
    write_json(directory / "selection.json", {"seed": seed, "choices": choices, "entries": entries})
    cache = StructureCache(repo / "data/cache")
    audits_dir = directory / "audits"
    audits_dir.mkdir(exist_ok=True)

    def acquire(entry):
        try:
            record = cache.get_pdb(entry)
            audit_path = audits_dir / f"{entry}.json"
            if audit_path.exists():
                audit = json.loads(audit_path.read_text())
                if audit["exclusion_fingerprint"] != exclusion["fingerprint"]:
                    raise ValueError("stale exclusion audit")
                if sha256_bytes(Path(record.path).read_bytes()) != audit["source_file_sha256"]:
                    raise ValueError("source coordinates changed since the frozen audit")
            else:
                audit = audit_source(record, exclusion, mapping)
                write_json(audit_path, audit)
            return entry, record, audit, None
        except Exception as exc:
            return (
                entry,
                None,
                None,
                {"entry": entry, "reason": "acquisition_or_audit_failed", "error": str(exc)},
            )

    records, audits, failures = {}, {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for number, (entry, record, audit, error) in enumerate(executor.map(acquire, entries), 1):
            if error:
                failures.append(error)
            elif audit["allowed"]:
                records[entry], audits[entry] = record, audit
            else:
                failures.append({"entry": entry, "reason": "benchmark_overlap", "audit": audit})
            if number % 50 == 0:
                print("acquired", number, "/", len(entries), "clean", len(records), flush=True)
    specs = []
    for entity, roles in sorted(entity_roles.items()):
        entry, entity_id = entity.split("_")
        if entry not in records:
            continue
        record = records[entry]
        block = gemmi.cif.read_file(record.path).sole_block()
        ids = list(block.find_values("_entity_poly.entity_id"))
        if entity_id not in ids:
            continue
        strands = list(block.find_values("_entity_poly.pdbx_strand_id"))
        chain_names = strands[ids.index(entity_id)].strip("'\"").split(",")
        # No source identity is put in the model-visible text.
        spec = {
            "id": f"train_{entry.lower()}_{entity_id}",
            "source_type": "pdb",
            "entry": entry,
            "chains": [chain_names[0]],
            "cluster": mapping[entity],
            "roles": roles,
            "entity": entity,
            "sequence_sha256s": audits[entry]["sequence_sha256s"],
        }
        specs.append(spec)
        if "interfaces" in roles:
            specs.append(
                {
                    **spec,
                    "id": spec["id"] + "_assembly1",
                    "chains": None,
                    "assembly_id": "1",
                    "roles": ["assembly"],
                }
            )
    write_json(
        directory / "sources.json",
        {
            "seed": seed,
            "specs": specs,
            "exclusion_fingerprint": exclusion["fingerprint"],
            "source_count": len(records),
        },
    )
    write_json(directory / "source_rejections.json", failures)
    print("source pool complete:", len(specs), "views from", len(records), "clean entries", flush=True)
