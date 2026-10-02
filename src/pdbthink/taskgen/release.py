"""Publishable Parquet shards and an auditable release manifest."""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..generators import V1_FAMILIES
from ..prompts.library import PROMPT_VERSION
from ..scoring.scorers import SCORING_VERSION
from ..util import derive_seed, sha256_bytes, stable_hash, write_json
from . import TASKGEN_VERSION
from .build import select_tasks
from .harbor import row_for
from .revisions import read_parent, update_prompt


def _package(args):
    task, split, group, version = args
    update_prompt(task)
    return row_for(task, split, group, version=version)


def source_groups(directory: Path) -> dict[str, str]:
    """Keep entries, all protein partners and shared sequences in one split."""
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        x, y = sorted((find(a), find(b)))
        parent[y] = x

    for path in sorted((directory / "audits").glob("*.json")):
        audit = json.loads(path.read_text())
        if not audit["allowed"]:
            continue
        entry = "entry:" + audit["entry"]
        for key in audit["rcsb30_clusters"] + ["sequence:" + s for s in audit["sequence_sha256s"]]:
            union(entry, key)
    if (directory / "pairs.json").exists():
        for pair in json.loads((directory / "pairs.json").read_text())["pairs"]:
            union("entry:" + pair["first"]["entry"], "entry:" + pair["second"]["entry"])
    members = defaultdict(list)
    for key in sorted(parent):
        members[find(key)].append(key)
    names = {root: "group-" + stable_hash(keys)[:24] for root, keys in members.items()}
    return {key[6:]: names[find(key)] for key in parent if key.startswith("entry:")}


def export_release(
    repo: Path,
    directory: Path,
    output: Path,
    *,
    total: int = 10000,
    parent: Path | None = None,
    version: str = TASKGEN_VERSION,
    workers: int = 8,
) -> dict:
    retained = read_parent(parent) if parent else []
    tasks = select_tasks(directory, total, retained=retained)
    groups = source_groups(directory)
    exclusion = json.loads((directory / "exclusion.json").read_text())
    if stable_hash({k: v for k, v in exclusion.items() if k != "fingerprint"}) != exclusion["fingerprint"]:
        raise ValueError("exclusion snapshot fingerprint mismatch")
    if (
        sha256_bytes((directory / "clusters-by-entity-30.txt").read_bytes())
        != exclusion["cluster_file_sha256"]
    ):
        raise ValueError("sequence clusters differ from the exclusion snapshot")
    source = json.loads((directory / "sources.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    if any((output / "data").glob("*.parquet")):
        raise ValueError("release output already contains data; use a new output directory")
    (output / "data").mkdir(exist_ok=True)
    audits = {p.stem: json.loads(p.read_text()) for p in (directory / "audits").glob("*.json")}
    rows = defaultdict(list)
    entries, sequence_hashes, cluster_ids = set(), set(), set()
    families, labels, split_counts, split_families = (
        Counter(),
        defaultdict(Counter),
        Counter(),
        defaultdict(Counter),
    )
    prompt_hashes, semantic_keys = set(), set()
    task_digests = []
    jobs = []
    for task in tasks:
        instance = task["instance"]
        task_groups = {groups[e] for e in instance["source_entries"]}
        if len(task_groups) != 1:
            raise ValueError("paired structures do not share a leakage group")
        group = next(iter(task_groups))
        bucket = derive_seed(source["seed"], "split", group) % 100
        split = "train" if bucket < 90 else ("validation" if bucket < 95 else "test")
        if "parent_split" in task and (task["parent_split"] != split or task["parent_source_group"] != group):
            raise ValueError("parent task split or source group changed")
        for entry in instance["source_entries"]:
            audit = audits[entry]
            if not audit["allowed"] or audit["exclusion_fingerprint"] != exclusion["fingerprint"]:
                raise ValueError("missing or stale source exclusion audit")
            entries.add(entry)
            sequence_hashes.update(audit["sequence_sha256s"])
            cluster_ids.update(audit["rcsb30_clusters"])
        if set(instance["source_file_sha256s"]) != {
            audits[entry]["source_file_sha256"] for entry in instance["source_entries"]
        }:
            raise ValueError("generated coordinates do not match the audited source files")
        jobs.append((task, split, group, version))
        if instance["answer_schema"] == "category":
            labels[instance["question_family"]][instance["gold_answer"]["value"]] += 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for n, row in enumerate(executor.map(_package, jobs, chunksize=8), 1):
            if row["prompt_sha256"] in prompt_hashes or row["semantic_key"] in semantic_keys:
                raise ValueError("duplicate prompt or semantic question")
            split = row["split"]
            prompt_hashes.add(row["prompt_sha256"])
            semantic_keys.add(row["semantic_key"])
            rows[split].append(row)
            task_digests.append([row["path"], row["task_sha256"]])
            families[row["family"]] += 1
            split_counts[split] += 1
            split_families[split][row["family"]] += 1
            if n % 500 == 0:
                print("packaged", n, "/", total, flush=True)
    if entries & {s.split(":", 1)[1] for s in exclusion["sources"] if s.startswith("pdb:")}:
        raise ValueError("benchmark entry overlap")
    if sequence_hashes & set(exclusion["sequence_sha256s"]):
        raise ValueError("benchmark sequence overlap")
    if cluster_ids & set(exclusion["rcsb30_clusters"]):
        raise ValueError("benchmark homology cluster overlap")
    if set(families) != set(V1_FAMILIES):
        raise ValueError("not all 19 coordinate families are present")
    for family, count in labels.items():
        expected = 2 if family == "S03" else 3
        if len(count) != expected:
            raise ValueError(f"categorical family {family} lacks all {expected} labels")
    for split, data in sorted(rows.items()):
        data.sort(key=lambda r: r["path"])
        for offset in range(0, len(data), 500):
            dest = output / "data" / f"{split}-{offset // 500:05d}.parquet"
            pq.write_table(
                pa.Table.from_pylist(data[offset : offset + 500]), dest, compression="zstd", row_group_size=50
            )
    ledger = []
    for path in sorted((directory / "shards").glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            shard = json.load(stream)
        for rejection in shard["rejections"]:
            ledger.append(
                {
                    "source": path.stem.removesuffix(".json"),
                    "family": rejection["family"],
                    "status": "rejected",
                    "reason": rejection["reason"],
                    "detail": json.dumps(rejection["detail"], sort_keys=True),
                }
            )
    acquisition_rejections = json.loads((directory / "source_rejections.json").read_text())
    if (directory / "pairs.json").exists():
        acquisition_rejections += json.loads((directory / "pairs.json").read_text())["rejections"]
    for rejection in acquisition_rejections:
        ledger.append(
            {
                "source": rejection["entry"],
                "family": "*",
                "status": "rejected",
                "reason": rejection["reason"],
                "detail": json.dumps(rejection, sort_keys=True),
            }
        )
    pq.write_table(pa.Table.from_pylist(ledger), output / "ledger.parquet", compression="zstd")
    (output / "audits").mkdir(exist_ok=True)
    write_json(output / "audits/exclusion.json", exclusion)
    write_json(output / "audits/sources.json", {entry: audits[entry] for entry in sorted(entries)})
    write_json(output / "audits/source_groups.json", {entry: groups[entry] for entry in sorted(entries)})
    for name in ("selection.json", "sources.json", "phosphoproteins.query.json", "pool_summary.json"):
        if (directory / name).exists():
            destination = "source_pool.json" if name == "sources.json" else name
            shutil.copyfile(directory / name, output / "audits" / destination)
    snapshot_files = [
        *repo.glob("src/pdbthink/**/*.py"),
        *repo.glob("src/pdbthink/data/*"),
        repo / "configs/definitions_v1.yaml",
        repo / "pyproject.toml",
        repo / "LICENSE",
        repo / "README.md",
        repo / "docs/coordinate-task-generation.md",
        repo / "tests/test_taskgen.py",
        repo / "tests/test_scoring.py",
        repo / "docs/teacher-traces-glm53/clash-audit-task.json",
        repo / "docs/teacher-traces-glm53/tolerance-boundary-audit.json",
    ]
    from .harbor import archive

    code = {str(p.relative_to(repo)): p.read_bytes() for p in sorted(snapshot_files) if p.is_file()}
    (output / "generator_source.tar.gz").write_bytes(archive(code))
    shutil.copyfile(repo / "LICENSE", output / "LICENSE")
    shutil.copyfile(repo / "docs/coordinate-task-generation.md", output / "USAGE.md")
    reproduction = {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.glob("*.json"))}
    reproduction.update({p.name: p.read_bytes() for p in sorted(directory.glob("clusters-*.txt"))})
    reproduction.update(
        {str(p.relative_to(directory)): p.read_bytes() for p in sorted((directory / "audits").glob("*.json"))}
    )
    (output / "reproduction_inputs.tar.gz").write_bytes(archive(reproduction))
    manifest = {
        "dataset": "open-athena/pdbthink-coordinate-tasks",
        "version": version,
        "prompt_version": PROMPT_VERSION,
        "scoring_version": SCORING_VERSION,
        "retained_parent_tasks": len(retained),
        "parent_task_set_fingerprint": (
            json.loads((parent / "manifest.json").read_text())["task_set_fingerprint"] if parent else None
        ),
        "seed": source["seed"],
        "task_count": total,
        "family_counts": dict(sorted(families.items())),
        "split_counts": dict(split_counts),
        "split_family_counts": {k: dict(v) for k, v in split_families.items()},
        "category_counts": {k: dict(v) for k, v in labels.items()},
        "source_entries": len(entries),
        "source_groups": len({groups[e] for e in entries}),
        "exclusion_fingerprint": exclusion["fingerprint"],
        "audit": {
            "entry_overlap": 0,
            "sequence_overlap": 0,
            "rcsb30_overlap": 0,
            "duplicate_prompts": 0,
            "duplicate_semantic_tasks": 0,
            "serialised_coordinate_oracle_checks": total,
            "gold_reward_passes": total,
            "empty_reward_rejections": total,
            "wrong_answer_reward_rejections": total,
        },
        "task_set_fingerprint": stable_hash(sorted(task_digests)),
        "code_hashes": {k: sha256_bytes(v) for k, v in code.items()},
        "data_hashes": {
            str(p.relative_to(output)): sha256_bytes(p.read_bytes())
            for p in sorted((output / "data").glob("*.parquet"))
        },
        "format": "Harbor 1.4; TaskTrove-compatible binary archive columns",
        "reward": "exact_correct; partial credit in verifier details only",
        "token_counts": "cl100k_base reference counts, not native model counts",
        "limitations": [
            "Experimental structures may occur in foundation-model pretraining.",
            "This is an automatically verified task release, not a curator-reviewed benchmark.",
            "Multiple tasks can share a source protein; use the supplied grouped splits.",
            "Tool-free evaluation requires the supplied adapter or equivalent enforced controls.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    write_card(output, manifest)
    return manifest


def write_card(output: Path, manifest: dict) -> None:
    counts = "\n".join(f"| {k} | {v} |" for k, v in manifest["family_counts"].items())
    splits = "\n".join(f"| {k} | {v} |" for k, v in manifest["split_counts"].items())
    card = f"""---
pretty_name: PDBThink Coordinate Tasks
license: apache-2.0
language: [en]
task_categories: [text-generation]
tags: [protein, structural-biology, coordinate-reasoning, harbor, reinforcement-learning, no-tools]
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: validation
        path: data/validation-*.parquet
      - split: test
        path: data/test-*.parquet
---

# PDBThink Coordinate Tasks

{manifest["task_count"]:,} new coordinate-interpretation tasks across all 19 active PDBThink families,
from {manifest["source_entries"]:,} experimental PDB entries in {manifest["source_groups"]:,} source groups.
Version {manifest["version"]}; deterministic seed {manifest["seed"]}.

The model receives sanitised, rotated, rounded protein coordinates and a question.
It must answer without tools. This release contains **no sequence-to-structure
prediction tasks and no retired MECH tasks**. It is intended for additional evaluation,
RL with deterministic rewards, and generating oracle-checked SFT demonstrations.

## Revision history

- **v1.3.0:** retains all 100,000 v1.2.0 task identities, displayed coordinates,
  gold answers and grouped splits. Prompt v5 states the clash exclusions and
  ranking rule for G04. Scorer 1.1.0 compares the decimal representations of
  parsed numeric values exactly at the inclusive tolerance boundary; it adds
  no tolerance slack. Both coordinate triples and distances use this rule.
- **v1.2.0:** expands the corrected v1.1.0 release from 10,000 to 100,000 tasks.
- **v1.1.0:** corrects the S03/S05/S09 answer-format examples with prompt v4.

The release comparison and native-verifier regression evidence are in
`audits/contract_revision/`. The geometry definitions and gold labels are unchanged.
Historical releases remain available by their version tags. The published
[GLM teacher traces](https://huggingface.co/datasets/open-athena/pdbthink-glm53-teacher-traces/tree/v1.0.0)
were generated and scored against v1.2.0; their prompts, scores and retry histories
have not been rewritten. In particular, they do not constitute new evaluations of
the clarified G04 prompts. Use the verifier bundled with each task release.

## Relationship to the benchmark

These are new tasks, not the published benchmark questions. Acquisition excluded every
source configured in the frozen benchmark inventory, exact full-entity and observed-chain
protein sequences, and RCSB 30% sequence clusters associated with those sources. Every
protein partner in a candidate entry was audited, even when only one chain was shown.
The frozen inventory, source checks and file hashes are in `audits/` and `manifest.json`.
All released source, sequence and cluster intersections with that inventory are zero.
This does **not** establish absence from foundation-model pretraining.

Gold answers are recomputed from the coordinates actually shown, after transformation,
rounding and permitted cropping, and checked again after parsing the displayed PDB text.
Existing PDBThink operational definitions and family oracles are used unchanged.
Scientific ambiguities and failed generation criteria are recorded in `ledger.parquet`.
This is an automatically verified training/evaluation release, not a curator-reviewed benchmark.

## Format

The archive columns follow [Open Athena Task Trove](https://huggingface.co/datasets/open-athena/task-trove):
`path`, `source`, `family`, `template_id`, `converter`, `mode`, `dockerfile_id`, `language`,
`tags`, `has_solution`, `task_binary`, and `solution_binary`.
Each binary is a gzip-compressed tar archive with deterministic member order and timestamps.

`task_binary` contains `instruction.md`, `task.toml`, `prompt.json`, `environment/Dockerfile`,
`tests/test.sh`, `tests/verifier.toml`, and a self-contained Python verifier with hidden gold.
**Oracle solutions are separate in `solution_binary`**, never baked into the environment.
The custom `pdbthink-coordinate` verifier is bundled; it is not a built-in tasktrove-verify mode.
The verifier writes exact correctness to `/logs/verifier/reward.txt` and diagnostic partial
credit to `details.json`. Contact sets must be completely correct for unit reward.

Additional columns include `split`, `source_group`, `input_tokens`, `atom_count`,
`answer_schema`, `prompt_sha256`, `semantic_key`, `task_sha256`, `source_entries`, and
`provenance`. Provenance and test/solution files are evaluator-only and must never be
included in a model prompt. Token counts use cl100k_base as a reference; compute exact
native token counts before selecting an evaluation cohort or assigning output budgets.

```python
from datasets import load_dataset
tasks = load_dataset("open-athena/pdbthink-coordinate-tasks", split="train")
contacts = tasks.filter(lambda row: row["family"] in ["S06", "I01"])
```

## Tool-free execution

Ordinary terminal agents do not satisfy this scientific protocol. The source snapshot includes
`pdbthink.taskgen.no_tools_agent:CoordinateNoToolsAgent`, a Harbor adapter which sends only
the system/user prompt, explicitly disables model tools and the OpenRouter web plugin,
retains raw responses, audits tool events, and writes the returned answer to `/app/answer.txt`
on the host's behalf. It never executes a model-generated command or tool call.
Use an exact per-prompt budget manifest containing native input counts, context and output
limits, tokenizer revision and endpoint-limit provenance. The adapter allocates the largest
supported output budget that fits. Truncated responses remain flagged for budget audit.

For offline oracle validation only, unpack `solution_binary` beside the task and run Harbor's
oracle agent. Do not unpack oracle solutions into an evaluated model's environment.
For SFT generation, provide only `prompt.json` to the teacher; check its response using the
same verifier. Oracle answer files are final answers, not teacher-written worked solutions.

## Splits and composition

All entries connected by exact protein sequences or RCSB 30% clusters stay in one split.
The target allocation is 90% train / 5% validation / 5% test at source-group level, so task
counts need not have those exact proportions. Do not randomly split rows or rotate a
training structure and call it an independent held-out example. Multiple tasks may share
a source structure; no task is counted solely because of a different rigid rotation.

| Split | Tasks |
|---|---:|
{splits}

| Family | Tasks |
|---|---:|
{counts}

The selector balances families subject to the available unambiguous, non-overlapping
source pool. Family counts and categorical label distributions are recorded in the manifest.

## Reproduction and licence

`generator_source.tar.gz` contains the Python package, versioned definitions and project
metadata. `reproduction_inputs.tar.gz` contains the frozen source selections, exclusion
inventory, sequence-cluster snapshots and acquisition audits; unpack it into the work directory.
Acquisition is the network stage; generation, verification and packaging run offline.
Source selections, raw-source hashes and exclusion snapshots are retained for audit.
See the bundled `pdbthink.taskgen` CLI and `manifest.json` for build settings and code hashes.

Task-generation code is distributed under the repository's Apache 2.0 licence. Protein coordinates
come from the public Protein Data Bank; original entries are identified in evaluator-only
provenance. No assertion of experimental folding ground truth is made beyond the supplied
coordinate-reasoning definitions.
"""
    (output / "README.md").write_text(card)
