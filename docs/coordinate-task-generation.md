# Coordinate tasks for evaluation and post-training

The `pdbthink.taskgen` pipeline packages new tasks for all 19 active coordinate
families. It reuses the existing generators, operational definitions and answer
scorers. It does not modify the frozen benchmark or include MECH/F01/F02.

Install `.[taskgen,tokenizer]` for acquisition, generation and Parquet export.
Harbor integration additionally needs Python 3.12+ and `.[harbor]`.

The [published release](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks)
contains 10,000 tasks from 1,922 PDB entries, with all 19 families in every split:
9,157 train, 458 validation and 385 test. The release record is in
`docs/coordinate-tasks-v1.json` for the original v1.0.0 release. Version 1.1.0
corrects the category-format examples in S03, S05 and S09 with prompt version v4;
all 10,000 task identities, displayed coordinates, answers and splits are retained.
Its pinned release record is `docs/coordinate-tasks-v1.1.json`.
Historical releases remain available by their Hugging Face revision or version tag.
Data archives and acquisition snapshots live on
Hugging Face; generator code, tests and usage instructions live in this repository.

## Build stages

On a fresh checkout, initialise the work directory with the published, frozen
source selections and exclusion inventory. This supplies the RCSB query results
and cluster files otherwise read from a local `data/expansion_sources` cache.

```bash
hf download open-athena/pdbthink-coordinate-tasks reproduction_inputs.tar.gz --repo-type dataset --revision c18c100cd68f55e5260ea09d2882908962532dd8 --local-dir /tmp/pdbthink-v1-inputs
mkdir -p data/taskgen_v1
tar -xzf /tmp/pdbthink-v1-inputs/reproduction_inputs.tar.gz -C data/taskgen_v1
```

For byte-for-byte reproduction of the published release, also use
`generator_source.tar.gz` at that same Hub revision. It freezes the complete
scientific implementation and prompt versions used for v1; the GitHub code can
continue to evolve independently. Dependency versions are in `validation.json`.

```bash
python -m pdbthink.taskgen acquire --work data/taskgen_v1 --workers 12
python -m pdbthink.taskgen acquire-pairs --work data/taskgen_v1 --workers 8
python -m pdbthink.taskgen build --work data/taskgen_v1 --workers 24
python -m pdbthink.taskgen build-pairs --work data/taskgen_v1 --workers 16
python -m pdbthink.taskgen export --work data/taskgen_v1 --count 10000
python -m pdbthink.taskgen validate --dataset data/taskgen_v1/release --workers 16
```

Only acquisition uses the network. Source selections, sequence clusters and
benchmark exclusions are frozen before generation. Source-view shards make
builds resumable; use a fresh work directory when changing the seed, per-source
sampling or scientific implementation. A release directory cannot be silently
overwritten.

To revise or extend a published release, download that release and pass its
local directory with `export --parent PATH --version VERSION`. Every parent task
is retained; export fails if any parent split or source group would change.
The v3-to-v4 migration replaces only the known category-format suffix and
recounts prompt tokens. It never regenerates coordinates or changes an answer.

For a larger pool, copy the frozen acquisition inputs to a fresh work directory,
then run `build --per-family 8` and `build-pairs`. The post-training sampler
extends the benchmark's short proposal list with further distinct admissible
questions. Use `export --count 100000 --parent PATH --version 1.2.0` to retain
the corrected 10,000 tasks and add 90,000 new semantic questions. No acquisition
is needed when using the same frozen pool. This increases questions per source;
it does not promise ten times as many independent protein groups or ten times
as many examples of every rare family.

The source pool uses the existing frozen RCSB expansion queries for small
proteins, ligands and interfaces, with a separate query for phosphoproteins.
The exclusion snapshot covers every configured benchmark source, its complete
and observed protein sequences, and every RCSB 30% sequence cluster containing
one of those entries. Candidate entries are checked across all protein
partners, even when the task shows one chain. The snapshot includes its input
file hashes and fails closed if a benchmark source cannot be sequence-audited.
Published audit records contain sequence hashes; they do not distribute the
excluded benchmark's raw sequences.

T01 uses additional experimental structures in the same RCSB 90% sequence
cluster, then applies the benchmark's sequence mapping, alignment and
unambiguous contact-change checks. Distinct candidate-pair lists are distinct
questions; different rigid rotations alone never count as new instances.

Every retained task has its gold recomputed from the rotated, rounded, possibly
cropped coordinates. A separate parser reconstructs the structures from the
actual PDB text and runs the oracle again. Disagreement rejects the task.
The verifier also checks that the oracle answer receives exact reward 1 and
that empty and deliberately incorrect answers receive exact reward 0.

## Release format

Parquet rows use Task Trove's metadata columns, `task_binary`, and separate
`solution_binary`. Archives have deterministic timestamps and contain normal
Harbor 1.4 task directories. Hidden tests include a self-contained copy of the
coordinate answer parser/scorer; no installation or model judge is needed at
verification time. The custom `pdbthink-coordinate` verifier mode is bundled
and is not a built-in mode of the separate `tasktrove-verify` package.

The primary reward is exact correctness, including the benchmark's numerical
tolerances. Partial-set F1 and other diagnostics are written separately. Gold,
provenance, oracle solutions and source identifiers are evaluator-only data.
Teacher/Snowball inputs must come exclusively from `prompt.json` or the exact
Harbor instruction.

All entries connected by protein sequences, RCSB 30% clusters, or a two-state
pair belong to one source group. Groups are assigned deterministically to
train/validation/test in a target 90/5/5 allocation. Multiple questions may
share a protein, but related source groups never cross these splits. These
checks establish separation from the frozen PDBThink inventory, not freedom
from all foundation-model pretraining data.

## Tool-free Harbor adapter

For Snowball's native 32,768-token window, install `.[taskgen-context]` and
cache the pinned checkpoint's tokenizer files, then run:

```bash
python -m pdbthink.taskgen.context --dataset coordinate-tasks --model open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38 --revision cfc1d845dae89b067cdc7250d0164abefa5a69cf
```

This offline audit uses the native chat template with thinking enabled. It
writes `snowball_context.parquet`, keyed by task and prompt hash, and a summary
of how many tasks leave 1K, 4K, 8K or 16K output tokens. These are cohort filters;
evaluation must still allocate the maximum available endpoint output budget.
For SFT, check the actual teacher completion length, including reasoning and
training-template overhead, before admitting a complete training example.

Download and unpack a chosen evaluation cohort:

```bash
hf download open-athena/pdbthink-coordinate-tasks --repo-type dataset --local-dir coordinate-tasks
python -m pdbthink.taskgen unpack --dataset coordinate-tasks --output harbor-tasks --split test --families G01 S06 I01 --limit 100
```

This leaves oracle solutions out of the task directories. Add `--include-solutions`
only when preparing an offline Harbor oracle check.

Use `pdbthink.taskgen.no_tools_agent:CoordinateNoToolsAgent` with Harbor's custom
agent import option. It accepts `endpoint`, `api_key_env`, `budget_manifest`,
and optional `request_extra`. The host sends one OpenAI-compatible chat
completion with tools explicitly disabled, audits the raw response, and writes
the returned text to `/app/answer.txt`. No model-generated tool request or
command is executed. Generic terminal agents do not satisfy this protocol.

The budget manifest is a JSON object keyed by the row's `prompt_sha256`:

```json
{
  "<prompt_sha256>": {
    "input_tokens": 15000,
    "context_window": 32768,
    "native_output_limit": 32768,
    "tokenizer_revision": "<exact native tokenizer and chat-template revision>",
    "endpoint_limit_source": "<verified endpoint configuration or documentation>"
  }
}
```

These numbers are illustrative. Populate them by tokenising each exact prompt
with the endpoint's actual tokenizer, chat template and thinking configuration.
The adapter requests `min(native_output_limit, context_window - input_tokens)`;
it never substitutes the dataset's reference cl100k count. Raw requests,
responses, budgets, usage and termination reasons are retained. Truncation
requires an endpoint/budget audit before an evaluation is treated as final.
No provider call is made by task generation or release validation.

For mechanical Harbor validation, merge the task and solution archives on the
host and use the oracle agent. Do not merge oracle archives for actual model
evaluation. For teacher-generated SFT, retain only verified outputs, inspect
important intermediate calculations, and keep the grouped split boundaries.
