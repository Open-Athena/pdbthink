# pdbthink

**Evaluating tool-free reasoning over protein structures.**

pdbthink is a reproducible, single-turn benchmark that measures how well language
models parse and reason about protein structures when they are given nothing but
the coordinates: no tools, no internet, no precomputed annotation. Each item
shows a model one or two sanitised, randomly rotated structures in minimal PDB
format and asks a question whose answer is recomputed from exactly those
coordinates. Grading is deterministic — there is no LLM judge anywhere in the
primary score.

The benchmark separates capabilities that are usually tangled together:

| level | what it isolates | families |
| --- | --- | --- |
| parsing | reading a fixed-column coordinate file | P01–P03 |
| geometry | elementary 3D arithmetic | G01–G04 |
| local structure | turning geometry into structural-biology concepts | S01–S09 |
| interface & networks | reasoning across chains and contact graphs | I01, N01 |
| two-state | comparing two conformations | T01 |
| mechanism | connecting a local change to a functional consequence | 6 curated episodes |

## Quick start

```bash
pip install -e ".[dev,tokenizer]"
```

```bash
structural-reasoning acquire --config configs/dataset_smoke.yaml
```

```bash
structural-reasoning build --config configs/dataset_smoke.yaml --output datasets/smoke
```

```bash
structural-reasoning validate --dataset datasets/smoke --config configs/dataset_smoke.yaml
```

Everything after `acquire` runs offline.

### Check the pipeline without spending anything

The `mock-gold` provider answers every prompt with its own gold label, so a run
must score exactly 1.0. This is the no-model validation step of the
cost-controlled workflow and the fastest way to confirm an environment is sane.

```bash
structural-reasoning evaluate --dataset datasets/smoke --model-config configs/models/mock_gold.yaml --output runs/mock
```

```bash
structural-reasoning score --dataset datasets/smoke --responses runs/mock --output scores/mock
```

### Evaluate a real model

Serve a local model with any OpenAI-compatible server (vLLM, llama.cpp, …):

```bash
vllm serve Qwen/Qwen3-8B --port 8000
```

```bash
structural-reasoning evaluate --dataset datasets/smoke --model-config configs/models/local_vllm.yaml --output runs/qwen3-8b --resume
```

```bash
structural-reasoning score --dataset datasets/smoke --responses runs/qwen3-8b --output scores/qwen3-8b
```

```bash
structural-reasoning report --scores scores/qwen3-8b --output reports/qwen3-8b
```

`--resume` reuses every completion already stored, so an interrupted run never
repeats a paid call. API models use the same commands with
`configs/models/anthropic_opus.yaml` or `configs/models/openai_gpt.yaml`.

### Review candidates

```bash
structural-reasoning review --dataset data/datasets/candidates_v1 --decisions data/review_decisions/v1.jsonl
```

The review interface opens at <http://127.0.0.1:8787>; to let colleagues in, see
[docs/deployment.md](docs/deployment.md). Each candidate shows an
interactive 3D view with the queried residues, gold answer and evidence
highlighted, the exact model-visible prompt, the continuous measurements and
ambiguity margins behind the label, and — curator-only — the source entry,
release date, method and file hashes. `a` accepts, `r` rejects (a reason is
required), `j`/`k` move through the list. Decisions append to a JSONL file that
the builder consumes:

```bash
structural-reasoning build --config configs/dataset_v1.yaml --output datasets/final --decisions data/review_decisions/v1.jsonl --accepted-only
```

## Command surface

| command | what it does | network |
| --- | --- | --- |
| `acquire` | download and cache PDB/AFDB sources, recording hashes and metadata | yes |
| `build` | generate semantic instances and rendered variants | no |
| `validate` | check schemas, provenance, budgets, gold consistency | no |
| `review` | launch the curator interface | no¹ |
| `evaluate` | run a model over the dataset, resumable | model calls |
| `score` | score stored responses | no |
| `report` | aggregate metrics, bootstrap CIs, controls | no |

¹ the 3D viewer library is fetched from a CDN the first time; everything else,
including the prompts and decisions, is local.

`pdbthink` and `structural-reasoning` are the same program.

## How an item is made

1. **Acquire** the mmCIF from RCSB or AlphaFold DB; store it with its sha256,
   release date, method and resolution.
2. **Sanitise** it: build the biological assembly when the task needs one, drop
   hydrogens, waters and crystallisation additives, resolve alternate locations
   per residue, anonymise ligand component codes, renumber serials, flatten
   occupancy and zero B-factors (which also removes AlphaFold pLDDT).
3. **Propose** a question. Each generator searches for parameters that satisfy
   the Appendix A ambiguity margins and records, in machine-readable form, why
   every candidate was accepted or rejected.
4. **Render**: apply a proper random rotation derived from the instance's seed,
   round to three decimals, crop if the family allows it and the budget requires
   it, and emit minimal PDB plus a matched normalized-coordinate table.
5. **Recompute** the gold answer from the rendered coordinates, and confirm it is
   unchanged under a second rotation and in the paired representation.
6. **Review**: a curator accepts, rejects or annotates. Only accepted instances
   enter the final dataset.

Steps 3–5 are why the guarantees are mechanical rather than aspirational: gold
answers are computed from the same numbers the model reads, so a rotation
variant, a cropped variant and a normalized-coordinate variant either agree or
the build fails.

## Documentation

- [docs/architecture.md](docs/architecture.md) — package layout and data flow
- [docs/definitions.md](docs/definitions.md) — the operational definitions and
  where each one is enforced
- [docs/evaluation.md](docs/evaluation.md) — protocol, metrics and controls
- [docs/evalchemy.md](docs/evalchemy.md) — running the benchmark under Evalchemy
- [docs/deployment.md](docs/deployment.md) — letting colleagues reach the review interface

## Repository layout

```
configs/          versioned definitions, dataset configs, model configs
data/manifests/   the reproducible list of source structures
data/datasets/    built datasets (gitignored except the frozen manifests)
src/pdbthink/     the package
tests/            unit, golden and end-to-end tests
```

## Development

```bash
pytest -q
```

The suite runs entirely offline against committed fixtures and covers geometry,
preprocessing, prompt rendering, answer parsing, every scorer, rotation
invariance, rebuild determinism and the full build → validate → evaluate → score
→ report path.

Release dates are recorded for every source structure as a contamination
covariate. They are not a guarantee.

## License

Apache 2.0. Structures are redistributed only as small test fixtures; everything
else is downloaded from RCSB and the AlphaFold DB under their own terms.
