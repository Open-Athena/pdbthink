# CLAUDE.md

Guidance for agents working in this repository.

## What this is

A benchmark for tool-free reasoning over protein structures. The scientific
contract is narrow and strict: **a gold answer must be recomputable from exactly
the coordinates the model is shown.** Most of the design follows from that.

## Setup

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,tokenizer]"
```

```bash
.venv/bin/python -m pytest -q
```

Tests run offline against committed fixtures and take a few seconds. Run them
before and after any change to geometry, prompts or scoring.

## Rules that are not negotiable

1. **Never change a gold label without bumping a version.** Anything in
   `configs/definitions_v1.yaml`, `prompts/library.py` or `chem.py` that can move
   a label requires a `definition_version` or `PROMPT_VERSION` bump. The prompt
   library and the chemistry tables are content-hashed for exactly this reason.

2. **Gold answers come from oracles, never from proposals.** `propose` finds
   question parameters; `oracle` recomputes the answer from a displayed
   structure. If you find yourself storing a computed answer during `propose`,
   stop: that answer would have come from the pre-rotation coordinates.

3. **Never silently resolve a scientific ambiguity.** Yield a `Rejection` with a
   machine-readable reason and let it reach `rejections.jsonl`. A near-tie that
   is quietly broken is worse than a missing instance.

4. **Nothing about the source may reach a prompt.** Entry IDs, publications,
   original ligand component codes, secondary-structure records, B-factors,
   pLDDT. `validate` checks this; do not weaken the check.

5. **Determinism is a feature.** Derive seeds with `derive_seed(...)` from stable
   identifiers. No wall-clock, no global RNG, no reliance on dict or filesystem
   ordering.

## Where things live

`docs/architecture.md` has the full map. The parts most often edited:

- a new question family → `src/pdbthink/generators/`, then `V1_FAMILIES`, a target
  in `configs/dataset_v1.yaml`, and a golden test;
- a new geometric definition → `src/pdbthink/geometry/`, with its constants in
  `configs/definitions_v1.yaml`, never hard-coded at the call site;
- prompt wording → `src/pdbthink/prompts/library.py` only.

## Checking your work

```bash
.venv/bin/python -m pdbthink.cli build --config configs/dataset_smoke.yaml --output /tmp/smoke
```

```bash
.venv/bin/python -m pdbthink.cli validate --dataset /tmp/smoke --config configs/dataset_smoke.yaml
```

Then the no-model check, which must score exactly 1.0:

```bash
.venv/bin/python -m pdbthink.cli evaluate --dataset /tmp/smoke --model-config configs/models/mock_gold.yaml --output /tmp/run
```

```bash
.venv/bin/python -m pdbthink.cli score --dataset /tmp/smoke --responses /tmp/run --output /tmp/scores
```

A full build of `configs/dataset_v1.yaml` takes a few minutes and needs the
structure cache; `acquire` fills it and is the only step that touches the network.

## Debugging a family that produces nothing

Read `rejections.jsonl` first — it is grouped by family, protein and reason and
usually answers the question directly. Common causes: the structure exceeds
`MAX_UNCROPPABLE_ATOMS` for a family that may not be cropped, a margin in
Appendix A is genuinely not satisfied by that structure, or the protein pool for
that family in `family_proteins` is too small.

## Style

Match the surrounding code: type hints, docstrings that say *why* rather than
*what*, and comments reserved for decisions a reader would otherwise have to
reverse-engineer. British-neutral spelling in prose; identifiers follow whatever
the surrounding module already uses.
