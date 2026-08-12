# Architecture

## Data flow

```
sources.yaml ──acquire──▶ data/cache/{structures,metadata}
                              │
              dataset.yaml ───┼───▶ DatasetBuilder
        definitions_v1.yaml ──┘        │
                                       ├─ load_processed   sanitise, altlocs, anonymise
                                       ├─ Generator.propose  parameters + margins + rejections
                                       ├─ transform + crop   rotate, round, excerpt
                                       ├─ Generator.oracle   gold from displayed coordinates
                                       └─ build_prompt       system + user prompt
                                       ▼
                          instances.jsonl · renders.jsonl · prompts/ · rejections.jsonl
                                       │
                     ┌─────────────────┼──────────────────┐
                     ▼                 ▼                  ▼
                  validate          review            evaluate
                                       │                  │
                              decisions.jsonl        results.jsonl
                                       │                  │
                                 build --accepted-only    score ──▶ report
```

## Package layout

| module | responsibility |
| --- | --- |
| `config` | loads the versioned definitions and dataset configuration, hashing both |
| `chem` | static chemistry: one-letter codes, vdW radii, max ASA, the `ccd_lite_v1` bond dictionary |
| `acquisition` | download, cache and record provenance for PDB and AlphaFold DB entries |
| `preprocessing` | the internal structure model, mmCIF loading and A.2/A.3 sanitisation, rigid transforms, cropping |
| `geometry` | DSSP, SASA, contacts, salt bridges, disulfides, metals, clashes, torsions, alignment |
| `representations` | minimal PDB and normalized-coordinate renderers, token counting |
| `prompts` | the versioned prompt library and prompt assembly |
| `generators` | one generator per question family, plus the shared framework |
| `mechanistic` | the six curated episodes and their processing pipeline |
| `dataset` | the builder that turns proposals into instances and rendered variants |
| `validate` | the definition-of-done checks |
| `scoring` | answer parsing, canonicalisation and every deterministic scorer |
| `evaluation` | the resumable model runner and the offline scorer |
| `reporting` | metrics, clustered bootstrap and report rendering |
| `review_ui` | the curator interface |
| `evalchemy_task` | the Evalchemy `BaseBenchmark` adapter |

## Two ideas hold the design together

### 1. Propose and oracle are separate

A generator does two things, and keeping them apart is what makes the
correctness guarantees mechanical rather than aspirational:

```python
class Generator:
    def propose(self, ctx) -> Iterator[Proposal | Rejection]: ...
    def oracle(self, structure, parameters, definitions, analysis) -> OracleResult: ...
```

`propose` searches one structure for question parameters that satisfy the
Appendix A margins. `oracle` recomputes the answer from *any* displayed
structure given those parameters. Because the oracle takes the structure as an
argument, the builder can:

- compute the gold answer from the rotated, rounded, possibly cropped
  coordinates the model actually reads;
- re-run it on a second rotation and fail the build if the label moves;
- re-run it on the cropped structure and fail if the crop changed the answer;
- let `validate` re-derive labels from a shipped dataset.

A generator that cannot recompute its own answer cannot ship.

### 2. A displayed structure is the unit of truth

`DisplayedStructure` is a structure after the rigid transform and after rounding
to three decimals. Everything downstream — oracles, evidence, rendering, hashes
— consumes that object, so there is no path by which a gold label can be derived
from coordinates that differ from the ones in the prompt.

## Adding a question family

1. Subclass `Generator` in `src/pdbthink/generators/`, set `family`, `version`,
   `answer_schema` and `level`.
2. Implement `propose`, `oracle` and `question`; return `Rejection` objects with
   a machine-readable reason whenever a candidate fails a criterion.
3. Add the question text to `prompts/library.py` and bump `PROMPT_VERSION` if any
   existing text changes.
4. Register it, add it to `V1_FAMILIES`, give it a target in the dataset config.
5. Add a golden test that checks the oracle against an independently known fact
   and confirms rotation invariance.

The extension backlog in specification section 8 (polymer content, complete
disulfide lists, proline cis/trans, 3-10 helices, articulation residues, further
two-state comparisons) needs nothing beyond this: the geometry for most of it is
already implemented and tested in `pdbthink.geometry`.

## Determinism

Every seed is derived from stable identifiers rather than iteration order:

```python
rotation_seed = derive_seed(dataset_seed, semantic_instance_id, "rotation", variant_index)
```

`derive_seed` hashes its arguments, so rebuilding after adding an unrelated
protein leaves existing instances byte-identical. Rotations use PCG64 seeded that
way (`pcg64_unit_quaternion_v1`), and no code calls the global RNG, the clock or
the filesystem order.
