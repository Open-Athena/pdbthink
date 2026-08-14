# Evaluation protocol

## Recommended final protocol

- three independent completions for every rendered prompt;
- five to ten completions on a stratified 20-instance reliability subset;
- the mean score across completions as the primary model score — never pass@k;
- hierarchical bootstrap confidence intervals clustered by protein or paper;
- paired confidence intervals for the representation and rotation contrasts.

Set `completions: 3` in the model config for the main run, then a second run with
`completions: 10 --families ...` restricted to the reliability subset.

## Cost-controlled workflow

The order below is designed so that anything that can fail cheaply fails before
a frontier model is charged for it.

1. **No-model validation.** `mock_gold.yaml` answers every prompt with its own
   gold label. Scoring the run must return exactly 1.0; anything else means a
   scorer, a prompt or a gold label is wrong.
2. **Smoke set.** `configs/dataset_smoke.yaml` — one instance per family, every
   answer schema, small structures.
3. **Development set.** A 25–30 instance build on proteins disjoint from the
   final set where possible.
4. **Local model.** An open 8–32B instruct model through vLLM.
5. **Cheap API model**, smoke set only.
6. **Freeze** definitions, prompts and scoring.
7. **Build and review** the final set.
8. **Frontier models**, all effort levels, repeated completions.

## What is scored

| answer schema | primary metric | secondary |
| --- | --- | --- |
| residue, atom, category, boolean, multiple choice, integer | exact accuracy | — |
| numeric coordinate triple | all three components within 0.001 Å | components within tolerance |
| distance | absolute error ≤ 0.02 Å | absolute error |
| residue / interaction set | set F1 | exact-set accuracy, precision, recall |
| two-state gained/lost | mean of the two set F1 scores | exact set for both arms |
| ordered path | exact ordered accuracy | per-position accuracy |
| mechanistic episode | per-field scores | their mean |

Malformed, refused and truncated answers score zero and are reported under their
own failure category, so a low score caused by formatting is never mistaken for a
low score caused by reasoning.

The headline number is the **macro average across question families**. Micro
averages are reported too, but families with long answer lists must not dominate
the headline.

## Controls

**Representation.** A matched subset of each instance is rendered both as minimal
PDB and as a normalized coordinate table containing exactly the same atoms,
coordinates and labels. The paired difference isolates the cost of parsing
fixed-column PDB syntax from spatial reasoning itself. Note that the two formats
also differ in token cost — about 41 versus 22 `cl100k_base` tokens per atom —
which is worth stating alongside the accuracy gap.

**Rotation.** About 20% of instances get a second variant under a different
proper random rotation. Distances and contacts are invariant by construction, so
any paired difference measures the model's sensitivity to coordinate frame rather
than to structure. For two-state items both states receive the *same* transform:
independently rotating paired states is a harder alignment task and is out of
scope for V1.

**Context-only.** The same question with the coordinates removed. It applies to
mechanistic episodes and to `P01`, `P02`, `S03`, `S04`, `S05`, `S08` and `S09`,
and it measures a different thing in each case.

For a mechanistic episode the prompt still describes the experiment — the
receptor, the allosteric ligand, what the binding assay showed — so a model that
recognises the system can answer from the literature. The gain over
context-only there separates genuine use of the supplied structure from recall
of a famous result.

For the seven automatic families the sanitised question identifies nothing: no
entry ID, no organism, no ligand code. A model given `How many protein residues
are present in chain A?` and no coordinates cannot know which protein it is
being asked about. What the control measures there is the **guessing floor** —
what the question text alone is worth, including whatever prior the model has
over answers. That floor is neither uniform nor negligible:

| family | answer space | why the floor is worth knowing |
| --- | --- | --- |
| `P01` | chain identifiers | answering `A` is right much of the time |
| `P02` | an integer | unconstrained; the floor should be near zero |
| `S03` | buried / exposed | a coin flip, before any prior over which is commoner |
| `S04` | helix / strand / coil | three-way, and `coil` is the plurality class |
| `S05` | three fold classes | three-way |
| `S08` | a residue label | unconstrained; should sit near zero |
| `S09` | g+ / t / g- | three-way, and `g-` is the commonest rotamer |

A family's coordinate score only means something relative to its floor. The
classification families are where this bites hardest: a model scoring 0.55 on
`S03` has demonstrated nothing at all.

The seven were chosen because they ask for a property of the molecule over a
small answer space. `P03` is excluded on principle rather than by taste — its
answer is a function of the displayed frame, so a context-only variant would
have no well-defined gold answer, and the builder rejects any family whose
answer schema is coordinate-dependent. The list lives in
`CONTEXT_ONLY_FAMILIES` in `dataset.py`.

Context-only renders are excluded from the primary score. `report` prints the
per-family table under *What the coordinates are worth*.

**Reversed state order.** Mechanistic episodes are rendered with the two states
swapped, and the gold labels are transformed accordingly, so an item cannot be
answered from ordering conventions.

## The response cache

Every completion is stored in a content-addressed cache, by default
`data/response_cache/`, keyed on the provider endpoint, model, its revision,
sampling parameters, output budget, completion index and the exact system and
user prompt text — and on nothing else. In particular it is **not** keyed on
`render_id` or on the semantic instance identifier, both of which move whenever
the dataset is rebuilt with a different seed or protein pool.

That is the whole point. Adding a question to the benchmark costs the calls for
that question; removing one costs nothing, and its responses stay on disk.
Changing the output budget *does* invalidate an entry, deliberately: truncation
and inability score identically, so a 8k-budget answer must never be silently
reused for a 64k run.

`--resume` is stricter than the shared response cache. It resumes an interrupted
run only when the model configuration and the fingerprint of the complete
accepted dataset still match. After rebuilding a dataset, use a new output
directory; unchanged prompts will still be recovered safely from the exact-
prompt response cache.

Each entry holds the provider's full response body, so reasoning traces survive
for inspection rather than being reduced to the answer text at call time.
Prompts are stored by hash only — they are large, they are already in the
dataset's `prompts/` directory, and duplicating them would multiply the cache
size for no gain. Results files record the `cache_key` so a row can be traced
back to the response that produced it.

```bash
structural-reasoning evaluate --dataset datasets/final --model-config configs/models/x.yaml --output runs/x --cache-dir data/response_cache
```

`--no-cache` bypasses it entirely. The mock providers are never cached: they are
free and deterministic.

Cache-format upgrades preserve only identities that can be reconstructed
exactly. Migration uses each format-2 entry's stored completion count and request
encoding, so an identical wire seed remains reusable when a later run asks for
more repeats or changes between generated and explicit seed provenance.
Validated entries are promoted atomically to the current key and record the
source cache key in provenance so result rows remain directly auditable.
Fallback discovery checks only small format headers and does not decode
unrelated current-format response bodies. Its first lookup takes one
process-lifetime snapshot of a quiescent format-2 cache, so a large new batch
does not rescan every shard once per prompt. Stop evaluators that still write
format 2 before starting current code; compatibility migration does not
coordinate with active old-format writers. A transiently incomplete snapshot is
retried and then fails closed rather than authorising a paid call. Current-format
writers coordinate by exact request key, and a present but corrupt exact entry
must be moved or removed explicitly instead of being overwritten. OpenAI-shaped
legacy responses containing an in-band API error are rejected rather than
promoted as answers.

## Batch inference

Together's Batch API offers discounts of up to 50% for selected eligible
models; models not listed for batch discounts use standard rates. Check the
[current eligibility and pricing](https://docs.together.ai/docs/inference/batch/overview)
before submitting. The completion window is measured in hours, which suits a
non-interactive benchmark.

A batch run does exactly one thing: **fill the response cache**. `evaluate` then
runs as usual and finds every completion already there. Scoring, reporting and
the results schema never learn that batching happened, a partially returned
batch simply leaves some prompts uncached, and a batch that fails outright costs
nothing but time.

```bash
structural-reasoning batch --dataset datasets/final --model-config configs/models/together_deepseek_v4_flash.yaml --state-dir runs/batch-deepseek
```

```bash
structural-reasoning evaluate --dataset datasets/final --model-config configs/models/together_deepseek_v4_flash.yaml --output runs/deepseek
```

`--stage submit|poll|fetch` splits the three steps so that a long completion
window can be picked up by a later invocation; the batch ids live in
`<state-dir>/batch_state.json`. Prompts already in the cache are never
submitted, so re-running after adding questions submits only those questions.
Batching needs the Together client: `pip install -e ".[batch]"`.
Each `--state-dir` is bound to one complete model request configuration and must
not be reused for another model or sampling setup. Submission is serialized per
state directory. Each exact request set is reserved before its paid provider
create call; the state directory and newly created parent links are fsynced where
the filesystem supports it. The returned batch id is then persisted immediately.
If the process dies between those writes, the reservation blocks automatic
resubmission because the provider may already have accepted it. Inspect the
Together account first. If the batch exists, rerun `--stage submit` with
`--recover-ambiguous-batch-id <id>`; its input file is verified before the ID is
attached. Recovery is attach-only: it does not create batches for any new
questions present in that invocation. Run an ordinary submit afterward so its
reachability preflight applies before any additional paid work. Pass
`--confirm-ambiguous-resubmit` only after confirming that no corresponding
batch exists.
Completed state can be reused when the same dataset grows: only new, uncached,
never-submitted request digests are appended as new jobs. Pre-v2 single-
completion state can still be polled and fetched, but additions require a new
state directory because the old file did not record the complete request
configuration. Pre-v2 multi-completion state cannot be mapped into the current
seeded identities and is rejected rather than given false provenance.
Batch fetches share the synchronous evaluator's per-request lock and preserve
the first valid response already stored at each cache key. In addition, a batch
submission durably claims its response-cache directory until every tracked job
has reached a terminal state and `fetch` has run. While that marker exists, a
synchronous paid evaluation or a batch using another state directory is rejected
before any model call. This deliberately conservative cache-wide guard closes the
race between a batch cache miss and its later provider submission. Use a separate
cache directory only when the runs are intentionally independent and duplicate
requests are acceptable. If a batch is abandoned, inspect the provider account
first and use the original state directory to recover or fetch it; remove the
`.active_batch` marker manually only after confirming that no paid job exists.

The fetch and all stages exit nonzero whenever a selected completion failed, was
unrecognised, or remains uncached. Do not start synchronous evaluation until
`batch --stage fetch` succeeds unless paying to retry missing requests is
intentional.

## Statistics

Three sources of variation are kept apart:

1. **Benchmark composition** — bootstrap resampling of whole protein clusters.
   Instances from the same structure are not independent draws, so resampling
   individual instances would understate the interval. Structures of the same
   protein (an apo/holo pair, two states of adenylate kinase, the two TRPV1
   episodes) share a `cluster` key and are always resampled together.
2. **Model stochasticity** — repeated completions of identical prompts, reported
   as pairwise exact agreement for scalar and categorical answers and pairwise
   Jaccard agreement for set answers.
3. **Representation sensitivity** — the paired contrasts above.

## Mechanistic reporting

Reported separately, never merged into one number:

- observation, integration and mechanism sub-scores;
- mechanism accuracy conditional on a correct observation — a model that picks
  the right mechanism while misidentifying the residue has probably recognised
  the system rather than read the structure;
- gain over the context-only control;
- the minimal-PDB versus normalized-coordinate gap;
- consistency across rotations and repeated runs.

The two TRPV1 episodes share a source protein and therefore a bootstrap cluster,
so they do not receive independent full weight.
