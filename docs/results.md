# First results

Eight models over the 117-instance candidate set. Numbers are reproducible from
the response cache; the commands are at the end.

| model | renders | as scored | with budget re-run | completed only | truncated |
| --- | --- | --- | --- | --- | --- |
| **Kimi K3** | 247 | 0.684 | **0.785** | 0.830 | 60 |
| DeepSeek V4 Flash | 222 | 0.615 | — | 0.756 | 45 |
| Gemma 4 31B | 155 | 0.578 | — | 0.730 | 19 |
| MiniMax M3 | 247 | 0.453 | 0.528 | 0.757 | 102 |
| Qwen3.5 9B | 246 | 0.294 | 0.337 | 0.583 | 159 |
| gpt-oss-120b | 229 | 0.257 | 0.259 | 0.270 | 11 |
| gpt-oss-20b | 242 | 0.180 | 0.197 | 0.256 | 97 |
| Marin 32B *(base, 4k ctx)* | 36 | 0.053 | — | 0.142 | 22 |

All seven Together models cover all twenty families. Marin covers eight, for
reasons given below. The two right-hand columns bracket the same quantity from
opposite sides — what a model scores when allowed to finish — one by buying the
missing answers, the other by dropping them.

## The headline is not a leaderboard

The most useful thing this sweep produced is not a ranking. It is that **a
single number per model is misleading on this benchmark**, for two reasons that
both have the same shape: something that is not a reasoning failure is being
scored as one.

## 1. Scores are output budgets

Truncation and inability score identically — zero. If a model runs out of output
tokens mid-answer, it never emits the `FINAL:` line, and the scorer sees a
format error worth nothing. That is correct behaviour for a scorer, and
catastrophic for interpretation.

The best model in the sweep, Kimi K3, scores **0.668** across the twenty
families. Restricted to responses that actually terminated, it scores **0.839**.
Its per-family results split cleanly in two:

| | families |
| --- | --- |
| at 0.95 or above | `G01` `G02` `G03` `N01` `P01` `P02` `P03` `S01` `S02` `S07` `S08` `S09` `T01` |
| below | `I01` `G04` `S03` `S04` `MECH` `S06` `S05` |

In every family in the second row, **the format-error count equals the
truncation count**. `I01` — list the residues of one chain within 4 Å of another
— scores 0.000 with 10 of 10 responses truncated. That is not an inability to
reason about interfaces. It is a model that never finished writing.

This is not a new observation on this benchmark, and that is the point: an
earlier run of DeepSeek V4 Flash moved **0.475 → 0.638 → 0.710** on an identical
dataset with nothing changed but its output budget, 8k → 32k → 64k. Any
benchmark of reasoning models that reports one number without a truncation count
is reporting a budget.

### The experiment

Every prompt that hit the cap was re-run at a larger budget, unchanged in every
other respect. Each scored exactly 0.000 the first time, by construction — a
truncated response never reaches a `FINAL` line.

| model | prompts | new budget | before | after | still cut off |
| --- | --- | --- | --- | --- | --- |
| Kimi K3 | 63 | 64k | 0.000 | **0.644** | 15 |
| MiniMax M3 | 35 | 128k | 0.000 | **0.591** | 4 |
| Qwen3.5 9B | 30 | 128k | 0.000 | **0.450** | 0 |
| gpt-oss-120b | 10 | 40k | 0.000 | 0.267 | 3 |
| gpt-oss-20b | 97 | 40k | 0.000 | 0.034 | 66 |

A zero from a cut-off answer is recoverable; a zero from a wrong answer is not.
For the stronger models most of that zero was budget — Kimi K3's headline moves
from 0.668 to 0.805 on the strength of 63 prompts it had already been asked.

**The two gpt-oss rows are inconclusive rather than negative.** Their
131,072-token context sits beside an 87,500-token prompt, so the largest budget
they can be given is a 25% increase, and two thirds of gpt-oss-20b's prompts
still run out. Nothing there says those models cannot do the task; it says the
experiment cannot be run on them at this prompt length. Answering that would
need a smaller-structure variant of the dataset — the same thing Marin 32B
needs, below.

## 2. The context-only control can invert its own sign

Seven automatic families and the six mechanistic episodes are also rendered with
the coordinates removed. For a mechanistic episode the prompt still describes the
experiment, so a model that recognises the system can answer from the literature
and the gain over context-only separates structure use from recall. For the
automatic families the sanitised question names no protein at all, so what the
control measures is the **guessing floor** — and the floors are not small: `S03`
is a coin flip, `S04`, `S05` and `S09` are three-way, and answering `A` to `P01`
is right much of the time.

Read naively, Kimi K3's `S04` says the coordinates make it **worse**:

| family | floor | with coordinates | naive gain | truncated | gain, completed only |
| --- | --- | --- | --- | --- | --- |
| `S03` | 0.167 | 0.167 | 0.000 | 83% | **+0.833** |
| `S04` | 0.667 | 0.167 | **−0.500** | 83% | **+0.333** |
| `S05` | 0.333 | 0.500 | +0.167 | 50% | **+0.667** |
| `MECH` | 0.522 | 0.368 | **−0.154** | 58% | **+0.361** |
| `P01` `P02` `S08` `S09` | 0.000 | 1.000 | +1.000 | 0% | +1.000 |

The cause is an asymmetry nobody designs in on purpose. **A context-only prompt
is about 200 tokens and never truncates. A coordinate prompt runs to 87,500
tokens and truncates at up to 83%.** The naive difference therefore compares a
finished cheap answer against a cut-off expensive one. Conditioned on
completion, every one of those families shows the gain the control exists to
measure.

Any published context-only gain has to be conditioned on completion. This is now
computed in `collect_results.py` and shown in the deck.

## 3. Marin 32B: the benchmark does not fit

`marin-community/marin-32b-base` was served on Modal with vLLM and evaluated at
its real context length. It cannot be evaluated here, and the reason is
structural rather than a matter of tuning:

**The smallest prompt in the candidate set that contains any coordinates is
7,132 tokens. Marin 32B's context is 4,096.**

Of 247 renders, 48 fit — and all 48 are context-only controls, which show no
structure by construction. Its 0.053 macro is a guessing floor over prompts
containing no protein.

Two further points, both properties of the model:

- **It is a base model**; `marin-32b-instruct` does not exist. It answered `P01`
  with `FINAL: A,B,C,...,Z` — the entire alphabet, which set F1 scores at 0.094
  for perfect recall and hopeless precision — and elsewhere repeated the prompt
  back until it hit the cap, which accounts for all 34 truncations.
- **Its config contradicts itself.** `config.json` carries a llama3
  `rope_scaling` block implying 65,536 while declaring
  `max_position_embeddings: 4096`; the model card settles it at 4,096. vLLM
  refuses the override except behind `VLLM_ALLOW_LONG_MAX_MODEL_LEN`, whose
  warning is that RoPE positions past the derived length return NaN. A zero
  produced by NaN activations is indistinguishable from a zero produced by
  inability, so the override was not used.

Evaluating 4k-context models on structural reasoning needs a dataset built to a
~3,000-token budget: single small chains, aggressive cropping, normalized
coordinates throughout at ~22 tokens per atom against minimal PDB's ~41. That
caps structures near 130 atoms — enough for the parsing and geometry families,
not for folds, interfaces or binding sites. It is a separate dataset, not a
filter on this one.

## What went wrong, and what it cost

Three things are worth recording because they are the sort of thing that
silently corrupts a result set.

**Models that are listed but not servable.** DeepSeek V3.1, GLM-4.7 and
Qwen3-235B all appear in Together's catalogue and all refuse work on this
account. Neither `pricing` nor `running` predicts it — DeepSeek V4 Flash reports
`running: false` and works. The expensive version of this failure is
Qwen3-235B: it *accepted* a 247-request batch, reported `COMPLETED` at 100%, and
put every request in the error file with *"Unable to access non-serverless
model"* — discovered only after the completion window. `batch` now spends one
token on a synchronous probe before submitting.

**A credit limit mid-sweep.** Three runs were cut off. Because batches are
processed in render-id order, a truncated run covers the alphabetically early
families and nothing else: at the time, DeepSeek's 80 renders spanned 8 of 20
families and Gemma's 60 spanned a different 8. `evaluate --cache-only` exists so
that undelivered prompts are reported as missing rather than scored as wrong.

Those runs have since been completed and the bias is worth recording, because it
was large and in the flattering direction. **Gemma 4 31B scored 0.808 on its
first 60 renders and 0.577 on 175.** Had that first number been published it
would have led the table. A macro average over a partial run is not an estimate
of the full one.

**A cache format change.** An upstream change added the provider endpoint to the
cache key and bumped the entry format, which made all 1,284 responses bought
overnight unreadable — right data, wrong digest. They were re-keyed rather than
re-bought, by walking the same (model, render, completion) space the runner
walks and asking it for the key rather than reconstructing one by hand. Nothing
was paid for twice.

## Reproducing

```bash
structural-reasoning build --config configs/dataset_v1.yaml --output data/datasets/candidates_v1
```

```bash
structural-reasoning batch --dataset data/datasets/candidates_v1 --model-config configs/models/together_kimi_k3.yaml --state-dir runs/batch-kimi
```

```bash
structural-reasoning evaluate --dataset data/datasets/candidates_v1 --model-config configs/models/together_kimi_k3.yaml --output runs/kimi
```

```bash
structural-reasoning score --dataset data/datasets/candidates_v1 --responses runs/kimi --output scores/kimi
```

Re-running only what hit the output cap, which is how a budget-limited score is
told apart from a capability-limited one:

```bash
structural-reasoning evaluate --dataset data/datasets/candidates_v1 --model-config configs/models/together_kimi_k3_hi.yaml --output runs/kimi-hi --rerun-truncated scores/kimi
```

## What balancing the categorical families changed

`S03`, `S04` and `S09` each realised a single gold label for every instance
until the selection bug behind it was fixed. The clearest evidence that it is
fixed is not the leaderboard — which barely moved, the ranking not at all — but
the **context-only floors**, which are what a blind model scores on the question
text alone:

| family | answer space | floor before | floor after | what the floor should be |
| --- | --- | --- | --- | --- |
| `S03` | buried / exposed | 0.167 | **0.500** | 0.5, a coin flip |
| `S04` | helix / strand / coil | **0.667** | **0.167** | ~0.33 |
| `S05` | three fold classes | 0.333 | 0.333 | ~0.33 |
| `S09` | g+ / t / g- | 0.000 | **0.333** | ~0.33 |

`S04` is the striking one. Its floor was **0.667** — a blind model answering
`coil` was right two times in three, because two of every three labels *were*
`coil`. After balancing, guessing `coil` wins one time in six and the floor
falls to **0.167**. The number was never measuring a prior over secondary
structure; it was measuring the label distribution.

`S03` moves the other way for the same reason: a balanced binary should sit at
0.5, and now does.

None of these families can now be won by a constant, and their scores mean what
they say for the first time.

## S02: thin, and narrower than it looks

`S02` realises two instances against a target of four, and the reason has three
layers. The first is the one that actually binds today, and it is not scarcity.

**Four of its six candidate proteins are on the burned list.** Every one of them
produced a valid candidate, and every one was rejected with
`answer_previously_published`:

```
S02-lck_sh2_phosphopeptide-19a604a9
S02-lck_sh2_phosphopeptide_2-e91a4ae4
S02-sh2_phosphopeptide_shc-5e795158
S02-src_sh2_phosphopeptide-9afa2733
```

Those answers reached a public commit before gold answers were split out of the
repository, so they are excluded by design — see
[contamination.md](contamination.md). Lifting the target would be the wrong fix;
these questions are burnt, not missing.

**Underneath that, phospho-residues are genuinely rare.** Of the 181 structures
in the cache, exactly 8 contain a phosphorylated residue at all:

| entry | component | what it is |
| --- | --- | --- |
| 1AYB | PTR | phosphopeptide complex, SH2 |
| 1LCJ | PTR | Lck SH2 domain |
| 1LKK | PTR | Lck SH2 domain |
| 1SHA | PTR | Src phosphotyrosine recognition domain |
| 1SHB | PTR | Src phosphotyrosine recognition domain |
| 1SPS | PTR | high-affinity phosphotyrosyl peptide, SH2 |
| 1X27 | PTR | SH2 domain complex |
| 2PLD | PTR | SH2 domain, NMR |

**And that is the third layer, which no amount of selection can fix.** Every one
is `PTR` — phosphotyrosine. There is not a single `SEP` (phosphoserine) or `TPO`
(phosphothreonine) anywhere in the cache, and all eight are the same structural
motif: an SH2 domain bound to a phosphopeptide.

The question reads:

> Exactly one protein residue in this structure is phosphorylated, that is, it
> is a phosphoserine, phosphothreonine or phosphotyrosine residue. Which residue
> is it?

It offers three possibilities and the answer is always the third, in the same
domain family. This is the same defect the categorical families had — a question
whose answer space is wider than its realised labels — except that here it lives
in the protein pool rather than in selection, so `_rotate_by_tag` and
`_balance_labels` cannot reach it. A model could learn "find the SH2 domain,
name the phosphotyrosine" and score perfectly without ever recognising a
phosphate group.

Fixing it needs new source structures, chosen for the classes that are missing:
a kinase activation loop carrying `TPO`, a 14-3-3 or phosphatase complex
carrying `SEP`, and ideally a phosphoprotein that is not an SH2 complex at all.
That is an `acquire` change plus new `family_proteins` entries, not a builder
change.
