# Coordinate benchmark report

The [live HTML report](https://openathena.ai/pdbthink/) and
[repository copy](benchmark-report/index.html) compare the same 88 questions across
19 coordinate families. The report keeps sequence prediction and retired MECH
experiments separate.

Figures are available as SVG and PDF vectors and 300-dpi PNGs, individually or in
`figures.zip`. They contain axes, model labels and measured values; explanations
and qualifications are in the HTML captions. A report snapshot may be published
before every response is collected. Missing answers remain explicitly unresolved,
with lower–upper accuracy bounds; they are never counted as model errors.

## Scientific revision

Report v5 uses the scientific contract at commit
`c00271fe049d60dc0d671b999d8c2fb595086166`: prompt v6, scorer 1.2.0,
and the current chemistry fingerprint. It preserves every displayed coordinate
record from the previously published panel. The current oracles admit all 88
questions and reproduce all 88 gold answers. The system instruction changes on
all 88 questions and user wording changes on 43, so old model responses cannot
serve as results for this revision.

The November 9 corrections state the protein-only interface criterion, expose
nonstandard residue identifier mappings, reject unknown clash topology, and repair
historical mechanistic contact sets and coordinate-derived answers. Review found
two adjacent reproducibility defects: historical dataset loads replaced the system
instruction with the current one, and the task-export helper could relabel older
prompts after only suffix edits. Frozen system messages are now persisted; legacy
versions resolve to their historical message. Exports reject older scientific
contracts that need a full rebuild. These changes do not alter the current v6
questions or gold labels.

## Model conditions

Report v6 adds `hero-sft-step400407` (Mochi) in BF16 at the user's selected
temperature **0.7**, with thinking enabled and the full remaining 262,144-token
context allowance. Top-p is 1.0, top-k is disabled, and the per-question seed is
`derive_seed(20260929, render_id)`. All 88 prompts fit unchanged. The v5 results
remain available in the [pre-Mochi snapshot](https://openathena.ai/pdbthink/reports/20261116-before-mochi/).

Mochi uses Marin IAP credentials refreshed for each request, sent only in
`Proxy-Authorization`. No OpenAI Authorization header is supplied, and login
redirects are rejected. Live metadata confirms the checkpoint path, dtype and
context limit. The runtime, topology and tokenizer revision reported by the
operator are distinguished from those live checks in `configs/report_serving_v6.json`.
The client records exact token IDs and counts using the server's `/tokenize` route.

The `marin-thinking-v1` response adapter preserves raw completions and extracts
only text after a complete `<|start_think|>…<|end_think|>` block. It cannot score
candidate answers inside unfinished reasoning. Malformed delimiter sequences
produce no parseable answer. Both structured and text-delimited tool calls are
audited; no tool is executed. The first four questions validate the transport at
concurrency four, and the remaining questions use concurrency eight. Completed
answers are retained regardless of score; transport retries have separate records.
The provider's development backend has documented load-dependent numerical
variation, so these settings do not guarantee bitwise reproducibility.

The configuration is `configs/report_models_v6.json`; run it with
`scripts/evaluate_coordinate_report.py --models mochi` in an environment that
also supplies Marin's `rigging.credentials` module and cached IAP credentials.
The scientific panel, gold answers and scorer remain identical to report v5.

Snowball is **Marin-Snowball-67B-A2B-v0.1.1**, pinned to
`59852d6f07261722bbec1edb6ab148317156b58e`, served with a **262,144-token context**.
The October report used Step38 at 32,768 tokens; its limited subset is historical
and is not the v0.1.1 context capability. The current panel includes all 88 questions.
The later November 9 plumbing experiment is not used. The server uses Marin vLLM
`0.0.0.dev20260924+marin.01911be34fac`, BF16 inference, two H100 GPUs per replica with expert
parallelism, the supplied tokenizer and the `grug` reasoning parser. The live
model registry confirms the full context limit. Prefix caching is disabled. All
88 complete prompts fit: exact Snowball input counts range from 12,671 to 63,024
tokens. Versions, configuration and tokenizer hashes for both self-hosted models
are recorded in `configs/report_serving_v5.json`. The initial Snowball collection
assigned three questions to a replica with one request per engine and the other
85 to an identical replica with up to four per engine. Qwen similarly used a
second replica for unstarted questions. Exact assignment counts, completed
answers, interruptions and concurrency settings are retained in the serving
manifest.

Local collection stopped on November 13 at 00:42 UTC, preserving 10 Snowball
and 62 Qwen answers. Thirteen partial streams remain in the private archive.
After confirming that the original clients and servers had stopped, recovery
was restricted to the 78 missing Snowball and 26 missing Qwen answers. The
replacement clients run beside their servers in the cluster and persist raw
responses to the existing volume, so local session interruptions no longer
stop collection. Snowball recovery uses two identical replicas: 57 questions
on the first and 21 unsubmitted questions transferred to the second, with an
atomic assignment record preventing duplicate submissions. Completed answers
are never selected by score or rerun.

Qwen's completed comparison uses its native **262,144-token context on all 88
questions**. One S05 response exhausted its full 206,951-token output allowance
after a 55,193-token prompt. That response remains a recorded context-limit
truncation. The publisher's YaRN extension is outside this report's fixed 256K
Qwen condition; no larger-context answer replaces it.

DeepSeek is the explicitly dated `deepseek/deepseek-v4.1-flash-1017` endpoint.
The provider route and maximum output allowance are recorded alongside every
model's condition. Native DeepSeek routing was unavailable under the account's
data policy; the Fireworks route preserves that policy. GLM 5.3 uses the
Parasail FP8 route with maximum reasoning effort, a 1,048,576-token context,
and a requested 943,718-token output ceiling. An earlier high-effort run showed
repeated provider clamps at 131,072 output tokens despite that larger request.
It is preserved separately and excluded from the headline; the maximum-effort
condition is run on all 88 questions. The maximum-effort run also encountered
131,072-token provider clamps, so changing reasoning effort does not resolve
the endpoint limit. GMICloud also stopped at 131,072 reasoning tokens with an
empty answer, despite reporting `stop`; that response is quarantined as another
provider cutoff. Here, OpenRouter is the routing service and providers such as Fireworks run the
model inference. The direct GLM deployment is accessed without OpenRouter; it
is not Z.ai's official API. Direct GLM recovery uses the documented FP8/vLLM endpoint with
a 262,144-token total context and the full remaining allowance for each question.
The direct endpoint subsequently went offline with one I01 response still
uncollected. That question uses Fireworks through OpenRouter, retaining maximum
reasoning effort and temperature zero, with the full advertised 943,718-token
output allowance in a 1,048,576-token context. Its provider does not expose an
immutable weight revision or verified quantization; this is a separate recovery
condition, configured in `configs/report_glm_recovery_v5.json`. The report records
both contexts and every request's actual route and budget. The first Fireworks
stream ended after about 50 minutes without a finish event or token accounting;
it is retained as a provider interruption. Two further attempts used the
`fireworks/fast` priority route with the same full generation budget. One ended
with a provider connection error; the other ended without a finish event or
token accounting. The remaining I01 response is missing coverage, not an
incorrect answer, and contributes to the bounds shown for GLM.
The report retains the budget comparison and does not describe an observed
provider clamp as the model’s native limit. Its earlier self-hosted batch
metering requests were withdrawn while still queued, before any benchmark
evaluation began. The revised provider
condition is explicit; historical self-hosted scores are not relabelled.

Sampling settings differ between models. Snowball uses temperature 0.6 and
Qwen uses 1.0; both use `top_p=0.95`, `top_k=20` and seed `20260929`. GLM uses
temperature 0.0 across its primary and recovery routes. Astra, Opus, Kimi and
DeepSeek omit temperature, top-p, top-k and seed, leaving sampling to their
providers; their effective numeric defaults were not verified. This is a
comparison of the recorded model configurations, not a temperature-matched study.
Qwen's settings match its pinned publisher recommendation for general thinking
tasks. Snowball's pinned card and generation config do not prescribe 0.6; this
was an explicit run choice. GLM's zero-temperature condition is a greedy baseline.
Optimal sampling settings have not been established for these models.

Every new request disables tools, tool choice and the OpenRouter web plugin.
Full provider JSON responses and raw streamed events are retained for a
tool-event audit. The first non-streaming probes retain the complete JSON object;
later requests also retain the HTTP body. The evidence identifies the format
and hash of each retained response. Partial provider responses remain failures to collect evidence, rather than
incorrect model answers. Retries have separate records. Each question receives the
largest endpoint-supported output allowance that fits its exact input, including
reasoning tokens. Exact server tokenisation or a separately recorded metering call
establishes input usage. An output clamp below the requested allowance requires
investigation before inclusion. The result table separates endpoint output-limit
truncations from exhaustion of the full context window. Both are scored as
incorrect returned answers under this benchmark's exact-answer rule. Lower
provider clamps and interrupted streams are collection failures, not model
errors, and their attempts remain in the provenance archive.

## Reproduction

Install the package with `pip install -e '.[dev,tokenizer,report]'`. Keep the previous
published dataset and its exact-record archive as immutable inputs. The following
commands run from the repository root; dataset and run directories are private.

```bash
PYTHONPATH=src python scripts/refresh_coordinate_report.py \
  --parent /path/to/clean_coordinate_v4 \
  --archive /path/to/previous/exact-records.zip \
  --output data/datasets/report_coordinate_v5
```

The refresh verifies the parent archive's file hashes, recomputes every gold answer
from displayed coordinates, validates the dataset and records any rejection.
Existing frozen output directories are never overwritten.

```bash
PYTHONPATH=src python scripts/preflight_report_context.py \
  --dataset data/datasets/report_coordinate_v5 \
  --config configs/report_models_v5.json \
  --output runs/report_sync_v5/discovery

PYTHONPATH=src python scripts/evaluate_coordinate_report.py \
  --dataset data/datasets/report_coordinate_v5 \
  --config configs/report_models_v5.json \
  --output runs/report_sync_v5/api --workers 8
```

Provide credentials through the environment and start the pinned self-hosted
servers first. Provider discovery and exact token counts must be refreshed before
launching a future run under a new version. The checked-in model settings describe
this evaluation, not indefinitely valid service limits. Resolve uncertain
submissions before retrying; do not delete their intent or raw response records.

```bash
PYTHONPATH=src python scripts/build_coordinate_report.py \
  --dataset data/datasets/report_coordinate_v5 \
  --runs runs/report_sync_v5 \
  --config configs/report_models_v5.json \
  --output docs/benchmark-report
```

The public report publishes aggregate results, budgets, usage and hashes. Fresh
prompts, gold answers and raw model responses stay outside Git. Earlier versions
retain their original scores and request conditions. Differences from historical
scores do not isolate a model improvement: the scientific prompt revision and
serving conditions also changed. The predecessor’s prompts and gold answers were
already public; the refreshed comparison therefore uses a previously exposed
panel, even though the new run archive stays private.

## Publication

Build and audit all seven complete rows before publishing. Commit the source and
report to `main`. In a clean checkout of the existing Pages branch, stage the
publication with the following command; it preserves the entire previous report
under `historical/v4/`, keeps existing download URLs working, and records source
and artifact hashes in `publication.json`.

```bash
PYTHONPATH=src python scripts/prepare_report_publication.py \
  --source docs/benchmark-report --site /path/to/pages-checkout --archive-name v4
```

Review and commit that checkout, then push it to `codex/snowball-report-site`.
GitHub Pages serves that branch at the live URL. Later revisions must choose a
new archive name and keep earlier archives intact.
