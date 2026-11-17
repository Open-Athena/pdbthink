# Standardized coordinate comparison

This separate condition reruns the same frozen 88 questions used by the
[original report](https://openathena.ai/pdbthink/). It includes Astra, Opus,
Kimi, DeepSeek, GLM, Qwen, Snowball, Mochi and Nemotron 3 Super. Variant names
are `standardized-astra`, `standardized-opus`, and so on. Original responses
and report snapshots are preserved; no original answer becomes a standardized
answer by relabelling.

## Agreed controls

Every variant has a **262,144-token total budget**, including the exact prompt,
reasoning and final answer. The output allowance is
`min(endpoint output ceiling, 262144 - exact input tokens)`. The user explicitly
selected this policy instead of imposing a common 128,000-token output cap.
Hosted servers may have larger native windows; the client constrains the total
allowance without changing the server's architecture or shortening prompts.

Adjustable models use temperature **0.7**, top-p **1.0**, disabled top-k,
frequency/presence penalties **0**, and repetition penalty **1**. OpenRouter
encodes disabled top-k as `0`; vLLM uses `-1`. Local min-p is explicitly `0`.
No seed is requested for any model. Hosted reasoning effort is `high`; local
thinking is enabled. These controls do not imply equal reasoning computation,
tokenization, numerical precision or repeatability.

Astra and Opus cannot accept custom temperature, top-p or top-k. Live
temperature-0.7 probes with `require_parameters=true` failed at parameter
routing. This agrees with the [OpenAI documentation](https://developers.openai.com/api/docs/changelog)
and [Anthropic documentation](https://platform.claude.com/docs/en/build-with-claude/thinking).
The user explicitly chose to include fresh 256K runs using fixed provider
sampling, marked as exceptions. Their omitted controls must never be described
as verified temperature 0.7.

| Variant | Provider | Advertised output ceiling | Sampling |
| --- | --- | ---: | --- |
| standardized-astra | OpenAI flex via OpenRouter | 128,000 | Fixed provider sampling; exception |
| standardized-opus | Anthropic via OpenRouter | 128,000 | Fixed provider sampling; exception |
| standardized-kimi | Fireworks via OpenRouter | 943,718 | Explicit standardized controls |
| standardized-deepseek | Fireworks via OpenRouter | 943,718 | Explicit standardized controls |
| standardized-glm | Fireworks priority via OpenRouter | 943,718 | Explicit standardized controls |
| standardized-qwen | Pinned local vLLM | 262,144 | Explicit standardized controls |
| standardized-snowball | Pinned Marin vLLM | 262,144 | Explicit standardized controls |
| standardized-mochi | Marin vLLM, Hero SFT step 400407 | 262,144 | Explicit standardized controls |
| standardized-nemotron | DeepInfra BF16 priority via OpenRouter | 235,929 | Explicit standardized controls |

The larger advertised ceilings do not override the 262,144-token total budget.
Kimi changes from Moonshot to Fireworks to expose top-k. GLM uses Fireworks
because its earlier direct endpoint is unavailable. Nemotron is exactly
`nvidia/nemotron-3-super-120b-a12b`, not the separate 3.5 Super VL model.
OpenRouter routes are pinned, disable fallback, and require all submitted
parameters. Provider labels are not independent verification of weights or dtype.

## Collection and reporting

`configs/report_models_v7.json` freezes the request conditions. The runner
`scripts/run_standardized_condition.py` records separate initial and recovery
stages. Disjoint Snowball shards cover each question exactly once. Transport
failures retain their partial streams; completed answers are not retried based
on correctness. An uncertain submission requires reconciliation before retry.
Provider clamps and tool events require review rather than automatic recovery.

Every request explicitly disables tools and OpenRouter web search. No runner
executes tools. Mochi retains the `marin-thinking-v1` protocol: preserve raw
thinking delimiters and score only the final answer after a complete block.

The report distinguishes native endpoint output limits, total-context exhaustion,
format errors and uncollected requests. A returned truncated answer is incorrect;
an interrupted stream is a collection failure. Missing responses produce
lower–upper bounds, not confidence intervals. Actual per-question budgets,
token usage, request controls and response hashes remain in the evidence.

`scripts/build_comparison_report.py` builds separate original and standardized
panels, with clean SVG/PDF/PNG figures for each. The landing page links both and
shows the standardized results alongside the preserved original configurations.
Publication archives the preceding report and includes both panels in the new
snapshot. Fresh prompts, gold answers and raw model output remain private.
