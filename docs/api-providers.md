# Running models through APIs

The native evaluator can call any provider that implements the OpenAI-compatible
`/v1/chat/completions` interface. The benchmark prompts, response cache and
scorer do not depend on which compatible provider serves the model.

## Model configuration

An API model is selected with a small YAML file:

```yaml
model_id: provider/model-name
provider: openai_chat
base_url: https://provider.example/v1
api_key_env: PROVIDER_API_KEY
max_output_tokens: 8192
output_token_parameter: max_tokens
temperature: 0.0
completions: 1
concurrency: 1
max_retries: 1
label: short-run-label
```

`model_id`, output limits and optional reasoning parameters are model-specific.
Most gateways use `max_tokens`; direct OpenAI reasoning models use
`max_completion_tokens`, selected with `output_token_parameter` as in
`configs/models/openai_gpt.yaml`.

Changing gateways normally requires only `base_url`, `api_key_env` and the model
identifier. A gateway can only expose the models and features it supports, so a
direct provider configuration may still be needed for a model or reasoning mode
that the gateway does not offer. The endpoint is part of the run and response-
cache identity, so responses from two gateways are never silently interchanged.

Anthropic's current adaptive-thinking models use a different request shape from
older manual-thinking models. `configs/models/anthropic_opus_5_max.yaml` shows
the native Messages API configuration: `thinking_mode: adaptive` maps effort to
`output_config.effort` when supplied. Adaptive configurations require both
`temperature` and `top_p` to be `null`, matching Anthropic's default-sampling
requirement, and adaptive thinking is still transmitted when no explicit effort
is set. For older manual-thinking models, pdbthink assigns half of
`max_output_tokens` to `budget_tokens`, so the
configuration requires at least 2,048 output tokens to satisfy Anthropic's 1,024-
token minimum; an explicit `top_p` must be between 0.95 and 1. See Anthropic's
[thinking contract](https://platform.claude.com/docs/en/about-claude/models/extended-thinking-models).

Keep the key out of the YAML and the repository:

```bash
export PROVIDER_API_KEY=...
```

Put that export in the shell's private startup file, a password-manager shell
plugin or the CI secret store if it should be available in future sessions.

## OpenRouter smoke test

OpenRouter exposes its models through the same chat-completions interface. Set
the key and use the checked-in free-model configuration:

```bash
export OPENROUTER_API_KEY=...
```

```bash
structural-reasoning evaluate \
    --dataset datasets/smoke \
    --model-config configs/models/openrouter_nemotron_3_5_lightning_free.yaml \
    --output runs/openrouter-canary \
    --limit 1 \
    --resume
```

```bash
structural-reasoning score \
    --dataset datasets/smoke \
    --responses runs/openrouter-canary \
    --output scores/openrouter-canary
```

This one-render run checks authentication, endpoint compatibility, response
storage and strict `FINAL:` parsing. It is a transport smoke test, not a useful
estimate of scientific performance. Free endpoint availability and rate limits
are controlled by OpenRouter and may change. HTTP-200 responses carrying an
in-band provider error are recorded as API failures and are never cached as
answers. Any API failure makes the command exit nonzero.

To switch models, copy the config and change `model_id`, `max_output_tokens`,
sampling or reasoning parameters, and `label`. Remove `--limit 1` only after the
canary succeeds. A model's context window must cover the input prompt plus its
output budget; `--max-input-tokens` can explicitly skip prompts that are too
large rather than sending requests that cannot fit.

Use a separate `--output` directory for each model configuration. `--resume`
means resume the same model run in the same directory; the evaluator refuses to
mix rows from a different run or dataset build. Expanding a partial selection is
safe (for example, `--limit 1` followed by the full dataset). Narrowing or
switching to a disjoint family selection requires a separate output directory,
so the stored run never silently becomes a union of unrelated invocations.

Automatic retries improve reliability but cannot guarantee exactly-once billing:
a provider may finish a paid request even when its response is lost in transit.
Set `max_retries: 1` in a paid-model config to disable automatic resubmission,
accepting that a transient failure will then need manual recovery. Retriable
rate-limit and service errors respect a numeric `Retry-After` header; terminal
payment, authentication, request-size and validation errors are not retried.

The shared response cache serializes identical cache misses across evaluator
processes on the same filesystem. This prevents two sessions from knowingly
paying for the same exact request. It cannot eliminate the provider-side
ambiguity after a connection fails without returning a response.

### OpenRouter routing and data policy

OpenRouter is a gateway: both OpenRouter and the upstream inference provider see
the prompts. By default, OpenRouter may load-balance and fall back among upstream
providers. That is convenient for a smoke test but can add provider-to-provider
variation to a benchmark. Routing policy belongs in `extra_body`, for example:

```yaml
extra_body:
  provider:
    only: [together]
    allow_fallbacks: false
    require_parameters: true
    data_collection: deny
    zdr: true
```

Choose the actual upstream slug supported by the model. `only` and
`allow_fallbacks` control where inference runs; `data_collection: deny` filters
providers by collection policy; `zdr: true` requires a zero-data-retention
endpoint. These constraints can reduce availability, including for free models.
For reproducible published results, record the routing policy and returned
provider metadata from the cached raw response.
See OpenRouter's official [provider-routing](https://openrouter.ai/docs/guides/routing/provider-selection)
and [zero-data-retention](https://openrouter.ai/docs/guides/features/zdr) guides
before choosing a publication or privacy policy.

## What crosses the API boundary

Only the system prompt and user prompt are sent to the provider. The user prompt
contains the question, displayed coordinates and required answer format. Gold
answers are not sent: they remain local and are used later by `score`.

The evaluator stores the returned answer, token usage, truncation status and
latency in the run. Its content-addressed response cache also preserves the full
provider response, including a reasoning field when the provider supplies one.
This lets a report distinguish a wrong answer from a response that exhausted its
output budget. Changing the output budget intentionally creates a different
cache entry.

## Other API paths

- `configs/models/openai_gpt.yaml` uses OpenAI's compatible chat endpoint.
- `configs/models/anthropic_opus.yaml` and
  `configs/models/anthropic_opus_5_max.yaml` use the native Anthropic Messages
  API with their respective manual/default and adaptive thinking contracts.
- Together configurations use the compatible synchronous path, and can also use
  PDBThink's Together-specific batch command for lower-cost full runs.
- Evalchemy is optional. It is useful when integrating with Marin's evaluation
  stack, but is not required to run or score this benchmark through an API.
