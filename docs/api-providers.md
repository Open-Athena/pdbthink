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
temperature: 0.0
completions: 1
concurrency: 1
label: short-run-label
```

`model_id`, output limits and optional reasoning parameters are model-specific.
Changing gateways normally requires only `base_url`, `api_key_env` and the model
identifier. A gateway can only expose the models and features it supports, so a
direct provider configuration may still be needed for a model or reasoning mode
that the gateway does not offer. The endpoint is part of the run and response-
cache identity, so responses from two gateways are never silently interchanged.

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
are controlled by OpenRouter and may change.

To switch models, copy the config and change `model_id`, `max_output_tokens`,
sampling or reasoning parameters, and `label`. Remove `--limit 1` only after the
canary succeeds. A model's context window must cover the input prompt plus its
output budget; `--max-input-tokens` can explicitly skip prompts that are too
large rather than sending requests that cannot fit.

Use a separate `--output` directory for each model configuration. `--resume`
means resume the same model run in the same directory; the evaluator refuses to
mix rows from a different run.

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
- `configs/models/anthropic_opus.yaml` uses the native Anthropic Messages API.
- Together configurations use the compatible synchronous path, and can also use
  PDBThink's Together-specific batch command for lower-cost full runs.
- Evalchemy is optional. It is useful when integrating with Marin's evaluation
  stack, but is not required to run or score this benchmark through an API.
