# Running under Evalchemy

pdbthink ships an [Evalchemy](https://github.com/marin-community/evalchemy)
`BaseBenchmark` adapter. The package directory doubles as the benchmark
directory, so installation is a symlink.

## Install

```bash
pip install -e /path/to/pdbthink
```

```bash
ln -s "$(python -c 'import pdbthink.evalchemy_task as m; print(m.__path__[0])')" "$EVALCHEMY/eval/chat_benchmarks/PDBThink"
```

## Run

Point the task at a built dataset and launch it like any other Evalchemy task:

```bash
PDBTHINK_DATASET=/path/to/datasets/smoke python -m eval.eval --model vllm --tasks PDBThink --model_args "pretrained=Qwen/Qwen3-8B" --output_path logs
```

API models work the same way:

```bash
PDBTHINK_DATASET=/path/to/datasets/final python -m eval.eval --model openai-chat-completions --tasks PDBThink --model_args "model=gpt-5,num_concurrent=16" --output_path logs
```

## What the adapter does

`generate_responses` loads the dataset, drops instances a curator rejected,
turns every remaining rendered variant into one `generate_until` request with the
versioned system prompt, and repeats the pass `n_repeat` times.

`evaluate_responses` scores the completions with `pdbthink.scoring` — the same
code path as `structural-reasoning score`. Evalchemy numbers and CLI numbers
therefore cannot drift apart, and the per-completion rows it returns have the
same shape the reporting module consumes.

The lm-eval and Evalchemy imports are deferred to call time, so building,
validating, scoring and reporting never require Evalchemy to be installed.

## Constructor options

| option | default | meaning |
| --- | --- | --- |
| `dataset_dir` | `$PDBTHINK_DATASET` | the dataset to evaluate |
| `max_tokens` | 8192 | generation budget per item |
| `n_repeat` | 1 | completions per prompt; use 3 for the final protocol |
| `families` | all | restrict to specific question families |
| `debug` | `False` | evaluate two items only |

## Standalone alternative

If you would rather not install Evalchemy, `structural-reasoning evaluate` speaks
the same OpenAI-compatible protocol, is resumable, stores raw responses and full
run configuration, and feeds the same scorer:

```bash
structural-reasoning evaluate --dataset datasets/final --model-config configs/models/local_vllm.yaml --output runs/local --resume
```
