# GLM teacher traces for Snowball

The teacher pipeline samples the frozen coordinate tasks and scores each answer
with the verifier bundled in that task's released Harbor archive. Only training
tasks whose Snowball prompts leave at least 8,192 tokens in a 32,768-token context
are included: 28,045 tasks across 18 families in release v1.2.0. T01 has no eligible
tasks. The parent validation and test splits are never sent to the teacher.

Generation uses GLM-5.3, high reasoning effort, temperature 1.0 and top-p 0.95.
Every request explicitly disables tools. Each retry contains the same frozen
prompt and a fresh deterministic seed, without previous responses, gold answers
or verifier feedback. Stop at the first correct answer or ten total scored
attempts. Transport failures are recorded separately and do not consume a scored
attempt. The teacher receives its full remaining 262,144-token served context;
the 8K reserve selects tasks, rather than capping teacher generation.

## Running the pipeline

Install the project with its `taskgen-context` dependencies, plus Matplotlib and
the Hugging Face CLI. Download the pinned tokenizer files into the local cache:

```bash
hf download zai-org/GLM-5.3 \
  --revision aca966e4e02791568aa6a4ced368624b3d897f42 \
  --include '*token*' '*template*' 'config.json'
hf download open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38 \
  --revision cfc1d845dae89b067cdc7250d0164abefa5a69cf \
  --include '*token*' '*template*' 'config.json'
hf download open-athena/pdbthink-coordinate-tasks --repo-type dataset \
  --revision fbd07fe7255f1f65d4d860c7561d5ae03110d1e9 \
  --local-dir data/coordinate-tasks-v1.2.0

python -m pdbthink.taskgen.teacher_data \
  --dataset data/coordinate-tasks-v1.2.0 --root runs/teacher-v1 --workers 8
```

Supply `TEACHER_BASE_URL` and `TEACHER_API_KEY` through the environment. The current
client expects a batch service that accepts raw JSONL uploads at `/files`,
OpenAI-compatible request bodies and `/batches` jobs with `priority: bulk`.
Other services may need a transport adapter. The GLM endpoint used for this run
accepts `tools: null` with `tool_choice: none`; it rejects an empty tool array.

```bash
python -m pdbthink.taskgen.teacher_run run --root runs/teacher-v1 \
  --batch-size 128 --active-jobs 1024 --max-inflight 2048
python -m pdbthink.taskgen.teacher_run status --root runs/teacher-v1
python -m pdbthink.taskgen.teacher_export --root runs/teacher-v1 \
  --output runs/teacher-v1/release --workers 8
```

The controller resumes from SQLite and saved batch receipts. Server jobs remain
active when the controller stops; restart with the same root to collect them.
A lock prevents concurrent controllers. Unknown submission outcomes are
reconciled by input-file identity before any resubmission. The final export
refuses to proceed until every task is solved or has exhausted ten attempts.
`--allow-partial` is available only for labelled local previews.

## Using the traces

The published dataset has three configurations:

- `sft`: first correct response per solved task when the complete conversation
  fits Snowball's 32,768-token context.
- `attempts`: every scored response, including incorrect and overlong correct
  responses, with native scores, raw API responses and token counts.
- `outcomes`: every task in the original cohort, with its attempt count, first
  success or exhaustion, and SFT eligibility.

Reasoning and the final answer are joined with a blank line in a single
assistant message. Apply the pinned Snowball chat template with
`enable_thinking=True` and assistant-only loss. Exact lengths include delimiters
and the end token. No traces are truncated. `completion_within_8k` selects the
stricter subset whose assistant completion is at most 8,192 Snowball tokens.
`has_reasoning` distinguishes separately returned reasoning from answer-only
responses; the latter remain useful verified targets and contain no invented
reasoning. Original reasoning and answer text are also kept in separate columns.

The report measures observed success by attempt under stop-on-success sampling.
It does not use the fixed-sample pass@k estimator. A correct final answer does not
certify all reasoning steps; ten failures do not prove a task is impossible.
Family sizes and shared structures matter when interpreting the counts and
choosing an SFT mixture. The source benchmark exclusions do not establish absence
from the teacher's pretraining. The served teacher's immutable weight revision
is unavailable; the tokenizer, request policy, task release and verifier code
are pinned independently.
