# Coordinate task corrections in v1.3.0

The [v1.3.0 coordinate task release](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/tree/v1.3.0)
fixes two defects found while auditing GLM teacher traces. It retains all 100,000
task identities, displayed coordinates, gold answers and source-group splits.
The [release record](coordinate-tasks-v1.3.json) pins the data revision, validation
evidence and source commit.

## Numeric tolerance boundary

The old scorer subtracted binary floating-point values before comparing against
the inclusive tolerance. For example, a predicted coordinate of −1.710 against
gold −1.711 should pass a 0.001 tolerance, but subtraction produced
0.001000000000000112 and rejected it. The teacher audit found 11 such rejected
responses across two tasks.

Scorer 1.1.0 compares exact rational differences of the decimal representations
of parsed numeric values. Both coordinate triples and scalar distances use this
comparison. The allowed tolerance is unchanged; answers outside it still fail.
The parser continues to parse numeric answers as floats, so this is not an
arbitrary-precision parser for the original response text.

## Missing clash exclusion rule

The G04 oracle excludes every sulfur SG–SG atom pair, including pairs farther
apart than the threshold used to identify a disulfide bond. The old question
did not state that exclusion. In one audited example the strongest apparent
overlap was an excluded SG–SG pair, making a reasonable reading of the question
disagree with its gold answer.

Prompt v5 now states the overlap formula, Bondi radii, covalent-neighbour and
metal exclusions, the unconditional SG–SG exclusion, and the residue-pair ranking
rule. All 232 G04 prompts were migrated with strict checks against the old
wording. The geometric definitions and gold answers are unchanged.

## Verification

The source suite passed 397 tests. Every packaged task passed coordinate-oracle
recomputation and solution verification. A separate comparison checked all
100,000 old and revised task archives: only the 232 G04 prompt texts changed;
all coordinates, gold answers, splits, source groups and solution archives were
preserved.

The regression audit also ran the actual packaged `tests/test.sh` inside the
release's pinned Docker image with networking disabled:

| Case | Old reward | Revised reward |
|---|---:|---:|
| First reported numeric boundary answer | 0 | 1 |
| Second reported numeric boundary answer | 0 | 1 |
| Two answers just outside tolerance | Not rerun | 0 each |
| Clarified G04 gold residue pair | Not rerun | 1 |
| Excluded G04 sulfur pair | Not rerun | 0 |

This is eight container invocations and is a direct packaged-verifier test, not
a new Harbor Trial orchestration run. No model calls were made. Detailed rewards,
diagnostics and archive hashes are in the published
[regression evidence](https://huggingface.co/datasets/open-athena/pdbthink-coordinate-tasks/blob/v1.3.0/audits/contract_revision/regressions.json).

The native Snowball context recount preserves exactly the same 28,045 training
task identities with room for 8,192 reasoning/output tokens in its 32,768-token
window. Across all splits, 30,418 tasks meet that criterion. The longer G04
question reduces prompt-only fit by one task and 4K-headroom fit by one task.

## Historical data

Earlier dataset tags and the
[v1.0.0 GLM teacher traces](https://huggingface.co/datasets/open-athena/pdbthink-glm53-teacher-traces/tree/v1.0.0)
remain unchanged. Those traces were generated and scored against task v1.2.0;
their retry counts and scores have not been silently rewritten. They do not
constitute evaluations of the clarified G04 prompts. A future corrected teacher
release should be versioned separately, recording any rescoring and regenerating
G04 traces against the revised question.
