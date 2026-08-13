# Review decisions

`structural-reasoning review --dataset <dir> --decisions <file>.jsonl` appends one
JSON object per curator decision here. The dataset builder consumes the file:

```bash
structural-reasoning build --config configs/dataset_v1.yaml --output datasets/final \
    --decisions data/review_decisions/v1.jsonl --accepted-only
```

Each line is a `ReviewDecision`:

```json
{"semantic_instance_id": "S08-crambin-1a2b3c4d", "decision": "accept",
 "reason": "", "notes": "", "label_override": null,
 "curator": "initials", "timestamp": "2026-08-12T17:40:00+00:00"}
```

The last decision for an instance wins, so a curator can revisit an item. A
rejection requires a reason, and overriding a generated label requires one too.

## Decisions are tied to the question, not to the family

The trailing hash in a `semantic_instance_id` covers the question parameters, so
a decision only applies to the exact question that was reviewed. Anything that
changes which parameters a generator settles on — bumping `seed`, adding a
protein to a `family_proteins` pool, changing a margin in `definitions_v1.yaml` —
retires the old identifiers and the decisions attached to them. They are not
silently reapplied to the replacement question, because it is a different
question.

This is the intended behaviour, but it means the pool should be frozen before
curation starts in earnest. The 2026-08-13 FoldBench expansion orphaned two
accepts this way (`N01-adenylate_kinase_closed-bf56ab5b`, whose protein no longer
wins an N01 slot, and `T01-hras-5826dd83`, which now has a different candidate
list). Orphaned lines are kept rather than deleted: they are the record of what
was reviewed and when.
