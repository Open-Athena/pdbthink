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
