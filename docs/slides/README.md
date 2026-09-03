# Result decks

| deck | covers |
| --- | --- |
| [pdbthink-results-2026-09-03.pptx](pdbthink-results-2026-09-03.pptx) | Current. Eight models, all twenty families, including the output-budget experiment. |
| [pdbthink-results-2026-08-15.pptx](pdbthink-results-2026-08-15.pptx) | Superseded. Same sweep before the budget re-run and before three runs were completed. |
| [pdbthink-results-2026-08-13.pptx](pdbthink-results-2026-08-13.pptx) | Superseded. First results, 20-instance smoke set, 97-instance dataset, prompt v1. |

The deck is generated from a collected results file rather than hard-coded,
because these numbers moved five times while the sweep was running:

```bash
python docs/slides/collect_results.py <runs-dir> results.json
```

```bash
python docs/slides/build_deck_v2.py results.json docs/slides/pdbthink-results-2026-09-03.pptx
```

`collect_results.py` computes four things a plain score does not: every score
restricted to responses that terminated, the same score with a higher-budget
re-run folded in, the context-only gain conditioned on completion, and per-run
family coverage so an incomplete run is never averaged against a complete one.
See [../results.md](../results.md) for what those corrections change — in one
case a partial run read 0.808 where the completed run reads 0.577.
