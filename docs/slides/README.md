# Result decks

| deck | covers |
| --- | --- |
| [pdbthink-results-2026-08-15.pptx](pdbthink-results-2026-08-15.pptx) | Eight models over the full 117-instance set. Leads with the truncation finding. |
| [pdbthink-results-2026-08-13.pptx](pdbthink-results-2026-08-13.pptx) | Superseded. First results, 20-instance smoke set, 97-instance dataset, prompt v1. |

The current deck is generated from a collected results file rather than
hard-coded, because these numbers moved four times while the sweep was running:

```bash
python docs/slides/collect_results.py <runs-dir> results.json
```

```bash
python docs/slides/build_deck_v2.py results.json docs/slides/pdbthink-results-2026-08-15.pptx
```

`collect_results.py` computes three things a plain score does not: every score
restricted to responses that terminated, the context-only gain conditioned on
completion, and per-run family coverage so an incomplete run is never averaged
against a complete one. See [../results.md](../results.md) for what those
corrections change.
