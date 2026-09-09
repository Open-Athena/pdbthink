# Result decks

| deck | covers |
| --- | --- |
| [pdbthink-results-2026-09-09.pptx](pdbthink-results-2026-09-09.pptx) · [PDF](pdbthink-results-2026-09-09.pdf) | Current. Ten models including GPT-6 Astra and Claude Opus 5, both budget ladders complete, refusals counted apart from format errors. |
| [pdbthink-results-2026-09-08.pptx](pdbthink-results-2026-09-08.pptx) · [PDF](pdbthink-results-2026-09-08.pdf) | Superseded — its headline scores were diluted by the context-only controls. |
| [pdbthink-results-2026-09-04.pptx](pdbthink-results-2026-09-04.pptx) | Superseded. Before the frontier pair. |
| [pdbthink-results-2026-09-03.pptx](pdbthink-results-2026-09-03.pptx) | Superseded. Before S03/S04/S09 were rebalanced. |
| [pdbthink-results-2026-08-15.pptx](pdbthink-results-2026-08-15.pptx) | Superseded. Before the budget re-runs. |
| [pdbthink-results-2026-08-13.pptx](pdbthink-results-2026-08-13.pptx) | Superseded. First results, smoke set, prompt v1. |

Generated from a collected results file rather than hard-coded, because these
numbers moved eight times while the work was running:

```bash
python docs/slides/collect_results.py <runs-dir> results.json
```

```bash
python docs/slides/build_deck_v2.py results.json docs/slides/pdbthink-results-2026-09-09.pptx
```

```bash
libreoffice --headless --convert-to pdf --outdir docs/slides docs/slides/pdbthink-results-2026-09-09.pptx
```

`collect_results.py` computes five things a plain score does not: the macro over
**primary renders only** (coordinate renders, excluding the context-only
controls and rotation variants, matching what `report` has always done), every
score restricted to responses that terminated, higher-budget re-runs folded in
as a ladder, refusals counted apart from format errors, and per-run family
coverage so an incomplete run is never averaged against a complete one.
