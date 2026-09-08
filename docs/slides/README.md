# Result decks

| deck | covers |
| --- | --- |
| [pdbthink-results-2026-09-08.pptx](pdbthink-results-2026-09-08.pptx) · [PDF](pdbthink-results-2026-09-08.pdf) | Current. Eight models, label-balanced dataset, two-tier output-budget ladder for Kimi K3. |
| [pdbthink-results-2026-09-04.pptx](pdbthink-results-2026-09-04.pptx) | Superseded. Before the 256k budget tier. |
| [pdbthink-results-2026-09-03.pptx](pdbthink-results-2026-09-03.pptx) | Superseded. Before S03/S04/S09 were rebalanced. |
| [pdbthink-results-2026-08-15.pptx](pdbthink-results-2026-08-15.pptx) | Superseded. Before the budget re-run and before three runs completed. |
| [pdbthink-results-2026-08-13.pptx](pdbthink-results-2026-08-13.pptx) | Superseded. First results, 20-instance smoke set, 97-instance dataset, prompt v1. |

The deck is generated from a collected results file rather than hard-coded,
because these numbers moved seven times while the work was running:

```bash
python docs/slides/collect_results.py <runs-dir> results.json
```

```bash
python docs/slides/build_deck_v2.py results.json docs/slides/pdbthink-results-2026-09-08.pptx
```

A PDF is committed alongside the pptx so the deck can be read in a browser
without PowerPoint. Regenerate it after any change to the slides:

```bash
libreoffice --headless --convert-to pdf --outdir docs/slides docs/slides/pdbthink-results-2026-09-08.pptx
```

`collect_results.py` computes four things a plain score does not: every score
restricted to responses that terminated, the same score with higher-budget
re-runs folded in as a ladder (a prompt cut off at 32k may be re-run at 64k and
again at 256k), the context-only gain conditioned on completion, and per-run
family coverage so an incomplete run is never averaged against a complete one.

The runs directory is expected to hold `f3_scores_<label>/` for the main run and
optionally `hi_scores_<label>/` and `max_scores_<label>/` for the budget tiers.
See [../results.md](../results.md) for what these corrections change — in one
case a partial run read 0.808 where the completed run reads 0.578, and in
another a model read 0.684 where it scores 0.804 once allowed to finish.
