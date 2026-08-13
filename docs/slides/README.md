# Result decks

| deck | covers |
| --- | --- |
| [pdbthink-results-2026-08-13.pptx](pdbthink-results-2026-08-13.pptx) | First results: DeepSeek V4 Flash 0731 and Qwen3-1.7B on the 20-instance smoke set |

Rebuild with:

```bash
python docs/slides/build_deck.py docs/slides/pdbthink-results-2026-08-13.pptx
```

The numbers are hard-coded in the generator rather than read from a scores
directory, because runs live outside the repository. Each one is sourced from a
`scores/*/scores.jsonl` produced by `structural-reasoning score`; regenerate the
run before changing a figure.
