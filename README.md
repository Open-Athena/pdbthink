# pdbthink evaluation report

[Read the public report](https://open-athena.github.io/pdbthink/).

This branch publishes the static Snowball Step 38 comparison report through GitHub Pages. It includes the full benchmark comparison for the hosted baselines, the matched 25-question comparison with Snowball, separate F01/F02 sequence-prediction results, and complete prompts and model outputs. Qwen is excluded from this edition pending complete results.

- [Exact prompts, outputs and score records](https://open-athena.github.io/pdbthink/exact-records.zip)
- [Metrics CSV](https://open-athena.github.io/pdbthink/metrics.csv)
- [Evidence JSON](https://open-athena.github.io/pdbthink/evidence.json)
- [Benchmark source repository](https://github.com/Open-Athena/pdbthink/tree/main)

The methods section records the tool-use audit and its limits. The HTML is self-contained and can also be downloaded and opened offline. Plot downloads are in `plots/`; exact-record hashes and validation counts are in `validation.json`.

To update this publication, regenerate `docs/snowball-comparison/` in the research checkout with `python -m scripts.build_snowball_writeup`, copy its contents to this branch, and push. GitHub Pages publishes this branch's root. Keep `.nojekyll` in place. Add `--include-qwen` only after all Qwen panels are complete.
