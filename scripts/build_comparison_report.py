"""Publish both conditions together without pooling their scores or reusing responses."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_coordinate_report import build


def build_comparison(dataset: Path, runs: Path, output: Path) -> dict:
    configs = Path(__file__).parents[1] / "configs"
    standardized = build(dataset, runs, configs / "report_models_v7.json", output / "standardized")
    return build(dataset, runs, configs / "report_models_v6.json", output, standardized_report=standardized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_comparison(args.dataset, args.runs, args.output)
