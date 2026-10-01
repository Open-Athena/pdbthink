"""Separate commands keep network acquisition out of generation and scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..util import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["acquire", "acquire-pairs", "build", "build-pairs", "export", "unpack", "validate"],
    )
    parser.add_argument("--work", type=Path, default=REPO_ROOT / "data/taskgen_v1")
    parser.add_argument("--seed", type=int, default=2026100101)
    parser.add_argument("--source-limit", type=int, default=1100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--include-solutions", action="store_true", help="Oracle validation only")
    args = parser.parse_args()
    if args.command == "acquire":
        from .acquire import acquire_pool

        acquire_pool(REPO_ROOT, args.work, seed=args.seed, limit=args.source_limit, workers=args.workers)
    elif args.command == "build":
        from .build import build_pool

        build_pool(REPO_ROOT, args.work, workers=args.workers, per_family=args.per_family, limit=args.limit)
    elif args.command == "acquire-pairs":
        from .pairs import acquire_pairs

        acquire_pairs(REPO_ROOT, args.work, limit=args.limit or 350, workers=args.workers)
    elif args.command == "build-pairs":
        from .pairs import build_pairs

        build_pairs(REPO_ROOT, args.work, workers=args.workers)
    elif args.command == "validate":
        from .validate import validate_release

        if not args.dataset:
            parser.error("validate requires --dataset")
        print(json.dumps(validate_release(args.dataset, workers=args.workers), indent=2))
    elif args.command == "unpack":
        from .unpack import unpack_release

        if not args.dataset or not args.output:
            parser.error("unpack requires --dataset and --output")
        print(
            unpack_release(
                args.dataset,
                args.output,
                split=args.split,
                families=args.families,
                limit=args.limit,
                include_solutions=args.include_solutions,
            )
        )
    else:
        from .release import export_release

        manifest = export_release(
            REPO_ROOT, args.work, args.output or args.work / "release", total=args.count
        )
        print(
            json.dumps(
                {k: manifest[k] for k in ("task_count", "family_counts", "split_counts", "audit")}, indent=2
            )
        )


if __name__ == "__main__":
    main()
