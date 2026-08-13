"""Command-line surface.

The specification's command names are provided verbatim through the
``structural-reasoning`` console script; ``pdbthink`` is the same program under
the repository's own name.

    structural-reasoning acquire  --manifest <sources.yaml>
    structural-reasoning build    --config <dataset.yaml> --output <dataset_dir>
    structural-reasoning validate --dataset <dataset_dir>
    structural-reasoning review   --dataset <dataset_dir> --decisions <decisions.jsonl>
    structural-reasoning evaluate --dataset <dataset_dir> --model-config <model.yaml> \
                                  --output <run_dir> --resume
    structural-reasoning score    --dataset <dataset_dir> --responses <run_dir> \
                                  --output <scores_dir>
    structural-reasoning report   --scores <scores_dir> --output <report_dir>

Every command exits nonzero on schema violations or inconsistent gold labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .acquisition.cache import AcquisitionError, StructureCache
from .acquisition.manifest import SourceManifest, manifest_from_dataset
from .config import ConfigError, DatasetConfig, Definitions
from .util import read_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except (ConfigError, AcquisitionError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structural-reasoning",
        description="Protein Structural Reasoning Benchmark (pdbthink)",
    )
    parser.add_argument("--version", action="version", version=f"pdbthink {__version__}")
    parser.add_argument(
        "--definitions",
        default=None,
        help="operational-definitions YAML (default configs/definitions_v1.yaml)",
    )
    parser.add_argument("--cache", default=None, help="structure cache directory")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("acquire", help="download and cache source structures")
    p.add_argument("--manifest", help="sources YAML listing PDB and AFDB entries")
    p.add_argument("--config", help="dataset config; its sources are acquired")
    p.add_argument("--write-manifest", help="write the resolved manifest to this path")
    p.set_defaults(handler=cmd_acquire)

    p = sub.add_parser("build", help="build a dataset from cached structures")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--decisions", help="curator decisions JSONL consumed by the build")
    p.add_argument(
        "--accepted-only",
        action="store_true",
        help="emit only instances a curator accepted (required for the final dataset)",
    )
    p.add_argument("--families", nargs="*", help="restrict to these question families")
    p.set_defaults(handler=cmd_build)

    p = sub.add_parser("validate", help="validate a built dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", help="dataset config, enabling composition checks")
    p.add_argument("--final", action="store_true", help="apply final-manifest rules")
    p.set_defaults(handler=cmd_validate)

    p = sub.add_parser("review", help="launch the curator review interface")
    p.add_argument("--dataset", required=True)
    p.add_argument("--decisions", required=True)
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument(
        "--auth-token",
        help="require this shared token (query string, cookie or bearer header). "
        "Also read from PDBTHINK_REVIEW_TOKEN. Required in practice whenever the "
        "server is bound beyond localhost",
    )
    p.add_argument("--export", help="export decisions to JSON and exit")
    p.set_defaults(handler=cmd_review)

    p = sub.add_parser("evaluate", help="run a model over a dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, help="evaluate at most this many renders")
    p.add_argument("--families", nargs="*")
    p.add_argument(
        "--max-input-tokens",
        type=int,
        help="skip renders whose prompt exceeds this many reference tokens, for "
        "models with a short context window",
    )
    p.add_argument("--cache-dir", help="response cache directory (default data/response_cache)")
    p.add_argument("--no-cache", action="store_true", help="ignore the response cache entirely")
    p.set_defaults(handler=cmd_evaluate)

    p = sub.add_parser(
        "batch",
        help="run a model through a Batch API at roughly half price, filling the "
        "response cache so that `evaluate` then needs no calls at all",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--state-dir", required=True, help="where batch ids and inputs are kept")
    p.add_argument("--families", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-input-tokens", type=int)
    p.add_argument("--cache-dir")
    p.add_argument(
        "--stage",
        choices=("submit", "poll", "fetch", "all"),
        default="all",
        help="all submits, waits and fetches; the separate stages let a long "
        "completion window be picked up by a later invocation",
    )
    p.add_argument("--poll-interval", type=float, default=60.0)
    p.set_defaults(handler=cmd_batch)

    p = sub.add_parser("score", help="score stored responses without calling a model")
    p.add_argument("--dataset", required=True)
    p.add_argument("--responses", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(handler=cmd_score)

    p = sub.add_parser("report", help="produce the benchmark report")
    p.add_argument("--scores", required=True, nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.set_defaults(handler=cmd_report)

    p = sub.add_parser("families", help="list the implemented question families")
    p.set_defaults(handler=cmd_families)
    return parser


# --------------------------------------------------------------------------- #

def _definitions(args) -> Definitions:
    return Definitions.load(args.definitions)


def cmd_acquire(args) -> int:
    cache = StructureCache(args.cache)
    if args.manifest:
        manifest = SourceManifest.load(args.manifest)
    elif args.config:
        manifest = manifest_from_dataset(DatasetConfig.load(args.config))
    else:
        print("error: pass --manifest or --config", file=sys.stderr)
        return 2

    if args.write_manifest:
        manifest.save(args.write_manifest)
        print(f"wrote manifest with {len(manifest)} entries to {args.write_manifest}")

    failures = 0
    for entry in manifest.pdb_entries:
        try:
            record = cache.get_pdb(entry)
            print(f"pdb  {entry}  {record.sha256[:12]}  {record.release_date}  {record.experimental_method}")
        except AcquisitionError as exc:
            failures += 1
            print(f"pdb  {entry}  FAILED: {exc}", file=sys.stderr)
    for entry in manifest.afdb_entries:
        try:
            record = cache.get_afdb(entry)
            print(f"afdb {entry}  {record.sha256[:12]}  v{record.extra.get('afdb_version')}")
        except AcquisitionError as exc:
            failures += 1
            print(f"afdb {entry}  FAILED: {exc}", file=sys.stderr)
    print(f"{len(manifest) - failures}/{len(manifest)} sources cached in {cache.root}")
    return 1 if failures else 0


def cmd_build(args) -> int:
    from .dataset import DatasetBuilder, write_dataset

    config = DatasetConfig.load(args.config)
    definitions = _definitions(args)
    cache = StructureCache(args.cache)
    decisions = _load_decisions(args.decisions) if args.decisions else {}

    builder = DatasetBuilder(
        config,
        definitions,
        cache,
        decisions=decisions,
        accepted_only=args.accepted_only,
    )
    result = builder.build(args.families)
    stats = write_dataset(result, args.output)
    print(f"built {stats['n_instances']} semantic instances / {stats['n_renders']} renders")
    print(f"  realised: {stats['realised_counts']}")
    print(f"  targets:  {stats['target_counts']}")
    print(f"  rejections recorded: {stats['n_rejection_records']}")
    print(f"  written to {args.output}")
    if not result.instances:
        print("error: build produced no instances", file=sys.stderr)
        return 1
    return 0


def cmd_validate(args) -> int:
    from .validate import validate_dataset

    config = DatasetConfig.load(args.config) if args.config else None
    report = validate_dataset(
        args.dataset,
        config=config,
        require_final_size=args.final,
        require_reviewed=args.final,
    )
    for warning in report.warnings:
        print(f"warning: {warning}")
    for error in report.errors:
        print(f"error: {error}", file=sys.stderr)
    print(
        f"{report.stats.get('n_instances', 0)} instances, "
        f"{report.stats.get('n_renders', 0)} renders, "
        f"{len(report.errors)} errors, {len(report.warnings)} warnings"
    )
    print(f"  realised counts: {report.stats.get('realised_counts')}")
    return 0 if report.ok else 1


def cmd_review(args) -> int:
    from .review_ui.server import export_decisions, serve

    if args.export:
        count = export_decisions(args.decisions, args.export)
        print(f"exported {count} decisions to {args.export}")
        return 0
    serve(
        dataset_dir=args.dataset,
        decisions_path=args.decisions,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        auth_token=args.auth_token,
    )
    return 0


def cmd_evaluate(args) -> int:
    from .evaluation.cache import ResponseCache
    from .evaluation.runner import EvaluationRunner, ModelConfig, ResumeError

    model = ModelConfig.load(args.model_config)
    cache = ResponseCache(
        getattr(args, "cache_dir", None), enabled=not getattr(args, "no_cache", False)
    )
    runner = EvaluationRunner(
        dataset_dir=args.dataset,
        model=model,
        output_dir=args.output,
        resume=args.resume,
        cache=cache,
    )
    try:
        summary = runner.run(
            limit=args.limit, families=args.families, max_input_tokens=args.max_input_tokens
        )
    except ResumeError as exc:
        raise ConfigError(str(exc)) from exc
    print(
        f"run {summary['run_id']}: {summary['completed']} completions "
        f"({summary['skipped']} reused, {summary['errors']} errors) -> {args.output}"
    )
    stats = summary.get("cache") or {}
    if stats.get("enabled"):
        print(
            f"  response cache: {stats['hits']} hits, {stats['misses']} misses, "
            f"{stats['writes']} written -> {stats['directory']}"
        )
    if summary.get("skipped_over_input_limit"):
        print(
            f"  {summary['skipped_over_input_limit']} renders skipped: prompt longer "
            f"than --max-input-tokens {args.max_input_tokens}"
        )
    return 1 if summary["errors"] and not summary["completed"] else 0


def cmd_batch(args) -> int:
    from .dataset import load_dataset
    from .evaluation.batch import BatchRun
    from .evaluation.cache import ResponseCache
    from .evaluation.runner import ModelConfig

    model = ModelConfig.load(args.model_config)
    cache = ResponseCache(args.cache_dir)
    instances, renders = load_dataset(args.dataset)
    accepted = {i.semantic_instance_id for i in instances if i.curation_status != "rejected"}
    renders = [r for r in renders if r.semantic_instance_id in accepted]
    if args.families:
        renders = [r for r in renders if r.question_family in set(args.families)]
    if args.max_input_tokens is not None:
        renders = [r for r in renders if (r.input_token_count or 0) <= args.max_input_tokens]
    renders.sort(key=lambda r: r.render_id)
    if args.limit:
        renders = renders[: args.limit]

    run = BatchRun(model, cache, args.state_dir)
    if args.stage in ("submit", "all"):
        run.preflight()
        pending = run.pending(renders)
        jobs = run.submit(renders)
        print(
            f"{len(renders)} renders, {len(pending)} uncached -> "
            f"{len(jobs)} batch(es): {', '.join(j.batch_id for j in jobs) or 'nothing to submit'}"
        )
    if args.stage in ("poll", "all"):
        jobs = run.wait(interval=args.poll_interval) if args.stage == "all" else run.poll()
        for job in jobs:
            print(f"  {job.batch_id}: {job.status} ({job.n_requests} requests)")
    if args.stage in ("fetch", "all"):
        if args.stage == "fetch":
            run.poll()   # otherwise the stored status predates the batch finishing
        result = run.fetch(renders)
        print(
            f"cached {result['stored']} completions "
            f"({result['failed']} failed, {result['unknown']} unrecognised)"
        )
        if not result["stored"]:
            for message, count in sorted(
                run.errors().items(), key=lambda kv: -kv[1]
            )[:3]:
                print(f"  {count} x {message}")
    return 0


def cmd_score(args) -> int:
    from .evaluation.score import score_run

    summary = score_run(args.dataset, args.responses, args.output)
    print(
        f"scored {summary['n_results']} completions across "
        f"{summary['n_renders']} renders -> {args.output}"
    )
    print(f"  macro score: {summary['macro_score']:.4f}   micro score: {summary['micro_score']:.4f}")
    print(f"  format errors: {summary['format_errors']}  refusals: {summary['refusals']}")
    return 0


def cmd_report(args) -> int:
    from .reporting.report import build_report

    summary = build_report(args.scores, args.output, bootstrap_samples=args.bootstrap)
    print(f"report for {summary['n_runs']} run(s) written to {args.output}")
    for row in summary["headline"]:
        print(
            f"  {row['model_id']:<40} macro={row['macro_score']:.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]  n={row['n_instances']}"
        )
    return 0


def cmd_families(args) -> int:
    from .generators import V1_FAMILIES, all_generators
    from .mechanistic.episodes import EPISODES

    print(f"{'family':<8}{'schema':<24}{'level':<12}{'croppable':<11}version")
    for family in V1_FAMILIES:
        if family == "T01":
            from .generators.two_state import T01

            generator = T01
        else:
            generator = all_generators()[family]
        print(
            f"{generator.family:<8}{generator.answer_schema:<24}{generator.level:<12}"
            f"{str(getattr(generator, 'croppable', False)):<11}{generator.version}"
        )
    print("\nmechanistic episodes:")
    for episode in EPISODES.values():
        fields = ", ".join(f.name for f in episode.fields)
        print(f"  {episode.id:<22}{'/'.join(episode.entries):<12}{fields}")
    return 0


def _load_decisions(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in read_jsonl(path):
        out[row["semantic_instance_id"]] = row
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
