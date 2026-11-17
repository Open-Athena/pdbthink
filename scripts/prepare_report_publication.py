"""Publish an audited report snapshot while preserving preceding reports and download URLs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from pdbthink.util import read_json, sha256_bytes, write_json

ARTIFACTS = (
    "index.html", "evidence.json", "metrics.csv", "figures.zip",
    *(f"{name}.{extension}" for name in ("accuracy", "question-accuracy", "family-accuracy", "accuracy-cost")
      for extension in ("svg", "pdf", "png")),
)


def artifact_paths(source: Path) -> tuple[str, ...]:
    evidence = read_json(source / "evidence.json")
    if not evidence.get("standardized_comparison"):
        return ARTIFACTS
    if evidence["standardized_comparison"]["path"] != "standardized/":
        raise ValueError("Unexpected standardized report destination")
    comparison = read_json(source / "standardized/evidence.json")
    if comparison["dataset_fingerprint"] != evidence["dataset_fingerprint"]:
        raise ValueError("Comparison report uses a different frozen question panel")
    if comparison["metrics"] != evidence["standardized_comparison"]["metrics"]:
        raise ValueError("Landing page differs from the standardized results")
    return (*ARTIFACTS, *("standardized/" + name for name in ARTIFACTS))


def git(site: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(site), *args], text=True).strip()


def prepare(
    source: Path, site: Path, archive_name: str, *, allow_incomplete: bool = False,
    snapshot_name: str | None = None,
) -> None:
    if source == site or site in source.parents:
        raise ValueError("Report source must be outside the publication checkout")
    if not archive_name or Path(archive_name).name != archive_name or archive_name in (".", ".."):
        raise ValueError("Archive name must be one path component")
    evidence = read_json(source / "evidence.json")
    if not evidence["complete"] and not allow_incomplete:
        raise ValueError("Cannot publish an incomplete head-to-head comparison")
    if evidence.get("standardized_comparison", {}).get("complete") is False and not allow_incomplete:
        raise ValueError("Cannot publish an incomplete standardized comparison")
    artifacts = artifact_paths(source)
    if snapshot_name and (
        Path(snapshot_name).name != snapshot_name or snapshot_name in (".", "..")
    ):
        raise ValueError("Snapshot name must be one path component")
    snapshot = site / "reports" / snapshot_name if snapshot_name else None
    if snapshot and snapshot.exists():
        raise ValueError("Report snapshot already exists; never overwrite it")
    if git(source, "status", "--porcelain"):
        raise ValueError("Commit the source and audited report before preparing publication")
    if git(site, "status", "--porcelain"):
        raise ValueError("Publication checkout must be clean")
    archive = site / "historical" / archive_name
    if archive.exists():
        raise ValueError("Historical report already exists; never overwrite it")
    for name in artifacts:
        if not (source / name).is_file():
            raise ValueError(f"Missing report artifact: {name}")
    source_commit = git(source, "rev-parse", "HEAD")
    previous_commit = git(site, "rev-parse", "HEAD")
    paths = [name for name in git(site, "ls-files", "-z").split("\0") if name]
    for name in paths:
        if name.startswith(("historical/", "reports/")):
            continue
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Keep existing download URLs working while freezing the old landing page.
        shutil.copy2(site / name, destination)
    for name in artifacts:
        (site / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, site / name)
    if snapshot:
        snapshot.mkdir(parents=True)
        for name in artifacts:
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, snapshot / name)
    (site / ".nojekyll").touch()
    (site / "README.md").write_text(
        "# PDBThink coordinate benchmark\n\n"
        "[Read the current report](https://openathena.ai/pdbthink/).\n\n"
        f"[Previous report](historical/{archive_name}/index.html). "
        "Historical scores and request conditions are preserved.\n"
    )
    write_json(
        site / "publication.json",
        {
            "source_commit": source_commit,
            "previous_site_commit": previous_commit,
            "scientific_commit": evidence["scientific_commit"],
            "dataset_fingerprint": evidence["dataset_fingerprint"],
            "complete": evidence["complete"],
            "snapshot": f"reports/{snapshot_name}/" if snapshot_name else None,
            "artifacts": {name: sha256_bytes((source / name).read_bytes()) for name in artifacts},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument(
        "--allow-incomplete", action="store_true", help="Publish explicitly marked coverage gaps"
    )
    parser.add_argument("--snapshot-name", help="Also preserve this publication at a stable report URL")
    args = parser.parse_args()
    prepare(args.source.resolve(), args.site.resolve(), args.archive_name,
            allow_incomplete=args.allow_incomplete, snapshot_name=args.snapshot_name)
