"""Refresh a published coordinate panel against the checked-out scientific contract.

Parent prompts and coordinates are frozen inputs. Every new gold answer is computed
from those displayed coordinates; rejected questions never silently enter a score.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pdbthink.chem import chemistry_fingerprint
from pdbthink.config import Definitions
from pdbthink.dataset import BuildResult, write_dataset
from pdbthink.evaluation.runner import dataset_fingerprint
from pdbthink.generators import T01, Analysis, get_generator
from pdbthink.prompts.library import PROMPT_VERSION, answer_format, prompt_fingerprint
from pdbthink.prompts.render import StructureBlock, build_prompt
from pdbthink.representations.tokens import count_tokens
from pdbthink.schemas import RejectionRecord, RenderedVariant, SemanticInstance
from pdbthink.scoring import SCORING_VERSION
from pdbthink.taskgen.coordinates import parse_structures, recompute
from pdbthink.util import gold_hash, read_jsonl, sha256_bytes, write_json
from pdbthink.validate import validate_dataset

VERSION = "coordinate-report-v5.0.0"
COORDINATES = re.compile(r"^(?:ATOM  |HETATM|TER   |END$).*$", re.M)


def parent_records(dataset: Path, archive_path: Path) -> Iterator[tuple[SemanticInstance, RenderedVariant]]:
    gold = {
        r["semantic_instance_id"]: r
        for r in read_jsonl(dataset / "gold.jsonl")
        if "semantic_instance_id" in r
    }
    instances = {}
    for row in read_jsonl(dataset / "instances.jsonl"):
        if "semantic_instance_id" not in row:
            continue
        if row.pop("task_track", "coordinate_reasoning") != "coordinate_reasoning":
            raise ValueError("Only coordinate reasoning belongs in this report")
        row.update({k: gold[row["semantic_instance_id"]][k] for k in ("gold_answer", "gold_evidence")})
        instances[row["semantic_instance_id"]] = SemanticInstance(**row)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for name, expected in manifest.items():
            if sha256_bytes(archive.read(name)) != expected["sha256"]:
                raise ValueError(f"Parent archive hash mismatch: {name}")
        records = {}
        for name in archive.namelist():
            if re.fullmatch(r"questions/q\d+\.json", name):
                record = json.loads(archive.read(name))
                records[record["render_id"]] = record
    for row in read_jsonl(dataset / "renders.jsonl"):
        if "render_id" not in row:
            continue
        if row.pop("task_track", "coordinate_reasoning") != "coordinate_reasoning":
            raise ValueError("Only coordinate reasoning belongs in this report")
        record = records[row["render_id"]]
        prompt = (dataset / "prompts" / (row["render_id"].replace("::", "__") + ".txt")).read_text()
        if prompt != record["user_prompt"] or gold_hash(record["gold"]) != row["gold_sha256"]:
            raise ValueError("Parent dataset differs from its published archive")
        row.update(system_prompt=record["system_prompt"], user_prompt=prompt, gold_answer=record["gold"])
        yield instances[row["semantic_instance_id"]], RenderedVariant(**row)


def revised(
    instance: SemanticInstance, old: RenderedVariant, definitions: Definitions
) -> tuple[SemanticInstance, RenderedVariant]:
    structures = parse_structures(old.user_prompt, definitions)
    generator = T01 if old.question_family == "T01" else get_generator(old.question_family)
    params = instance.question_parameters
    blocks = [
        StructureBlock(label, s, cropped=old.crop is not None)
        for label, s in zip(old.state_order, structures, strict=True)
    ]
    prompt = build_prompt(
        representation="minimal_pdb",
        blocks=blocks,
        context=generator.context(params),
        question=generator.question(params, structures[0]),
        answer_schema=old.answer_schema,
        format_instructions=answer_format(old.answer_schema, old.question_family),
    )
    if COORDINATES.findall(prompt.user_prompt) != COORDINATES.findall(old.user_prompt):
        raise ValueError("coordinate_roundtrip_changed")
    gold = recompute(prompt.user_prompt, old.question_family, params, definitions)
    if old.question_family == "T01":
        evidence = T01.oracle(tuple(structures), params, definitions)["evidence"]
    else:
        evidence = generator.oracle(
            structures[0], params, definitions, Analysis(structures[0], definitions)
        ).evidence
    identifier = instance.semantic_instance_id + "::report-v5"
    new_instance = instance.model_copy(
        update={
            "semantic_instance_id": identifier,
            "question_version": generator.version,
            "gold_answer": gold,
            "gold_evidence": evidence,
            "gold_sha256": gold_hash(gold),
            "definition_version": definitions.version,
        }
    )
    tokens, tokenizer = count_tokens(prompt.system_prompt + "\n" + prompt.user_prompt, old.tokenizer)
    new = old.model_copy(
        update={
            "semantic_instance_id": identifier,
            "render_id": old.render_id + "::report-v5",
            "prompt_version": PROMPT_VERSION,
            "system_prompt": prompt.system_prompt,
            "user_prompt": prompt.user_prompt,
            "gold_answer": gold,
            "gold_sha256": gold_hash(gold),
            "input_token_count": tokens,
            "tokenizer": tokenizer,
        }
    )
    return new_instance, new


def freeze(parent: Path, archive: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen dataset; use a new output directory")
    definitions = Definitions.load()
    result = BuildResult()
    changes = []
    for instance, old in parent_records(parent, archive):
        try:
            new_instance, new = revised(instance, old, definitions)
        except ValueError as exc:
            reason = str(exc).split(":", 1)[0]
            result.rejections.append(
                RejectionRecord(
                    question_family=old.question_family,
                    protein_group_id=old.protein_group_id,
                    reason=reason,
                    detail={"parent_render_id": old.render_id, "error": str(exc)},
                )
            )
            changes.append(
                {
                    "parent_render_id": old.render_id,
                    "family": old.question_family,
                    "status": "rejected",
                    "reason": reason,
                }
            )
            print(old.question_family, "REJECTED", str(exc), flush=True)
            continue
        result.instances.append(new_instance)
        result.renders.append(new)
        changes.append(
            {
                "parent_render_id": old.render_id,
                "render_id": new.render_id,
                "family": old.question_family,
                "status": "admitted",
                "system_changed": old.system_prompt != new.system_prompt,
                "user_changed": old.user_prompt != new.user_prompt,
                "gold_changed": old.gold_answer != new.gold_answer,
                "parent_system_sha256": sha256_bytes(old.system_prompt.encode()),
                "parent_user_sha256": sha256_bytes(old.user_prompt.encode()),
                "system_sha256": sha256_bytes(new.system_prompt.encode()),
                "user_sha256": sha256_bytes(new.user_prompt.encode()),
                "gold_sha256": new.gold_sha256,
            }
        )
        print(
            old.question_family,
            "admitted",
            "gold_changed=" + str(old.gold_answer != new.gold_answer),
            flush=True,
        )
    result.stats = {
        "version": VERSION,
        "scientific_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint(),
        "chemistry_fingerprint": chemistry_fingerprint(),
        "scoring_version": SCORING_VERSION,
        "definitions_sha256": definitions.content_sha256,
        "parent_archive_sha256": sha256_bytes(archive.read_bytes()),
        "dataset_fingerprint": dataset_fingerprint(result.renders),
        "questions": len(result.renders),
        "families": dict(Counter(r.question_family for r in result.renders)),
        "changes": changes,
    }
    write_dataset(result, output)
    validation = validate_dataset(output)
    write_json(
        output / "validation.json",
        {"ok": validation.ok, "errors": validation.errors, "warnings": validation.warnings},
    )
    if not validation.ok:
        raise ValueError(validation.errors)
    return result.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = freeze(args.parent, args.archive, args.output)
    print(json.dumps({k: v for k, v in data.items() if k != "changes"}, indent=2))
