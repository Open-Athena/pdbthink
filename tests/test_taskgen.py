"""Post-training exports must preserve both benchmark separation and its reward contract."""

from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import tomllib

from pdbthink.dataset import Candidate, DatasetBuilder
from pdbthink.generators import GenerationContext, Proposal, get_generator
from pdbthink.taskgen.build import configuration
from pdbthink.taskgen.coordinates import parse_structures, recompute
from pdbthink.taskgen.exclusion import audit_source, polymer_sequences, sequence_hash
from pdbthink.taskgen.harbor import gold_response, negative_response, task_archives, validate_reward
from pdbthink.util import rng_for, sha256_bytes


@pytest.mark.parametrize("family", ["S03", "S04", "S05", "S09"])
def test_category_example_is_a_valid_family_answer(family):
    from pdbthink.prompts.library import answer_format
    from pdbthink.scoring import score_response

    categories = get_generator(family).prompt_parameters({})["categories"]
    example = answer_format("category", family).split("Example: ", 1)[1]
    assert any(
        score_response(example, "category", {"value": category}, parameters={"categories": categories})[
            "score"
        ]["correct"]
        for category in categories
    )


def test_prompt_revision_preserves_coordinates_gold_and_identity():
    from copy import deepcopy

    from pdbthink.prompts.library import PROMPT_VERSION, answer_format
    from pdbthink.representations.tokens import count_tokens
    from pdbthink.taskgen.revisions import update_prompt

    old = "Answer with exactly one of the listed categories.\nExample: FINAL: helix\n"
    for family in ("S03", "S04", "S05", "S09"):
        task = {
            "instance": {"question_family": family, "gold_answer": {"value": "unchanged"}},
            "semantic_key": "unchanged",
            "render": {
                "answer_schema": "category",
                "prompt_version": "v3",
                "tokenizer": "cl100k_base",
                "system_prompt": "System",
                "user_prompt": "Coordinates and question\n\n" + old,
                "gold_answer": {"value": "unchanged"},
                "displayed_coordinates_sha256": "unchanged",
            },
        }
        update_prompt(task)
        assert task["render"]["prompt_version"] == PROMPT_VERSION
        assert (
            task["render"]["user_prompt"]
            == "Coordinates and question\n\n" + answer_format("category", family) + "\n"
        )
        assert (
            task["render"]["input_token_count"] == count_tokens("System\n" + task["render"]["user_prompt"])[0]
        )
        assert task["render"]["gold_answer"] == task["instance"]["gold_answer"] == {"value": "unchanged"}
        assert task["semantic_key"] == task["render"]["displayed_coordinates_sha256"] == "unchanged"
        first = deepcopy(task)
        update_prompt(task)
        assert task == first


def test_expansion_retains_parent_when_a_family_has_no_new_tasks(tmp_path, monkeypatch):
    from copy import deepcopy

    from pdbthink.taskgen.build import select_tasks
    from pdbthink.util import stable_hash

    monkeypatch.setattr("pdbthink.taskgen.build.V1_FAMILIES", ["G01", "S03"])
    tasks = []
    for family in ("G01", "S03"):
        instance = {
            "question_family": family,
            "source_entries": ["NEW1"],
            "selected_chains": ["A"],
            "biological_assembly_ids": [],
            "question_parameters": {},
            "gold_evidence": {"cluster": "new-cluster"},
        }
        tasks.append({"instance": instance, "semantic_key": stable_hash(family, ["NEW1"], ["A"], [], {})})
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards/one.json.gz").write_bytes(gzip.compress(json.dumps({"tasks": tasks}).encode()))
    parent = deepcopy(tasks[1])
    parent["parent_split"] = "test"
    selected = select_tasks(tmp_path, 2, retained=[parent])
    assert selected[0] == parent
    assert len({t["semantic_key"] for t in selected}) == 2
    with pytest.raises(ValueError, match="smaller"):
        select_tasks(tmp_path, 0, retained=[parent])


def test_post_training_sampler_extends_benchmark_shortlist(tmp_path):
    from pdbthink.acquisition.cache import StructureCache
    from pdbthink.config import Definitions
    from pdbthink.taskgen.build import ordered_proposals
    from pdbthink.util import derive_seed

    builder = DatasetBuilder(configuration(8237), Definitions.load(), StructureCache(tmp_path, offline=True))
    proposals = [Proposal(parameters={"residue": f"A:A{i}"}, tag="one") for i in range(20)]
    ordered = ordered_proposals(builder, proposals, "S03", "new")
    original = builder._diversify(builder._seeded_choice(proposals, "S03", "new"))
    offset = derive_seed(8237, "tag-start", "S03", "new") % len(original)
    assert ordered[: len(original)] == original[offset:] + original[:offset]
    assert len({p.key() for p in ordered}) == 20


def test_native_context_counts_ids_not_tokenizer_result_fields(coordinate_task, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from pdbthink.taskgen.context import _count
    from pdbthink.taskgen.harbor import row_for

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] and kwargs["enable_thinking"]
            assert kwargs["tools"] == []
            assert [m["role"] for m in messages] == ["system", "user"]
            return {"input_ids": [1] * 24576, "attention_mask": [1] * 24576}

    monkeypatch.setattr("pdbthink.taskgen.context.TOKENIZER", Tokenizer())
    path = tmp_path / "one.parquet"
    pq.write_table(pa.Table.from_pylist([row_for(coordinate_task, "train", "group")]), path)
    assert _count(path)[0]["input_tokens"] == 24576


@pytest.fixture
def coordinate_task(crambin, definitions, tmp_path):
    from pdbthink.acquisition.cache import StructureCache

    builder = DatasetBuilder(configuration(8237), definitions, StructureCache(tmp_path, offline=True))
    displayed, analysis = builder._proposal_frame(crambin.spec, crambin)
    generator = get_generator("P03")
    ctx = GenerationContext(crambin.spec, crambin, displayed, definitions, rng_for(8237), analysis)
    proposal = next(p for p in generator.propose(ctx) if isinstance(p, Proposal))
    instance, renders = builder._materialise(
        Candidate("P03", generator, crambin.spec, crambin, proposal, crambin.structure)
    )
    render = next(r for r in renders if r.representation == "minimal_pdb")
    return {"instance": instance.model_dump(), "render": render.model_dump(), "semantic_key": "abc123"}


def test_gold_recomputes_from_serialised_coordinates(coordinate_task, definitions):
    task = coordinate_task
    assert (
        recompute(task["render"]["user_prompt"], "P03", task["instance"]["question_parameters"], definitions)
        == task["render"]["gold_answer"]
    )
    assert parse_structures(task["render"]["user_prompt"], definitions)[0].atom_count == 327
    altered = task["render"]["user_prompt"].replace("  1.00  0.00", "  1.00 42.00", 1)
    with pytest.raises(ValueError, match="non-sanitised"):
        parse_structures(altered, definitions)


def test_strict_task_generation_rejects_source_identity_in_prompt(monkeypatch, request):
    from pdbthink.dataset import BuildRejection

    original = DatasetBuilder._render

    def leaking_render(self, *args, **kwargs):
        render = original(self, *args, **kwargs)
        render.user_prompt += "\nSource entry: 1CRN\n"
        return render

    monkeypatch.setattr(DatasetBuilder, "_render", leaking_render)
    with pytest.raises(BuildRejection, match="prompt_provenance_leak"):
        request.getfixturevalue("coordinate_task")


def test_different_entry_with_benchmark_sequence_is_excluded(crambin):
    record = crambin.record
    seqs = polymer_sequences(record.path)
    exclusion = {
        "sources": [],
        "source_file_sha256s": [],
        "sequence_sha256s": [sequence_hash(s) for s in seqs.values()],
        "rcsb30_clusters": [],
        "fingerprint": "frozen",
    }
    audit = audit_source(record, exclusion, {})
    assert not audit["allowed"]
    assert "benchmark_protein_sequence" in audit["reasons"]


def test_benchmark_source_hash_is_excluded_even_without_id_match(crambin):
    exclusion = {
        "sources": [],
        "source_file_sha256s": [sha256_bytes(Path(crambin.record.path).read_bytes())],
        "sequence_sha256s": [],
        "rcsb30_clusters": [],
        "fingerprint": "frozen",
    }
    assert "benchmark_source_file" in audit_source(crambin.record, exclusion, {})["reasons"]


def test_nonselected_partner_homology_excludes_entire_entry(crambin, monkeypatch):
    monkeypatch.setattr(
        "pdbthink.taskgen.exclusion.polymer_sequences",
        lambda _: {"entity:1": "ACDEF", "entity:2": "HIKLM"},
    )
    exclusion = {
        "sources": [],
        "source_file_sha256s": [],
        "sequence_sha256s": [],
        "rcsb30_clusters": ["benchmark-cluster"],
        "fingerprint": "frozen",
    }
    entry = crambin.record.entry.upper()
    audit = audit_source(
        crambin.record, exclusion, {f"{entry}_1": "fresh-cluster", f"{entry}_2": "benchmark-cluster"}
    )
    assert not audit["allowed"]
    assert "benchmark_rcsb30_cluster" in audit["reasons"]


def test_source_groups_union_sequences_clusters_and_state_pairs(tmp_path):
    pytest.importorskip("pyarrow")
    from pdbthink.taskgen.release import source_groups

    (tmp_path / "audits").mkdir()
    for entry, cluster, sequence in [
        ("A", "cluster1", "seq1"),
        ("B", "cluster1", "seq2"),
        ("C", "cluster2", "seq2"),
        ("D", "cluster3", "seq3"),
        ("E", "cluster4", "seq4"),
    ]:
        (tmp_path / "audits" / f"{entry}.json").write_text(
            json.dumps(
                {
                    "entry": entry,
                    "allowed": True,
                    "rcsb30_clusters": [cluster],
                    "sequence_sha256s": [sequence],
                }
            )
        )
    (tmp_path / "pairs.json").write_text(
        json.dumps({"pairs": [{"first": {"entry": "C"}, "second": {"entry": "D"}}]})
    )
    groups = source_groups(tmp_path)
    assert len({groups[e] for e in "ABCD"}) == 1
    assert groups["E"] != groups["A"]


def test_archives_are_deterministic_and_solutions_separate(coordinate_task):
    first, solution = task_archives(coordinate_task, "pdbthink-test")
    assert (first, solution) == task_archives(coordinate_task, "pdbthink-test")
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as tar:
        names = tar.getnames()
        assert "solution/solve.sh" not in names
        assert "tests/gold.json" in names
        config = tomllib.loads(tar.extractfile("task.toml").read().decode())
        assert config["metadata"]["requires_tool_free_runner"]
        assert config["environment"]["network_mode"] == "no-network"
        assert b"tests" not in tar.extractfile("environment/Dockerfile").read()
        assert all(m.mtime == 0 and not m.issym() for m in tar.getmembers())
    with tarfile.open(fileobj=io.BytesIO(solution), mode="r:gz") as tar:
        assert tar.getnames() == ["solution/answer.txt", "solution/solve.sh"]
    assert gzip.decompress(first)


def test_portable_verifier_positive_negative_empty_and_truncation(coordinate_task, tmp_path):
    binary, _ = task_archives(coordinate_task, "pdbthink-test")
    with tarfile.open(fileobj=io.BytesIO(binary), mode="r:gz") as tar:
        tar.extractall(tmp_path, filter="data")
    answer = tmp_path / "answer.txt"
    reward = tmp_path / "reward"
    env = {
        **os.environ,
        "PDBTHINK_TEST_ROOT": str(tmp_path / "tests"),
        "PDBTHINK_ANSWER": str(answer),
        "PDBTHINK_REWARD_DIR": str(reward),
    }
    gold = gold_response(coordinate_task["render"]["answer_schema"], coordinate_task["render"]["gold_answer"])
    for text, expected, status in [
        (gold, 1, {}),
        (negative_response(coordinate_task), 0, {}),
        ("", 0, {}),
        (gold, 0, {"truncated": True}),
        (gold, 0, {"tool_events": ["tool_calls"]}),
    ]:
        answer.write_text(text)
        answer.with_name("answer_status.json").write_text(json.dumps(status))
        subprocess.run([sys.executable, str(tmp_path / "tests/verify.py")], check=True, env=env)
        assert float((reward / "reward.txt").read_text()) == expected
    validate_reward(coordinate_task)


def test_two_state_answer_contract():
    from pdbthink.scoring import score_response

    gold = {"gained": ["A:A1--A:V9"], "lost": ["A:G2--A:W8"]}
    text = gold_response("two_interaction_sets", gold)
    assert score_response(text, "two_interaction_sets", gold)["score"]["correct"]


def test_published_coordinate_track_is_compatible_with_coordinate_only_checkout(coordinate_task):
    from pdbthink.taskgen.harbor import contract_for

    task = {**coordinate_task, "instance": dict(coordinate_task["instance"])}
    task["instance"].pop("task_track", None)
    expected = contract_for(task)
    task["instance"]["task_track"] = "coordinate_reasoning"
    assert contract_for(task) == expected
    task["instance"]["task_track"] = "sequence_prediction"
    with pytest.raises(ValueError, match="different task track"):
        contract_for(task)


def test_packaged_release_validation_and_solution_opt_in(coordinate_task, tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from pdbthink.taskgen.harbor import row_for
    from pdbthink.taskgen.unpack import unpack_release
    from pdbthink.taskgen.validate import validate_release

    release = tmp_path / "release"
    (release / "data").mkdir(parents=True)
    (release / "audits").mkdir()
    row = row_for(coordinate_task, "test", "test-group")
    parquet = release / "data/test-00000.parquet"
    pq.write_table(pa.Table.from_pylist([row]), parquet)
    (release / "manifest.json").write_text(
        json.dumps(
            {
                "task_count": 1,
                "family_counts": {"P03": 1},
                "data_hashes": {"data/test-00000.parquet": sha256_bytes(parquet.read_bytes())},
            }
        )
    )
    (release / "audits/exclusion.json").write_text(
        json.dumps(
            {
                "sources": [],
                "sequence_sha256s": [],
                "rcsb30_clusters": [],
            }
        )
    )
    (release / "audits/sources.json").write_text(
        json.dumps(
            {
                e: {"sequence_sha256s": ["new-sequence"], "rcsb30_clusters": ["new-cluster"]}
                for e in row["source_entries"]
            }
        )
    )
    assert validate_release(release, workers=1)["packaged_coordinate_oracle_checks"] == 1
    for oracle in (False, True):
        output = tmp_path / str(oracle)
        assert unpack_release(release, output, split="test", include_solutions=oracle) == 1
        assert (output / row["path"] / "solution").exists() == oracle
    parquet.write_bytes(parquet.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_release(release, workers=1)


def test_tool_free_request_uses_full_available_budget_and_rejects_overrides():
    from pdbthink.taskgen.protocol import request_payload, tool_events

    budget = {
        "input_tokens": 20000,
        "context_window": 32768,
        "native_output_limit": 32768,
        "tokenizer_revision": "native-test",
        "endpoint_limit_source": "test",
    }
    payload = request_payload("test-model", [], budget)
    assert payload["max_tokens"] == 12768
    assert payload["tools"] == [] and payload["tool_choice"] == "none"
    assert payload["plugins"] == [{"id": "web", "enabled": False}]
    for field in ["tools", "functions", "function_call", "max_tokens", "messages", "plugins"]:
        with pytest.raises(ValueError, match="override"):
            request_payload("test-model", [], budget, extra={field: []})
    with pytest.raises(ValueError, match="context exclusion"):
        request_payload("test-model", [], {**budget, "input_tokens": 32768})
    assert tool_events({"choices": [{"message": {"tool_calls": [{"id": "x"}]}}]})
    assert not tool_events({"choices": [{"message": {"content": "FINAL: A"}}]})
