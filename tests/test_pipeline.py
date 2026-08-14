"""End-to-end pipeline: build, validate, evaluate offline, score, report.

The whole test runs without network access and without a model: it builds a tiny
dataset from the committed fixtures, then answers every prompt with the gold
label through the ``mock`` provider, which must score exactly 1.0.
"""

from __future__ import annotations

import json

import pytest
import yaml

from pdbthink.acquisition.cache import StructureCache
from pdbthink.config import DatasetConfig, Definitions
from pdbthink.dataset import DatasetBuilder, load_dataset, write_dataset
from pdbthink.evaluation.runner import EvaluationRunner, ModelConfig, ResumeError
from pdbthink.evaluation.score import score_run
from pdbthink.prompts.library import SYSTEM_PROMPT, prompt_fingerprint
from pdbthink.reporting.report import build_report
from pdbthink.validate import format_gold_answer, validate_dataset

FIXTURE_CONFIG = {
    "name": "pdbthink_test",
    "version": "0.0.1",
    "seed": 7,
    "token_budget": {"automatic": 64000, "mechanistic": 96000, "tokenizer": "cl100k_base"},
    "rotation_variant_fraction": 1.0,
    "representation_variant_fraction": 1.0,
    "max_instances_per_protein": 12,
    "family_targets": {"P01": 1, "P02": 1, "P03": 1, "G01": 1, "S04": 1, "S05": 1, "S08": 1, "S09": 1},
    "proteins": [
        {"id": "crambin", "source_type": "pdb", "entry": "1CRN"},
    ],
    "episodes": [],
}


class FixtureCache(StructureCache):
    """Serves the committed fixture files instead of downloading anything."""

    FILES = {"1CRN": "crambin_processed.pdb", "1CA2": "zinc_site.pdb"}

    def get_pdb(self, entry: str):
        from tests.conftest import fixture_record

        return fixture_record(self.FILES[entry.upper()], entry.upper())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    config_path = root / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(FIXTURE_CONFIG), encoding="utf-8")
    config = DatasetConfig.load(config_path)
    definitions = Definitions.load()
    builder = DatasetBuilder(config, definitions, FixtureCache(root / "cache"))
    result = builder.build()
    dataset_dir = root / "dataset"
    write_dataset(result, dataset_dir)
    return {"root": root, "config": config, "dataset_dir": dataset_dir, "result": result}


class TestBuild:
    def test_every_requested_family_is_realised(self, built):
        realised = built["result"].realised_counts()
        assert set(realised) == set(FIXTURE_CONFIG["family_targets"])

    def test_instances_record_full_provenance(self, built):
        for instance in built["result"].instances:
            assert instance.source_entries and instance.source_file_sha256s
            assert instance.definition_version
            assert instance.gold_evidence
            assert instance.selection_margins or instance.question_family == "P01"

    def test_prompts_never_mention_the_source_entry(self, built):
        for render in built["result"].renders:
            assert "1CRN" not in render.user_prompt.upper()

    def test_system_prompt_is_the_versioned_one(self, built):
        for render in built["result"].renders:
            assert render.system_prompt == SYSTEM_PROMPT

    def test_paired_representations_share_gold_and_atoms(self, built):
        by_key: dict[tuple[str, int], list] = {}
        for render in built["result"].renders:
            by_key.setdefault((render.semantic_instance_id, render.rotation_seed), []).append(render)
        for group in by_key.values():
            if len(group) < 2:
                continue
            assert len({json.dumps(r.gold_answer, sort_keys=True) for r in group}) == 1
            assert len({r.atom_count for r in group}) == 1

    def test_memorisable_families_get_a_context_only_baseline(self, built):
        """The floor a coordinate score has to beat: same question, no structure."""
        from pdbthink.dataset import CONTEXT_ONLY_FAMILIES

        renders = built["result"].renders
        by_family: dict[str, list] = {}
        for render in renders:
            by_family.setdefault(render.question_family, []).append(render)
        # The fixture protein cannot support every family, so check the ones it built.
        covered = [f for f in CONTEXT_ONLY_FAMILIES if f in by_family]
        assert len(covered) >= 4

        for family in covered:
            blind = [r for r in by_family[family] if r.representation == "context_only"]
            instances = {r.semantic_instance_id for r in by_family[family]}
            assert len(blind) == len(instances), family
            for render in blind:
                assert "ATOM" not in render.user_prompt
                assert "HETATM" not in render.user_prompt
                assert "No coordinates are provided" in render.user_prompt
                # The question and its answer are unchanged; only the input differs.
                primary = next(
                    r
                    for r in by_family[family]
                    if r.semantic_instance_id == render.semantic_instance_id
                    and r.representation == "minimal_pdb"
                    and not r.is_rotation_variant
                )
                assert render.gold_answer == primary.gold_answer

        for family, group in by_family.items():
            if family not in CONTEXT_ONLY_FAMILIES:
                assert not [r for r in group if r.representation == "context_only"], family

    def test_the_context_only_baseline_is_reported_per_family(self, built):
        from pdbthink.dataset import CONTEXT_ONLY_FAMILIES
        from pdbthink.reporting.metrics import context_only_baseline

        rows = [
            {
                "question_family": r.question_family,
                "semantic_instance_id": r.semantic_instance_id,
                "representation": r.representation,
                "is_rotation_variant": r.is_rotation_variant,
                "score": 0.0 if r.representation == "context_only" else 1.0,
            }
            for r in built["result"].renders
        ]
        baseline = context_only_baseline(rows)
        built_families = {r.question_family for r in built["result"].renders}
        assert set(baseline) == set(CONTEXT_ONLY_FAMILIES) & built_families
        for stats in baseline.values():
            assert stats["gain_over_context_only"] == pytest.approx(1.0)

    def test_rotation_variants_use_a_different_seed(self, built):
        rotated = [r for r in built["result"].renders if r.is_rotation_variant]
        assert rotated
        for render in rotated:
            siblings = [
                other
                for other in built["result"].renders
                if other.semantic_instance_id == render.semantic_instance_id
                and not other.is_rotation_variant
            ]
            assert all(other.rotation_seed != render.rotation_seed for other in siblings)


class TestValidate:
    def test_dataset_validates_cleanly(self, built):
        report = validate_dataset(built["dataset_dir"], config=built["config"])
        assert report.errors == []

    def test_gold_answers_score_themselves(self, built):
        _, renders = load_dataset(built["dataset_dir"])
        from pdbthink.scoring import score_response

        for render in renders:
            text = format_gold_answer(render.answer_schema, render.gold_answer)
            outcome = score_response(text, render.answer_schema, render.gold_answer)
            assert outcome["score"]["score"] == 1.0, render.render_id

    def test_a_corrupted_gold_label_is_caught(self, built, tmp_path):
        import shutil

        broken = tmp_path / "broken"
        shutil.copytree(built["dataset_dir"], broken)
        # Gold now lives in the sidecar; corrupt exactly one variant there so it
        # disagrees with its own siblings.
        rows = [json.loads(line) for line in (broken / "gold.jsonl").read_text().splitlines()]
        renders = {
            r["render_id"]: r
            for r in (json.loads(line) for line in (broken / "renders.jsonl").read_text().splitlines())
            if "_canary" not in r
        }
        for row in rows:
            rid = row.get("render_id")
            if rid and renders[rid]["answer_schema"] == "integer" and renders[rid]["is_rotation_variant"]:
                row["gold_answer"] = {"value": row["gold_answer"]["value"] + 1}
                break
        (broken / "gold.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        report = validate_dataset(broken, config=built["config"])
        assert any("gold answer changes under rotation" in e for e in report.errors), report.errors

    def test_withheld_gold_is_reported_clearly(self, built, tmp_path):
        """A manifest without its sidecar must say how to regenerate it."""
        import shutil

        from pdbthink.dataset import MissingGold

        stripped = tmp_path / "stripped"
        shutil.copytree(built["dataset_dir"], stripped)
        (stripped / "gold.jsonl").unlink()
        with pytest.raises(MissingGold, match="structural-reasoning build"):
            load_dataset(stripped)

    def test_every_manifest_carries_the_canary(self, built):
        from pdbthink.canary import CANARY_GUID

        for name in ("instances.jsonl", "renders.jsonl", "gold.jsonl"):
            first = json.loads((built["dataset_dir"] / name).read_text().splitlines()[0])
            assert first["_canary"] == CANARY_GUID, name


class TestDeterminism:
    def test_rebuilding_reproduces_byte_identical_prompts(self, built):
        config = built["config"]
        builder = DatasetBuilder(config, Definitions.load(), FixtureCache(built["root"] / "cache2"))
        again = builder.build()

        first = {r.render_id: r.user_prompt for r in built["result"].renders}
        second = {r.render_id: r.user_prompt for r in again.renders}
        assert set(first) == set(second)
        assert all(first[k] == second[k] for k in first)

    def test_rebuilding_reproduces_gold_labels(self, built):
        builder = DatasetBuilder(
            built["config"], Definitions.load(), FixtureCache(built["root"] / "cache3")
        )
        again = builder.build()
        first = {i.semantic_instance_id: i.gold_answer for i in built["result"].instances}
        second = {i.semantic_instance_id: i.gold_answer for i in again.instances}
        assert first == second

    def test_prompt_library_is_fingerprinted(self):
        assert len(prompt_fingerprint()) == 16


class TestEvaluateScoreReport:
    def test_gold_answering_model_scores_one(self, built, tmp_path):
        model = ModelConfig(
            model_id="mock-gold", provider="mock", completions=1, extra_body={"answer_gold": True}
        )
        run_dir = tmp_path / "run"
        runner = EvaluationRunner(built["dataset_dir"], model, run_dir)
        summary = runner.run()
        assert summary["errors"] == 0
        assert summary["completed"] == len(built["result"].renders)

        scores_dir = tmp_path / "scores"
        scored = score_run(built["dataset_dir"], run_dir, scores_dir)
        assert scored["macro_score"] == pytest.approx(1.0)
        assert scored["micro_score"] == pytest.approx(1.0)
        assert scored["format_errors"] == 0

        report_dir = tmp_path / "report"
        report = build_report([scores_dir], report_dir, bootstrap_samples=64)
        assert report["headline"][0]["macro_score"] == pytest.approx(1.0)
        assert (report_dir / "report.md").exists()
        assert (report_dir / "report.html").exists()

    def test_runs_are_resumable(self, built, tmp_path):
        model = ModelConfig(
            model_id="mock-gold", provider="mock", completions=1, extra_body={"answer_gold": True}
        )
        run_dir = tmp_path / "resume"
        first = EvaluationRunner(built["dataset_dir"], model, run_dir).run(limit=3)
        assert first["completed"] == 3

        second = EvaluationRunner(built["dataset_dir"], model, run_dir, resume=True).run()
        assert second["skipped"] == 3
        assert second["completed"] == len(built["result"].renders) - 3

    def test_resume_rejects_results_from_another_model(self, built, tmp_path):
        run_dir = tmp_path / "mixed-models"
        first_model = ModelConfig(model_id="mock-one", provider="mock", completions=1)
        EvaluationRunner(built["dataset_dir"], first_model, run_dir).run(limit=1)

        second_model = ModelConfig(model_id="mock-two", provider="mock", completions=1)
        with pytest.raises(ResumeError, match="separate --output directory"):
            EvaluationRunner(
                built["dataset_dir"], second_model, run_dir, resume=True
            ).run(limit=1)

    def test_resume_rejects_a_rebuilt_dataset(self, built, tmp_path):
        import shutil

        dataset_dir = tmp_path / "rebuilt"
        shutil.copytree(built["dataset_dir"], dataset_dir)
        run_dir = tmp_path / "stale-prompts"
        model = ModelConfig(model_id="mock", provider="mock", completions=1)
        EvaluationRunner(dataset_dir, model, run_dir).run(limit=1)

        prompt = next((dataset_dir / "prompts").glob("*.txt"))
        prompt.write_text(prompt.read_text() + "\nChanged prompt.\n")
        with pytest.raises(ResumeError, match="separate --output"):
            EvaluationRunner(dataset_dir, model, run_dir, resume=True).run(limit=1)

    def test_endpoint_changes_the_run_identity(self, built):
        direct = ModelConfig(
            model_id="same-model", base_url="https://provider.example/v1"
        )
        gateway = ModelConfig(
            model_id="same-model", base_url="https://gateway.example/v1"
        )
        assert direct.run_id("dataset-fingerprint") != gateway.run_id("dataset-fingerprint")

    def test_endpoint_identity_ignores_a_trailing_slash(self, built):
        without_slash = ModelConfig(model_id="same-model", base_url="https://example.test/v1")
        with_slash = ModelConfig(model_id="same-model", base_url="https://example.test/v1/")
        assert without_slash.run_id("dataset-fingerprint") == with_slash.run_id(
            "dataset-fingerprint"
        )

    def test_scoring_needs_no_model(self, built, tmp_path):
        """`score` reads stored responses only; it never calls a provider."""
        model = ModelConfig(model_id="mock", provider="mock", completions=1)
        run_dir = tmp_path / "run2"
        EvaluationRunner(built["dataset_dir"], model, run_dir).run()
        scores = score_run(built["dataset_dir"], run_dir, tmp_path / "scores2")
        assert scores["n_results"] > 0
