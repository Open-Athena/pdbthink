"""The response cache and the batch path that fills it.

The cache exists so that adding a question to the benchmark costs the calls for
that question and nothing else, so the tests here are mostly about what does
*not* invalidate an entry (a rebuilt dataset, a moved render id) and what does
(a different output budget, a changed prompt).
"""

from __future__ import annotations

import json

import pytest

from pdbthink.evaluation.cache import CachedResponse, CacheKey, ResponseCache, extract_reasoning


def key(**overrides) -> CacheKey:
    base = {
        "provider": "openai_chat",
        "endpoint": "https://provider.example/v1",
        "model_id": "some/model",
        "model_revision": None,
        "reasoning_effort": None,
        "max_output_tokens": 4096,
        "sampling_parameters": {"temperature": 0.0},
        "system_prompt": "system",
        "user_prompt": "user",
        "completion_index": 0,
    }
    return CacheKey(**{**base, **overrides})


class TestKeying:
    def test_the_same_request_is_the_same_entry(self):
        assert key().digest == key().digest

    @pytest.mark.parametrize(
        "field,value",
        [
            ("endpoint", "https://another-provider.example/v1"),
            ("model_id", "other/model"),
            ("max_output_tokens", 8192),
            ("user_prompt", "a different question"),
            ("system_prompt", "a different system prompt"),
            ("completion_index", 1),
            ("reasoning_effort", "high"),
            ("model_revision", "2026-08-01"),
            ("sampling_parameters", {"temperature": 0.7}),
        ],
    )
    def test_anything_that_changes_the_answer_changes_the_key(self, field, value):
        assert key().digest != key(**{field: value}).digest

    def test_the_output_budget_is_part_of_the_key(self):
        """Truncation scores zero, so a short-budget answer must never be reused."""
        assert key(max_output_tokens=8192).digest != key(max_output_tokens=65536).digest

    def test_endpoint_is_stored_for_provider_attribution(self, tmp_path):
        cache = ResponseCache(tmp_path)
        cache.put(key(), CachedResponse(text="FINAL: A"))
        stored = json.loads(next(iter(cache.entries()))[0].read_text())
        assert stored["request"]["endpoint"] == "https://provider.example/v1"

    def test_legacy_digest_is_not_endpoint_scoped(self):
        """Old batch state can be fetched, but ordinary v2 keys stay isolated."""
        direct = key(endpoint="https://provider.example/v1")
        gateway = key(endpoint="https://gateway.example/v1")
        assert direct.legacy_v1_digest == gateway.legacy_v1_digest
        assert direct.digest != gateway.digest

    def test_the_prompt_is_stored_by_hash_only(self, tmp_path):
        """Prompts are large and already in the dataset; the cache holds hashes."""
        cache = ResponseCache(tmp_path)
        secret = "ATOM      1  N   ILE A  72     -12.424   2.081   4.751"
        cache.put(key(user_prompt=secret), CachedResponse(text="FINAL: A"))
        stored = json.loads(next(iter(cache.entries()))[0].read_text())
        assert secret not in json.dumps(stored)
        assert stored["request"]["user_prompt_sha256"]


class TestRoundTrip:
    def test_a_stored_response_comes_back_whole(self, tmp_path):
        cache = ResponseCache(tmp_path)
        response = CachedResponse(
            text="FINAL: A:C40",
            usage={"completion_tokens": 900},
            truncated=False,
            reasoning="First I list the cysteines...",
            raw={"choices": [{"message": {"content": "FINAL: A:C40"}}]},
        )
        cache.put(key(), response, provenance={"render_id": "S08-x::minimal_pdb::1"})
        loaded = cache.get(key())
        assert loaded is not None
        assert loaded.text == response.text
        assert loaded.reasoning == response.reasoning
        assert loaded.raw == response.raw
        assert loaded.usage == response.usage
        assert cache.hits == 1

    def test_a_miss_is_a_miss(self, tmp_path):
        cache = ResponseCache(tmp_path)
        assert cache.get(key()) is None
        assert cache.misses == 1

    def test_a_corrupt_entry_is_a_miss_rather_than_a_crash(self, tmp_path):
        cache = ResponseCache(tmp_path)
        path = cache.path_for(key())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not json")
        assert cache.get(key()) is None

    def test_disabling_the_cache_stores_and_returns_nothing(self, tmp_path):
        cache = ResponseCache(tmp_path, enabled=False)
        assert cache.put(key(), CachedResponse(text="x")) is None
        assert cache.get(key()) is None


class TestReasoningExtraction:
    @pytest.mark.parametrize(
        "choice,expected",
        [
            ({"message": {"reasoning_content": "because"}}, "because"),
            ({"message": {"reasoning": "because"}}, "because"),
            ({"message": {"content": "FINAL: A"}}, ""),
            ({"message": {"reasoning": [{"text": "a"}, {"text": "b"}]}}, "a\nb"),
            ({"message": {"reasoning_content": "   "}, "reasoning": "fallback"}, "fallback"),
        ],
    )
    def test_providers_disagree_about_the_field_name(self, choice, expected):
        assert extract_reasoning(choice) == expected


class TestSurvivesARebuild:
    def test_a_rebuilt_dataset_reuses_every_completion(self, tmp_path):
        """The point of the whole module: identical prompts, moved identifiers."""
        cache = ResponseCache(tmp_path)
        prompts = [f"question {i}" for i in range(5)]
        for prompt in prompts:
            cache.put(key(user_prompt=prompt), CachedResponse(text="FINAL: A"))
        cache.hits = cache.misses = 0

        # A rebuild renames every render and reorders the set; the text is the same.
        for prompt in reversed(prompts):
            assert cache.get(key(user_prompt=prompt)) is not None
        assert (cache.hits, cache.misses) == (5, 0)

        # Adding a question costs exactly one call.
        assert cache.get(key(user_prompt="a newly added question")) is None
        assert cache.misses == 1


class FakeBatchClient:
    """Stands in for Together: records what was submitted, replays a canned output."""

    def __init__(self, output_lines: list[dict]) -> None:
        self.output_lines = output_lines
        self.uploaded: list[str] = []
        self.created: list[str] = []

    def upload(self, path, purpose="batch-api"):
        self.uploaded.append(path.read_text())
        return "file-1"

    def create(self, input_file_id, *, endpoint="/v1/chat/completions", completion_window="24h"):
        self.created.append(input_file_id)
        return {"id": "batch-1", "status": "VALIDATING"}

    def retrieve(self, batch_id):
        return {"id": batch_id, "status": "COMPLETED", "output_file_id": "file-out"}

    def content(self, file_id):
        return "\n".join(json.dumps(row) for row in self.output_lines).encode()


class TestBatch:
    @pytest.fixture
    def pieces(self, tmp_path):
        from types import SimpleNamespace

        from pdbthink.evaluation.runner import ModelConfig

        model = ModelConfig(
            model_id="some/model", provider="openai_chat",
            base_url="https://example.invalid/v1", api_key_env="NOPE",
            max_output_tokens=4096, temperature=0.0, completions=1,
        )
        renders = [
            SimpleNamespace(
                render_id=f"P01-x-{i}::minimal_pdb::1", semantic_instance_id=f"P01-x-{i}",
                question_family="P01", protein_group_id="x", representation="minimal_pdb",
                input_token_count=100, system_prompt="system", user_prompt=f"question {i}",
            )
            for i in range(3)
        ]
        return model, renders, ResponseCache(tmp_path / "cache")

    def test_only_uncached_prompts_are_submitted(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        run = BatchRun(model, cache, tmp_path / "state", client=FakeBatchClient([]))
        cache.put(run.key_for(renders[0], 0), CachedResponse(text="already answered"))
        assert len(run.pending(renders)) == 2

    def test_a_finished_batch_lands_in_the_cache(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        jobs = run.submit(renders)
        assert len(jobs) == 1 and jobs[0].n_requests == 3

        # Reply to each submitted custom_id the way the provider would.
        client.output_lines = [
            {
                "custom_id": custom_id,
                "response": {"body": {
                    "choices": [{
                        "message": {"content": "FINAL: A", "reasoning_content": "thinking"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"completion_tokens": 7},
                }},
            }
            for custom_id in jobs[0].custom_ids
        ]
        run.poll()
        assert run.fetch(renders) == {"stored": 3, "failed": 0, "unknown": 0}
        for render in renders:
            entry = cache.get(run.key_for(render, 0))
            assert entry is not None and entry.text == "FINAL: A"
            assert entry.reasoning == "thinking"

    def test_a_batch_submitted_with_v1_keys_lands_in_the_v2_cache(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        jobs = run.submit(renders)
        job = jobs[0]
        legacy_by_current = {
            run.key_for(render, 0).digest: run.key_for(render, 0).legacy_v1_digest
            for render in renders
        }
        for custom_id, digest in list(job.custom_ids.items()):
            job.custom_ids[custom_id] = legacy_by_current[digest]
        run._save_jobs(jobs)
        client.output_lines = [
            {
                "custom_id": custom_id,
                "response": {"body": {
                    "choices": [{
                        "message": {"content": "FINAL: A"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"completion_tokens": 2},
                }},
            }
            for custom_id in job.custom_ids
        ]
        run.poll()
        assert run.fetch(renders) == {"stored": 3, "failed": 0, "unknown": 0}
        for render in renders:
            assert cache.get(run.key_for(render, 0)) is not None

    def test_a_failed_request_is_counted_not_cached(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        jobs = run.submit(renders)
        client.output_lines = [
            {"custom_id": custom_id, "error": {"message": "upstream refused"}}
            for custom_id in jobs[0].custom_ids
        ]
        run.poll()
        result = run.fetch(renders)
        assert result["stored"] == 0 and result["failed"] == 3
        # Nothing cached means the next submission retries exactly these prompts.
        assert len(run.pending(renders)) == 3

    def test_resubmission_is_refused_while_a_batch_is_outstanding(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        run.submit(renders)
        run.submit(renders)
        assert len(client.created) == 1, "a second submit would pay for the same prompts twice"
