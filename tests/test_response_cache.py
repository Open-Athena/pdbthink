"""The response cache and the batch path that fills it.

The cache exists so that adding a question to the benchmark costs the calls for
that question and nothing else, so the tests here are mostly about what does
*not* invalidate an entry (a rebuilt dataset, a moved render id) and what does
(a different output budget, a changed prompt).
"""

from __future__ import annotations

import json

import pytest

from pdbthink.evaluation.cache import (
    CacheDiscoveryError,
    CachedResponse,
    CacheKey,
    ResponseCache,
    extract_reasoning,
)


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
        """Old batch state can be fetched, but ordinary current keys stay isolated."""
        direct = key(endpoint="https://provider.example/v1")
        gateway = key(endpoint="https://gateway.example/v1")
        assert direct.legacy_v1_digest == gateway.legacy_v1_digest
        assert direct.digest != gateway.digest

    def test_legacy_v2_migration_preserves_an_explicit_seed(self):
        generated = key(
            sampling_parameters={"temperature": 0.0, "seed": 1001},
            completion_index=1,
            legacy_v2_generated_seed=True,
        )
        explicit_same_value = key(
            sampling_parameters={"temperature": 0.0, "seed": 1001},
            completion_index=1,
            legacy_v2_generated_seed=False,
        )
        assert (
            generated.legacy_v2_digest(3)
            != explicit_same_value.legacy_v2_digest(3)
        )

    def test_cache_identity_tracks_the_effective_repeat_request(self):
        from pdbthink.evaluation.runner import ModelConfig

        single = ModelConfig(model_id="same", completions=1)
        three = ModelConfig(model_id="same", completions=3)
        ten = ModelConfig(model_id="same", completions=10)
        assert key(sampling_parameters=single.sampling_parameters_for(0)).digest != key(
            sampling_parameters=three.sampling_parameters_for(0)
        ).digest
        assert key(sampling_parameters=three.sampling_parameters_for(0)).digest == key(
            sampling_parameters=ten.sampling_parameters_for(0)
        ).digest
        assert three.run_id("dataset") != ten.run_id("dataset")

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
            refusal=True,
            reasoning="First I list the cysteines...",
            raw={"choices": [{"message": {"content": "FINAL: A:C40"}}]},
        )
        cache.put(key(), response, provenance={"render_id": "S08-x::minimal_pdb::1"})
        loaded = cache.get(key())
        assert loaded is not None
        assert loaded.text == response.text
        assert loaded.reasoning == response.reasoning
        assert loaded.refusal is True
        assert loaded.raw == response.raw
        assert loaded.usage == response.usage
        assert cache.hits == 1

    @pytest.mark.parametrize(
        ("stop_reason", "truncated", "refusal"),
        [
            ("model_context_window_exceeded", True, False),
            ("refusal", False, True),
        ],
    )
    def test_current_cache_derives_new_terminal_flags_from_old_raw_response(
        self, tmp_path, stop_reason, truncated, refusal
    ):
        cache = ResponseCache(tmp_path)
        current = key()
        path = cache.put(
            current,
            CachedResponse(
                text="FINAL: A",
                raw={"content": [], "stop_reason": stop_reason},
            ),
        )
        entry = json.loads(path.read_text())
        entry["derived"].pop("refusal")
        entry["derived"]["truncated"] = False
        path.write_text(json.dumps(entry))

        loaded = cache.get(current)
        assert loaded is not None
        assert loaded.truncated is truncated
        assert loaded.refusal is refusal

    def test_empty_openai_refusal_is_not_derived_as_a_refusal(self, tmp_path):
        cache = ResponseCache(tmp_path)
        current = key()
        path = cache.put(
            current,
            CachedResponse(
                text="FINAL: A",
                raw={"choices": [{"message": {
                    "content": "FINAL: A",
                    "refusal": "",
                }}]},
            ),
        )
        entry = json.loads(path.read_text())
        entry["derived"].pop("refusal")
        path.write_text(json.dumps(entry))

        loaded = cache.get(current)
        assert loaded is not None
        assert loaded.refusal is False

    def test_a_miss_is_a_miss(self, tmp_path):
        cache = ResponseCache(tmp_path)
        assert cache.get(key()) is None
        assert cache.misses == 1

    def test_unseeded_format_2_entry_is_reused(self, tmp_path, monkeypatch):
        cache = ResponseCache(tmp_path)
        current = key()
        legacy_digest = current.legacy_v2_digest(1)
        legacy_path = (
            cache.directory
            / current.provider
            / legacy_digest[:2]
            / f"{legacy_digest}.json"
        )
        request = current.request_fingerprint()
        request["sampling_parameters"] = {
            **current.sampling_parameters,
            "completions": 1,
        }
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(json.dumps({
            "cache_format": 2,
            "cache_key": legacy_digest,
            "created_at": "2026-08-01T00:00:00+0000",
            "request": request,
            "derived": {
                "text": "FINAL: A",
                "usage": {"completion_tokens": 2},
                "truncated": False,
                "reasoning": "",
            },
            "response": {"choices": [{"message": {"content": "FINAL: A"}}]},
        }))

        monkeypatch.setattr(
            cache, "_legacy_v2_index",
            lambda provider: pytest.fail("direct migration scanned the cache"),
        )
        loaded = cache.get(current)
        assert loaded is not None and loaded.text == "FINAL: A"
        assert cache.hits == 1 and cache.misses == 0
        promoted = json.loads(cache.path_for(current).read_text())
        assert promoted["cache_key"] == current.digest
        assert promoted["provenance"]["migrated_from_cache_key"] == legacy_digest

    def test_seeded_format_2_entry_survives_repeat_count_change(
        self, tmp_path, monkeypatch
    ):
        cache = ResponseCache(tmp_path)
        current = key(
            sampling_parameters={"temperature": 0.0, "seed": 1001},
            completion_index=1,
            legacy_v2_completions=10,
        )
        legacy_digest = current.legacy_v2_digest(3)
        legacy_path = (
            cache.directory
            / current.provider
            / legacy_digest[:2]
            / f"{legacy_digest}.json"
        )
        request = current.request_fingerprint()
        request["sampling_parameters"] = {
            "temperature": 0.0,
            "completions": 3,
        }
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(json.dumps({
            "cache_format": 2,
            "cache_key": legacy_digest,
            "created_at": "2026-08-01T00:00:00+0000",
            "request": request,
            "provenance": {"run_id": "old-run"},
            "derived": {
                "text": "FINAL: B",
                "usage": {},
                "truncated": False,
                "reasoning": "",
            },
            "response": {"choices": [{"message": {"content": "FINAL: B"}}]},
        }))

        unrelated_path = cache.put(
            key(user_prompt="unrelated"),
            CachedResponse(
                text="FINAL: OTHER",
                raw={"choices": [{"message": {"content": "FINAL: OTHER"}}]},
            ),
        )
        assert unrelated_path is not None
        original_read = cache._read_entry_with_status

        def guarded_read(path):
            if path == unrelated_path:
                pytest.fail("format-3 response body was decoded during migration")
            return original_read(path)

        monkeypatch.setattr(cache, "_read_entry_with_status", guarded_read)
        loaded = cache.get(current)
        assert loaded is not None and loaded.text == "FINAL: B"
        promoted = json.loads(cache.path_for(current).read_text())
        assert promoted["cache_key"] == current.digest
        assert promoted["provenance"]["run_id"] == "old-run"
        assert promoted["provenance"]["migrated_from_cache_key"] == legacy_digest

    @pytest.mark.parametrize(
        ("current_generated", "stored_generated"),
        [(True, False), (False, True)],
    )
    def test_same_wire_seed_migrates_between_generated_and_explicit_provenance(
        self, tmp_path, current_generated, stored_generated
    ):
        from dataclasses import replace

        cache = ResponseCache(tmp_path)
        current = key(
            sampling_parameters={"temperature": 0.0, "seed": 1001},
            completion_index=1,
            legacy_v2_completions=10,
            legacy_v2_generated_seed=current_generated,
        )
        stored = replace(
            current,
            legacy_v2_completions=3,
            legacy_v2_generated_seed=stored_generated,
        )
        legacy_digest = stored.legacy_v2_digest(3)
        legacy_path = (
            cache.directory
            / stored.provider
            / legacy_digest[:2]
            / f"{legacy_digest}.json"
        )
        request = stored.request_fingerprint()
        request["sampling_parameters"] = stored._legacy_v2_sampling_parameters(3)
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(json.dumps({
            "cache_format": 2,
            "cache_key": legacy_digest,
            "created_at": "2026-08-01T00:00:00+0000",
            "request": request,
            "derived": {
                "text": "FINAL: SAME",
                "usage": {},
                "truncated": False,
                "reasoning": "",
            },
            "response": {"choices": [{"message": {"content": "FINAL: SAME"}}]},
        }))

        loaded = cache.get(current)
        assert loaded is not None and loaded.text == "FINAL: SAME"
        promoted = json.loads(cache.path_for(current).read_text())
        assert promoted["provenance"]["migrated_from_cache_key"] == legacy_digest

    def test_non_object_exact_cache_entry_fails_closed(self, tmp_path):
        cache = ResponseCache(tmp_path)
        current = key()
        path = cache.path_for(current)
        path.parent.mkdir(parents=True)
        path.write_text("[1]")

        with pytest.raises(CacheDiscoveryError, match="move or remove"):
            cache.get(current)
        assert cache.misses == 0

    @pytest.mark.parametrize(
        ("field", "value"),
        [("derived", [1]), ("response", [1])],
    )
    def test_malformed_nested_exact_cache_data_fails_closed(
        self, tmp_path, field, value
    ):
        cache = ResponseCache(tmp_path)
        current = key()
        path = cache.put(current, CachedResponse(text="FINAL: A"))
        assert path is not None
        entry = json.loads(path.read_text())
        entry[field] = value
        path.write_text(json.dumps(entry))

        with pytest.raises(CacheDiscoveryError, match="move or remove"):
            cache.get(current)
        assert cache.misses == 0

    def test_invalid_usage_metadata_fails_closed(self, tmp_path):
        cache = ResponseCache(tmp_path)
        current = key()
        path = cache.put(current, CachedResponse(text="FINAL: A"))
        assert path is not None
        entry = json.loads(path.read_text())
        entry["derived"]["usage"] = []
        path.write_text(json.dumps(entry))

        with pytest.raises(CacheDiscoveryError, match="corrupt"):
            cache.get(current)

    def test_invalid_truncation_metadata_fails_closed(self, tmp_path):
        cache = ResponseCache(tmp_path)
        current = key()
        path = cache.put(current, CachedResponse(text="FINAL: A"))
        assert path is not None
        entry = json.loads(path.read_text())
        entry["derived"]["truncated"] = "false"
        path.write_text(json.dumps(entry))

        with pytest.raises(CacheDiscoveryError, match="corrupt"):
            cache.get(current)

    def test_mismatched_exact_cache_identity_fails_closed(self, tmp_path):
        cache = ResponseCache(tmp_path)
        current = key()
        path = cache.put(current, CachedResponse(text="FINAL: A"))
        assert path is not None
        entry = json.loads(path.read_text())
        entry["cache_key"] = "wrong"
        path.write_text(json.dumps(entry))

        with pytest.raises(CacheDiscoveryError, match="does not match"):
            cache.get(current)

    def test_put_syncs_the_cache_directory(self, tmp_path, monkeypatch):
        import pdbthink.evaluation.cache as cache_module

        synced = []
        monkeypatch.setattr(cache_module, "fsync_directory", synced.append)
        cache = ResponseCache(tmp_path)
        path = cache.put(key(), CachedResponse(text="FINAL: A"))

        assert path is not None
        assert synced[-1] == path.parent

    def test_malformed_legacy_json_is_a_complete_scan(self, tmp_path, monkeypatch):
        cache = ResponseCache(tmp_path)
        malformed = cache.directory / key().provider / "aa" / "broken.json"
        malformed.parent.mkdir(parents=True)
        malformed.write_text('{"cache_format": 2,')

        scans = 0
        original_refresh = cache._refresh_legacy_v2_index

        def counted_refresh(provider):
            nonlocal scans
            scans += 1
            return original_refresh(provider)

        monkeypatch.setattr(cache, "_refresh_legacy_v2_index", counted_refresh)
        assert cache.get(key(user_prompt="first new prompt")) is None
        assert cache.get(key(user_prompt="second new prompt")) is None
        assert scans == 1

    def test_unreadable_current_entry_fails_closed(self, tmp_path, monkeypatch):
        cache = ResponseCache(tmp_path)
        current = key()
        original_read = cache._read_entry_with_status

        def unreadable(path):
            if path == cache.path_for(current):
                return None, False, True
            return original_read(path)

        monkeypatch.setattr(cache, "_read_entry_with_status", unreadable)
        with pytest.raises(CacheDiscoveryError, match="refusing"):
            cache.get(current)

    def test_confirmed_legacy_misses_share_one_snapshot(
        self, tmp_path, monkeypatch
    ):
        cache = ResponseCache(tmp_path)
        scans = 0
        original_refresh = cache._refresh_legacy_v2_index

        def counted_refresh(provider):
            nonlocal scans
            scans += 1
            return original_refresh(provider)

        monkeypatch.setattr(cache, "_refresh_legacy_v2_index", counted_refresh)
        assert cache.get(key(user_prompt="first new prompt")) is None
        assert cache.get(key(user_prompt="second new prompt")) is None
        assert scans == 1

    def test_a_new_process_snapshots_entries_written_after_an_old_snapshot(
        self, tmp_path
    ):
        from dataclasses import replace
        cache = ResponseCache(tmp_path)
        current = key(
            sampling_parameters={"temperature": 0.0, "seed": 1001},
            completion_index=1,
            legacy_v2_completions=10,
            legacy_v2_generated_seed=True,
        )
        stored = replace(current, legacy_v2_completions=3)
        legacy_digest = stored.legacy_v2_digest(3)
        legacy_path = (
            cache.directory
            / stored.provider
            / legacy_digest[:2]
            / f"{legacy_digest}.json"
        )
        legacy_path.parent.mkdir(parents=True)
        (legacy_path.parent / "unrelated.json").write_text("[1]")

        assert cache.get(current) is None

        request = stored.request_fingerprint()
        request["sampling_parameters"] = stored._legacy_v2_sampling_parameters(3)
        legacy_path.write_text(json.dumps({
            "cache_format": 2,
            "cache_key": legacy_digest,
            "created_at": "2026-08-01T00:00:00+0000",
            "request": request,
            "derived": {
                "text": "FINAL: REFRESHED",
                "usage": {},
                "truncated": False,
                "reasoning": "",
            },
            "response": {
                "choices": [{"message": {"content": "FINAL: REFRESHED"}}]
            },
        }))
        # Format-2 writers must stop before a current evaluator starts. A new
        # cache instance represents that explicit process boundary.
        loaded = ResponseCache(tmp_path).get(current)
        assert loaded is not None and loaded.text == "FINAL: REFRESHED"

    @pytest.mark.parametrize("failure_stage", ["header", "body"])
    def test_transient_legacy_scan_error_is_retried(
        self, tmp_path, monkeypatch, failure_stage
    ):
        from dataclasses import replace

        cache = ResponseCache(tmp_path)
        current = key(
            sampling_parameters={"temperature": 0.0, "seed": 1001},
            completion_index=1,
            legacy_v2_completions=10,
            legacy_v2_generated_seed=True,
        )
        stored = replace(current, legacy_v2_completions=3)
        legacy_digest = stored.legacy_v2_digest(3)
        legacy_path = (
            cache.directory
            / stored.provider
            / legacy_digest[:2]
            / f"{legacy_digest}.json"
        )
        request = stored.request_fingerprint()
        request["sampling_parameters"] = stored._legacy_v2_sampling_parameters(3)
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(json.dumps({
            "cache_format": 2,
            "cache_key": legacy_digest,
            "created_at": "2026-08-01T00:00:00+0000",
            "request": request,
            "derived": {
                "text": "FINAL: RETRIED",
                "usage": {},
                "truncated": False,
                "reasoning": "",
            },
            "response": {
                "choices": [{"message": {"content": "FINAL: RETRIED"}}]
            },
        }))

        attempts = 0
        if failure_stage == "header":
            original_header = cache._looks_like_legacy_v2

            def flaky_header(path):
                nonlocal attempts
                if path == legacy_path and attempts < 2:
                    attempts += 1
                    return None
                return original_header(path)

            monkeypatch.setattr(cache, "_looks_like_legacy_v2", flaky_header)
        else:
            original_read = cache._read_entry_with_status

            def flaky_read(path):
                nonlocal attempts
                if path == legacy_path and attempts < 2:
                    attempts += 1
                    return None, False, True
                return original_read(path)

            monkeypatch.setattr(cache, "_read_entry_with_status", flaky_read)

        with pytest.raises(CacheDiscoveryError, match="refusing"):
            cache.get(current)
        assert attempts == 2
        loaded = cache.get(current)
        assert loaded is not None and loaded.text == "FINAL: RETRIED"

    def test_repeat_migration_does_not_reuse_a_different_seeded_request(
        self, tmp_path
    ):
        cache = ResponseCache(tmp_path)
        current = key(
            sampling_parameters={"temperature": 0.0, "seed": 1000},
            completion_index=0,
            legacy_v2_completions=3,
            legacy_v2_generated_seed=True,
        )
        legacy_digest = current.legacy_v2_digest(1)
        legacy_path = (
            cache.directory
            / current.provider
            / legacy_digest[:2]
            / f"{legacy_digest}.json"
        )
        request = current.request_fingerprint()
        request["sampling_parameters"] = {
            "temperature": 0.0,
            "completions": 1,
        }
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(json.dumps({
            "cache_format": 2,
            "cache_key": legacy_digest,
            "created_at": "2026-08-01T00:00:00+0000",
            "request": request,
            "derived": {
                "text": "FINAL: OLD",
                "usage": {},
                "truncated": False,
                "reasoning": "",
            },
            "response": {"choices": [{"message": {"content": "FINAL: OLD"}}]},
        }))

        assert cache.get(current) is None
        assert not cache.path_for(current).exists()

    def test_unseeded_format_2_entry_survives_repeat_count_change(self, tmp_path):
        cache = ResponseCache(tmp_path)
        current = key(
            provider="anthropic_messages",
            endpoint="https://api.anthropic.com",
            completion_index=1,
            legacy_v2_completions=10,
        )
        legacy_digest = current.legacy_v2_digest(3)
        legacy_path = (
            cache.directory
            / current.provider
            / legacy_digest[:2]
            / f"{legacy_digest}.json"
        )
        request = current.request_fingerprint()
        request["sampling_parameters"] = {
            **current.sampling_parameters,
            "completions": 3,
        }
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(json.dumps({
            "cache_format": 2,
            "cache_key": legacy_digest,
            "created_at": "2026-08-01T00:00:00+0000",
            "request": request,
            "derived": {
                "text": "FINAL: C",
                "usage": {},
                "truncated": False,
                "reasoning": "",
            },
            "response": {"content": [{"type": "text", "text": "FINAL: C"}]},
        }))

        loaded = cache.get(current)
        assert loaded is not None and loaded.text == "FINAL: C"
        promoted = json.loads(cache.path_for(current).read_text())
        assert promoted["provenance"]["migrated_from_cache_key"] == legacy_digest

    @pytest.mark.parametrize(
        "response",
        [
            {"error": {"message": "gateway failed"}},
            {
                "choices": [{
                    "finish_reason": "error",
                    "error": {"message": "generation failed"},
                }]
            },
        ],
    )
    def test_in_band_error_in_format_2_entry_is_not_promoted(
        self, tmp_path, response
    ):
        cache = ResponseCache(tmp_path)
        current = key()
        legacy_digest = current.legacy_v2_digest(1)
        legacy_path = (
            cache.directory
            / current.provider
            / legacy_digest[:2]
            / f"{legacy_digest}.json"
        )
        request = current.request_fingerprint()
        request["sampling_parameters"] = {
            **current.sampling_parameters,
            "completions": 1,
        }
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(json.dumps({
            "cache_format": 2,
            "cache_key": legacy_digest,
            "created_at": "2026-08-01T00:00:00+0000",
            "request": request,
            "derived": {
                "text": "FINAL: BAD",
                "usage": {},
                "truncated": False,
                "reasoning": "",
            },
            "response": response,
        }))

        assert cache.get(current) is None
        assert cache.misses == 1
        assert not cache.path_for(current).exists()

    def test_matching_corrupt_format_2_entry_fails_closed(self, tmp_path):
        cache = ResponseCache(tmp_path)
        current = key()
        legacy_digest = current.legacy_v2_digest(1)
        legacy_path = (
            cache.directory
            / current.provider
            / legacy_digest[:2]
            / f"{legacy_digest}.json"
        )
        request = current.request_fingerprint()
        request["sampling_parameters"] = {
            **current.sampling_parameters,
            "completions": 1,
        }
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(json.dumps({
            "cache_format": 2,
            "cache_key": legacy_digest,
            "created_at": "2026-08-01T00:00:00+0000",
            "request": request,
            "derived": {
                "text": "FINAL: A",
                "usage": [],
                "truncated": False,
                "reasoning": "",
            },
            "response": {"choices": [{"message": {"content": "FINAL: A"}}]},
        }))

        with pytest.raises(CacheDiscoveryError, match="matching legacy"):
            cache.get(current)
        assert not cache.path_for(current).exists()

    def test_directory_fsync_unsupported_operation_is_a_safe_fallback(
        self, tmp_path, monkeypatch
    ):
        import errno

        import pdbthink.evaluation.locking as locking

        def unsupported(descriptor):
            raise OSError(errno.EINVAL, "directory fsync unsupported")

        monkeypatch.setattr(locking.os, "fsync", unsupported)
        locking.fsync_directory(tmp_path)

    def test_durable_mkdir_syncs_each_new_parent_link(self, tmp_path, monkeypatch):
        import pdbthink.evaluation.locking as locking

        synced = []
        monkeypatch.setattr(locking, "fsync_directory", synced.append)
        target = tmp_path / "parent" / "state"
        locking.durable_mkdir(target)
        assert target.is_dir()
        assert synced == [tmp_path, tmp_path / "parent"]

    def test_first_request_lock_durably_creates_the_cache_root(
        self, tmp_path, monkeypatch
    ):
        import pdbthink.evaluation.locking as locking

        synced = []
        monkeypatch.setattr(locking, "fsync_directory", synced.append)
        cache = ResponseCache(tmp_path / "new-cache")

        with cache.request_lock(key()):
            cache.put(key(), CachedResponse(text="FINAL: A"))

        assert tmp_path in synced

    def test_cli_import_does_not_require_posix_fcntl(self):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['fcntl'] = None; import pdbthink.cli",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_malformed_json_at_the_exact_key_fails_closed(self, tmp_path):
        cache = ResponseCache(tmp_path)
        path = cache.path_for(key())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not json")
        with pytest.raises(CacheDiscoveryError, match="move or remove"):
            cache.get(key())

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


class TestOpenAIRequestBody:
    def test_configured_output_token_parameter_is_sent(self, monkeypatch):
        import pdbthink.evaluation.runner as runner

        captured = {}

        def fake_post(url, payload, headers, model):
            captured.update(payload)
            return {"choices": [{"message": {"content": "FINAL: A"}}]}

        monkeypatch.setattr(runner, "_post", fake_post)
        model = runner.ModelConfig(
            model_id="reasoning-model",
            output_token_parameter="max_completion_tokens",
            max_output_tokens=1234,
        )
        runner._openai_chat(model, "system", "user", 0)
        assert captured["max_completion_tokens"] == 1234
        assert "max_tokens" not in captured

    def test_repeated_completions_send_distinct_seeds(self, monkeypatch):
        import pdbthink.evaluation.runner as runner

        seeds = []

        def fake_post(url, payload, headers, model):
            seeds.append(payload.get("seed"))
            return {"choices": [{"message": {"content": "FINAL: A"}}]}

        monkeypatch.setattr(runner, "_post", fake_post)
        model = runner.ModelConfig(model_id="repeated", completions=3)
        runner._openai_chat(model, "system", "user", 0)
        runner._openai_chat(model, "system", "user", 1)
        assert seeds == [1000, 1001]

    def test_anthropic_repeats_do_not_claim_or_send_seeds(self, monkeypatch):
        import pdbthink.evaluation.runner as runner

        captured = {}

        def fake_post(url, payload, headers, model):
            captured.update(payload)
            return {"content": [{"type": "text", "text": "FINAL: A"}]}

        monkeypatch.setattr(runner, "_post", fake_post)
        model = runner.ModelConfig(
            model_id="claude-opus",
            provider="anthropic_messages",
            completions=3,
        )
        runner._anthropic(model, "system", "user", 1)
        assert "seed" not in captured
        assert "seed" not in model.sampling_parameters_for(1)

    def test_ollama_generated_and_top_level_seeds_keep_distinct_identity(self, monkeypatch):
        import pdbthink.evaluation.runner as runner

        captured = {}

        def fake_post(url, payload, headers, model):
            captured.update(payload)
            return {"message": {"content": "FINAL: A"}}

        monkeypatch.setattr(runner, "_post", fake_post)
        single = runner.ModelConfig(
            model_id="ollama",
            provider="ollama_chat",
            completions=1,
            extra_body={"seed": 42},
        )
        repeated = runner.ModelConfig(
            model_id="ollama",
            provider="ollama_chat",
            completions=3,
            extra_body={"seed": 42},
        )
        runner._ollama(repeated, "system", "user", 0)
        assert captured["seed"] == 42
        assert captured["options"]["seed"] == 1000
        assert single.sampling_parameters_for(0) != repeated.sampling_parameters_for(0)
        assert repeated.sampling_parameters_for(0) == {
            **repeated.sampling_parameters,
            "ollama_options_seed": 1000,
        }

    def test_an_in_band_gateway_error_is_not_an_answer(self, monkeypatch):
        import pdbthink.evaluation.runner as runner

        monkeypatch.setattr(runner, "_post", lambda *args: {
            "choices": [{
                "message": {"content": "FINAL: A"},
                "finish_reason": "error",
                "error": {"message": "upstream failed"},
            }]
        })
        with pytest.raises(runner.ProviderError, match="upstream failed"):
            runner._openai_chat(runner.ModelConfig(model_id="gateway"), "system", "user", 0)

    @pytest.mark.parametrize(
        "payload",
        [
            {"choices": [1]},
            {"choices": "not-a-list"},
            {"choices": [{"message": []}]},
            {"choices": [{"message": {"content": []}}]},
            {"choices": [{"message": {"content": "FINAL: A"}}], "usage": []},
            ["not-an-object"],
        ],
    )
    def test_malformed_openai_response_is_a_provider_error(self, monkeypatch, payload):
        import pdbthink.evaluation.runner as runner

        monkeypatch.setattr(runner, "_post", lambda *args: payload)
        with pytest.raises(runner.ProviderError, match="response"):
            runner._openai_chat(
                runner.ModelConfig(model_id="gateway"), "system", "user", 0
            )

    def test_native_openai_refusal_text_is_preserved(self, monkeypatch):
        import pdbthink.evaluation.runner as runner
        from pdbthink.scoring import looks_like_refusal

        monkeypatch.setattr(
            runner,
            "_post",
            lambda *args: {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "refusal": "I cannot provide that answer.",
                        }
                    }
                ]
            },
        )

        response = runner._openai_chat(
            runner.ModelConfig(model_id="gateway"), "system", "user", 0
        )

        assert response.text == "I cannot provide that answer."
        assert looks_like_refusal(response.text)

    @pytest.mark.parametrize(
        "payload",
        [
            {"error": {"message": "overloaded"}},
            ["not-an-object"],
            {"content": "not-a-list"},
            {"content": [1]},
            {"content": [], "usage": []},
            {"content": [{"type": "text", "text": 5}]},
            {"content": [{"type": "thinking", "thinking": []}]},
        ],
    )
    def test_malformed_anthropic_response_is_a_provider_error(
        self, monkeypatch, payload
    ):
        import pdbthink.evaluation.runner as runner

        monkeypatch.setattr(runner, "_post", lambda *args: payload)
        model = runner.ModelConfig(
            model_id="claude", provider="anthropic_messages"
        )

        with pytest.raises(runner.ProviderError, match="Anthropic"):
            runner._anthropic(model, "system", "user", 0)

    @pytest.mark.parametrize(
        "payload",
        [
            {"error": "model not found"},
            ["not-an-object"],
            {"message": []},
            {"message": {"content": 5}},
            {"message": {"content": "FINAL: A", "thinking": []}},
            {"message": {"content": "FINAL: A"}, "eval_count": []},
        ],
    )
    def test_malformed_ollama_response_is_a_provider_error(
        self, monkeypatch, payload
    ):
        import pdbthink.evaluation.runner as runner

        monkeypatch.setattr(runner, "_post", lambda *args: payload)
        model = runner.ModelConfig(model_id="ollama", provider="ollama_chat")

        with pytest.raises(runner.ProviderError, match="Ollama"):
            runner._ollama(model, "system", "user", 0)

    @pytest.mark.parametrize(
        ("stop_reason", "truncated", "refusal"),
        [
            ("refusal", False, True),
            ("model_context_window_exceeded", True, False),
            ("max_tokens", True, False),
        ],
    )
    def test_anthropic_terminal_stop_reasons_are_preserved(
        self, monkeypatch, stop_reason, truncated, refusal
    ):
        import pdbthink.evaluation.runner as runner

        monkeypatch.setattr(runner, "_post", lambda *args: {
            "content": [{"type": "text", "text": "FINAL: A:V22"}],
            "stop_reason": stop_reason,
        })
        response = runner._anthropic(
            runner.ModelConfig(model_id="claude", provider="anthropic_messages"),
            "system",
            "user",
            0,
        )
        assert response.truncated is truncated
        assert response.refusal is refusal

    def test_adaptive_anthropic_thinking_does_not_require_effort(self, monkeypatch):
        import pdbthink.evaluation.runner as runner

        captured = {}

        def fake_post(url, payload, headers, model):
            captured.update(payload)
            return {"content": [{"type": "text", "text": "FINAL: A"}]}

        monkeypatch.setattr(runner, "_post", fake_post)
        model = runner.ModelConfig(
            model_id="claude-opus",
            provider="anthropic_messages",
            thinking_mode="adaptive",
            temperature=None,
        )
        runner._anthropic(model, "system", "user", 0)
        assert captured["thinking"] == {"type": "adaptive"}
        assert "temperature" not in captured

    def test_anthropic_top_p_and_manual_effort_reach_the_wire(self, monkeypatch):
        import pdbthink.evaluation.runner as runner

        captured = {}

        def fake_post(url, payload, headers, model):
            captured.update(payload)
            return {"content": [{"type": "text", "text": "FINAL: A"}]}

        monkeypatch.setattr(runner, "_post", fake_post)
        model = runner.ModelConfig(
            model_id="claude-opus-4-5",
            provider="anthropic_messages",
            thinking_mode="manual",
            reasoning_effort="medium",
            top_p=0.95,
        )
        runner._anthropic(model, "system", "user", 0)
        assert captured["top_p"] == 0.95
        assert captured["thinking"]["type"] == "enabled"
        assert captured["output_config"] == {"effort": "medium"}

    def test_adaptive_anthropic_thinking_uses_effort_not_a_budget(self, monkeypatch):
        import pdbthink.evaluation.runner as runner

        captured = {}

        def fake_post(url, payload, headers, model):
            captured.update(payload)
            return {"content": [{"type": "text", "text": "FINAL: A"}]}

        monkeypatch.setattr(runner, "_post", fake_post)
        model = runner.ModelConfig(
            model_id="claude-opus-5",
            provider="anthropic_messages",
            reasoning_effort="max",
            thinking_mode="adaptive",
            temperature=None,
        )
        runner._anthropic(model, "system", "user", 0)
        assert captured["thinking"] == {"type": "adaptive"}
        assert captured["output_config"] == {"effort": "max"}
        assert "temperature" not in captured
        assert "budget_tokens" not in json.dumps(captured)

    def test_cache_discovery_error_never_reaches_the_provider(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        import pdbthink.evaluation.runner as runner

        cache = ResponseCache(tmp_path / "cache")
        monkeypatch.setattr(
            cache,
            "get",
            lambda cache_key: (_ for _ in ()).throw(
                CacheDiscoveryError("cache state unknown")
            ),
        )
        monkeypatch.setattr(
            runner,
            "call_model",
            lambda *args, **kwargs: pytest.fail("provider was called"),
        )
        evaluation = runner.EvaluationRunner(
            tmp_path / "dataset",
            runner.ModelConfig(model_id="paid"),
            tmp_path / "run",
            cache=cache,
        )
        evaluation.run_id = "test-run"
        render = SimpleNamespace(
            render_id="r1",
            semantic_instance_id="i1",
            question_family="P01",
            protein_group_id="p",
            representation="minimal_pdb",
            input_token_count=10,
            system_prompt="system",
            user_prompt="user",
        )

        result = evaluation._one(render, 0)

        assert result.error == "CacheDiscoveryError: cache state unknown"
        assert not result.from_cache

    def test_batch_cache_owner_rejects_another_state_directory(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        first = tmp_path / "batch-one"
        second = tmp_path / "batch-two"
        assert cache.claim_batch(first) is True
        assert cache.claim_batch(first) is False

        with pytest.raises(CacheDiscoveryError, match="finish it"):
            cache.claim_batch(second)
        with pytest.raises(CacheDiscoveryError, match="outstanding batch"):
            with cache.synchronous_run_guard():
                pass

        cache.release_batch(first)
        with cache.synchronous_run_guard():
            pass

    def test_identical_concurrent_prompts_make_one_paid_call(self, monkeypatch, tmp_path):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        import pdbthink.evaluation.runner as runner

        entered = threading.Event()
        release = threading.Event()
        calls = 0

        def fake_call(*args, **kwargs):
            nonlocal calls
            calls += 1
            entered.set()
            assert release.wait(timeout=2)
            return CachedResponse(text="FINAL: A")

        monkeypatch.setattr(runner, "call_model", fake_call)
        cache = ResponseCache(tmp_path / "cache")
        evaluation = runner.EvaluationRunner(
            tmp_path / "dataset",
            runner.ModelConfig(model_id="paid", concurrency=2),
            tmp_path / "run",
            cache=cache,
        )
        evaluation.run_id = "test-run"

        def render(number):
            return SimpleNamespace(
                render_id=f"r{number}",
                semantic_instance_id=f"i{number}",
                question_family="P01",
                protein_group_id="p",
                representation="minimal_pdb",
                input_token_count=10,
                system_prompt="same system",
                user_prompt="same user",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(evaluation._one, render(1), 0)
            assert entered.wait(timeout=2)
            second = pool.submit(evaluation._one, render(2), 0)
            release.set()
            results = [first.result(), second.result()]

        assert calls == 1
        assert not any(result.error for result in results)
        assert sorted(result.from_cache for result in results) == [False, True]


class TestRetryPolicy:
    def test_checked_in_paid_api_configs_do_not_automatically_resubmit(self):
        from pathlib import Path

        from pdbthink.evaluation.runner import ModelConfig

        paths = [
            Path("configs/models/openai_gpt.yaml"),
            *Path("configs/models").glob("anthropic_*.yaml"),
            *Path("configs/models").glob("together_*.yaml"),
        ]
        assert paths
        assert all(ModelConfig.load(path).max_retries == 1 for path in paths)

    def test_retry_after_is_respected(self, monkeypatch):
        import io
        import urllib.error

        import pdbthink.evaluation.runner as runner

        calls = 0
        delays = []

        class Reply:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices": [{"message": {"content": "ok"}}]}'

        def fake_urlopen(request, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise urllib.error.HTTPError(
                    request.full_url,
                    429,
                    "rate limited",
                    {"Retry-After": "7"},
                    io.BytesIO(b"busy"),
                )
            return Reply()

        monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(runner.time, "sleep", delays.append)
        model = runner.ModelConfig(model_id="retry", max_retries=2)
        runner._post("https://example.test", {}, {}, model)
        assert calls == 2
        assert delays == [7.0]

    @pytest.mark.parametrize("status", [402, 413])
    def test_terminal_payment_and_size_errors_are_not_retried(self, monkeypatch, status):
        import io
        import urllib.error

        import pdbthink.evaluation.runner as runner

        calls = 0

        def fake_urlopen(request, timeout):
            nonlocal calls
            calls += 1
            raise urllib.error.HTTPError(
                request.full_url, status, "terminal", {}, io.BytesIO(b"terminal")
            )

        monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)
        model = runner.ModelConfig(model_id="terminal", max_retries=4)
        with pytest.raises(runner.ProviderError, match=f"HTTP {status}"):
            runner._post("https://example.test", {}, {}, model)
        assert calls == 1


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
            base_url="https://api.together.ai/v1", api_key_env="NOPE",
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

    def test_pending_holds_the_request_lock_while_discovering_cache(
        self, pieces, tmp_path, monkeypatch
    ):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        run = BatchRun(model, cache, tmp_path / "state", client=FakeBatchClient([]))
        held = False
        locked = []

        class Lock:
            def __init__(self, digest):
                self.digest = digest

            def __enter__(self):
                nonlocal held
                assert not held
                held = True
                locked.append(self.digest)

            def __exit__(self, *args):
                nonlocal held
                held = False

        monkeypatch.setattr(cache, "request_lock", lambda cache_key: Lock(cache_key.digest))

        def get(cache_key):
            assert held
            return None

        monkeypatch.setattr(cache, "get", get)
        assert len(run.pending(renders)) == 3
        assert locked == [run.key_for(render, 0).digest for render in renders]

    def test_another_batch_state_is_rejected_before_upload(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        first = BatchRun(
            model, cache, tmp_path / "state-one", client=FakeBatchClient([])
        )
        first.submit(renders[:1])
        second_client = FakeBatchClient([])
        second = BatchRun(
            model, cache, tmp_path / "state-two", client=second_client
        )

        with pytest.raises(CacheDiscoveryError, match="finish it"):
            second.submit(renders[1:2])
        assert second_client.uploaded == []
        assert second_client.created == []

    def test_submit_preflights_only_when_uncached_work_exists(
        self, pieces, tmp_path, monkeypatch
    ):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        run = BatchRun(model, cache, tmp_path / "state", client=FakeBatchClient([]))
        for render in renders:
            cache.put(run.key_for(render, 0), CachedResponse(text="already answered"))
        monkeypatch.setattr(
            run,
            "preflight",
            lambda: pytest.fail("no-op submit contacted the provider"),
        )

        assert run.submit(renders, preflight=True) == []
        with cache.synchronous_run_guard():
            pass

    def test_cache_discovery_precedes_batch_preflight(
        self, pieces, tmp_path, monkeypatch
    ):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        run = BatchRun(model, cache, tmp_path / "state", client=FakeBatchClient([]))

        def unreadable(cache_key):
            raise CacheDiscoveryError("cache state unknown")

        monkeypatch.setattr(cache, "get", unreadable)
        monkeypatch.setattr(
            run,
            "preflight",
            lambda: pytest.fail("cache error still contacted the provider"),
        )
        with pytest.raises(CacheDiscoveryError, match="cache state unknown"):
            run.submit(renders, preflight=True)
        with cache.synchronous_run_guard():
            pass

    def test_new_batch_work_runs_one_preflight(self, pieces, tmp_path, monkeypatch):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        run = BatchRun(model, cache, tmp_path / "state", client=FakeBatchClient([]))
        calls = []
        monkeypatch.setattr(run, "preflight", lambda: calls.append("preflight"))

        jobs = run.submit(renders[:1], preflight=True)

        assert len(jobs) == 1
        assert calls == ["preflight"]

    def test_identical_prompts_are_submitted_once(self, pieces, tmp_path):
        from types import SimpleNamespace

        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        duplicate = SimpleNamespace(
            **{
                **vars(renders[0]),
                "render_id": "duplicate-render",
                "semantic_instance_id": "duplicate-instance",
            }
        )
        run = BatchRun(model, cache, tmp_path / "state", client=FakeBatchClient([]))
        jobs = run.submit([renders[0], duplicate])
        assert jobs[0].n_requests == 1

    def test_together_batch_jsonl_uses_the_native_chat_schema(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        run.submit(renders[:1])

        row = json.loads(client.uploaded[0])
        assert set(row) == {"custom_id", "body"}
        assert row["custom_id"] == "r00-00000"
        assert row["body"] == run.request_body(renders[0], 0)

    def test_a_finished_batch_lands_in_the_cache(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        jobs = run.submit(renders)
        assert len(jobs) == 1 and jobs[0].n_requests == 3
        with pytest.raises(CacheDiscoveryError, match="outstanding batch"):
            with cache.synchronous_run_guard():
                pass

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
        with cache.synchronous_run_guard():
            pass
        for render in renders:
            entry = cache.get(run.key_for(render, 0))
            assert entry is not None and entry.text == "FINAL: A"
            assert entry.reasoning == "thinking"

    def test_narrowed_split_stage_fetch_keeps_all_paid_results_replayable(
        self, pieces, tmp_path
    ):
        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        jobs = run.submit(renders[:2])
        client.output_lines = [
            {
                "custom_id": custom_id,
                "response": {"body": {
                    "choices": [{
                        "message": {"content": "FINAL: A"},
                        "finish_reason": "stop",
                    }]
                }},
            }
            for custom_id in jobs[0].custom_ids
        ]
        run.poll()

        with pytest.raises(BatchError, match="fetch selection omits 1 request"):
            run.fetch(renders[:1])
        assert cache.get(run.key_for(renders[0], 0)) is None
        with pytest.raises(CacheDiscoveryError, match="outstanding batch"):
            with cache.synchronous_run_guard():
                pass

        assert run.fetch(renders[:2]) == {"stored": 2, "failed": 0, "unknown": 0}
        with cache.synchronous_run_guard():
            pass
        assert all(cache.get(run.key_for(render, 0)) for render in renders[:2])

    def test_batch_fetch_preserves_the_first_valid_cached_response(
        self, pieces, tmp_path
    ):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        jobs = run.submit(renders[:1])
        key = run.key_for(renders[0], 0)
        cache.put(key, CachedResponse(text="FINAL: FIRST"))
        client.output_lines = [
            {
                "custom_id": custom_id,
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {"content": "FINAL: SECOND"},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                },
            }
            for custom_id in jobs[0].custom_ids
        ]
        run.poll()

        assert run.fetch(renders[:1]) == {"stored": 0, "failed": 0, "unknown": 0}
        assert cache.get(key).text == "FINAL: FIRST"

    def test_a_batch_submitted_with_v1_keys_lands_in_the_current_cache(self, pieces, tmp_path):
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
        old_job = job.as_dict()
        old_job.pop("custom_id_cache_format")
        run.state_path.write_text(json.dumps({
            "model_id": model.model_id,
            "provider": model.provider,
            "max_output_tokens": model.max_output_tokens,
            "jobs": [old_job],
        }))
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

    def test_legacy_state_is_bound_to_the_first_guarded_cache(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, renders, _ = pieces
        first_cache = ResponseCache(tmp_path / "cache-one")
        state_dir = tmp_path / "state"
        client = FakeBatchClient([])
        first = BatchRun(model, first_cache, state_dir, client=client)
        first.submit(renders[:1])
        state = json.loads(first.state_path.read_text())
        state.pop("batch_state_format")
        state.pop("request_identity")
        state.pop("custom_id_cache_format")
        for job in state["jobs"]:
            job.pop("custom_id_cache_format")
        first.state_path.write_text(json.dumps(state))

        first.poll()
        migrated = json.loads(first.state_path.read_text())
        assert migrated["cache_directory"] == str(first_cache.directory.resolve())

        second_cache = ResponseCache(tmp_path / "cache-two")
        second = BatchRun(model, second_cache, state_dir, client=client)
        with pytest.raises(BatchError, match="original --cache-dir"):
            second.fetch(renders[:1])
        with pytest.raises(CacheDiscoveryError, match="outstanding batch"):
            with first_cache.synchronous_run_guard():
                pass
        with second_cache.synchronous_run_guard():
            pass

    def test_cache_format_2_state_is_not_resubmitted(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        jobs = run.submit(renders)
        job = jobs[0]
        by_current = {
            run.key_for(render, 0).digest: run.key_for(render, 0).legacy_v2_digest(1)
            for render in renders
        }
        for custom_id, digest in list(job.custom_ids.items()):
            job.custom_ids[custom_id] = by_current[digest]
        old_sampling = {**model.sampling_parameters, "completions": 1}
        old_job = job.as_dict()
        old_job.pop("custom_id_cache_format")
        run.state_path.write_text(json.dumps({
            "custom_id_cache_format": 2,
            "request_identity": {
                "provider": model.provider,
                "endpoint": model.endpoint_identity,
                "model_id": model.model_id,
                "model_revision": model.model_revision,
                "sampling_parameters": old_sampling,
            },
            "jobs": [old_job],
        }))

        assert len(run.submit(renders)) == 1
        assert len(client.created) == 1

        from types import SimpleNamespace

        new_render = SimpleNamespace(
            **{
                **vars(renders[0]),
                "render_id": "new-render",
                "semantic_instance_id": "new-instance",
                "user_prompt": "new question",
            }
        )
        jobs = run.submit([*renders, new_render])
        assert [batch_job.custom_id_cache_format for batch_job in jobs] == [2, 3]
        jobs[0].output_file_id = None
        jobs[1].output_file_id = "file-out"
        run._save_jobs(jobs)
        client.output_lines = [
            {
                "custom_id": custom_id,
                "response": {"body": {
                    "choices": [{
                        "message": {"content": "FINAL: A"},
                        "finish_reason": "stop",
                    }]
                }},
            }
            for custom_id in jobs[1].custom_ids
        ]
        assert run.fetch([*renders, new_render]) == {
            "stored": 1,
            "failed": 0,
            "unknown": 0,
        }
        assert cache.get(run.key_for(new_render, 0)) is not None

    def test_completed_format_2_state_migrates_only_with_its_matching_cache(
        self, pieces, tmp_path
    ):
        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, renders, cache = pieces
        state_dir = tmp_path / "state"
        client = FakeBatchClient([])
        run = BatchRun(model, cache, state_dir, client=client)
        run.submit(renders[:1])
        cache.put(
            run.key_for(renders[0], 0),
            CachedResponse(
                text="FINAL: A",
                raw={"choices": [{"message": {"content": "FINAL: A"}}]},
            ),
        )
        state = json.loads(run.state_path.read_text())
        state["batch_state_format"] = 2
        state["request_identity"].pop("cache_directory")
        state["jobs"][0]["status"] = "completed"
        state["jobs"][0]["fetch_complete"] = True
        run.state_path.write_text(json.dumps(state))
        cache.release_batch(state_dir)

        wrong_cache = ResponseCache(tmp_path / "wrong-cache")
        wrong = BatchRun(model, wrong_cache, state_dir, client=client)
        with pytest.raises(BatchError, match="cannot be matched"):
            wrong.poll()
        with wrong_cache.synchronous_run_guard():
            pass

        jobs = run.poll()
        assert jobs[0].fetch_complete is True
        migrated = json.loads(run.state_path.read_text())
        assert migrated["batch_state_format"] == 3
        assert migrated["request_identity"]["cache_directory"] == str(
            cache.directory.resolve()
        )

    def test_split_stage_state_is_bound_to_its_original_cache(
        self, pieces, tmp_path
    ):
        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, renders, _ = pieces
        first_cache = ResponseCache(tmp_path / "cache-one")
        state_dir = tmp_path / "state"
        first = BatchRun(
            model, first_cache, state_dir, client=FakeBatchClient([])
        )
        first.submit(renders[:1])
        state = json.loads(first.state_path.read_text())
        assert state["request_identity"]["cache_directory"] == str(
            first_cache.directory.resolve()
        )

        second_cache = ResponseCache(tmp_path / "cache-two")
        second = BatchRun(
            model, second_cache, state_dir, client=FakeBatchClient([])
        )
        with pytest.raises(BatchError, match="separate --state-dir"):
            second.poll()
        with pytest.raises(CacheDiscoveryError, match="outstanding batch"):
            with first_cache.synchronous_run_guard():
                pass
        with second_cache.synchronous_run_guard():
            pass

    def test_missing_state_cannot_release_an_existing_batch_marker(
        self, pieces, tmp_path
    ):
        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, _, cache = pieces
        state_dir = tmp_path / "lost-state"
        cache.claim_batch(state_dir)
        run = BatchRun(model, cache, state_dir, client=FakeBatchClient([]))

        with pytest.raises(BatchError, match="state .* is missing"):
            run.poll()
        with pytest.raises(CacheDiscoveryError, match="outstanding batch"):
            with cache.synchronous_run_guard():
                pass

    def test_malformed_fetch_state_cannot_release_batch_ownership(
        self, pieces, tmp_path
    ):
        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, renders, cache = pieces
        run = BatchRun(model, cache, tmp_path / "state", client=FakeBatchClient([]))
        run.submit(renders[:1])
        state = json.loads(run.state_path.read_text())
        state["jobs"][0]["fetch_complete"] = "false"
        run.state_path.write_text(json.dumps(state))

        with pytest.raises(BatchError, match="fetch_complete must be boolean"):
            run.poll()
        with pytest.raises(CacheDiscoveryError, match="outstanding batch"):
            with cache.synchronous_run_guard():
                pass

    def test_state_dir_rejects_another_model_configuration(self, pieces, tmp_path):
        from dataclasses import replace

        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, renders, cache = pieces
        state_dir = tmp_path / "state"
        BatchRun(model, cache, state_dir, client=FakeBatchClient([])).submit(renders)
        other = replace(model, temperature=0.7)
        with pytest.raises(BatchError, match="separate --state-dir"):
            BatchRun(other, cache, state_dir, client=FakeBatchClient([])).poll()

    def test_batch_extra_requires_the_v2_together_sdk(self):
        from pathlib import Path

        project = Path("pyproject.toml").read_text()
        assert 'batch = ["together>=2.0.0"]' in project

    def test_together_upload_disables_fine_tuning_validation(self, tmp_path):
        from types import SimpleNamespace

        from pdbthink.evaluation.batch import TogetherBatchClient

        captured = {}

        class Files:
            def upload(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(id="file-1")

        client = TogetherBatchClient.__new__(TogetherBatchClient)
        client._client = SimpleNamespace(files=Files())
        batch_input = tmp_path / "batch.jsonl"
        batch_input.write_text('{"custom_id": "r00-00000"}\n')

        assert client.upload(batch_input) == "file-1"
        assert captured == {
            "file": batch_input,
            "purpose": "batch-api",
            "check": False,
        }

    def test_batch_command_accepts_the_legacy_together_endpoint(
        self, pieces, tmp_path
    ):
        from dataclasses import replace

        from pdbthink.evaluation.batch import BatchRun

        model, _, cache = pieces
        legacy = replace(model, base_url="https://api.together.xyz/v1")
        BatchRun(legacy, cache, tmp_path / "state", client=FakeBatchClient([]))

    def test_batch_command_rejects_a_non_together_endpoint(self, pieces, tmp_path):
        from dataclasses import replace

        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, _, cache = pieces
        other = replace(model, base_url="https://openrouter.ai/api/v1")
        with pytest.raises(BatchError, match="Together's Batch API"):
            BatchRun(other, cache, tmp_path / "state", client=FakeBatchClient([]))

    def test_batch_command_rejects_a_non_openai_provider(self, pieces, tmp_path):
        from dataclasses import replace

        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, _, cache = pieces
        other = replace(model, provider="mock")
        with pytest.raises(BatchError, match="provider: openai_chat"):
            BatchRun(other, cache, tmp_path / "state", client=FakeBatchClient([]))

    def test_batch_repeats_use_distinct_seeds(self, pieces, tmp_path):
        from dataclasses import replace

        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        repeated = replace(model, completions=3)
        run = BatchRun(repeated, cache, tmp_path / "state", client=FakeBatchClient([]))
        assert run.request_body(renders[0], 0)["seed"] == 1000
        assert run.request_body(renders[0], 1)["seed"] == 1001

    def test_in_band_batch_errors_are_not_cached(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        jobs = run.submit(renders)
        client.output_lines = [
            {
                "custom_id": custom_id,
                "response": {"body": {"choices": [{
                    "message": {"content": "FINAL: A"},
                    "finish_reason": "error",
                    "error": {"message": "upstream failed"},
                }]}},
            }
            for custom_id in jobs[0].custom_ids
        ]
        run.poll()
        assert run.fetch(renders) == {"stored": 0, "failed": 3, "unknown": 0}
        assert not list(cache.entries())

    def test_malformed_batch_choice_is_counted_as_failed(self, pieces, tmp_path):
        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        jobs = run.submit(renders[:1])
        client.output_lines = [
            {
                "custom_id": custom_id,
                "response": {"body": {"choices": [1]}},
            }
            for custom_id in jobs[0].custom_ids
        ]
        run.poll()

        assert run.fetch(renders[:1]) == {"stored": 0, "failed": 1, "unknown": 0}
        assert not list(cache.entries())

    def test_a_later_chunk_failure_preserves_created_batch_ids(
        self, pieces, tmp_path, monkeypatch
    ):
        import pdbthink.evaluation.batch as batch

        model, renders, cache = pieces

        class FailingClient(FakeBatchClient):
            def __init__(self):
                super().__init__([])
                self.calls = 0
                self.fail_second = True

            def create(self, input_file_id, **kwargs):
                self.calls += 1
                if self.calls == 2 and self.fail_second:
                    raise RuntimeError("provider failed during chunk 2")
                self.created.append(input_file_id)
                return {"id": f"batch-{self.calls}", "status": "VALIDATING"}

        monkeypatch.setattr(batch, "MAX_REQUESTS_PER_BATCH", 1)
        client = FailingClient()
        run = batch.BatchRun(model, cache, tmp_path / "state", client=client)
        with pytest.raises(RuntimeError, match="chunk 2"):
            run.submit(renders)
        state = json.loads(run.state_path.read_text())
        assert [job["batch_id"] for job in state["jobs"]] == ["batch-1", ""]
        assert state["jobs"][1]["status"] == "creating"

        client.fail_second = False
        with pytest.raises(batch.BatchError, match="confirm-ambiguous-resubmit"):
            run.submit(renders)
        jobs = run.submit(renders, confirm_ambiguous_resubmit=True)
        assert len(jobs) == 3
        assert jobs[0].batch_id == "batch-1"
        assert all(job.batch_id for job in jobs)

    def test_an_accepted_ambiguous_batch_can_be_attached(self, pieces, tmp_path):
        from types import SimpleNamespace

        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces

        class AcceptedThenLostClient(FakeBatchClient):
            def __init__(self):
                super().__init__([])
                self.create_calls = 0

            def create(self, input_file_id, **kwargs):
                self.create_calls += 1
                raise RuntimeError("response lost after provider acceptance")

            def retrieve(self, batch_id):
                return {
                    "id": batch_id,
                    "input_file_id": "file-1",
                    "status": "VALIDATING",
                }

        client = AcceptedThenLostClient()
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        with pytest.raises(RuntimeError, match="response lost"):
            run.submit(renders)

        new_render = SimpleNamespace(
            **{
                **vars(renders[0]),
                "render_id": "new-render",
                "semantic_instance_id": "new-instance",
                "user_prompt": "new question",
            }
        )
        jobs = run.submit(
            [*renders, new_render],
            recover_ambiguous_batch_id="batch-found-in-provider-account",
        )
        assert client.create_calls == 1
        assert len(jobs) == 1
        assert jobs[0].batch_id == "batch-found-in-provider-account"
        assert jobs[0].status == "validating"
        assert len(run.pending([new_render])) == 1

    @pytest.mark.parametrize("provider_input", [None, "another-file"])
    def test_ambiguous_recovery_requires_matching_input_identity(
        self, pieces, tmp_path, provider_input
    ):
        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, renders, cache = pieces

        class UnverifiableClient(FakeBatchClient):
            def create(self, input_file_id, **kwargs):
                raise RuntimeError("response lost")

            def retrieve(self, batch_id):
                payload = {"id": batch_id, "status": "VALIDATING"}
                if provider_input is not None:
                    payload["input_file_id"] = provider_input
                return payload

        run = BatchRun(
            model,
            cache,
            tmp_path / "state",
            client=UnverifiableClient([]),
        )
        with pytest.raises(RuntimeError, match="response lost"):
            run.submit(renders)
        with pytest.raises(BatchError, match="expected reserved input file"):
            run.submit(renders, recover_ambiguous_batch_id="wrong-or-unverifiable")

    def test_reservation_directory_is_synced_before_create(
        self, pieces, tmp_path, monkeypatch
    ):
        import pdbthink.evaluation.batch as batch

        model, renders, cache = pieces
        syncs = []
        monkeypatch.setattr(batch, "fsync_directory", lambda path: syncs.append(path))

        class InspectingClient(FakeBatchClient):
            def create(self, input_file_id, **kwargs):
                assert syncs == [tmp_path / "state"]
                return super().create(input_file_id, **kwargs)

        run = batch.BatchRun(
            model,
            cache,
            tmp_path / "state",
            client=InspectingClient([]),
        )
        run.submit(renders)
        assert syncs == [tmp_path / "state", tmp_path / "state"]

    def test_concurrent_submitters_share_one_state_lock(self, pieces, tmp_path):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        entered = threading.Event()
        release = threading.Event()

        class SlowClient(FakeBatchClient):
            def create(self, input_file_id, **kwargs):
                self.created.append(input_file_id)
                entered.set()
                assert release.wait(timeout=2)
                return {"id": "batch-1", "status": "VALIDATING"}

        client = SlowClient([])
        state_dir = tmp_path / "state"
        first_run = BatchRun(model, cache, state_dir, client=client)
        second_run = BatchRun(model, cache, state_dir, client=client)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(first_run.submit, renders)
            assert entered.wait(timeout=2)
            second = pool.submit(second_run.submit, renders)
            release.set()
            first.result()
            second.result()
        assert len(client.created) == 1

    def test_completed_state_accepts_only_new_prompts(self, pieces, tmp_path):
        from types import SimpleNamespace

        from pdbthink.evaluation.batch import BatchRun

        model, renders, cache = pieces
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        first_jobs = run.submit(renders)
        assert len(first_jobs) == 1
        for render in renders:
            cache.put(run.key_for(render, 0), CachedResponse(text="FINAL: A"))

        new_render = SimpleNamespace(
            **{
                **vars(renders[0]),
                "render_id": "P01-new::minimal_pdb::1",
                "semantic_instance_id": "P01-new",
                "user_prompt": "a newly added question",
            }
        )
        jobs = run.submit([*renders, new_render])
        assert len(jobs) == 2
        assert jobs[-1].n_requests == 1
        assert len(client.created) == 2

    def test_legacy_multi_completion_batch_is_rejected(self, pieces, tmp_path):
        from dataclasses import replace

        from pdbthink.evaluation.batch import BatchError, BatchRun

        model, renders, cache = pieces
        model = replace(model, completions=3)
        client = FakeBatchClient([])
        run = BatchRun(model, cache, tmp_path / "state", client=client)
        job = run.submit(renders)[0]
        legacy_by_current = {
            run.key_for(render, index).digest: run.key_for(render, index).legacy_v1_digest
            for render in renders
            for index in range(model.completions)
        }
        for custom_id, digest in list(job.custom_ids.items()):
            job.custom_ids[custom_id] = legacy_by_current[digest]
        run.state_path.write_text(json.dumps({
            "model_id": model.model_id,
            "provider": model.provider,
            "max_output_tokens": model.max_output_tokens,
            "jobs": [job.as_dict()],
        }))

        with pytest.raises(BatchError, match="were not seeded"):
            run.poll()

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
