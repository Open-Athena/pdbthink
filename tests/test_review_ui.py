"""Curator review interface: structure extraction, highlights and access control."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from pdbthink.review_ui.server import Handler, ReviewState, _extract_structures, _highlights


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    """A one-instance dataset built from the committed fixtures."""
    import yaml

    from pdbthink.config import DatasetConfig, Definitions
    from pdbthink.dataset import DatasetBuilder, write_dataset
    from tests.test_pipeline import FIXTURE_CONFIG, FixtureCache

    root = tmp_path_factory.mktemp("review")
    config_path = root / "dataset.yaml"
    config = dict(FIXTURE_CONFIG, family_targets={"P02": 1, "S08": 1})
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded = DatasetConfig.load(config_path)
    result = DatasetBuilder(loaded, Definitions.load(), FixtureCache(root / "cache")).build()
    out = root / "dataset"
    write_dataset(result, out)
    return out


def serve_in_thread(state, token):
    handler = type("Bound", (Handler,), {"state": state, "auth_token": token})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def status_of(url: str, headers: dict[str, str] | None = None, method: str = "GET") -> int:
    request = urllib.request.Request(url, headers=headers or {}, method=method)
    if method == "POST":
        request.data = b"{}"
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


class TestContent:
    def test_pdb_blocks_are_extracted_for_the_viewer(self, dataset_dir):
        state = ReviewState(dataset_dir, dataset_dir / "decisions.jsonl")
        render = next(
            r for r in state.renders if r.representation == "minimal_pdb"
        )
        structures = _extract_structures(render.user_prompt)
        assert structures
        assert structures[0]["pdb"].startswith(("ATOM", "HETATM"))

    def test_highlights_cover_query_gold_and_evidence(self, dataset_dir):
        state = ReviewState(dataset_dir, dataset_dir / "decisions.jsonl")
        instance = next(i for i in state.instances if i.question_family == "S08")
        roles = {h["role"] for h in _highlights(instance)}
        assert "gold" in roles
        assert all(isinstance(h["resi"], int) for h in _highlights(instance))


class TestAccessControl:
    def test_open_when_no_token_is_configured(self, dataset_dir):
        state = ReviewState(dataset_dir, dataset_dir / "decisions.jsonl")
        server, base = serve_in_thread(state, None)
        try:
            assert status_of(f"{base}/api/instances") == 200
        finally:
            server.shutdown()

    def test_token_is_required_for_reads_and_writes(self, dataset_dir):
        state = ReviewState(dataset_dir, dataset_dir / "decisions.jsonl")
        server, base = serve_in_thread(state, "secret")
        try:
            assert status_of(f"{base}/api/instances") == 401
            assert status_of(f"{base}/api/instances?token=wrong") == 401
            assert status_of(f"{base}/api/decision", method="POST") == 401
            assert status_of(f"{base}/api/instances?token=secret") == 200
            assert (
                status_of(f"{base}/api/instances", {"Authorization": "Bearer secret"}) == 200
            )
        finally:
            server.shutdown()

    def test_decisions_are_recorded_and_reloaded(self, dataset_dir, tmp_path):
        decisions = tmp_path / "decisions.jsonl"
        state = ReviewState(dataset_dir, decisions)
        instance_id = state.instances[0].semantic_instance_id
        state.record(
            {"semantic_instance_id": instance_id, "decision": "accept", "curator": "tester"}
        )
        rows = [json.loads(line) for line in decisions.read_text().splitlines()]
        assert rows[0]["semantic_instance_id"] == instance_id
        assert rows[0]["timestamp"]
        assert ReviewState(dataset_dir, decisions).decisions[instance_id]["decision"] == "accept"

    def test_a_rejection_needs_a_reason(self, dataset_dir, tmp_path):
        state = ReviewState(dataset_dir, tmp_path / "d.jsonl")
        instance_id = state.instances[0].semantic_instance_id
        with pytest.raises(ValueError):
            state.record(
                {
                    "semantic_instance_id": instance_id,
                    "decision": "reject",
                    "label_override": {"value": "A:V1"},
                }
            )
