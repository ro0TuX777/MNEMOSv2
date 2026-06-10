"""Unit tests for the Matryoshka migration tool's pure logic (no qdrant/model deps)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "mnemos_matryoshka_migrate", PROJECT_ROOT / "tools" / "mnemos_matryoshka_migrate.py"
)
assert _spec is not None and _spec.loader is not None
mig = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mig  # dataclasses resolve string annotations via sys.modules
_spec.loader.exec_module(mig)


# ---- jaccard ------------------------------------------------------------


def test_jaccard_basic():
    assert mig.jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert mig.jaccard(["a", "b"], ["c", "d"]) == 0.0
    assert mig.jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5)
    assert mig.jaccard([], []) == 1.0


def test_rank_displacement():
    assert mig.rank_displacement(["a", "b", "c"], ["b", "a", "z"]) == pytest.approx(1.0)
    assert mig.rank_displacement(["a"], ["z"]) is None


def test_mean_top_score_and_score_drop_guard():
    bge = [("a", 0.91), ("b", 0.89), ("c", 0.80), ("d", 0.1)]
    nomic = [("x", 0.70), ("y", 0.67), ("z", 0.65)]
    assert mig.mean_top_score(bge) == pytest.approx((0.91 + 0.89 + 0.80) / 3)
    assert mig.nomic_score_significantly_lower(mig.mean_top_score(bge), mig.mean_top_score(nomic)) is True
    assert mig.nomic_score_significantly_lower(0.80, 0.76) is False
    assert mig.nomic_score_significantly_lower(None, 0.50) is False


# ---- MRL slice ----------------------------------------------------------


def test_mrl_slice_shape_and_normalization():
    rng = np.random.default_rng(7)
    full = rng.normal(size=(5, 768)).astype(np.float32)
    sliced = mig.mrl_slice(full, 64)
    assert sliced.shape == (5, 64)
    np.testing.assert_allclose(np.linalg.norm(sliced, axis=1), 1.0, rtol=1e-5)


def test_mrl_slice_applies_layer_norm_not_plain_truncation():
    rng = np.random.default_rng(7)
    full = rng.normal(size=(3, 768)).astype(np.float32) + 10.0  # offset mean
    sliced = mig.mrl_slice(full, 64)
    plain = full[:, :64] / np.linalg.norm(full[:, :64], axis=1, keepdims=True)
    assert not np.allclose(sliced, plain, atol=1e-3)


# ---- promotion verdict matrix -------------------------------------------


def _row(cls, sim, regression=False):
    return {"query_class": cls, "jaccard_at_10": sim, "labeled_regression": regression}


def test_verdict_proceed_when_median_high_and_no_regression():
    verdict = mig.replay_verdict([_row("semantic", 0.8), _row("semantic", 0.7)])
    assert verdict["overall"] == "PROCEED"


def test_verdict_review_on_low_median_without_regression():
    verdict = mig.replay_verdict([_row("semantic", 0.4), _row("semantic", 0.5)])
    assert verdict["overall"] == "REVIEW"
    assert verdict["per_class"]["semantic"]["verdict"] == "REVIEW"


def test_verdict_block_is_absolute_even_with_high_similarity():
    verdict = mig.replay_verdict([
        _row("semantic", 0.9),
        _row("acronym", 0.9, regression=True),
    ])
    assert verdict["overall"] == "BLOCK"
    assert verdict["per_class"]["acronym"]["verdict"] == "BLOCK"
    assert verdict["per_class"]["semantic"]["verdict"] == "PROCEED"


# ---- MigrationState -----------------------------------------------------


def test_migration_state_roundtrip(tmp_path):
    state = mig.MigrationState(
        migration_id="m1",
        source_collection="src",
        target_collection="dst",
        batch_size=500,
        total_engrams=1000,
    )
    state.failed_ids.append("bad-1")
    state.record_batch(500, elapsed_s=2.0, next_offset="point-501", processed_ids=["e1", "e2"])
    path = tmp_path / "migration_checkpoint.json"
    state.save(path)

    loaded = mig.MigrationState.load(path)
    assert loaded.processed_count == 500
    assert loaded.processed_ids == ["e1", "e2"]
    assert loaded.next_offset == "point-501"
    assert loaded.failed_ids == ["bad-1"]
    assert loaded.throughput_samples == [250.0]
    assert loaded.current_model_version == mig.DEFAULT_TARGET_MODEL
    assert loaded.updated_at  # set by save()


def test_migration_state_save_is_atomic(tmp_path):
    path = tmp_path / "migration_checkpoint.json"
    state = mig.MigrationState(migration_id="m1")
    state.save(path)
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()  # tmp file replaced, not left behind


def test_eta_estimation():
    state = mig.MigrationState(total_engrams=10_000)
    state.record_batch(500, elapsed_s=5.0, next_offset=None)  # 100/sec
    eta = state.eta_seconds()
    assert eta == pytest.approx((10_000 - 500) / 100.0)

    empty = mig.MigrationState()
    assert empty.eta_seconds() is None


def test_throughput_samples_bounded_to_last_20():
    state = mig.MigrationState(total_engrams=100_000)
    for _ in range(30):
        state.record_batch(100, elapsed_s=1.0, next_offset="x")
    assert len(state.throughput_samples) == 20


def test_migration_state_load_backfills_new_fields(tmp_path):
    path = tmp_path / "migration_checkpoint.json"
    path.write_text('{"migration_id": "old", "target_model": "new-model"}', encoding="utf-8")
    loaded = mig.MigrationState.load(path)
    assert loaded.current_model_version == "new-model"
    assert loaded.processed_ids == []


def test_result_rows_uses_payload_mnemos_id_and_score():
    class Point:
        def __init__(self, pid, score, payload):
            self.id = pid
            self.score = score
            self.payload = payload

    class Response:
        points = [
            Point("uuid-1", 0.9, {"_mnemos_id": "engram-1"}),
            Point("uuid-2", None, {}),
        ]

    assert mig._result_rows(Response()) == [("engram-1", 0.9), ("uuid-2", None)]


def test_phase1_dry_run_samples_without_writes():
    class Count:
        count = 2

    class Point:
        def __init__(self, content):
            self.payload = {"content": content}

    class Client:
        def __init__(self):
            self.upserts = 0
            self.calls = 0

        def count(self, collection_name, exact):
            return Count()

        def scroll(self, **kwargs):
            self.calls += 1
            if self.calls > 1:
                return [], None
            return [Point("alpha"), Point("beta")], None

        def upsert(self, **kwargs):
            self.upserts += 1

    class Embedder:
        def embed_documents(self, texts):
            assert texts == ["alpha", "beta"]
            return np.ones((2, 768), dtype=np.float32)

    client = Client()
    state = mig.MigrationState(source_collection="src", target_collection="dst", batch_size=10)
    result = mig.phase1_dry_run(client, state, Embedder(), dry_run_limit=2)

    assert result["sampled_engrams"] == 2
    assert result["writes_performed"] == 0
    assert client.upserts == 0


# ---- prefix sentinel check ----------------------------------------------


class _FakeEmbedder:
    """Embedder double: honors_prefixes controls whether prefix changes output."""

    def __init__(self, honors_prefixes: bool):
        self._honors = honors_prefixes

    def embed_documents(self, texts):
        seed = 1 if self._honors else 0
        rng = np.random.default_rng(seed)
        return rng.normal(size=(len(texts), 8))

    def embed_raw(self, texts):
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), 8))


def test_prefix_sentinel_check_passes_when_prefixes_honored():
    assert mig.prefix_sentinel_check(_FakeEmbedder(honors_prefixes=True)) is True


def test_prefix_sentinel_check_fails_when_prefixes_ignored():
    assert mig.prefix_sentinel_check(_FakeEmbedder(honors_prefixes=False)) is False


# ---- anchor truthset shape ----------------------------------------------


def test_anchor_truthset_loads_and_has_required_fields():
    import json

    data = json.loads((PROJECT_ROOT / "benchmarks" / "truthsets" / "anchor_queries_v1.json").read_text(encoding="utf-8"))
    assert data["anchors"], "anchor list must not be empty"
    classes = set()
    for anchor in data["anchors"]:
        assert anchor["query_id"] and anchor["query"]
        assert anchor["class"] in {"semantic", "exact_term", "acronym", "long_context"}
        classes.add(anchor["class"])
    assert {"semantic", "acronym", "exact_term", "long_context"} <= classes


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
