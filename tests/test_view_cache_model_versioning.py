"""Cache-key embedding-model versioning (model-isolated derived-view lanes)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.memory_over_maps.view_cache import build_cache_key

def _key(embedding_model_name: str = "unversioned") -> str:
    return build_cache_key(
        view_type="evidence",
        query_fingerprint_value="qfp",
        artifact_ids=["a1"],
        chunk_ids=["c1"],
        governance_state_hash_value="g",
        synthesis_policy_version="default",
        embedding_model_name=embedding_model_name,
    )


def test_cache_key_isolated_per_embedding_model():
    assert _key("BAAI/bge-base-en-v1.5") != _key("nomic-embed-text-v1.5")


def test_cache_key_deterministic_within_model_lane():
    assert _key("BAAI/bge-base-en-v1.5") == _key("BAAI/bge-base-en-v1.5")


def test_cache_key_default_lane_is_stable():
    # Callers that do not pass a model land in the "unversioned" lane —
    # explicitly distinct from any named model lane.
    assert _key() != _key("BAAI/bge-base-en-v1.5")


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
