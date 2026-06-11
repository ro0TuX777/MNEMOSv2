"""Unit tests for Phase 9b: depth-2 recursion and summary isolation sentinel."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.engram.model import Engram
from mnemos.governance.hygiene.clustering_runner import HierarchicalClusteringRunner


class _CapturingIndexer:
    def __init__(self):
        self.indexed: List[Engram] = []

    def index(self, engrams: List[Engram]) -> int:
        self.indexed.extend(engrams)
        return len(engrams)


def _depth1_summary(i: int) -> Engram:
    return Engram(
        id=f"summary_cluster_{i:03d}_abc{i}",
        content=f"Thematic summary {i} covering governance topic {i}.",
        source=f"derived://hierarchy/cluster_{i:03d}",
        edges=[f"leaf_{i}_a", f"leaf_{i}_b"],
        metadata={"is_summary_engram": True, "depth": 1, "cluster_id": f"cluster_{i:03d}"},
    )


# ── depth-2 recursion ─────────────────────────────────────────────────────


def test_depth2_builds_single_root_with_recursive_lineage(monkeypatch):
    monkeypatch.delenv("SMC_LLM_BASE_URL", raising=False)  # force deterministic fallback
    summaries = [_depth1_summary(i) for i in range(4)]
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(4, 8)).astype(np.float32)

    indexer = _CapturingIndexer()
    report = HierarchicalClusteringRunner(n_clusters=4).run(
        summaries, vectors=vectors, dry_run=False, indexer=indexer, depth=2
    )

    assert report.cluster_count == 1  # depth=2 forces a single root cluster
    assert report.engrams_scanned == 4
    assert len(indexer.indexed) == 1

    root = indexer.indexed[0]
    assert root.id.startswith("summary_root_")
    assert root.source == "derived://hierarchy/root"
    assert root.metadata["depth"] == 2
    assert root.metadata["is_summary_engram"] is True
    assert sorted(root.edges) == sorted(s.id for s in summaries)  # recursive lineage


def test_depth2_excludes_leaves_and_prior_roots(monkeypatch):
    """Mixed input: only depth-1 summaries reach the depth-2 cluster."""
    monkeypatch.delenv("SMC_LLM_BASE_URL", raising=False)
    leaf = Engram(id="leaf_x", content="raw leaf")
    old_root = Engram(
        id="summary_root_old",
        content="old root",
        metadata={"is_summary_engram": True, "depth": 2},
    )
    summaries = [_depth1_summary(i) for i in range(3)]
    # Vectors align with the *selected* engrams (the 3 depth-1 summaries);
    # passing the full mixed list with explicit vectors is a caller error,
    # so the runner is fed the mixed list without vectors only when it can
    # embed. Here we verify selection by scan count using aligned input.
    rng = np.random.default_rng(5)

    indexer = _CapturingIndexer()
    report = HierarchicalClusteringRunner().run(
        summaries,
        vectors=rng.normal(size=(3, 8)).astype(np.float32),
        dry_run=False,
        indexer=indexer,
        depth=2,
    )
    assert report.engrams_scanned == 3

    # And a mixed list is rejected when vectors don't align with selection.
    with pytest.raises(ValueError, match="align"):
        HierarchicalClusteringRunner().run(
            [leaf, old_root, *summaries],
            vectors=rng.normal(size=(5, 8)).astype(np.float32),
            dry_run=True,
            depth=2,
        )


def test_depth_validation():
    with pytest.raises(ValueError, match="depth"):
        HierarchicalClusteringRunner().run([], depth=3)


# ── isolation sentinel ────────────────────────────────────────────────────


def test_qdrant_filter_sentinel_builds_summary_must_not():
    pytest.importorskip("qdrant_client")
    from mnemos.retrieval.qdrant_tier import QdrantTier

    flt = QdrantTier._build_filter({"__exclude_summaries__": True})
    assert flt is not None and flt.must_not is not None
    assert flt.must_not[0].key == "app_is_summary_engram"

    both = QdrantTier._build_filter({"__exclude_summaries__": True, "__exclude_derived__": True})
    keys = {c.key for c in both.must_not}
    assert keys == {"app_is_summary_engram", "app_is_derived_fact"}


def test_router_default_path_injects_summary_exclusion():
    from mnemos.retrieval.base import BaseRetriever, SearchResult
    from mnemos.retrieval.fusion import TierFusion
    from mnemos.retrieval.retrieval_router import RetrievalRouter

    class Recorder(BaseRetriever):
        def __init__(self):
            self.last_filters = None

        @property
        def tier_name(self):
            return "qdrant"

        def index(self, engrams):
            return len(engrams)

        def search(self, query, top_k=10, filters=None, query_vector=None):
            self.last_filters = dict(filters or {})
            return [SearchResult(engram=Engram(id="a", content="a"), score=1.0, tier="qdrant")]

        def delete(self, engram_ids):
            return 0

        def stats(self):
            return {}

    tier = Recorder()
    router = RetrievalRouter(semantic_fusion=TierFusion([tier]), lexical_tier=None)
    router.search(query="plain factual lookup", top_k=3)

    assert tier.last_filters is not None
    assert tier.last_filters.get("__exclude_summaries__") is True
    assert tier.last_filters.get("__exclude_derived__") is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
