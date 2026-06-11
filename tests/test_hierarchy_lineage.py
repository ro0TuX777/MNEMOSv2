"""
Phase 9 lineage verification (integration).

A summary engram is only useful if its provenance is auditable: a CLASS_C
query must surface a summary engram whose ``edges`` resolve to retrievable
leaf engrams via the public API.

Requires the live stack (Qdrant :6333 + service :8700) with the Phase 9
hierarchy applied; skips cleanly otherwise so CI without the stack stays
green.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

requests = pytest.importorskip("requests")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("MNEMOS_BASE_URL", "http://localhost:8700")
QDRANT_URL = os.getenv("MNEMOS_QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("MNEMOS_QDRANT_COLLECTION", "mnemos_engrams_nomic_mrl")
TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "query_complexity_v1.json"


def _stack_ready() -> bool:
    try:
        if requests.get(f"{BASE_URL}/health", timeout=3).status_code != 200:
            return False
        info = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=3).json()
        return info.get("result", {}).get("points_count", 0) > 100
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _stack_ready(),
    reason="live MNEMOS stack with seeded corpus not available",
)


def _summary_search_qdrant(query_text: str, top_k: int = 5):
    """Summary-layer search via the tier (service search uses the same path)."""
    from mnemos.retrieval.qdrant_tier import QdrantTier

    tier = QdrantTier(
        url=QDRANT_URL,
        collection_name=COLLECTION,
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_dim=768,
        gpu_device=os.getenv("MNEMOS_GPU_DEVICE", "cpu"),
    )
    return tier.search(query_text, top_k=top_k, filters={"metadata.is_summary_engram": True})


@pytest.fixture(scope="module")
def class_c_query() -> str:
    data = json.loads(TRUTHSET.read_text(encoding="utf-8"))
    rows = [q for q in data["queries"] if q["label"] == "CLASS_C"]
    assert rows, "truthset must contain CLASS_C queries"
    return rows[0]["query"]


def test_class_c_top_result_is_summary_engram(class_c_query):
    hits = _summary_search_qdrant(class_c_query)
    assert hits, "summary-layer search returned no results — hierarchy not applied?"
    top = hits[0]
    assert top.engram.metadata.get("is_summary_engram") is True
    assert top.engram.metadata.get("depth") == 1
    assert top.engram.edges, "summary engram must carry lineage edges"


def test_summary_edges_resolve_to_leaf_engrams_via_api(class_c_query):
    hits = _summary_search_qdrant(class_c_query)
    assert hits
    edges = list(hits[0].engram.edges)
    assert len(edges) >= 3, f"expected >=3 lineage edges, got {len(edges)}"

    resolved = 0
    for edge_id in edges:
        if resolved >= 3:
            break
        resp = requests.get(f"{BASE_URL}/v1/mnemos/engrams/{edge_id}", timeout=10)
        if resp.status_code != 200:
            continue
        engram = resp.json().get("engram") or {}
        if engram.get("content"):
            assert engram.get("metadata", {}).get("is_summary_engram") is not True
            resolved += 1

    assert resolved >= 3, (
        f"only {resolved} of {len(edges)} lineage edges resolved to raw leaf "
        "content via GET /v1/mnemos/engrams/{id}"
    )


def test_root_summary_recursive_lineage():
    """Phase 9b Target 3: grandparent -> parent -> child resolution.

    Root (depth 2) edges resolve to depth-1 summaries via the live API,
    whose own edges resolve to raw leaves.
    """
    from mnemos.retrieval.qdrant_tier import QdrantTier

    tier = QdrantTier(
        url=QDRANT_URL,
        collection_name=COLLECTION,
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_dim=768,
        gpu_device=os.getenv("MNEMOS_GPU_DEVICE", "cpu"),
    )
    hits = tier.search(
        "overall corpus posture summary",
        top_k=3,
        filters={"metadata.is_summary_engram": True, "metadata.depth": 2},
    )
    assert hits, "no depth-2 root summary found — run the dry-run tool with --depth 2 --apply"
    root = hits[0].engram
    assert root.metadata.get("depth") == 2
    assert root.source == "derived://hierarchy/root"
    edges = list(root.edges)
    assert len(edges) >= 2, "root must reference the depth-1 summaries"

    # Grandparent -> parent: root edges resolve to depth-1 summaries.
    parents = []
    for eid in edges:
        if len(parents) >= 3:
            break
        resp = requests.get(f"{BASE_URL}/v1/mnemos/engrams/{eid}", timeout=10)
        if resp.status_code != 200:
            continue
        engram = resp.json().get("engram") or {}
        meta = engram.get("metadata") or {}
        if meta.get("is_summary_engram") and meta.get("depth") == 1:
            parents.append(engram)
    assert len(parents) >= 3, f"only {len(parents)} root edges resolved to depth-1 summaries"

    # Parent -> child: a depth-1 summary's edges resolve to a raw leaf.
    leaf_resolved = False
    for leaf_id in (parents[0].get("edges") or [])[:5]:
        resp = requests.get(f"{BASE_URL}/v1/mnemos/engrams/{leaf_id}", timeout=10)
        if resp.status_code != 200:
            continue
        leaf = resp.json().get("engram") or {}
        if leaf.get("content") and not (leaf.get("metadata") or {}).get("is_summary_engram"):
            leaf_resolved = True
            break
    assert leaf_resolved, "depth-1 summary edges did not resolve to raw leaf content"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
