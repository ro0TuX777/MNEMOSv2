"""Phase 1 Memory Over Maps schema and lineage tests."""

from types import SimpleNamespace

import pytest

from mnemos.audit.forensic_ledger import ForensicLedger
from mnemos.memory_over_maps.models import Chunk, DerivedView, SourceArtifact
from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult

pytest.importorskip("flask")

from service.app import MnemosRuntime


class _StubRouter:
    def __init__(self, hits):
        self._hits = hits

    def search(self, **kwargs):
        return self._hits, {"retrieval_mode": "semantic", "lexical_available": False}


def _mk_runtime_with_flag(flag_value: bool, hits):
    rt = MnemosRuntime()
    rt._config = SimpleNamespace(
        retrieval_mode="semantic",
        fusion_policy="balanced",
        explain_default=False,
        lexical_top_k=25,
        semantic_top_k=25,
        memory_over_maps_phase1=flag_value,
        governance_mode="off",
    )
    rt._router = _StubRouter(hits)
    rt._semantic_fusion = None
    rt._lexical_tier = None
    rt._ledger = None
    rt._governor = None
    rt._status = "healthy"
    rt._error = None
    return rt


def test_source_artifact_schema_round_trip():
    artifact = SourceArtifact(
        artifact_id="art-1",
        artifact_type="markdown",
        source_uri="repo://docs/whitepaper.md",
        content_hash="abc123",
    )
    out = SourceArtifact.from_dict(artifact.to_dict())
    assert out.artifact_id == "art-1"
    assert out.version_id == "v1"


def test_chunk_schema_requires_artifact_linkage():
    chunk = Chunk(
        chunk_id="chunk-1",
        artifact_id="art-1",
        chunk_index=0,
    )
    out = Chunk.from_dict(chunk.to_dict())
    assert out.artifact_id == "art-1"
    assert out.chunk_id == "chunk-1"


def test_derived_view_requires_declared_inputs():
    view = DerivedView(view_type="evidence_bundle", inputs={"artifact_ids": ["art-1"]})
    out = DerivedView.from_dict(view.to_dict())
    assert out.view_type == "evidence_bundle"
    assert out.inputs["artifact_ids"] == ["art-1"]


def test_search_explain_includes_lineage_when_phase1_enabled():
    hit = SearchResult(engram=Engram(id="eng-1", content="hello"), score=0.9, tier="qdrant")
    rt = _mk_runtime_with_flag(True, [hit])
    out = rt.search_documents(
        query="hello",
        top_k=1,
        tiers=None,
        filters=None,
        retrieval_mode="semantic",
        fusion_policy="balanced",
        explain=True,
    )
    row = out["results"][0]["engram"]
    assert "_lineage" in row
    assert row["_lineage"]["chunk_id"] == "eng-1"


def test_search_explain_omits_lineage_when_phase1_disabled():
    hit = SearchResult(engram=Engram(id="eng-1", content="hello"), score=0.9, tier="qdrant")
    rt = _mk_runtime_with_flag(False, [hit])
    out = rt.search_documents(
        query="hello",
        top_k=1,
        tiers=None,
        filters=None,
        retrieval_mode="semantic",
        fusion_policy="balanced",
        explain=True,
    )
    row = out["results"][0]["engram"]
    assert "_lineage" not in row


def test_forensic_ledger_logs_derived_view_generation(tmp_path):
    db_path = str(tmp_path / "audit.db")
    ledger = ForensicLedger(db_path=db_path)
    tx_id = ledger.log_derived_view_generation(
        view_type="evidence_bundle",
        view_id="view-1",
        inputs={"artifact_ids": ["art-1"], "chunk_ids": ["chunk-1"]},
        query_fingerprint="qf-123",
        governance_state_hash="gov-abc",
    )
    assert tx_id > 0
    recent = ledger.get_recent_transactions(limit=1)
    assert recent[0]["action"] == "derived_view_generation"
    assert recent[0]["component"] == "memory-over-maps"
