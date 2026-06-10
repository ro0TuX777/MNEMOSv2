"""Unit tests for the PIT-1 leakage guard hardening and derived-fact filters."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult
from mnemos.retrieval.retrieval_router import RetrievalRouter


def _result(eid: str, derived: bool = False) -> SearchResult:
    metadata = {"is_derived_fact": True} if derived else {}
    return SearchResult(engram=Engram(id=eid, content=f"content {eid}", metadata=metadata), score=0.9)


def _stub_router(leakage_mode: str = "strip", tier_names=None):
    return SimpleNamespace(
        _config=SimpleNamespace(pit_leakage_mode=leakage_mode),
        _semantic_fusion=SimpleNamespace(tier_names=tier_names if tier_names is not None else ["qdrant"]),
        _is_derived=RetrievalRouter._is_derived,
    )


def test_is_derived_detects_metadata_flag_and_authority_type():
    assert RetrievalRouter._is_derived(_result("a", derived=True)) is True
    assert RetrievalRouter._is_derived(_result("b")) is False

    governed = _result("c")
    governed.engram.governance = SimpleNamespace(authority_type="MNEMOS_DERIVED_FACT")
    assert RetrievalRouter._is_derived(governed) is True


def test_strip_mode_removes_leaked_derived_facts():
    stub = _stub_router("strip")
    results = [_result("a"), _result("leak", derived=True), _result("b")]
    meta = {}

    cleaned = RetrievalRouter._apply_leakage_guard(stub, results, meta)

    assert [r.engram.id for r in cleaned] == ["a", "b"]
    assert meta["query.default_retrieval.derived_fact_count"] == 1
    assert meta["query.default_retrieval.derived_facts_stripped"] == 1


def test_canary_mode_returns_leaked_results_unmodified():
    stub = _stub_router("canary")
    results = [_result("a"), _result("leak", derived=True)]
    meta = {}

    returned = RetrievalRouter._apply_leakage_guard(stub, results, meta)

    assert [r.engram.id for r in returned] == ["a", "leak"]
    assert meta["query.default_retrieval.derived_fact_count"] == 1
    assert "query.default_retrieval.derived_facts_stripped" not in meta


def test_clean_results_pass_through_untouched():
    stub = _stub_router("strip")
    results = [_result("a"), _result("b")]
    meta = {}

    cleaned = RetrievalRouter._apply_leakage_guard(stub, results, meta)

    assert cleaned is results
    assert meta["query.default_retrieval.derived_fact_count"] == 0


def test_overfetch_factor_skipped_for_server_side_filter_tiers():
    assert RetrievalRouter._derived_overfetch_factor(_stub_router(tier_names=["qdrant"])) == 1
    assert RetrievalRouter._derived_overfetch_factor(_stub_router(tier_names=["pgvector"])) == 1
    assert RetrievalRouter._derived_overfetch_factor(_stub_router(tier_names=["qdrant", "pgvector"])) == 1
    assert RetrievalRouter._derived_overfetch_factor(_stub_router(tier_names=["qdrant", "lancedb"])) == 2
    assert RetrievalRouter._derived_overfetch_factor(_stub_router(tier_names=[])) == 2


def test_qdrant_filter_sentinel_builds_must_not():
    pytest.importorskip("qdrant_client")
    from mnemos.retrieval.qdrant_tier import QdrantTier

    flt = QdrantTier._build_filter({"__exclude_derived__": True})
    assert flt is not None
    assert flt.must_not is not None and len(flt.must_not) == 1
    assert flt.must_not[0].key == "app_is_derived_fact"
    assert flt.must is None

    # sentinel must not leak into must conditions alongside real filters
    flt = QdrantTier._build_filter({"__exclude_derived__": True, "source": "s3://x"})
    assert flt.must is not None and all(c.key != "__exclude_derived__" for c in flt.must)
    assert flt.must_not is not None

    # no sentinel -> behavior unchanged
    flt = QdrantTier._build_filter({"source": "s3://x"})
    assert flt.must_not is None


def test_pit_leakage_mode_env_validation(monkeypatch):
    from mnemos.config import MnemosConfig

    monkeypatch.setenv("MNEMOS_PIT_LEAKAGE_MODE", "canary")
    assert MnemosConfig.from_env().pit_leakage_mode == "canary"

    monkeypatch.setenv("MNEMOS_PIT_LEAKAGE_MODE", "yolo")
    assert MnemosConfig.from_env().pit_leakage_mode == "strip"

    monkeypatch.delenv("MNEMOS_PIT_LEAKAGE_MODE", raising=False)
    assert MnemosConfig.from_env().pit_leakage_mode == "strip"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
