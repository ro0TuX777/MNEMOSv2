"""Runtime tests for governance explainability traces."""

from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

from mnemos.engram.model import Engram
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.retrieval.base import SearchResult
from service.app import MnemosRuntime


class _StubRouter:
    def __init__(self, hits):
        self._hits = hits

    def search(self, **kwargs):
        return self._hits, {
            "retrieval_mode": kwargs.get("retrieval_mode", "semantic"),
            "fusion_policy": kwargs.get("fusion_policy"),
            "lexical_available": True,
        }


class _StubGovernor:
    def __init__(self, decisions, ordered_results):
        self._decisions = decisions
        self._ordered_results = ordered_results

    def has_policy_profile(self, profile_id: str) -> bool:
        return profile_id == "default"

    def govern(self, **kwargs):
        return self._ordered_results, self._decisions, []


def _mk_runtime(router, governor):
    rt = MnemosRuntime()
    rt._config = SimpleNamespace(
        retrieval_mode="semantic",
        fusion_policy="balanced",
        explain_default=False,
        lexical_top_k=25,
        semantic_top_k=25,
        has_compression=False,
        quant_bits=4,
        memory_over_maps_phase1=False,
        memory_over_maps_phase2=False,
        memory_over_maps_phase3=False,
        memory_over_maps_phase4=False,
        governance_mode="advisory",
    )
    rt._router = router
    rt._governor = governor
    rt._semantic_fusion = None
    rt._lexical_tier = None
    rt._ledger = None
    rt._status = "healthy"
    rt._error = None
    return rt


def test_search_explain_governance_includes_trace_and_suppressed_summary():
    r1 = SearchResult(engram=Engram(id="doc-1", content="first"), score=0.8, tier="qdrant")
    r2 = SearchResult(engram=Engram(id="doc-2", content="second"), score=0.9, tier="qdrant")
    router = _StubRouter([r1, r2])

    d1 = GovernanceDecision(
        engram_id="doc-1",
        retrieval_score=0.8,
        governed_score=0.0,
        veto_pass=False,
        veto_modifier=0.0,
        veto_reason="score 0.8 below threshold 0.9",
        suppressed=True,
        suppressed_reason="score veto",
        would_be_suppressed_in_enforced_mode=True,
    )
    d2 = GovernanceDecision(
        engram_id="doc-2",
        retrieval_score=0.9,
        governed_score=1.02,
        trust_modifier=1.1,
        utility_modifier=1.2,
        freshness_modifier=1.0,
        contradiction_modifier=1.0,
        veto_modifier=1.0,
        veto_pass=True,
        conflict_status="none",
        suppressed=False,
        would_be_suppressed_in_enforced_mode=False,
    )
    # Reordered output to exercise rank_shift.
    governor = _StubGovernor([d1, d2], [r2, r1])
    rt = _mk_runtime(router, governor)

    out = rt.search_documents(
        query="q",
        top_k=2,
        tiers=None,
        filters=None,
        retrieval_mode="semantic",
        fusion_policy="balanced",
        explain=False,
        governance="advisory",
        explain_governance=True,
    )

    rows = {row["engram"]["id"]: row for row in out["results"]}
    assert "governance_trace" in rows["doc-2"]
    assert rows["doc-2"]["governance_trace"]["outcome"] == "retained"
    assert rows["doc-2"]["governance_trace"]["raw_rank"] == 2
    assert rows["doc-2"]["governance_trace"]["final_rank"] == 1
    assert rows["doc-2"]["governance_trace"]["rank_shift"] == 1
    assert rows["doc-2"]["governance_trace"]["top_factors"][0]["name"] in {"utility", "trust"}

    assert "governance_trace" in rows["doc-1"]
    assert rows["doc-1"]["governance_trace"]["outcome"] == "vetoed"
    assert "threshold" in rows["doc-1"]["governance_trace"]["reason"]

    suppressed = out["meta"]["governance_explain"]["suppressed_candidates"]
    assert len(suppressed) == 1
    assert suppressed[0]["engram_id"] == "doc-1"
    assert suppressed[0]["vetoed"] is True
