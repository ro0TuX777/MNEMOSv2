"""
Tests for the MNEMOS Attention Contract builder.

Coverage targets
----------------
1. Pre-cognitive cache hit short-circuits to a single attention decision.
2. Semantic mode produces correct retrieval_mode decision.
3. Hybrid mode captures fusion policy and engine.
4. Candidate envelope applied/not_requested/disabled branches.
5. Summary inclusion eligible/blocked branches.
6. Graph expansion is always blocked (double opt-in not on HTTP surface).
7. Derived facts eligible/not_requested/disabled branches.
8. Governance gate off/advisory/enforced branches.
9. Forecast advisory present/absent branches.
10. Shadow search / intent forecast annotations.
11. Rerank applied / skipped annotations.
12. All returned objects are AttentionDecision instances with non-empty fields.
"""

from __future__ import annotations

from typing import Any, Dict, List

from mnemos.cognitive.attention import build_attention_decisions
from mnemos.cognitive.cycle import AttentionDecision


# ── Helpers ───────────────────────────────────────────────────────────────────


def _dims(decisions: List[AttentionDecision]) -> Dict[str, str]:
    """Map dimension → decision for assertion convenience."""
    return {d.dimension: d.decision for d in decisions}


def _reasons(decisions: List[AttentionDecision]) -> Dict[str, str]:
    return {d.dimension: d.reason for d in decisions}


def _base_mode_meta(
    *,
    mode: str = "semantic",
    policy: str = "balanced",
    engine: str = "",
    envelope_enabled: bool = False,
    lexical_available: bool = True,
    forecast_plan: str = "",
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "retrieval_mode": mode,
        "fusion_policy": policy if mode == "hybrid" else None,
        "lexical_available": lexical_available,
        "candidate_envelope": {
            "enabled": envelope_enabled,
            "initial_candidate_count": 30 if envelope_enabled else 0,
            "final_candidate_count": 15 if envelope_enabled else 0,
            "suppression_summary": {
                "duplicate_similarity": 3 if envelope_enabled else 0,
                "source_cap_exceeded": 2 if envelope_enabled else 0,
                "bounded_limit_exceeded": 10 if envelope_enabled else 0,
            },
        },
        "forecast_advisory": (
            {
                "suggested_plan": forecast_plan,
                "forecast_pressure": 0.75,
                "confidence_score": 0.80,
                "forecast_reason": "predicted p95 spike",
                "actions_mode": "advisory",
            }
            if forecast_plan
            else {}
        ),
    }
    if engine:
        meta["fusion_engine"] = engine
    return meta


# ── Pre-cognitive cache hit ───────────────────────────────────────────────────


class TestPreCognitiveHit:
    def test_short_circuits_to_one_decision(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="advisory",
            governance_profile="default",
            pre_cognitive_hit=True,
        )
        assert len(decisions) == 1
        assert decisions[0].dimension == "pre_cognitive_routing"
        assert decisions[0].decision == "cache_hit"

    def test_no_other_dimensions_when_cache_hit(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(mode="hybrid", policy="balanced"),
            governance_mode="enforced",
            governance_profile="strict",
            phase1_enabled=True,
            phase3_enabled=True,
            derive_views_requested=["summary"],
            pre_cognitive_hit=True,
        )
        dims = {d.dimension for d in decisions}
        assert dims == {"pre_cognitive_routing"}


# ── Query classification ──────────────────────────────────────────────────────


class TestQueryClassification:
    def test_family_present(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            query_family="factoid",
            query_family_confidence=0.91,
        )
        d = _dims(decisions)
        assert d["query_classification"] == "factoid"
        assert "0.91" in _reasons(decisions)["query_classification"]

    def test_family_absent_no_classification_decision(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            query_family=None,
        )
        dims = {d.dimension for d in decisions}
        assert "query_classification" not in dims


# ── Retrieval mode ────────────────────────────────────────────────────────────


class TestRetrievalMode:
    def test_semantic_lexical_unavailable(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(mode="semantic", lexical_available=False),
            governance_mode="off",
            governance_profile="default",
        )
        d = _dims(decisions)
        assert d["retrieval_mode"] == "semantic"
        assert "unavailable" in _reasons(decisions)["retrieval_mode"].lower()

    def test_semantic_lexical_available(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(mode="semantic", lexical_available=True),
            governance_mode="off",
            governance_profile="default",
        )
        d = _dims(decisions)
        assert d["retrieval_mode"] == "semantic"
        assert "semantic-only" in _reasons(decisions)["retrieval_mode"].lower()

    def test_hybrid_python_rrf(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(mode="hybrid", policy="balanced"),
            governance_mode="off",
            governance_profile="default",
        )
        d = _dims(decisions)
        assert "hybrid:balanced" in d["retrieval_mode"]

    def test_hybrid_qdrant_rrf(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(mode="hybrid", policy="qdrant_rrf", engine="qdrant_rrf"),
            governance_mode="off",
            governance_profile="default",
        )
        d = _dims(decisions)
        assert "hybrid:" in d["retrieval_mode"]
        r = _reasons(decisions)["retrieval_mode"]
        assert "qdrant_rrf" in r


# ── Candidate envelope ────────────────────────────────────────────────────────


class TestCandidateEnvelope:
    def test_applied(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(envelope_enabled=True),
            governance_mode="off",
            governance_profile="default",
            phase2_enabled=True,
        )
        d = _dims(decisions)
        assert d["candidate_envelope"] == "applied"
        r = _reasons(decisions)["candidate_envelope"]
        assert "30" in r   # initial count
        assert "15" in r   # final count

    def test_not_requested(self):
        """Phase2 enabled but bounded_envelope not sent."""
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(envelope_enabled=False),
            governance_mode="off",
            governance_profile="default",
            phase2_enabled=True,
        )
        d = _dims(decisions)
        assert d["candidate_envelope"] == "not_requested"

    def test_disabled(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(envelope_enabled=False),
            governance_mode="off",
            governance_profile="default",
            phase2_enabled=False,
        )
        d = _dims(decisions)
        assert d["candidate_envelope"] == "disabled"


# ── Reranking ─────────────────────────────────────────────────────────────────


class TestReranking:
    def test_applied(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            rerank_applied=True,
        )
        d = _dims(decisions)
        assert d["cross_encoder_rerank"] == "applied"

    def test_skipped(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            rerank_applied=False,
            rerank_skipped_reason="no_reranker_configured",
        )
        d = _dims(decisions)
        assert d["cross_encoder_rerank"] == "skipped"
        assert "no_reranker_configured" in _reasons(decisions)["cross_encoder_rerank"]

    def test_neither_rerank_nor_skip_no_decision(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            rerank_applied=False,
            rerank_skipped_reason=None,
        )
        dims = {d.dimension for d in decisions}
        assert "cross_encoder_rerank" not in dims


# ── Summary inclusion ─────────────────────────────────────────────────────────


class TestSummaryInclusion:
    def test_eligible_when_phase1_enabled(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            phase1_enabled=True,
        )
        d = _dims(decisions)
        assert d["summary_inclusion"] == "eligible"

    def test_blocked_when_phase1_disabled(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            phase1_enabled=False,
        )
        d = _dims(decisions)
        assert d["summary_inclusion"] == "blocked"


# ── Graph expansion ───────────────────────────────────────────────────────────


class TestGraphExpansion:
    def test_always_blocked(self):
        """Graph expansion requires double opt-in; never eligible on HTTP surface."""
        for phase1, phase2, phase3 in [
            (False, False, False),
            (True, True, True),
        ]:
            decisions = build_attention_decisions(
                mode_meta=_base_mode_meta(),
                governance_mode="off",
                governance_profile="default",
                phase1_enabled=phase1,
                phase2_enabled=phase2,
                phase3_enabled=phase3,
            )
            d = _dims(decisions)
            assert d["graph_expansion"] == "blocked", (
                f"graph_expansion should be blocked (phase1={phase1})"
            )


# ── Derived facts ─────────────────────────────────────────────────────────────


class TestDerivedFacts:
    def test_eligible(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            phase3_enabled=True,
            derive_views_requested=["summary"],
        )
        d = _dims(decisions)
        assert d["derived_facts"] == "eligible"

    def test_not_requested(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            phase3_enabled=True,
            derive_views_requested=None,
        )
        d = _dims(decisions)
        assert d["derived_facts"] == "not_requested"

    def test_blocked(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            phase3_enabled=False,
        )
        d = _dims(decisions)
        assert d["derived_facts"] == "blocked"


# ── Governance gate ───────────────────────────────────────────────────────────


class TestGovernanceGate:
    def test_off(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
        )
        d = _dims(decisions)
        assert d["governance_gate"] == "off"

    def test_advisory(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="advisory",
            governance_profile="default",
        )
        d = _dims(decisions)
        assert d["governance_gate"] == "advisory:default"

    def test_enforced_custom_profile(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="enforced",
            governance_profile="strict",
        )
        d = _dims(decisions)
        assert d["governance_gate"] == "enforced:strict"


# ── Forecast advisory ─────────────────────────────────────────────────────────


class TestForecastAdvisory:
    def test_conservative_plan(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(forecast_plan="conservative"),
            governance_mode="off",
            governance_profile="default",
        )
        d = _dims(decisions)
        assert d["forecast_advisory"] == "conservative"
        r = _reasons(decisions)["forecast_advisory"]
        assert "pressure" in r.lower()

    def test_standard_plan(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(forecast_plan="standard"),
            governance_mode="off",
            governance_profile="default",
        )
        d = _dims(decisions)
        assert d["forecast_advisory"] == "standard"

    def test_no_forecast(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(forecast_plan=""),
            governance_mode="off",
            governance_profile="default",
        )
        d = _dims(decisions)
        assert d["forecast_advisory"] == "none"


# ── Shadow / intent ───────────────────────────────────────────────────────────


class TestShadowAndIntent:
    def test_shadow_present(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            shadow_search_present=True,
        )
        d = _dims(decisions)
        assert d["shadow_search"] == "triggered"

    def test_intent_forecast_present(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            intent_forecast_present=True,
        )
        d = _dims(decisions)
        assert d["intent_trajectory"] == "forecast_emitted"

    def test_neither_absent(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            shadow_search_present=False,
            intent_forecast_present=False,
        )
        dims = {d.dimension for d in decisions}
        assert "shadow_search" not in dims
        assert "intent_trajectory" not in dims


# ── Structural sanity ─────────────────────────────────────────────────────────


class TestStructuralSanity:
    def test_all_decisions_have_non_empty_fields(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(
                mode="hybrid",
                policy="balanced",
                envelope_enabled=True,
                forecast_plan="conservative",
            ),
            governance_mode="advisory",
            governance_profile="default",
            phase1_enabled=True,
            phase2_enabled=True,
            phase3_enabled=True,
            derive_views_requested=["summary"],
            query_family="multi_hop",
            query_family_confidence=0.77,
            rerank_applied=True,
            shadow_search_present=True,
            intent_forecast_present=True,
        )
        for dec in decisions:
            assert dec.dimension, f"Empty dimension in {dec}"
            assert dec.decision, f"Empty decision in {dec}"
            assert dec.reason, f"Empty reason in {dec}"

    def test_all_objects_are_attention_decisions(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
        )
        for d in decisions:
            assert isinstance(d, AttentionDecision)


# ── Pattern advisory (dimension 12) ──────────────────────────────────────────


class TestPatternAdvisory:
    def test_disabled_when_store_not_configured(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            pattern_store_enabled=False,
        )
        d = _dims(decisions)
        assert d["pattern_advisory"] == "disabled"

    def test_recalled_when_store_enabled_and_candidates_found(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            pattern_store_enabled=True,
            pattern_advisory_count=2,
        )
        d = _dims(decisions)
        assert d["pattern_advisory"] == "recalled"
        r = _reasons(decisions)["pattern_advisory"]
        assert "2" in r
        assert "advisory" in r.lower()

    def test_none_when_store_enabled_but_no_candidates(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            pattern_store_enabled=True,
            pattern_advisory_count=0,
        )
        d = _dims(decisions)
        assert d["pattern_advisory"] == "none"

    def test_pattern_advisory_always_present(self):
        """pattern_advisory decision is always emitted (unlike shadow/intent which are optional)."""
        for enabled in [True, False]:
            decisions = build_attention_decisions(
                mode_meta=_base_mode_meta(),
                governance_mode="off",
                governance_profile="default",
                pattern_store_enabled=enabled,
                pattern_advisory_count=0,
            )
            dims = {d.dimension for d in decisions}
            assert "pattern_advisory" in dims

    def test_recalled_reason_mentions_governance(self):
        decisions = build_attention_decisions(
            mode_meta=_base_mode_meta(),
            governance_mode="off",
            governance_profile="default",
            pattern_store_enabled=True,
            pattern_advisory_count=1,
        )
        r = _reasons(decisions)["pattern_advisory"]
        assert "governance" in r.lower()
        assert "authoritative" in r.lower()
