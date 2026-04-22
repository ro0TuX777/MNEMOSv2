"""
Tests for the Wave 1 governance layer.

Covers:
  - GovernanceMeta serialisation / deserialisation
  - RelevanceVetoPolicy: score floor, deletion state, toxic flag, freshness
  - UtilityPolicy: trust and utility modifier bounds
  - PolicyRegistry: pipeline execution, short-circuit, score formula
  - ReadPath: advisory (no suppression) and enforced (suppress + rerank) modes
  - Governor: govern() dispatch, stats tracking
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mnemos.engram.model import Engram
from mnemos.governance import GovernanceMeta, GovernanceDecision
from mnemos.governance.governor import Governor
from mnemos.governance.read_path import GOVERNANCE_MODES
from mnemos.governance.policies.relevance_veto_policy import (
    RelevanceVetoPolicy,
    _compute_freshness,
)
from mnemos.governance.policies.utility_policy import UtilityPolicy
from mnemos.governance.policy_registry import PolicyRegistry
from mnemos.governance.read_path import ReadPath
from mnemos.retrieval.base import SearchResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_engram(
    confidence: float = 1.0,
    created_days_ago: float = 0.0,
    governance: GovernanceMeta | None = None,
) -> Engram:
    created_at = (
        datetime.now(tz=timezone.utc) - timedelta(days=created_days_ago)
    ).isoformat().replace("+00:00", "Z")
    e = Engram(content="test", confidence=confidence, created_at=created_at)
    e.governance = governance
    return e


def _make_result(
    score: float = 0.8,
    confidence: float = 1.0,
    created_days_ago: float = 0.0,
    governance: GovernanceMeta | None = None,
) -> SearchResult:
    return SearchResult(
        engram=_make_engram(confidence, created_days_ago, governance),
        score=score,
        tier="qdrant",
    )


def _make_registry(
    min_score: float = 0.0,
    half_life: float = 180.0,
) -> PolicyRegistry:
    reg = PolicyRegistry()
    reg.register(RelevanceVetoPolicy(min_score_threshold=min_score, freshness_half_life_days=half_life))
    reg.register(UtilityPolicy())
    return reg


# ── GovernanceMeta ────────────────────────────────────────────────────────────


class TestGovernanceMeta:
    def test_defaults(self):
        gm = GovernanceMeta()
        assert gm.memory_type == "episodic"
        assert gm.lifecycle_state == "active"
        assert gm.trust_score == 1.0
        assert gm.utility_score == 1.0
        assert gm.deletion_state == "active"
        assert gm.policy_flags == []
        assert gm.derived_from == []

    def test_roundtrip(self):
        gm = GovernanceMeta(
            memory_type="semantic",
            source_id="src-1",
            derived_from=["a", "b"],
            trust_score=0.7,
            utility_score=0.4,
            policy_flags=["stale"],
            deletion_state="soft_deleted",
        )
        restored = GovernanceMeta.from_dict(gm.to_dict())
        assert restored.memory_type == "semantic"
        assert restored.source_id == "src-1"
        assert restored.derived_from == ["a", "b"]
        assert restored.trust_score == pytest.approx(0.7)
        assert restored.utility_score == pytest.approx(0.4)
        assert restored.policy_flags == ["stale"]
        assert restored.deletion_state == "soft_deleted"

    def test_from_dict_missing_keys(self):
        gm = GovernanceMeta.from_dict({})
        assert gm.trust_score == 1.0
        assert gm.lifecycle_state == "active"


# ── GovernanceDecision ────────────────────────────────────────────────────────


class TestGovernanceDecision:
    def test_to_dict_compact(self):
        dec = GovernanceDecision(
            engram_id="x",
            retrieval_score=0.8,
            governed_score=0.6,
            veto_pass=True,
        )
        d = dec.to_dict()
        assert d["retrieval_score"] == pytest.approx(0.8)
        assert d["governed_score"] == pytest.approx(0.6)
        assert d["veto_pass"] is True
        assert "modifiers" not in d

    def test_to_dict_full(self):
        dec = GovernanceDecision(
            engram_id="x",
            retrieval_score=0.8,
            governed_score=0.6,
            trust_modifier=1.1,
            freshness_modifier=0.75,
        )
        d = dec.to_dict_full()
        assert "modifiers" in d
        assert d["modifiers"]["trust"] == pytest.approx(1.1)
        assert d["modifiers"]["freshness"] == pytest.approx(0.75)


# ── Freshness ─────────────────────────────────────────────────────────────────


class TestFreshness:
    def test_brand_new(self):
        ts = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
        f = _compute_freshness(ts, half_life_days=180.0)
        assert f == pytest.approx(1.0, abs=0.01)

    def test_half_life(self):
        ts = (
            datetime.now(tz=timezone.utc) - timedelta(days=180)
        ).isoformat().replace("+00:00", "Z")
        f = _compute_freshness(ts, half_life_days=180.0)
        assert f == pytest.approx(0.5, abs=0.01)

    def test_very_old(self):
        ts = (
            datetime.now(tz=timezone.utc) - timedelta(days=3650)
        ).isoformat().replace("+00:00", "Z")
        f = _compute_freshness(ts, half_life_days=180.0)
        assert f < 0.001

    def test_bad_timestamp(self):
        assert _compute_freshness("not-a-date", 180.0) == pytest.approx(1.0)


# ── RelevanceVetoPolicy ───────────────────────────────────────────────────────


class TestRelevanceVetoPolicy:
    def test_no_veto_by_default(self):
        policy = RelevanceVetoPolicy(min_score_threshold=0.0)
        result = _make_result(score=0.01)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.01, governed_score=0.01)
        out = policy.evaluate(result, dec, {})
        assert out.veto_pass is True
        assert out.suppressed is False

    def test_veto_below_threshold(self):
        policy = RelevanceVetoPolicy(min_score_threshold=0.3)
        result = _make_result(score=0.1)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.1, governed_score=0.1)
        out = policy.evaluate(result, dec, {})
        assert out.veto_pass is False
        assert out.veto_modifier == 0.0
        assert out.suppressed is True
        assert "threshold" in out.veto_reason

    def test_no_veto_at_threshold(self):
        policy = RelevanceVetoPolicy(min_score_threshold=0.3)
        result = _make_result(score=0.3)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.3, governed_score=0.3)
        out = policy.evaluate(result, dec, {})
        assert out.veto_pass is True

    def test_veto_soft_deleted(self):
        gov = GovernanceMeta(deletion_state="soft_deleted")
        result = _make_result(score=0.9, governance=gov)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.9, governed_score=0.9)
        out = RelevanceVetoPolicy().evaluate(result, dec, {})
        assert out.veto_pass is False
        assert "soft_deleted" in out.veto_reason

    def test_veto_tombstone(self):
        gov = GovernanceMeta(deletion_state="tombstone")
        result = _make_result(score=0.9, governance=gov)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.9, governed_score=0.9)
        out = RelevanceVetoPolicy().evaluate(result, dec, {})
        assert out.veto_pass is False

    def test_veto_toxic_flag(self):
        gov = GovernanceMeta(policy_flags=["toxic"])
        result = _make_result(score=0.9, governance=gov)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.9, governed_score=0.9)
        out = RelevanceVetoPolicy().evaluate(result, dec, {})
        assert out.veto_pass is False
        assert "toxic" in out.veto_reason

    def test_active_with_other_flags_not_vetoed(self):
        gov = GovernanceMeta(policy_flags=["stale"])
        result = _make_result(score=0.9, governance=gov)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.9, governed_score=0.9)
        out = RelevanceVetoPolicy().evaluate(result, dec, {})
        assert out.veto_pass is True

    def test_freshness_modifier_set(self):
        result = _make_result(score=0.8, created_days_ago=180.0)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.8, governed_score=0.8)
        out = RelevanceVetoPolicy(freshness_half_life_days=180.0).evaluate(result, dec, {})
        assert out.veto_pass is True
        assert out.freshness_modifier == pytest.approx(0.5, abs=0.05)


# ── UtilityPolicy ─────────────────────────────────────────────────────────────


class TestUtilityPolicy:
    def test_full_confidence_neutral(self):
        result = _make_result(confidence=1.0)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.8, governed_score=0.8)
        out = UtilityPolicy().evaluate(result, dec, {})
        assert out.trust_modifier == pytest.approx(1.25)
        assert out.utility_modifier == pytest.approx(1.25)

    def test_zero_confidence(self):
        result = _make_result(confidence=0.0)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.8, governed_score=0.8)
        out = UtilityPolicy().evaluate(result, dec, {})
        assert out.trust_modifier == pytest.approx(0.75)

    def test_with_governance_scores(self):
        gov = GovernanceMeta(trust_score=0.5, utility_score=0.0)
        result = _make_result(governance=gov)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.8, governed_score=0.8)
        out = UtilityPolicy().evaluate(result, dec, {})
        # trust: 0.75 + 0.5*0.5 = 1.0
        assert out.trust_modifier == pytest.approx(1.0)
        # utility: clamp(0.5 + 0.75*0.0, 0.5, 1.25) = 0.5
        assert out.utility_modifier == pytest.approx(0.5)

    def test_modifiers_clamped(self):
        gov = GovernanceMeta(trust_score=2.0, utility_score=2.0)
        result = _make_result(governance=gov)
        dec = GovernanceDecision(engram_id=result.engram.id, retrieval_score=0.8, governed_score=0.8)
        out = UtilityPolicy().evaluate(result, dec, {})
        assert out.trust_modifier <= 1.25
        assert out.utility_modifier <= 1.25


# ── PolicyRegistry ────────────────────────────────────────────────────────────


class TestPolicyRegistry:
    def test_governed_score_formula(self):
        reg = _make_registry()
        result = _make_result(score=0.8, confidence=1.0, created_days_ago=0.0)
        dec = reg.evaluate(result, {"query": "test", "all_candidate_ids": [], "governance_mode": "advisory"})
        # freshness ~1.0, trust_mod=1.25, utility_mod=1.25 → score > 0.8
        assert dec.governed_score > 0.0
        assert dec.governed_score == pytest.approx(
            dec.retrieval_score
            * dec.trust_modifier
            * dec.utility_modifier
            * dec.freshness_modifier
            * dec.contradiction_modifier
            * dec.veto_modifier,
            abs=1e-5,
        )

    def test_short_circuit_on_veto(self):
        reg = _make_registry(min_score=0.5)
        result = _make_result(score=0.1)
        ctx = {"query": "q", "all_candidate_ids": [], "governance_mode": "advisory"}
        dec = reg.evaluate(result, ctx)
        assert dec.veto_modifier == 0.0
        assert dec.governed_score == pytest.approx(0.0, abs=1e-9)

    def test_disable_policy(self):
        reg = _make_registry(min_score=0.5)
        reg.disable("relevance_veto")
        result = _make_result(score=0.1)
        ctx = {"query": "q", "all_candidate_ids": [], "governance_mode": "advisory"}
        dec = reg.evaluate(result, ctx)
        assert dec.veto_pass is True  # veto not run

    def test_run_all_on_suppressed(self):
        reg = PolicyRegistry(run_all_on_suppressed=True)
        reg.register(RelevanceVetoPolicy(min_score_threshold=0.5))
        reg.register(UtilityPolicy())
        result = _make_result(score=0.1, confidence=1.0)
        ctx = {"query": "q", "all_candidate_ids": [], "governance_mode": "advisory"}
        dec = reg.evaluate(result, ctx)
        # UtilityPolicy ran despite veto → utility_modifier is 1.25, not default 1.0
        assert dec.utility_modifier == pytest.approx(1.25)


# ── ReadPath ──────────────────────────────────────────────────────────────────


class TestReadPath:
    def _make_read_path(self, min_score: float = 0.0) -> ReadPath:
        return ReadPath(_make_registry(min_score=min_score))

    def test_off_mode_returns_unchanged(self):
        rp = self._make_read_path()
        results = [_make_result(0.9), _make_result(0.5)]
        out, decisions, records = rp.apply(results, "q", governance_mode="off")
        assert out is results
        assert decisions == []
        assert records == []

    def test_advisory_returns_all(self):
        rp = self._make_read_path(min_score=0.5)
        # One result above threshold, one below
        r_high = _make_result(score=0.9)
        r_low = _make_result(score=0.1)
        out, decisions, records = rp.apply([r_high, r_low], "q", governance_mode="advisory")
        # Advisory: no suppression, both returned
        assert len(out) == 2
        assert len(decisions) == 2
        assert records == []

    def test_advisory_reranks_by_governed_score(self):
        rp = self._make_read_path()
        # r1 is older (lower freshness) but high raw score; r2 is newer, lower raw
        r1 = _make_result(score=0.9, created_days_ago=360.0)
        r2 = _make_result(score=0.5, created_days_ago=0.0)
        out, decisions, records = rp.apply([r1, r2], "q", governance_mode="advisory")
        # Both returned; decisions are in input order, out is sorted by governed_score
        assert len(out) == 2
        assert len(decisions) == 2
        dec_map = {d.engram_id: d for d in decisions}
        # out[0] should have the higher governed_score
        assert dec_map[out[0].engram.id].governed_score >= dec_map[out[1].engram.id].governed_score

    def test_enforced_suppresses_vetoed(self):
        rp = self._make_read_path(min_score=0.5)
        r_high = _make_result(score=0.9)
        r_low = _make_result(score=0.1)
        out, decisions, records = rp.apply([r_high, r_low], "q", governance_mode="enforced", top_k=10)
        assert len(out) == 1
        assert out[0].engram.id == r_high.engram.id

    def test_enforced_respects_top_k(self):
        rp = self._make_read_path()
        results = [_make_result(score=0.8 - i * 0.05) for i in range(10)]
        out, decisions, records = rp.apply(results, "q", governance_mode="enforced", top_k=3)
        assert len(out) == 3

    def test_empty_results(self):
        rp = self._make_read_path()
        out, decisions, records = rp.apply([], "q", governance_mode="advisory")
        assert out == []
        assert decisions == []
        assert records == []


# ── Governor ──────────────────────────────────────────────────────────────────


class TestGovernor:
    def test_govern_off_returns_unchanged(self):
        gov = Governor()
        results = [_make_result(0.8)]
        out, decisions, records = gov.govern(results, "q", governance_mode="off")
        assert out is results
        assert decisions == []
        assert records == []

    def test_govern_advisory(self):
        gov = Governor()
        results = [_make_result(0.8), _make_result(0.4)]
        out, decisions, records = gov.govern(results, "q", governance_mode="advisory")
        assert len(out) == 2
        assert len(decisions) == 2

    def test_govern_enforced(self):
        gov = Governor(min_score_threshold=0.5)
        r_keep = _make_result(score=0.9)
        r_drop = _make_result(score=0.1)
        out, decisions, records = gov.govern([r_keep, r_drop], "q", governance_mode="enforced")
        assert len(out) == 1
        assert out[0].engram.id == r_keep.engram.id

    def test_invalid_mode_raises(self):
        gov = Governor()
        with pytest.raises(ValueError, match="Invalid governance_mode"):
            gov.govern([], "q", governance_mode="unknown")

    def test_stats_accumulate(self):
        gov = Governor(min_score_threshold=0.5)
        gov.govern([_make_result(0.9)], "q", governance_mode="advisory")
        gov.govern([_make_result(0.1)], "q", governance_mode="enforced")
        s = gov.stats()
        assert s["total_governed_queries"] == 2
        assert s["advisory_queries"] == 1
        assert s["enforced_queries"] == 1
        assert s["total_candidates_evaluated"] == 2
        assert s["total_vetoed"] == 1
        assert "veto_rate" in s
        assert "active_policies" in s
        assert "total_contradictions_detected" in s
        assert "total_contradiction_suppressed" in s

    def test_governance_modes_constant(self):
        assert "off" in GOVERNANCE_MODES
        assert "advisory" in GOVERNANCE_MODES
        assert "enforced" in GOVERNANCE_MODES


# ── Engram governance field ───────────────────────────────────────────────────


class TestEngramGovernanceField:
    def test_default_governance_none(self):
        e = Engram(content="hello")
        assert e.governance is None

    def test_to_dict_excludes_governance_by_default(self):
        e = Engram(content="hello", governance=GovernanceMeta())
        d = e.to_dict()
        assert "_governance" not in d

    def test_to_dict_includes_governance_when_requested(self):
        e = Engram(content="hello", governance=GovernanceMeta(trust_score=0.7))
        d = e.to_dict(include_governance=True)
        assert "_governance" in d
        assert d["_governance"]["trust_score"] == pytest.approx(0.7)

    def test_from_dict_restores_governance(self):
        e = Engram(content="hello", governance=GovernanceMeta(memory_type="semantic"))
        d = e.to_dict(include_governance=True)
        e2 = Engram.from_dict(d)
        assert e2.governance is not None
        assert e2.governance.memory_type == "semantic"

    def test_from_dict_no_governance_key(self):
        d = {"content": "hello", "id": "abc"}
        e = Engram.from_dict(d)
        assert e.governance is None
