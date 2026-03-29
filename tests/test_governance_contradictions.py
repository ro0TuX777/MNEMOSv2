"""
Tests for Wave 2 contradiction detection and resolution.

Covers:
  - ContradictionPolicy.detect_and_resolve():
      winner selection by trust / recency / utility / source_authority / id
      same entity + different attribute (two independent contradictions)
      same attribute + different entity (no cross-group interference)
      single-member group (not a contradiction)
      all-same-value group (not a contradiction)
      no governance metadata (ignored)
  - ReadPath integration: advisory retains losers / enforced removes them
  - Governor stats: contradictions_detected / contradiction_suppressed
  - GovernanceDecision fields: suppressed_by_contradiction, would_be_suppressed_in_enforced_mode
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.policies.contradiction_policy import ContradictionPolicy
from mnemos.governance.governor import Governor
from mnemos.governance.read_path import ReadPath
from mnemos.governance.policy_registry import PolicyRegistry
from mnemos.retrieval.base import SearchResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ts(days_ago: float = 0.0) -> str:
    return (
        datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    ).isoformat().replace("+00:00", "Z")


def _make_result(
    eid: str = "",
    score: float = 0.8,
    entity_key: str = "",
    attribute_key: str = "",
    normalized_value: str = "",
    trust_score: float = 1.0,
    utility_score: float = 1.0,
    source_authority: float = 0.5,
    created_days_ago: float = 0.0,
) -> SearchResult:
    gov = GovernanceMeta(
        entity_key=entity_key,
        attribute_key=attribute_key,
        normalized_value=normalized_value,
        trust_score=trust_score,
        utility_score=utility_score,
        source_authority=source_authority,
    ) if entity_key else None

    e = Engram(
        content="test",
        confidence=1.0,
        created_at=_ts(created_days_ago),
    )
    if eid:
        e.id = eid
    e.governance = gov
    return SearchResult(engram=e, score=score, tier="qdrant")


def _make_decision(result: SearchResult) -> GovernanceDecision:
    return GovernanceDecision(
        engram_id=result.engram.id,
        retrieval_score=result.score,
        governed_score=result.score,
    )


def _apply_policy(
    results: list[SearchResult],
) -> tuple[list[ContradictionPolicy], list[GovernanceDecision]]:
    """Run ContradictionPolicy and return (records, decisions)."""
    decisions = [_make_decision(r) for r in results]
    policy = ContradictionPolicy()
    records = policy.detect_and_resolve(results, decisions)
    return records, decisions


# ── ContradictionPolicy unit tests ───────────────────────────────────────────


class TestContradictionPolicyBasic:
    def test_no_governance_metadata_ignored(self):
        """Candidates without GovernanceMeta are not grouped."""
        r1 = _make_result(score=0.9)  # no entity_key
        r2 = _make_result(score=0.8)
        records, decisions = _apply_policy([r1, r2])
        assert records == []
        assert all(d.conflict_status == "none" for d in decisions)

    def test_single_member_group_not_a_contradiction(self):
        """A group with only one candidate cannot contradict itself."""
        r1 = _make_result(entity_key="user:alice", attribute_key="city", normalized_value="london")
        records, decisions = _apply_policy([r1])
        assert records == []
        assert decisions[0].conflict_status == "none"

    def test_same_value_group_not_a_contradiction(self):
        """Two candidates agreeing on the same value is not a contradiction."""
        r1 = _make_result(entity_key="user:alice", attribute_key="city", normalized_value="london")
        r2 = _make_result(entity_key="user:alice", attribute_key="city", normalized_value="london")
        records, decisions = _apply_policy([r1, r2])
        assert records == []
        assert all(d.conflict_status == "none" for d in decisions)

    def test_basic_contradiction_produces_record(self):
        r1 = _make_result(
            entity_key="user:alice", attribute_key="city",
            normalized_value="london", trust_score=0.9,
        )
        r2 = _make_result(
            entity_key="user:alice", attribute_key="city",
            normalized_value="paris", trust_score=0.7,
        )
        records, decisions = _apply_policy([r1, r2])
        assert len(records) == 1
        rec = records[0]
        assert rec.entity_key == "user:alice"
        assert rec.attribute_key == "city"
        assert rec.status == "resolved"
        assert rec.winner_memory_id == r1.engram.id
        assert r2.engram.id in rec.loser_memory_ids


class TestWinnerSelection:
    def test_winner_by_trust_score(self):
        r_high = _make_result(
            eid="high", entity_key="user:alice", attribute_key="city",
            normalized_value="london", trust_score=0.9,
        )
        r_low = _make_result(
            eid="low", entity_key="user:alice", attribute_key="city",
            normalized_value="paris", trust_score=0.5,
        )
        records, decisions = _apply_policy([r_low, r_high])
        assert records[0].winner_memory_id == "high"
        dec_map = {d.engram_id: d for d in decisions}
        assert dec_map["high"].conflict_status == "winner"
        assert dec_map["low"].conflict_status == "suppressed"
        assert dec_map["low"].suppressed_by_contradiction is True
        assert dec_map["low"].contradiction_winner == "high"
        assert "trust_score" in records[0].resolution_reason

    def test_winner_by_recency_when_trust_tied(self):
        r_new = _make_result(
            eid="new", entity_key="user:bob", attribute_key="employer",
            normalized_value="acme", trust_score=0.8, created_days_ago=10,
        )
        r_old = _make_result(
            eid="old", entity_key="user:bob", attribute_key="employer",
            normalized_value="globex", trust_score=0.8, created_days_ago=90,
        )
        records, decisions = _apply_policy([r_old, r_new])
        assert records[0].winner_memory_id == "new"
        assert "timestamp" in records[0].resolution_reason

    def test_winner_by_utility_when_trust_and_recency_tied(self):
        ts = _ts(0)
        r_hi_util = _make_result(
            eid="u1", entity_key="proj:x", attribute_key="status",
            normalized_value="active", trust_score=0.8, utility_score=0.9,
            created_days_ago=0,
        )
        r_lo_util = _make_result(
            eid="u2", entity_key="proj:x", attribute_key="status",
            normalized_value="inactive", trust_score=0.8, utility_score=0.3,
            created_days_ago=0,
        )
        # Force identical timestamps by overwriting created_at
        r_hi_util.engram.created_at = ts
        r_lo_util.engram.created_at = ts
        records, decisions = _apply_policy([r_lo_util, r_hi_util])
        assert records[0].winner_memory_id == "u1"
        assert "utility_score" in records[0].resolution_reason

    def test_winner_by_source_authority_tiebreaker(self):
        ts = _ts(0)
        r_auth = _make_result(
            eid="auth", entity_key="doc:1", attribute_key="version",
            normalized_value="v2", trust_score=0.8, utility_score=0.8,
            source_authority=0.9, created_days_ago=0,
        )
        r_low_auth = _make_result(
            eid="noauth", entity_key="doc:1", attribute_key="version",
            normalized_value="v1", trust_score=0.8, utility_score=0.8,
            source_authority=0.3, created_days_ago=0,
        )
        r_auth.engram.created_at = ts
        r_low_auth.engram.created_at = ts
        records, decisions = _apply_policy([r_low_auth, r_auth])
        assert records[0].winner_memory_id == "auth"
        assert "source_authority" in records[0].resolution_reason

    def test_winner_by_id_tiebreaker(self):
        """When all scores and timestamps are equal, lower id wins."""
        ts = _ts(0)
        r_a = _make_result(
            eid="aaa", entity_key="e", attribute_key="k",
            normalized_value="x", trust_score=0.5, utility_score=0.5,
            source_authority=0.5, created_days_ago=0,
        )
        r_b = _make_result(
            eid="zzz", entity_key="e", attribute_key="k",
            normalized_value="y", trust_score=0.5, utility_score=0.5,
            source_authority=0.5, created_days_ago=0,
        )
        r_a.engram.created_at = ts
        r_b.engram.created_at = ts
        records, decisions = _apply_policy([r_b, r_a])
        assert records[0].winner_memory_id == "aaa"
        assert "tie-breaker" in records[0].resolution_reason


class TestContradictionModifiers:
    def test_winner_modifier_is_one(self):
        r1 = _make_result(
            entity_key="u", attribute_key="k", normalized_value="a", trust_score=0.9,
        )
        r2 = _make_result(
            entity_key="u", attribute_key="k", normalized_value="b", trust_score=0.5,
        )
        records, decisions = _apply_policy([r1, r2])
        dec_map = {d.engram_id: d for d in decisions}
        assert dec_map[r1.engram.id].contradiction_modifier == pytest.approx(1.0)

    def test_loser_modifier_is_0_25(self):
        r1 = _make_result(
            entity_key="u", attribute_key="k", normalized_value="a", trust_score=0.9,
        )
        r2 = _make_result(
            entity_key="u", attribute_key="k", normalized_value="b", trust_score=0.5,
        )
        records, decisions = _apply_policy([r1, r2])
        dec_map = {d.engram_id: d for d in decisions}
        assert dec_map[r2.engram.id].contradiction_modifier == pytest.approx(0.25)

    def test_loser_governed_score_reflects_modifier(self):
        r1 = _make_result(
            entity_key="u", attribute_key="k", normalized_value="a",
            trust_score=0.9, score=0.8,
        )
        r2 = _make_result(
            entity_key="u", attribute_key="k", normalized_value="b",
            trust_score=0.5, score=0.8,
        )
        records, decisions = _apply_policy([r1, r2])
        dec_map = {d.engram_id: d for d in decisions}
        # Loser: governed_score ≈ retrieval_score * 1.0 * 1.0 * 1.0 * 0.25 * 1.0
        assert dec_map[r2.engram.id].governed_score == pytest.approx(0.8 * 0.25, abs=1e-5)

    def test_loser_marked_suppressed(self):
        r1 = _make_result(entity_key="u", attribute_key="k", normalized_value="a", trust_score=0.9)
        r2 = _make_result(entity_key="u", attribute_key="k", normalized_value="b", trust_score=0.5)
        records, decisions = _apply_policy([r1, r2])
        dec_map = {d.engram_id: d for d in decisions}
        assert dec_map[r2.engram.id].suppressed is True
        assert dec_map[r2.engram.id].suppressed_by_contradiction is True


class TestMultipleGroups:
    def test_same_entity_different_attributes_are_independent(self):
        """city and employer for same entity are two separate conflict groups."""
        r_city_a = _make_result(
            entity_key="user:alice", attribute_key="city",
            normalized_value="london", trust_score=0.9,
        )
        r_city_b = _make_result(
            entity_key="user:alice", attribute_key="city",
            normalized_value="paris", trust_score=0.5,
        )
        r_emp_a = _make_result(
            entity_key="user:alice", attribute_key="employer",
            normalized_value="acme", trust_score=0.9,
        )
        r_emp_b = _make_result(
            entity_key="user:alice", attribute_key="employer",
            normalized_value="globex", trust_score=0.4,
        )
        records, decisions = _apply_policy(
            [r_city_a, r_city_b, r_emp_a, r_emp_b]
        )
        assert len(records) == 2
        group_ids = {rec.attribute_key for rec in records}
        assert group_ids == {"city", "employer"}

    def test_same_attribute_different_entities_no_interference(self):
        """alice.city and bob.city are separate groups that don't affect each other."""
        r_alice = _make_result(
            entity_key="user:alice", attribute_key="city",
            normalized_value="london", trust_score=0.9,
        )
        r_alice_b = _make_result(
            entity_key="user:alice", attribute_key="city",
            normalized_value="paris", trust_score=0.5,
        )
        r_bob = _make_result(
            entity_key="user:bob", attribute_key="city",
            normalized_value="tokyo", trust_score=0.8,
        )
        r_bob_b = _make_result(
            entity_key="user:bob", attribute_key="city",
            normalized_value="osaka", trust_score=0.6,
        )
        records, decisions = _apply_policy(
            [r_alice, r_alice_b, r_bob, r_bob_b]
        )
        assert len(records) == 2
        entities = {rec.entity_key for rec in records}
        assert entities == {"user:alice", "user:bob"}


# ── ReadPath integration ──────────────────────────────────────────────────────


class TestReadPathContradiction:
    def _make_registry(self) -> PolicyRegistry:
        return PolicyRegistry()

    def _make_read_path(self) -> ReadPath:
        from mnemos.governance.policies.contradiction_policy import ContradictionPolicy
        return ReadPath(self._make_registry(), contradiction_policy=ContradictionPolicy())

    def test_advisory_retains_losers(self):
        rp = self._make_read_path()
        r_winner = _make_result(
            eid="win", entity_key="u", attribute_key="k",
            normalized_value="a", trust_score=0.9, score=0.8,
        )
        r_loser = _make_result(
            eid="lose", entity_key="u", attribute_key="k",
            normalized_value="b", trust_score=0.5, score=0.8,
        )
        out, decisions, records = rp.apply(
            [r_winner, r_loser], "q", governance_mode="advisory"
        )
        # Advisory: both returned even though loser is suppressed
        assert len(out) == 2
        assert len(records) == 1

    def test_enforced_removes_losers(self):
        rp = self._make_read_path()
        r_winner = _make_result(
            eid="win", entity_key="u", attribute_key="k",
            normalized_value="a", trust_score=0.9, score=0.8,
        )
        r_loser = _make_result(
            eid="lose", entity_key="u", attribute_key="k",
            normalized_value="b", trust_score=0.5, score=0.8,
        )
        out, decisions, records = rp.apply(
            [r_winner, r_loser], "q", governance_mode="enforced"
        )
        assert len(out) == 1
        assert out[0].engram.id == "win"
        assert len(records) == 1

    def test_would_be_suppressed_set_in_advisory(self):
        rp = self._make_read_path()
        r_winner = _make_result(
            eid="win", entity_key="u", attribute_key="k",
            normalized_value="a", trust_score=0.9,
        )
        r_loser = _make_result(
            eid="lose", entity_key="u", attribute_key="k",
            normalized_value="b", trust_score=0.5,
        )
        out, decisions, records = rp.apply(
            [r_winner, r_loser], "q", governance_mode="advisory"
        )
        dec_map = {d.engram_id: d for d in decisions}
        assert dec_map["lose"].would_be_suppressed_in_enforced_mode is True
        assert dec_map["win"].would_be_suppressed_in_enforced_mode is False


# ── Governor stats ────────────────────────────────────────────────────────────


class TestGovernorContradictionStats:
    def test_contradiction_stats_tracked(self):
        gov = Governor()
        r_win = _make_result(
            entity_key="u", attribute_key="k",
            normalized_value="a", trust_score=0.9,
        )
        r_lose = _make_result(
            entity_key="u", attribute_key="k",
            normalized_value="b", trust_score=0.5,
        )
        gov.govern([r_win, r_lose], "q", governance_mode="advisory")
        s = gov.stats()
        assert s["total_contradictions_detected"] == 1
        assert s["total_contradiction_suppressed"] == 1

    def test_no_contradiction_stats_zero(self):
        gov = Governor()
        gov.govern([_make_result(score=0.8)], "q", governance_mode="advisory")
        s = gov.stats()
        assert s["total_contradictions_detected"] == 0
        assert s["total_contradiction_suppressed"] == 0
