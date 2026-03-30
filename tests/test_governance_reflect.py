"""
Tests for Wave 3 reflect path — trust reinforcement and utility learning.

Covers (8 cases per spec):
  Case 1 — explicit citation marks memory used
  Case 2 — retrieved but unused memory is ignored
  Case 3 — multiple used memories all reinforced
  Case 4 — contradiction loser gets appropriate penalty
  Case 5 — vetoed memory receives no reinforcement
  Case 6 — bounded updates: scores clamp at configured max/min
  Case 7 — advisory-safe: reflect works regardless of governance_mode
  Case 8 — stats increment correctly in Governor
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.usage_detector import UsageDetector, UsageLabel
from mnemos.governance.reinforcement import Reinforcement, ReinforcementConfig
from mnemos.governance.reflect_path import ReflectPath, ReflectResult
from mnemos.governance.governor import Governor
from mnemos.retrieval.base import SearchResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(
    eid: str = "",
    content: str = "some memory content about the topic",
    score: float = 0.8,
    trust_score: float = 0.8,
    utility_score: float = 0.8,
    stability_score: float = 0.8,
    deletion_state: str = "active",
    policy_flags: list | None = None,
    conflict_status: str = "none",
    usage_count: int = 0,
) -> SearchResult:
    gov = GovernanceMeta(
        trust_score=trust_score,
        utility_score=utility_score,
        stability_score=stability_score,
        deletion_state=deletion_state,
        policy_flags=policy_flags or [],
        conflict_status=conflict_status,
        usage_count=usage_count,
    )
    e = Engram(content=content)
    if eid:
        e.id = eid
    e.governance = gov
    return SearchResult(engram=e, score=score, tier="qdrant")


def _make_decision(
    result: SearchResult,
    veto_pass: bool = True,
    suppressed_by_contradiction: bool = False,
) -> GovernanceDecision:
    return GovernanceDecision(
        engram_id=result.engram.id,
        retrieval_score=result.score,
        governed_score=result.score,
        veto_pass=veto_pass,
        suppressed=not veto_pass or suppressed_by_contradiction,
        suppressed_by_contradiction=suppressed_by_contradiction,
        conflict_status="suppressed" if suppressed_by_contradiction else "none",
    )


def _reflect(
    query: str = "test query",
    answer: str = "test answer",
    results: list | None = None,
    decisions: list | None = None,
    cited_ids: list | None = None,
    governance_mode: str = "advisory",
) -> ReflectResult:
    rp = ReflectPath()
    return rp.reflect(
        query=query,
        answer=answer,
        results=results or [],
        decisions=decisions or [],
        cited_ids=cited_ids,
        governance_mode=governance_mode,
    )


# ── Case 1: Explicit citation marks memory used ───────────────────────────────


class TestCase1ExplicitCitation:
    def test_cited_id_classified_as_used(self):
        r = _make_result(eid="mem-001", trust_score=0.7, utility_score=0.6)
        d = _make_decision(r)
        result = _reflect(
            answer="The answer is here.",
            results=[r],
            decisions=[d],
            cited_ids=["mem-001"],
        )
        assert "mem-001" in result.used_ids

    def test_cited_memory_utility_increases(self):
        r = _make_result(eid="mem-001", utility_score=0.6)
        d = _make_decision(r)
        before = r.engram.governance.utility_score
        _reflect(results=[r], decisions=[d], cited_ids=["mem-001"])
        assert r.engram.governance.utility_score > before

    def test_cited_memory_trust_increases(self):
        r = _make_result(eid="mem-001", trust_score=0.7)
        d = _make_decision(r)
        before = r.engram.governance.trust_score
        _reflect(results=[r], decisions=[d], cited_ids=["mem-001"])
        assert r.engram.governance.trust_score > before

    def test_cited_memory_last_used_at_updated(self):
        r = _make_result(eid="mem-001")
        d = _make_decision(r)
        assert r.engram.governance.last_used_at is None
        _reflect(results=[r], decisions=[d], cited_ids=["mem-001"])
        assert r.engram.governance.last_used_at is not None

    def test_cited_memory_usage_count_increments(self):
        r = _make_result(eid="mem-001", usage_count=2)
        d = _make_decision(r)
        _reflect(results=[r], decisions=[d], cited_ids=["mem-001"])
        assert r.engram.governance.usage_count == 3


# ── Case 2: Retrieved but unused memory is ignored ────────────────────────────


class TestCase2IgnoredMemory:
    def test_non_cited_no_overlap_is_ignored(self):
        # Content has no overlap with the answer words
        r = _make_result(eid="mem-002", content="zzz yyy xxx quantum entanglement")
        d = _make_decision(r)
        result = _reflect(
            answer="The capital of France is Paris.",
            results=[r],
            decisions=[d],
            cited_ids=None,
        )
        assert "mem-002" in result.ignored_ids

    def test_ignored_memory_utility_decreases(self):
        r = _make_result(eid="mem-002", content="zzz yyy xxx quantum entanglement",
                         utility_score=0.8)
        d = _make_decision(r)
        before = r.engram.governance.utility_score
        _reflect(
            answer="The capital of France is Paris.",
            results=[r],
            decisions=[d],
        )
        assert r.engram.governance.utility_score < before

    def test_ignored_memory_trust_unchanged_by_default(self):
        r = _make_result(eid="mem-002", content="zzz yyy quantum entanglement",
                         trust_score=0.8)
        d = _make_decision(r)
        before = r.engram.governance.trust_score
        _reflect(
            answer="The capital of France is Paris.",
            results=[r],
            decisions=[d],
        )
        # Default trust_ignored == 0.0 → no change
        assert r.engram.governance.trust_score == pytest.approx(before)

    def test_ignored_memory_last_used_at_not_updated(self):
        r = _make_result(eid="mem-002", content="zzz yyy quantum entanglement")
        d = _make_decision(r)
        _reflect(
            answer="The capital of France is Paris.",
            results=[r],
            decisions=[d],
        )
        assert r.engram.governance.last_used_at is None


# ── Case 3: Multiple used memories all reinforced ─────────────────────────────


class TestCase3MultipleUsed:
    def test_all_cited_memories_are_reinforced(self):
        r1 = _make_result(eid="m1", utility_score=0.5)
        r2 = _make_result(eid="m2", utility_score=0.5)
        r3 = _make_result(eid="m3", utility_score=0.5)
        d1, d2, d3 = _make_decision(r1), _make_decision(r2), _make_decision(r3)

        result = _reflect(
            results=[r1, r2, r3],
            decisions=[d1, d2, d3],
            cited_ids=["m1", "m2", "m3"],
        )

        assert set(result.used_ids) == {"m1", "m2", "m3"}
        assert result.total_reinforced == 3
        assert result.total_penalized == 0

    def test_stats_count_all_used(self):
        results = [_make_result(eid=f"m{i}") for i in range(4)]
        decisions = [_make_decision(r) for r in results]
        cited = [r.engram.id for r in results]
        result = _reflect(results=results, decisions=decisions, cited_ids=cited)
        assert len(result.used_ids) == 4
        assert result.total_reinforced == 4

    def test_utility_deltas_all_positive(self):
        r1 = _make_result(eid="m1")
        r2 = _make_result(eid="m2")
        result = _reflect(
            results=[r1, r2],
            decisions=[_make_decision(r1), _make_decision(r2)],
            cited_ids=["m1", "m2"],
        )
        assert result.utility_deltas["m1"] > 0
        assert result.utility_deltas["m2"] > 0


# ── Case 4: Contradiction loser gets penalty ──────────────────────────────────


class TestCase4ContradictionLoser:
    def test_loser_classified_as_contradicted(self):
        r = _make_result(eid="loser", conflict_status="suppressed",
                         utility_score=0.8, trust_score=0.8)
        d = _make_decision(r, suppressed_by_contradiction=True)
        result = _reflect(results=[r], decisions=[d])
        assert "loser" in result.contradicted_ids

    def test_loser_utility_decreases(self):
        r = _make_result(eid="loser", conflict_status="suppressed",
                         utility_score=0.8)
        d = _make_decision(r, suppressed_by_contradiction=True)
        before = r.engram.governance.utility_score
        _reflect(results=[r], decisions=[d])
        assert r.engram.governance.utility_score < before

    def test_loser_trust_decreases(self):
        r = _make_result(eid="loser", conflict_status="suppressed",
                         trust_score=0.8)
        d = _make_decision(r, suppressed_by_contradiction=True)
        before = r.engram.governance.trust_score
        _reflect(results=[r], decisions=[d])
        assert r.engram.governance.trust_score < before

    def test_contradiction_winner_can_still_be_reinforced(self):
        winner = _make_result(eid="winner", conflict_status="winner",
                              utility_score=0.7)
        loser = _make_result(eid="loser", conflict_status="suppressed",
                             utility_score=0.7)
        dw = _make_decision(winner)
        dl = _make_decision(loser, suppressed_by_contradiction=True)

        result = _reflect(
            results=[winner, loser],
            decisions=[dw, dl],
            cited_ids=["winner"],
        )
        assert "winner" in result.used_ids
        assert "loser" in result.contradicted_ids
        assert winner.engram.governance.utility_score > 0.7
        assert loser.engram.governance.utility_score < 0.7


# ── Case 5: Vetoed memory receives no reinforcement ───────────────────────────


class TestCase5VetoedMemory:
    def test_vetoed_classified_correctly(self):
        r = _make_result(eid="toxic-mem", policy_flags=["toxic"])
        d = _make_decision(r, veto_pass=False)
        result = _reflect(results=[r], decisions=[d])
        assert "toxic-mem" in result.vetoed_ids

    def test_vetoed_utility_unchanged(self):
        r = _make_result(eid="deleted-mem", deletion_state="soft_deleted",
                         utility_score=0.6)
        d = _make_decision(r, veto_pass=False)
        before = r.engram.governance.utility_score
        _reflect(results=[r], decisions=[d], cited_ids=["deleted-mem"])
        # Even if cited, vetoed takes priority — no reinforcement
        assert r.engram.governance.utility_score == pytest.approx(before)

    def test_vetoed_trust_unchanged(self):
        r = _make_result(eid="deleted-mem", deletion_state="tombstone",
                         trust_score=0.5)
        d = _make_decision(r, veto_pass=False)
        before = r.engram.governance.trust_score
        _reflect(results=[r], decisions=[d])
        assert r.engram.governance.trust_score == pytest.approx(before)

    def test_vetoed_not_in_used_ids(self):
        r = _make_result(eid="vtx")
        d = _make_decision(r, veto_pass=False)
        result = _reflect(results=[r], decisions=[d], cited_ids=["vtx"])
        assert "vtx" not in result.used_ids


# ── Case 6: Bounded updates ───────────────────────────────────────────────────


class TestCase6BoundedUpdates:
    def test_utility_clamped_at_max(self):
        # Start at max; used update should stay at max
        r = _make_result(eid="at-max", utility_score=1.0)
        d = _make_decision(r)
        _reflect(results=[r], decisions=[d], cited_ids=["at-max"])
        assert r.engram.governance.utility_score <= 1.0

    def test_utility_clamped_at_min(self):
        # Start near zero; many ignore cycles should not go below 0
        cfg = ReinforcementConfig(utility_ignored=-0.5)
        rp = ReflectPath(reinforcement=Reinforcement(config=cfg))
        r = _make_result(eid="near-zero", utility_score=0.1,
                         content="zzz yyy quantum")
        d = _make_decision(r)
        rp.reflect(
            query="q", answer="Paris is the capital of France.",
            results=[r], decisions=[d],
        )
        assert r.engram.governance.utility_score >= 0.0

    def test_trust_clamped_at_max(self):
        r = _make_result(eid="at-max-trust", trust_score=1.0)
        d = _make_decision(r)
        _reflect(results=[r], decisions=[d], cited_ids=["at-max-trust"])
        assert r.engram.governance.trust_score <= 1.0

    def test_custom_config_deltas_respected(self):
        cfg = ReinforcementConfig(utility_used=0.1, trust_used=0.05)
        rp = ReflectPath(reinforcement=Reinforcement(config=cfg))
        r = _make_result(eid="m1", utility_score=0.5, trust_score=0.5)
        d = _make_decision(r)
        rp.reflect(query="q", answer="answer", results=[r], decisions=[d],
                   cited_ids=["m1"])
        assert r.engram.governance.utility_score == pytest.approx(0.6, abs=1e-5)
        assert r.engram.governance.trust_score == pytest.approx(0.55, abs=1e-5)


# ── Case 7: Advisory-safe behavior ───────────────────────────────────────────


class TestCase7AdvisorySafe:
    def test_reflect_runs_in_advisory_mode(self):
        r = _make_result(eid="m1", utility_score=0.5)
        d = _make_decision(r)
        result = _reflect(
            results=[r], decisions=[d],
            cited_ids=["m1"], governance_mode="advisory",
        )
        assert result.governance_mode == "advisory"
        assert "m1" in result.used_ids

    def test_reflect_runs_in_enforced_mode(self):
        r = _make_result(eid="m1", utility_score=0.5)
        d = _make_decision(r)
        result = _reflect(
            results=[r], decisions=[d],
            cited_ids=["m1"], governance_mode="enforced",
        )
        assert result.governance_mode == "enforced"
        assert "m1" in result.used_ids

    def test_reflect_does_not_alter_search_defaults(self):
        """Reflect mutates GovernanceMeta in place but does not affect
        the governor's query-time mode or active policy list."""
        gov = Governor()
        r = _make_result(eid="m1", utility_score=0.5)
        d = _make_decision(r)
        gov.reflect(
            query="q", answer="answer",
            results=[r], decisions=[d], cited_ids=["m1"],
        )
        s = gov.stats()
        # governance_mode of queries is unaffected — no governed queries yet
        assert s["total_governed_queries"] == 0
        assert "relevance_veto" in s["active_policies"]

    def test_reflect_result_has_ran_at_timestamp(self):
        result = _reflect()
        assert result.ran_at != ""
        # Should be a parseable ISO timestamp
        datetime.fromisoformat(result.ran_at.replace("Z", "+00:00"))


# ── Case 8: Stats increment correctly ────────────────────────────────────────


class TestCase8StatsIncrement:
    def test_reflect_stats_accumulate_in_governor(self):
        gov = Governor()
        r1 = _make_result(eid="m1")
        r2 = _make_result(eid="m2", content="zzz yyy quantum entanglement")
        d1, d2 = _make_decision(r1), _make_decision(r2)

        gov.reflect(
            query="q",
            answer="Paris is the capital of France.",
            results=[r1, r2],
            decisions=[d1, d2],
            cited_ids=["m1"],
        )

        s = gov.stats()
        assert s["total_reflect_runs"] == 1
        assert s["total_used_memories"] == 1
        assert s["total_ignored_memories"] == 1
        assert s["total_utility_reinforcements"] == 1
        assert s["total_utility_penalties"] == 1

    def test_multiple_reflect_runs_accumulate(self):
        gov = Governor()
        for _ in range(3):
            r = _make_result(eid="m1", utility_score=0.5)
            d = _make_decision(r)
            gov.reflect(
                query="q", answer="answer",
                results=[r], decisions=[d], cited_ids=["m1"],
            )
        s = gov.stats()
        assert s["total_reflect_runs"] == 3
        assert s["total_used_memories"] == 3

    def test_trust_reinforcement_counted(self):
        gov = Governor()
        r = _make_result(eid="m1", trust_score=0.5)
        d = _make_decision(r)
        gov.reflect(
            query="q", answer="a", results=[r], decisions=[d], cited_ids=["m1"]
        )
        s = gov.stats()
        assert s["total_trust_reinforcements"] == 1
        assert s["total_trust_penalties"] == 0

    def test_trust_penalty_counted_for_contradiction_loser(self):
        gov = Governor()
        r = _make_result(eid="loser", conflict_status="suppressed",
                         trust_score=0.8)
        d = _make_decision(r, suppressed_by_contradiction=True)
        gov.reflect(query="q", answer="a", results=[r], decisions=[d])
        s = gov.stats()
        assert s["total_trust_penalties"] == 1


# ── UsageDetector unit coverage ──────────────────────────────────────────────


class TestUsageDetector:
    def test_overlap_threshold_fires(self):
        """Memory content with sufficient word overlap → USED."""
        det = UsageDetector(overlap_threshold=0.15)
        # answer shares 3 of 4 unique memory words → 75% overlap
        r = _make_result(eid="m1", content="paris capital france travel")
        d = _make_decision(r)
        labels = det.detect(
            query="q",
            answer="Paris is the capital of France.",
            results=[r],
            decisions=[d],
        )
        assert labels["m1"] == UsageLabel.USED

    def test_below_threshold_is_ignored(self):
        det = UsageDetector(overlap_threshold=0.9)
        r = _make_result(eid="m1", content="paris capital france travel weather")
        d = _make_decision(r)
        labels = det.detect(
            query="q",
            answer="Paris.",  # Only 1 word matches
            results=[r],
            decisions=[d],
        )
        assert labels["m1"] == UsageLabel.IGNORED

    def test_no_decision_returns_unknown(self):
        det = UsageDetector()
        r = _make_result(eid="m1")
        labels = det.detect(query="q", answer="a", results=[r], decisions=[])
        assert labels["m1"] == UsageLabel.UNKNOWN
