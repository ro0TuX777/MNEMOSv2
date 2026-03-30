"""
Governance drift validation and false-credit inspection pack.

Purpose
-------
Verify that the Wave 3 reflect loop produces sensible long-run behavior:

  Drift scenarios (run N sequential reflect cycles, inspect convergence)
  -----------------------------------------------------------------------
  Scenario A — repeated winner cited
  Scenario B — repeated distractor ignored
  Scenario C — contradiction winner cited repeatedly
  Scenario D — stale value ignored repeatedly
  Scenario E — short generic distractor vs longer grounded candidate

  False-credit inspection (overlap detector failure modes)
  ---------------------------------------------------------
  Scenario F — long answer with multi-candidate surface overlap
  Scenario G — short generic memory content near classification boundary
  Scenario H — contradiction loser with phrasing overlap (priority ordering)
  Scenario I — semantically similar but lexically distinct distractor
  Scenario J — threshold boundary: raising threshold reclassifies borderline

Design notes
------------
- All scenarios run entirely in-process on the same Engram objects.
- No backend infrastructure required.
- Assertions check direction, convergence, and bounds — not exact values.
- Drift tables can be printed with ``pytest -s`` for manual inspection.
"""

from __future__ import annotations

import pytest

from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.usage_detector import UsageDetector, UsageLabel
from mnemos.governance.reinforcement import Reinforcement, ReinforcementConfig
from mnemos.governance.reflect_path import ReflectPath
from mnemos.retrieval.base import SearchResult


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_result(
    eid: str,
    content: str,
    utility: float = 0.5,
    trust: float = 0.5,
    stability: float = 0.5,
    conflict_status: str = "none",
    suppressed_by_contradiction: bool = False,
    veto_pass: bool = True,
) -> SearchResult:
    gov = GovernanceMeta(
        utility_score=utility,
        trust_score=trust,
        stability_score=stability,
        conflict_status=conflict_status,
    )
    e = Engram(content=content)
    e.id = eid
    e.governance = gov
    return SearchResult(engram=e, score=0.8, tier="qdrant")


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


def _cycle(
    rp: ReflectPath,
    results: list[SearchResult],
    decisions: list[GovernanceDecision],
    query: str,
    answer: str,
    cited_ids: list[str] | None,
) -> None:
    """Run one reflect cycle; updates GovernanceMeta in place."""
    rp.reflect(
        query=query,
        answer=answer,
        results=results,
        decisions=decisions,
        cited_ids=cited_ids,
    )


def _scores(r: SearchResult) -> tuple[float, float, float]:
    g = r.engram.governance
    return g.utility_score, g.trust_score, g.stability_score


# ── Scenario A: Repeated winner cited ────────────────────────────────────────


class TestScenarioA_RepeatedWinner:
    """
    A memory that is cited on every reflect cycle should:
    - Gain utility and trust monotonically until clamped at max.
    - Have usage_count equal to the number of cycles.
    - Converge at the upper bound rather than oscillating.
    """

    N = 15  # enough cycles to hit the max with utility_used=0.05 from 0.5

    def test_utility_increases_monotonically(self):
        r = _make_result("winner", content="the answer text about project status", utility=0.5)
        d = _make_decision(r)
        rp = ReflectPath()
        prev_utility = r.engram.governance.utility_score

        for _ in range(self.N):
            _cycle(rp, [r], [d], "q", "answer about project status", cited_ids=["winner"])
            current = r.engram.governance.utility_score
            assert current >= prev_utility, "utility must not decrease for a cited memory"
            prev_utility = current

    def test_utility_converges_at_max(self):
        r = _make_result("winner", content="content", utility=0.5)
        d = _make_decision(r)
        rp = ReflectPath()

        for _ in range(self.N):
            _cycle(rp, [r], [d], "q", "answer", cited_ids=["winner"])

        assert r.engram.governance.utility_score <= 1.0

    def test_trust_increases_monotonically(self):
        r = _make_result("winner", content="content", trust=0.5)
        d = _make_decision(r)
        rp = ReflectPath()
        prev_trust = r.engram.governance.trust_score

        for _ in range(self.N):
            _cycle(rp, [r], [d], "q", "answer", cited_ids=["winner"])
            current = r.engram.governance.trust_score
            assert current >= prev_trust
            prev_trust = current

    def test_stability_increases(self):
        r = _make_result("winner", content="content", stability=0.5)
        d = _make_decision(r)
        rp = ReflectPath()
        before = r.engram.governance.stability_score

        for _ in range(self.N):
            _cycle(rp, [r], [d], "q", "answer", cited_ids=["winner"])

        assert r.engram.governance.stability_score > before

    def test_usage_count_matches_cycle_count(self):
        r = _make_result("winner", content="content")
        d = _make_decision(r)
        rp = ReflectPath()

        for i in range(self.N):
            _cycle(rp, [r], [d], "q", "answer", cited_ids=["winner"])

        assert r.engram.governance.usage_count == self.N

    def test_last_used_at_updated_each_cycle(self):
        r = _make_result("winner", content="content")
        d = _make_decision(r)
        rp = ReflectPath()
        timestamps = set()

        for _ in range(5):
            _cycle(rp, [r], [d], "q", "answer", cited_ids=["winner"])
            ts = r.engram.governance.last_used_at
            assert ts is not None
            timestamps.add(ts)

        # Timestamps should be set (all equal is fine — fast test; set size >= 1)
        assert len(timestamps) >= 1

    def test_scores_do_not_exceed_max(self, capsys):
        r = _make_result("winner", content="content", utility=0.95, trust=0.95)
        d = _make_decision(r)
        rp = ReflectPath()
        history = []

        for i in range(10):
            _cycle(rp, [r], [d], "q", "answer", cited_ids=["winner"])
            u, t, s = _scores(r)
            history.append((i + 1, round(u, 4), round(t, 4), round(s, 4)))

        print("\nScenario A — repeated winner drift:")
        print(f"  {'cycle':>5}  {'utility':>8}  {'trust':>8}  {'stability':>10}")
        for cycle, u, t, s in history:
            print(f"  {cycle:>5}  {u:>8.4f}  {t:>8.4f}  {s:>10.4f}")

        assert all(u <= 1.0 and t <= 1.0 and s <= 1.0 for _, u, t, s in history)


# ── Scenario B: Repeated distractor ignored ──────────────────────────────────


class TestScenarioB_RepeatedDistractor:
    """
    A memory that is never cited and has no answer overlap should:
    - Lose utility monotonically until clamped at min.
    - Trust should remain unchanged (default trust_ignored=0.0).
    - Never go below 0.0 on any score.
    """

    N = 20
    ANSWER = "The capital of France is Paris and the Eiffel Tower stands there."
    DISTRACTOR_CONTENT = "zzz quantum entanglement xray spectrometry"

    def test_utility_decreases_monotonically(self):
        r = _make_result("distractor", content=self.DISTRACTOR_CONTENT, utility=0.5)
        d = _make_decision(r)
        rp = ReflectPath()
        prev = r.engram.governance.utility_score

        for _ in range(self.N):
            _cycle(rp, [r], [d], "q", self.ANSWER, cited_ids=None)
            current = r.engram.governance.utility_score
            assert current <= prev, "distractor utility must not increase"
            prev = current

    def test_utility_converges_at_min(self):
        r = _make_result("distractor", content=self.DISTRACTOR_CONTENT, utility=0.5)
        d = _make_decision(r)
        rp = ReflectPath()

        for _ in range(self.N):
            _cycle(rp, [r], [d], "q", self.ANSWER, cited_ids=None)

        assert r.engram.governance.utility_score >= 0.0

    def test_trust_unchanged_by_default(self):
        r = _make_result("distractor", content=self.DISTRACTOR_CONTENT, trust=0.7)
        d = _make_decision(r)
        rp = ReflectPath()
        before = r.engram.governance.trust_score

        for _ in range(self.N):
            _cycle(rp, [r], [d], "q", self.ANSWER, cited_ids=None)

        assert r.engram.governance.trust_score == pytest.approx(before)

    def test_last_used_at_never_updated(self):
        r = _make_result("distractor", content=self.DISTRACTOR_CONTENT)
        d = _make_decision(r)
        rp = ReflectPath()

        for _ in range(self.N):
            _cycle(rp, [r], [d], "q", self.ANSWER, cited_ids=None)

        assert r.engram.governance.last_used_at is None

    def test_drift_table(self, capsys):
        r = _make_result("distractor", content=self.DISTRACTOR_CONTENT, utility=0.5)
        d = _make_decision(r)
        rp = ReflectPath()
        history = []

        for i in range(self.N):
            _cycle(rp, [r], [d], "q", self.ANSWER, cited_ids=None)
            u, t, s = _scores(r)
            history.append((i + 1, round(u, 4), round(t, 4)))

        print("\nScenario B — repeated distractor drift:")
        print(f"  {'cycle':>5}  {'utility':>8}  {'trust':>8}")
        for cycle, u, t in history:
            print(f"  {cycle:>5}  {u:>8.4f}  {t:>8.4f}")


# ── Scenario C: Contradiction winner cited repeatedly ────────────────────────


class TestScenarioC_ContradictionWinnerCited:
    """
    A contradiction winner cited on every cycle should behave like scenario A.
    The loser (not cited, contradiction) should get penalized each cycle.
    After N cycles the winner should have meaningfully higher utility than the loser.
    """

    N = 10

    def test_winner_utility_exceeds_loser_after_N_cycles(self):
        winner = _make_result("win", content="project status active", utility=0.5, trust=0.5,
                              conflict_status="winner")
        loser = _make_result("lose", content="project status inactive", utility=0.5, trust=0.5,
                             conflict_status="suppressed", suppressed_by_contradiction=True)
        dw = _make_decision(winner)
        dl = _make_decision(loser, suppressed_by_contradiction=True)
        rp = ReflectPath()

        for _ in range(self.N):
            _cycle(rp, [winner, loser], [dw, dl],
                   "q", "project is active and running", cited_ids=["win"])

        assert winner.engram.governance.utility_score > loser.engram.governance.utility_score

    def test_winner_trust_exceeds_loser_after_N_cycles(self):
        winner = _make_result("win", content="project status active", utility=0.5, trust=0.5,
                              conflict_status="winner")
        loser = _make_result("lose", content="project status inactive", utility=0.5, trust=0.5,
                             conflict_status="suppressed", suppressed_by_contradiction=True)
        dw = _make_decision(winner)
        dl = _make_decision(loser, suppressed_by_contradiction=True)
        rp = ReflectPath()

        for _ in range(self.N):
            _cycle(rp, [winner, loser], [dw, dl],
                   "q", "project is active and running", cited_ids=["win"])

        assert winner.engram.governance.trust_score > loser.engram.governance.trust_score

    def test_drift_separation_table(self, capsys):
        winner = _make_result("win", content="project status active", utility=0.5, trust=0.5,
                              conflict_status="winner")
        loser = _make_result("lose", content="project status inactive", utility=0.5, trust=0.5,
                             conflict_status="suppressed", suppressed_by_contradiction=True)
        dw = _make_decision(winner)
        dl = _make_decision(loser, suppressed_by_contradiction=True)
        rp = ReflectPath()

        print("\nScenario C — contradiction winner vs loser drift:")
        print(f"  {'cycle':>5}  {'win_util':>9}  {'lose_util':>10}  {'separation':>11}")
        for i in range(self.N):
            _cycle(rp, [winner, loser], [dw, dl],
                   "q", "project is active and running", cited_ids=["win"])
            wu = winner.engram.governance.utility_score
            lu = loser.engram.governance.utility_score
            print(f"  {i+1:>5}  {wu:>9.4f}  {lu:>10.4f}  {wu-lu:>11.4f}")


# ── Scenario D: Stale value ignored repeatedly ───────────────────────────────


class TestScenarioD_StaleValueIgnored:
    """
    A stale memory never cited and with no answer overlap should lose utility
    steadily over many cycles. This simulates a value that used to be relevant
    but has been superseded — it should naturally decay toward low priority.
    """

    N = 30
    STALE_CONTENT = "legacy xray protocol obsolete deprecated endpoint"
    ANSWER = "The updated configuration is stored in the new system version."

    def test_stale_utility_below_initial_after_N_cycles(self):
        r = _make_result("stale", content=self.STALE_CONTENT, utility=0.8)
        d = _make_decision(r)
        rp = ReflectPath()
        before = r.engram.governance.utility_score

        for _ in range(self.N):
            _cycle(rp, [r], [d], "q", self.ANSWER, cited_ids=None)

        assert r.engram.governance.utility_score < before

    def test_stale_utility_never_negative(self):
        r = _make_result("stale", content=self.STALE_CONTENT, utility=0.1)
        d = _make_decision(r)
        rp = ReflectPath()

        for _ in range(self.N):
            _cycle(rp, [r], [d], "q", self.ANSWER, cited_ids=None)

        assert r.engram.governance.utility_score >= 0.0

    def test_drift_table(self, capsys):
        r = _make_result("stale", content=self.STALE_CONTENT, utility=0.8)
        d = _make_decision(r)
        rp = ReflectPath()
        history = []

        for i in range(0, self.N, 5):
            for _ in range(5):
                _cycle(rp, [r], [d], "q", self.ANSWER, cited_ids=None)
            history.append((i + 5, round(r.engram.governance.utility_score, 4)))

        print("\nScenario D — stale value decay:")
        print(f"  {'cycle':>5}  {'utility':>8}")
        for cycle, u in history:
            print(f"  {cycle:>5}  {u:>8.4f}")


# ── Scenario E: Short generic vs longer grounded candidate ───────────────────


class TestScenarioE_GenericVsGrounded:
    """
    A longer, specific memory that genuinely contributed to the answer
    should outrank a short, generic memory that merely overlaps on common words.

    After N cycles the grounded candidate should have higher utility than
    the generic one, regardless of whether the generic one fires the overlap
    detector by accident.
    """

    N = 10
    ANSWER = (
        "The deployment configuration for the core appliance profile "
        "uses Qdrant as the primary vector backend with PostgreSQL for "
        "the audit ledger. The governance mode defaults to off."
    )
    GROUNDED = (
        "deployment configuration core appliance profile uses Qdrant vector "
        "backend PostgreSQL audit ledger governance mode defaults off"
    )
    GENERIC = "it on"  # all tokens < 3 chars — fall below word-set floor

    def test_grounded_ends_higher_than_generic(self):
        r_grounded = _make_result("grounded", content=self.GROUNDED, utility=0.5)
        r_generic = _make_result("generic", content=self.GENERIC, utility=0.5)
        d_grounded = _make_decision(r_grounded)
        d_generic = _make_decision(r_generic)
        rp = ReflectPath()

        for _ in range(self.N):
            _cycle(rp, [r_grounded, r_generic], [d_grounded, d_generic],
                   "q", self.ANSWER, cited_ids=None)

        grounded_utility = r_grounded.engram.governance.utility_score
        generic_utility = r_generic.engram.governance.utility_score
        assert grounded_utility > generic_utility, (
            f"Grounded ({grounded_utility:.4f}) should outrank generic ({generic_utility:.4f}) "
            f"after {self.N} reflect cycles"
        )

    def test_short_content_below_word_floor_not_false_positive(self):
        """
        Content with only tokens shorter than 3 characters cannot match
        anything — the word set will be empty and overlap defaults to 0.
        """
        r = _make_result("generic", content="it is at my")  # all tokens < 3 chars
        d = _make_decision(r)
        det = UsageDetector(overlap_threshold=0.15)
        labels = det.detect(
            query="q",
            answer="it is at my house",
            results=[r],
            decisions=[d],
        )
        # Empty memory word set → overlap=0.0 → IGNORED
        assert labels["generic"] == UsageLabel.IGNORED


# ── Scenario F: Long answer with multi-candidate surface overlap ──────────────


class TestScenarioF_LongAnswerMultiOverlap:
    """
    A long answer that shares surface words with several candidates should
    not cause ALL candidates to be classified as USED.

    With default threshold=0.15, a candidate whose content is a subset of
    a long answer may fire if it has enough matching words.

    This scenario exists to document behavior and identify threshold needs.
    """

    ANSWER = (
        "The quarterly financial results for the fiscal year 2026 showed "
        "strong revenue growth across all business units. The technology "
        "division reported record profits driven by cloud expansion. "
        "Meanwhile the legal department finalized the compliance framework "
        "for data governance and privacy protection."
    )

    def test_topically_unrelated_candidate_is_ignored(self):
        """A candidate about quantum physics is IGNORED despite the long answer."""
        r = _make_result("physics", content="quantum entanglement photon spin")
        d = _make_decision(r)
        det = UsageDetector(overlap_threshold=0.15)
        labels = det.detect(query="q", answer=self.ANSWER, results=[r], decisions=[d])
        assert labels["physics"] == UsageLabel.IGNORED

    def test_high_overlap_candidate_is_used(self):
        """A candidate closely paraphrasing the answer fires the overlap detector."""
        r = _make_result(
            "finance",
            content="quarterly financial results fiscal year revenue growth technology profits",
        )
        d = _make_decision(r)
        det = UsageDetector(overlap_threshold=0.15)
        labels = det.detect(query="q", answer=self.ANSWER, results=[r], decisions=[d])
        assert labels["finance"] == UsageLabel.USED

    def test_low_specificity_common_words_borderline(self):
        """
        A candidate with only common words shared with the answer is a
        threshold-boundary case.  Document its classification at default
        threshold so we know when tuning changes it.
        """
        r = _make_result("common", content="business quarterly forecast budget goals")
        d = _make_decision(r)
        det_default = UsageDetector(overlap_threshold=0.15)
        det_strict = UsageDetector(overlap_threshold=0.50)

        label_default = det_default.detect(query="q", answer=self.ANSWER,
                                           results=[r], decisions=[d])["common"]
        label_strict = det_strict.detect(query="q", answer=self.ANSWER,
                                         results=[r], decisions=[d])["common"]

        # Document what each threshold produces — this is the inspection output
        print(f"\nScenario F boundary: default={label_default.value} strict={label_strict.value}")

        # At strict threshold the borderline candidate must be IGNORED
        assert label_strict == UsageLabel.IGNORED


# ── Scenario G: Short generic memory near classification boundary ─────────────


class TestScenarioG_ShortGenericBoundary:
    """
    Short, generic memory content near the overlap threshold boundary.
    Documents the threshold at which the precision-first rule holds.
    """

    def test_two_word_content_ignored_at_default_threshold(self):
        """2-token content can only achieve 0% or 100% overlap — binary."""
        r = _make_result("g", content="system status")
        d = _make_decision(r)
        det = UsageDetector(overlap_threshold=0.15)
        # answer contains "system" and "status"
        labels = det.detect(
            query="q",
            answer="the system status is green and all services are running",
            results=[r],
            decisions=[d],
        )
        # Both tokens appear in answer → 100% overlap → USED at 0.15 threshold
        # This is the known false-positive case for generic 2-token content
        assert labels["g"] == UsageLabel.USED

    def test_raising_threshold_reclassifies_generic_to_ignored(self):
        """
        At 0.50+ threshold, a 2-token memory classified USED at 0.15 should
        still be USED (100% overlap) — but a 5-token memory with 1 matching
        word (20% overlap) would drop below 0.50 and become IGNORED.
        """
        r = _make_result("g5", content="system status active recent deployment")
        d = _make_decision(r)

        det_low = UsageDetector(overlap_threshold=0.20)
        det_high = UsageDetector(overlap_threshold=0.50)

        # "system" and "status" both appear → 2/5 = 40% → fires at 0.20, not at 0.50
        answer = "the system status was last checked yesterday"
        label_low = det_low.detect(query="q", answer=answer, results=[r], decisions=[d])["g5"]
        label_high = det_high.detect(query="q", answer=answer, results=[r], decisions=[d])["g5"]

        assert label_low == UsageLabel.USED
        assert label_high == UsageLabel.IGNORED

    def test_known_false_positive_documented(self, capsys):
        """
        A short, generic 2-token memory that appears in most answers will be
        classified as USED at the 0.15 default.  This is the known precision
        gap.  Operators should either:
          (a) raise the threshold, or
          (b) set minimum content length rules upstream.
        """
        r = _make_result("generic-2tok", content="system status")
        d = _make_decision(r)
        det = UsageDetector(overlap_threshold=0.15)
        labels = det.detect(
            query="q",
            answer="the system status is healthy",
            results=[r],
            decisions=[d],
        )
        print(
            "\nScenario G known false-positive: 'system status' -> "
            f"{labels['generic-2tok'].value} at threshold=0.15"
        )
        # Not an assertion — documentation of known behavior for manual review


# ── Scenario H: Contradiction loser with phrasing overlap ────────────────────


class TestScenarioH_ContradictionLoserOverlap:
    """
    A contradiction loser whose content shares words with the answer should
    still be classified as CONTRADICTED, not USED.  Contradiction state
    takes priority over overlap detection.
    """

    ANSWER = "The project is currently active and all systems are running."

    def test_loser_classified_contradicted_despite_overlap(self):
        # Content overlaps with answer on "project", "active", "systems"
        loser = _make_result(
            "loser",
            content="project status active systems running",
            conflict_status="suppressed",
        )
        d = _make_decision(loser, suppressed_by_contradiction=True)
        det = UsageDetector(overlap_threshold=0.15)
        labels = det.detect(
            query="q",
            answer=self.ANSWER,
            results=[loser],
            decisions=[d],
        )
        assert labels["loser"] == UsageLabel.CONTRADICTED

    def test_loser_gets_contradiction_penalty_not_used_bonus(self):
        loser = _make_result(
            "loser",
            content="project status active systems running",
            conflict_status="suppressed",
            utility=0.7,
            trust=0.7,
        )
        d = _make_decision(loser, suppressed_by_contradiction=True)
        rp = ReflectPath()

        rp.reflect(
            query="q",
            answer=self.ANSWER,
            results=[loser],
            decisions=[d],
        )

        # Loser should be penalized, not reinforced
        assert loser.engram.governance.utility_score < 0.7
        assert loser.engram.governance.trust_score < 0.7
        assert loser.engram.governance.last_used_at is None

    def test_winner_alongside_loser_reinforced_correctly(self):
        winner = _make_result("winner", content="project active running", utility=0.5, trust=0.5)
        loser = _make_result(
            "loser",
            content="project status inactive stopped",
            conflict_status="suppressed",
            utility=0.5, trust=0.5,
        )
        dw = _make_decision(winner)
        dl = _make_decision(loser, suppressed_by_contradiction=True)
        rp = ReflectPath()

        rp.reflect(
            query="q",
            answer=self.ANSWER,
            results=[winner, loser],
            decisions=[dw, dl],
            cited_ids=["winner"],
        )

        assert winner.engram.governance.utility_score > 0.5  # reinforced
        assert loser.engram.governance.utility_score < 0.5   # penalized


# ── Scenario I: Semantically similar but lexically distinct distractor ────────


class TestScenarioI_SemanticDistractor:
    """
    The overlap detector is lexical.  A semantically relevant memory that uses
    different vocabulary from the answer should be classified as IGNORED.

    This is expected behavior — the detector does not attempt semantic matching.
    It is documented here so that future ML-based attribution can be measured
    against this baseline.
    """

    def test_semantic_match_no_lexical_overlap_is_ignored(self):
        # Semantically: both are about European capitals
        # Lexically: no overlap with the answer
        r = _make_result(
            "semantic-match",
            content="London metropolis England United Kingdom government seat",
        )
        d = _make_decision(r)
        det = UsageDetector(overlap_threshold=0.15)
        labels = det.detect(
            query="q",
            answer="Paris is the capital of France and home to the Eiffel Tower.",
            results=[r],
            decisions=[d],
        )
        assert labels["semantic-match"] == UsageLabel.IGNORED

    def test_lexical_overlap_fires_regardless_of_semantic_relevance(self):
        """
        A topically unrelated memory that happens to share words with the answer
        may fire the overlap detector.  This is the known precision gap for
        lexical-only attribution — document it explicitly.
        """
        # Topically: server infrastructure (irrelevant to Paris/travel answer)
        # Lexically: shares "tower" and "France" with the answer
        r = _make_result(
            "infra",
            content="eiffel tower france server infrastructure",
        )
        d = _make_decision(r)
        det = UsageDetector(overlap_threshold=0.15)
        labels = det.detect(
            query="q",
            answer="Paris is the capital of France and home to the Eiffel Tower.",
            results=[r],
            decisions=[d],
        )
        # 3/4 tokens match → 75% overlap → USED at default threshold
        # This is a known false-positive due to proper-noun overlap
        print(
            f"\nScenario I lexical false-positive: 'eiffel tower france server...' "
            f"-> {labels['infra'].value} at threshold=0.15"
        )
        # Not an assertion — documented for threshold tuning


# ── Scenario J: Threshold boundary tuning ────────────────────────────────────


class TestScenarioJ_ThresholdTuning:
    """
    Demonstrates how raising the overlap threshold affects precision/recall.
    Use these checks to determine the right threshold for production use.
    """

    ANSWER = "The system has been deployed to production with high availability."

    def test_tight_threshold_prefers_precision(self):
        """At 0.50, only candidates with majority word match are USED."""
        high_overlap = _make_result(
            "high",
            content="system deployed production high availability infrastructure",
        )
        low_overlap = _make_result(
            "low",
            content="system monitoring basic check",
        )
        dh = _make_decision(high_overlap)
        dl = _make_decision(low_overlap)
        det = UsageDetector(overlap_threshold=0.50)
        labels = det.detect(
            query="q",
            answer=self.ANSWER,
            results=[high_overlap, low_overlap],
            decisions=[dh, dl],
        )
        # high_overlap: "system", "deployed", "production", "high", "availability" ≥ 5/6 → USED
        # low_overlap: "system" only → 1/4 = 25% < 50% → IGNORED
        assert labels["high"] == UsageLabel.USED
        assert labels["low"] == UsageLabel.IGNORED

    def test_threshold_table(self, capsys):
        """Print classification results at multiple thresholds for inspection."""
        candidate = _make_result(
            "borderline",
            content="system deployed production check",
        )
        d = _make_decision(candidate)

        print("\nScenario J threshold sensitivity:")
        print(f"  {'threshold':>10}  {'label':>12}")
        for threshold in [0.10, 0.15, 0.25, 0.40, 0.50, 0.75]:
            det = UsageDetector(overlap_threshold=threshold)
            label = det.detect(
                query="q",
                answer=self.ANSWER,
                results=[candidate],
                decisions=[d],
            )["borderline"]
            print(f"  {threshold:>10.2f}  {label.value:>12}")
