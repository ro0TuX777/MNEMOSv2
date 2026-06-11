"""Unit tests for W5 counterfactual explainability (pure-arithmetic traces)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.governance.counterfactuals import compute_counterfactuals
from mnemos.governance.models.governance_decision import GovernanceDecision


def _decision(eid, retrieval=0.9, **mods):
    d = GovernanceDecision(engram_id=eid, retrieval_score=retrieval, governed_score=retrieval)
    for key, value in mods.items():
        setattr(d, key, value)
    d.governed_score = (
        d.retrieval_score
        * d.trust_modifier
        * d.utility_modifier
        * d.freshness_modifier
        * d.contradiction_modifier
        * d.veto_modifier
    )
    return d


def test_empty_when_no_included_candidates():
    vetoed = _decision("a", veto_pass=False, veto_modifier=0.0, suppressed=True)
    assert compute_counterfactuals([vetoed]) == []


def test_achievable_freshness_counterfactual():
    rank1 = _decision("winner", retrieval=0.9)                      # 0.9
    stale = _decision("stale", retrieval=0.9, freshness_modifier=0.4)  # 0.36
    entries = compute_counterfactuals([rank1, stale])

    assert len(entries) == 1
    entry = entries[0]
    assert entry["engram_id"] == "stale"
    assert entry["current_rank"] == 2
    assert abs(entry["rank1_gap"] - 0.54) < 1e-6

    freshness = next(c for c in entry["counterfactuals"] if c["modifier"] == "freshness")
    assert freshness["achievable"] is True
    # base = 0.36 / 0.4 = 0.9; required = 0.9 / 0.9 = 1.0
    assert abs(freshness["required_for_rank1"] - 1.0) < 1e-6
    assert "would tie rank 1" in freshness["statement"]


def test_unachievable_modifier_reports_best_rank():
    rank1 = _decision("winner", retrieval=0.9, trust_modifier=1.25)   # 1.125
    mid = _decision("mid", retrieval=0.9)                             # 0.9
    weak = _decision("weak", retrieval=0.5, trust_modifier=0.75)      # 0.375
    entries = compute_counterfactuals([rank1, mid, weak])

    weak_entry = next(e for e in entries if e["engram_id"] == "weak")
    trust = next(c for c in weak_entry["counterfactuals"] if c["modifier"] == "trust")
    # base = 0.375 / 0.75 = 0.5; required = 1.125 / 0.5 = 2.25 > 1.25 max
    assert trust["achievable"] is False
    assert abs(trust["required_for_rank1"] - 2.25) < 1e-6
    # at trust max 1.25: 0.5 * 1.25 = 0.625 → behind 1.125 and 0.9 → rank 3
    assert trust["best_achievable_rank"] == 3


def test_score_floor_veto_policy_hint():
    rank1 = _decision("winner")
    vetoed = _decision("floored", retrieval=0.55)
    vetoed.veto_pass = False
    vetoed.veto_modifier = 0.0
    vetoed.governed_score = 0.0
    vetoed.suppressed = True
    vetoed.veto_reason = "score 0.5500 below threshold 0.6000"

    entries = compute_counterfactuals([rank1, vetoed], min_score_threshold=0.6)
    entry = next(e for e in entries if e["engram_id"] == "floored")
    assert entry["current_rank"] is None
    assert entry["counterfactuals"] == []
    hint = entry["policy_hints"][0]
    assert "0.6000" in hint and "0.5500" in hint and "would have included it" in hint


def test_state_veto_is_not_counterfactable():
    rank1 = _decision("winner")
    deleted = _decision("deleted")
    deleted.veto_pass = False
    deleted.veto_modifier = 0.0
    deleted.governed_score = 0.0
    deleted.suppressed = True
    deleted.veto_reason = "deletion_state=tombstone"

    entries = compute_counterfactuals([rank1, deleted])
    entry = next(e for e in entries if e["engram_id"] == "deleted")
    assert "no modifier or threshold change" in entry["policy_hints"][0]


def test_contradiction_loser_gets_win_counterfactual_and_hint():
    rank1 = _decision("winner")                                        # 0.9
    loser = _decision("loser", retrieval=0.8, contradiction_modifier=0.25)
    loser.suppressed = True
    loser.suppressed_by_contradiction = True
    loser.contradiction_winner = "winner"

    entries = compute_counterfactuals([rank1, loser])
    entry = next(e for e in entries if e["engram_id"] == "loser")
    assert entry["current_rank"] is None

    contradiction = next(c for c in entry["counterfactuals"] if c["modifier"] == "contradiction")
    # base = 0.2 / 0.25 = 0.8; required = 0.9 / 0.8 = 1.125 > 1.0 max → not achievable
    assert contradiction["achievable"] is False
    assert contradiction["best_achievable_rank"] == 2
    assert any("contradiction loser to winner" in h for h in entry["policy_hints"])


def test_top_n_limits_targets():
    decisions = [_decision("winner")] + [
        _decision(f"c{i}", retrieval=0.8 - i * 0.1) for i in range(5)
    ]
    entries = compute_counterfactuals(decisions, top_n=3)
    assert len(entries) == 3
    assert [e["engram_id"] for e in entries] == ["c0", "c1", "c2"]


def test_freshness_age_inversion():
    from datetime import datetime, timedelta, timezone

    created = (datetime.now(tz=timezone.utc) - timedelta(days=200)).isoformat()
    rank1 = _decision("winner", retrieval=0.6)                          # 0.6
    stale = _decision("stale", retrieval=0.9, freshness_modifier=0.463)  # ~0.4167
    entries = compute_counterfactuals(
        [rank1, stale],
        created_at_by_id={"stale": created, "winner": created},
        freshness_half_life_days=180.0,
    )

    freshness = next(
        c for c in entries[0]["counterfactuals"] if c["modifier"] == "freshness"
    )
    # base = 0.4167/0.463 = 0.9; required = 0.6/0.9 = 0.6667
    # max_age = 180 * log2(1/0.6667) ≈ 105.3 days
    assert freshness["achievable"] is True
    assert abs(freshness["current_age_days"] - 200.0) < 1.0
    assert abs(freshness["max_age_days_for_rank1"] - 105.3) < 1.0
    assert "200 days old" in freshness["statement"]
    assert "younger than 105 days" in freshness["statement"]


def test_age_inversion_falls_back_without_timestamp():
    rank1 = _decision("winner", retrieval=0.9)
    stale = _decision("stale", retrieval=0.9, freshness_modifier=0.4)
    entries = compute_counterfactuals(
        [rank1, stale],
        created_at_by_id={},  # no timestamp for "stale"
        freshness_half_life_days=180.0,
    )
    freshness = next(
        c for c in entries[0]["counterfactuals"] if c["modifier"] == "freshness"
    )
    assert "current_age_days" not in freshness
    assert "would tie rank 1" in freshness["statement"]  # modifier-level statement


def test_age_inversion_handles_unparseable_timestamp():
    rank1 = _decision("winner", retrieval=0.9)
    stale = _decision("stale", retrieval=0.9, freshness_modifier=0.4)
    entries = compute_counterfactuals(
        [rank1, stale],
        created_at_by_id={"stale": "not-a-date"},
        freshness_half_life_days=180.0,
    )
    freshness = next(
        c for c in entries[0]["counterfactuals"] if c["modifier"] == "freshness"
    )
    assert "current_age_days" not in freshness


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
