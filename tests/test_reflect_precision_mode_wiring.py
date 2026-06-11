"""Wiring tests: reflect_precision_mode routes the reflect path to the NLI detector."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult
from mnemos.governance.governor import Governor
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.policy_profiles import GovernancePolicyProfile
from mnemos.governance.usage_detector import UsageDetector, UsageLabel


def _profile(mode: str, profile_id: str = "default") -> GovernancePolicyProfile:
    return GovernancePolicyProfile(profile_id=profile_id, reflect_precision_mode=mode).validate()


def _candidates():
    engram = Engram(id="m1", content="quarterly revenue grew by eight percent")
    results = [SearchResult(engram=engram, score=0.9, tier="t")]
    decisions = [GovernanceDecision(engram_id="m1", retrieval_score=0.9, governed_score=0.9)]
    return results, decisions


class _StubNLIDetector:
    """detect()-compatible double; records that it was invoked."""

    model_name = "stub-nli"

    def __init__(self):
        self.calls = 0

    def detect(self, query, answer, results, decisions, cited_ids=None) -> Dict[str, UsageLabel]:
        self.calls += 1
        return {r.engram.id: UsageLabel.USED for r in results}


# ── profile model ─────────────────────────────────────────────────────────


def test_profile_default_mode_is_lexical():
    assert GovernancePolicyProfile(profile_id="p").reflect_precision_mode == "lexical"


def test_profile_rejects_unknown_mode():
    with pytest.raises(ValueError, match="reflect_precision_mode"):
        _profile("yolo")


def test_profile_from_mapping_parses_and_normalizes_mode():
    profile = GovernancePolicyProfile.from_mapping("t1", {"reflect_precision_mode": " NLI "})
    assert profile.reflect_precision_mode == "nli"


# ── governor routing ──────────────────────────────────────────────────────


def test_lexical_profile_uses_lexical_detector():
    governor = Governor()
    detector = governor._resolve_usage_detector(_profile("lexical"))
    assert isinstance(detector, UsageDetector)


def test_nli_profile_routes_reflect_through_nli_detector():
    stub = _StubNLIDetector()
    governor = Governor(policy_profiles={"default": _profile("nli")})
    governor._get_nli_detector = lambda: stub

    results, decisions = _candidates()
    outcome = governor.reflect(
        query="q", answer="irrelevant words only", results=results, decisions=decisions
    )

    assert stub.calls == 1
    assert outcome.used_ids == ["m1"]  # stub labels everything USED; lexical would say IGNORED


def test_default_profile_nli_mode_is_honored():
    # Regression: the cached default reflect path used to bypass the profile.
    stub = _StubNLIDetector()
    governor = Governor(policy_profiles={"default": _profile("nli")})
    governor._get_nli_detector = lambda: stub

    results, decisions = _candidates()
    governor.reflect(query="q", answer="a", results=results, decisions=decisions)
    assert stub.calls == 1


def test_nli_unavailable_falls_back_to_lexical(monkeypatch):
    governor = Governor(policy_profiles={"default": _profile("nli")})
    monkeypatch.setattr(governor, "_get_nli_detector", lambda: None)

    detector = governor._resolve_usage_detector(_profile("nli"))
    assert isinstance(detector, UsageDetector)

    # reflect still completes end to end on the fallback
    results, decisions = _candidates()
    outcome = governor.reflect(query="q", answer="quarterly revenue grew by eight percent",
                               results=results, decisions=decisions)
    assert outcome.used_ids == ["m1"]  # lexical overlap fires on identical text


def test_nli_failure_is_cached_not_retried(monkeypatch):
    governor = Governor()
    attempts = {"n": 0}

    def _boom(*a, **k):
        attempts["n"] += 1
        raise RuntimeError("model load failed")

    import mnemos.governance.nli_usage_detector as nli_mod

    monkeypatch.setattr(nli_mod, "NLIUsageDetector", _boom)
    assert governor._get_nli_detector() is None
    assert governor._get_nli_detector() is None
    assert attempts["n"] == 1  # second call hits the cached failure


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
