from __future__ import annotations

from mnemos.engram.model import Engram
from mnemos.governance.governor import Governor
from mnemos.governance.policy_profiles import GovernancePolicyProfile, load_policy_profiles
from mnemos.retrieval.base import SearchResult


def _hit(doc_id: str, score: float) -> SearchResult:
    e = Engram(id=doc_id, content=f"content {doc_id}")
    return SearchResult(engram=e, score=score, tier="qdrant")


def test_load_policy_profiles_from_json():
    raw = """
    {
      "default": {"min_score_threshold": 0.1, "freshness_half_life_days": 120},
      "tenant_finance": {"min_score_threshold": 0.85, "overlap_threshold": 0.4},
      "tenant_legal": {"freshness_half_life_days": 45}
    }
    """
    profiles = load_policy_profiles(
        raw_json=raw,
        base_min_score_threshold=0.0,
        base_freshness_half_life_days=180.0,
    )
    assert "default" in profiles
    assert "tenant_finance" in profiles
    assert "tenant_legal" in profiles
    assert profiles["default"].min_score_threshold == 0.1
    assert profiles["tenant_finance"].min_score_threshold == 0.85
    assert profiles["tenant_finance"].freshness_half_life_days == 120.0
    assert profiles["tenant_legal"].freshness_half_life_days == 45.0


def test_govern_uses_tenant_profile_threshold():
    strict = GovernancePolicyProfile(
        profile_id="tenant_strict",
        min_score_threshold=0.9,
        freshness_half_life_days=180.0,
    ).validate()
    gov = Governor(policy_profiles={"default": GovernancePolicyProfile("default"), "tenant_strict": strict})
    r_hi = _hit("hi", 0.95)
    r_lo = _hit("lo", 0.4)
    out, decisions, _ = gov.govern(
        [r_hi, r_lo],
        query="q",
        governance_mode="enforced",
        governance_profile="tenant_strict",
        top_k=10,
    )
    assert [r.engram.id for r in out] == ["hi"]
    decision_map = {d.engram_id: d for d in decisions}
    assert decision_map["lo"].veto_pass is False


def test_unknown_profile_raises():
    gov = Governor()
    r = _hit("x", 0.9)
    try:
        gov.govern([r], query="q", governance_mode="advisory", governance_profile="unknown")
    except ValueError as exc:
        assert "Unknown governance_profile" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown governance profile")
