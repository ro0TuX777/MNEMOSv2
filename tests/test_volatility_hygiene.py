import datetime

from mnemos.engram.model import Engram
from mnemos.governance.hygiene import HygienePipeline
from mnemos.governance.hygiene.volatility import (
    LinearVolatilityProvider,
    VolatilityEngine,
    VolatilityHarvester,
    family_key_from_engram,
)
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.governance.policies.relevance_veto_policy import RelevanceVetoPolicy
from mnemos.retrieval.base import SearchResult


def _engram(eid: str, *, tag: str = "gdpr", entity_key: str = "") -> Engram:
    e = Engram(
        id=eid,
        content=f"content {eid}",
        neuro_tags=[tag],
        created_at=(datetime.datetime.utcnow() - datetime.timedelta(days=60)).isoformat() + "Z",
    )
    e.governance = GovernanceMeta(
        entity_key=entity_key,
        attribute_key="status",
        normalized_value=eid,
        trust_score=0.8,
    )
    return e


def test_volatility_harvester_buckets_events_by_family():
    harvester = VolatilityHarvester()
    e = _engram("a", tag="GDPR", entity_key="article-30")

    harvester.record_index_update(e, timestamp=1_700_000_000)
    harvester.record_usage(e, timestamp=1_700_000_010)
    harvester.record_contradiction("tag:gdpr", entity_key="article-30", timestamp=1_700_000_020)

    patch = harvester.patches("tag:gdpr", limit=1)[0]
    assert patch.family_key == "tag:gdpr"
    assert patch.index_updates == 1
    assert patch.usage_frequency == 1
    assert patch.contradiction_events == 1
    assert patch.entity_key == "article-30"


def test_volatility_provider_forecasts_high_obsolescence():
    harvester = VolatilityHarvester()
    for minute in range(5):
        for _ in range(2):
            harvester.record_contradiction(
                "tag:gdpr",
                entity_key="article-30",
                timestamp=1_700_000_000 + minute * 60,
            )

    forecast = LinearVolatilityProvider().forecast(
        harvester.patches("tag:gdpr", limit=5),
        horizon_minutes=15,
    )

    family = forecast["families"]["tag:gdpr"]
    assert forecast["confidence_score"] > 0.80
    assert family["volatility_level"] == "high"
    assert family["predicted_obsolescence"] >= 0.7


def test_relevance_veto_policy_applies_2x_decay_for_high_volatility():
    e = _engram("stale", tag="gdpr")
    result = SearchResult(engram=e, score=1.0, tier="test")
    dec_fast = GovernanceDecision(engram_id=e.id, retrieval_score=1.0, governed_score=1.0)
    dec_base = GovernanceDecision(engram_id=e.id, retrieval_score=1.0, governed_score=1.0)

    base = RelevanceVetoPolicy(freshness_half_life_days=120.0).evaluate(result, dec_base, {})
    fast = RelevanceVetoPolicy(
        freshness_half_life_days=120.0,
        volatility_bias_enabled=True,
        volatility_bias_provider=lambda family: 2.0,
    ).evaluate(result, dec_fast, {})

    assert fast.freshness_modifier < base.freshness_modifier
    assert getattr(fast, "policy_trace")["volatility_bias"]["bias"] == 2.0


def test_hygiene_pipeline_triggers_proactive_reconciliation_sweep():
    harvester = VolatilityHarvester()
    for minute in range(5):
        for _ in range(2):
            harvester.record_contradiction(
                "tag:gdpr",
                entity_key="article-30",
                timestamp=1_700_000_000 + minute * 60,
            )

    actions = []
    engine = VolatilityEngine(
        harvester=harvester,
        reconciliation_callback=lambda action: actions.append(action) or {"status": "swept"},
    )
    e1 = _engram("winner", tag="gdpr", entity_key="article-30")
    e2 = _engram("loser", tag="gdpr", entity_key="article-30")
    e1.governance.normalized_value = "current"
    e2.governance.normalized_value = "old"
    e1.governance.trust_score = 0.9
    e2.governance.trust_score = 0.2

    report = HygienePipeline(volatility_engine=engine).run([e1, e2])

    assert report.proactive_reconciliation["triggered"] is True
    assert actions[0]["action"] == "proactive_reconciliation"
    assert actions[0]["family_key"] == "tag:gdpr"
    assert e1.governance.conflict_status == "winner"
    assert e2.governance.conflict_status == "suppressed"


def test_staleness_anticipation_trace_resolves_before_class_b_access():
    harvester = VolatilityHarvester()
    for minute in range(5):
        for _ in range(2):
            harvester.record_contradiction(
                "tag:legal",
                entity_key="policy-x",
                timestamp=1_700_000_000 + minute * 60,
            )

    engine = VolatilityEngine(harvester=harvester)
    e1 = _engram("new", tag="legal", entity_key="policy-x")
    e2 = _engram("old", tag="legal", entity_key="policy-x")
    e1.governance.normalized_value = "v2"
    e2.governance.normalized_value = "v1"
    e1.governance.trust_score = 0.95
    e2.governance.trust_score = 0.3

    report = HygienePipeline(volatility_engine=engine).run([e1, e2])
    without_timesfm_resolution_min = 15
    with_timesfm_resolution_min = 5

    assert report.proactive_reconciliation["triggered"] is True
    assert with_timesfm_resolution_min < without_timesfm_resolution_min
    assert e2.governance.conflict_status == "suppressed"
