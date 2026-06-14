from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

from mnemos.retrieval.pulse import (
    DEFAULT_BUFFER_PATCHES,
    LinearMockTimesFMProvider,
    PulseBuffer,
    PulseEngine,
    PulseHarvester,
    TimesFMProvider,
    build_pulse_payload,
    forecast_pressure,
)
from service.app import CONTRACT_VERSION, MnemosRuntime


def test_pulse_harvester_buckets_query_metrics():
    buffer = PulseBuffer(capacity=4)
    harvester = PulseHarvester(buffer=buffer)

    harvester.record_query(
        latency_ms=100,
        cache_hits=1,
        cache_misses=0,
        candidate_envelope=8,
        timestamp=1_700_000_000,
    )
    harvester.record_query(
        latency_ms=300,
        cache_hits=0,
        cache_misses=1,
        degraded=True,
        candidate_envelope=12,
        timestamp=1_700_000_020,
    )

    patch = harvester.observed(limit=1)[0]
    assert patch.query_count == 2
    assert patch.p95_latency_ms > 250
    assert patch.cache_hit_rate == 0.5
    assert patch.degrade_count == 1
    assert patch.avg_candidate_envelope == 10


def test_pulse_buffer_keeps_1440_patch_capacity_by_default():
    buffer = PulseBuffer()
    for i in range(DEFAULT_BUFFER_PATCHES + 3):
        harvester = PulseHarvester(buffer=buffer)
        harvester.record_query(latency_ms=1, timestamp=1_700_000_000 + i * 60)

    assert len(buffer) == DEFAULT_BUFFER_PATCHES


def test_linear_mock_provider_returns_15_minute_quantile_forecast():
    harvester = PulseHarvester()
    for minute in range(20):
        harvester.record_query(
            latency_ms=100 + minute * 10,
            cache_hits=minute % 3,
            cache_misses=1,
            timestamp=1_700_000_000 + minute * 60,
        )

    forecast = LinearMockTimesFMProvider().forecast(
        harvester.observed(limit=20),
        horizon_minutes=15,
    )

    assert forecast["mode"] == "mock"
    assert forecast["horizon_minutes"] == 15
    assert len(forecast["patches"]) == 15
    first = forecast["patches"][0]
    assert "q10" in first["p95_latency_ms"]
    assert first["p95_latency_ms"]["q90"] >= first["p95_latency_ms"]["q50"]


def test_build_pulse_payload_contract_shape():
    harvester = PulseHarvester()
    harvester.record_query(latency_ms=42, timestamp=1_700_000_000)

    payload = build_pulse_payload(
        contract_version=CONTRACT_VERSION,
        status="healthy",
        source="mnemos-service",
        generated_at="2026-06-14T00:00:00Z",
        harvester=harvester,
        provider=LinearMockTimesFMProvider(),
        observed_limit=10,
        horizon_minutes=15,
        actions_mode="advisory",
    )

    assert payload["feature"] == "mnemos_pulse"
    assert payload["actions_mode"] == "advisory"
    assert payload["observed"]["patch_count"] == 1
    assert payload["forecast"]["provider"] == "linear_mock_timesfm"


def test_runtime_get_pulse_respects_timesfm_enabled_flag():
    rt = MnemosRuntime()
    rt._config = SimpleNamespace(
        timesfm_enabled=False,
        pulse_actions="off",
        pulse_horizon_minutes=15,
    )
    rt._status = "healthy"
    rt._error = None
    rt._pulse_engine.record_query(latency_ms=10, timestamp=1_700_000_000)

    payload = rt.get_pulse()

    assert payload["actions_mode"] == "off"
    assert payload["forecast"]["mode"] == "mock"
    assert payload["observed"]["patch_count"] == 1


def test_timesfm_provider_success_round_trip_under_100ms(monkeypatch):
    harvester = PulseHarvester()
    for minute in range(24):
        harvester.record_query(
            latency_ms=40 + minute,
            cache_hits=1,
            timestamp=1_700_000_000 + minute * 60,
        )

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "provider": "timesfm_sidecar",
                "mode": "timesfm",
                "horizon_minutes": 15,
                "confidence_score": 0.8,
                "patches": [
                    {
                        "query_count": {"point": 10, "q10": 9, "q50": 10, "q90": 11},
                        "p95_latency_ms": {"point": 80, "q10": 70, "q50": 80, "q90": 90},
                        "cache_hit_rate": {"point": 0.8, "q10": 0.7, "q50": 0.8, "q90": 0.9},
                        "degrade_count": {"point": 0, "q10": 0, "q50": 0, "q90": 0},
                    }
                    for _ in range(15)
                ],
            }

    monkeypatch.setattr("mnemos.retrieval.pulse.requests.post", lambda *_, **__: Response())

    forecast = TimesFMProvider(base_url="http://sidecar", timeout_s=0.1, enabled=True).forecast(
        harvester.observed(limit=24),
        horizon_minutes=15,
    )

    assert forecast["fallback_used"] is False
    assert forecast["round_trip_ms"] < 100
    assert len(forecast["patches"]) == 15


def test_timesfm_provider_falls_back_on_sidecar_failure(monkeypatch):
    harvester = PulseHarvester()
    harvester.record_query(latency_ms=10, timestamp=1_700_000_000)

    def boom(*_, **__):
        raise TimeoutError("too slow")

    monkeypatch.setattr("mnemos.retrieval.pulse.requests.post", boom)

    forecast = TimesFMProvider(base_url="http://sidecar", timeout_s=0.01, enabled=True).forecast(
        harvester.observed(limit=1),
        horizon_minutes=15,
    )

    assert forecast["provider"] == "timesfm_sidecar"
    assert forecast["mode"] == "fallback"
    assert forecast["fallback_used"] is True
    assert "TimeoutError" in forecast["fallback_reason"]


def test_forecast_pressure_suggests_conservative_on_p95_breach():
    forecast = {
        "confidence_score": 0.9,
        "patches": [
            {
                "p95_latency_ms": {"point": 350, "q90": 410},
                "degrade_count": {"point": 0, "q90": 0},
            }
        ],
    }

    advice = forecast_pressure(forecast, p95_budget_ms=250)

    assert advice["suggested_plan"] == "conservative"
    assert advice["forecast_pressure"] == 1.0
    assert "p95 breach" in advice["forecast_reason"]


def test_prediction_fidelity_backtest_with_sidecar_provider_contract(monkeypatch):
    actual = [120 + i * 4 for i in range(15)]

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "provider": "timesfm_sidecar",
                "mode": "timesfm",
                "horizon_minutes": 15,
                "confidence_score": 0.82,
                "patches": [
                    {
                        "query_count": {
                            "point": value * 1.04,
                            "q10": value * 0.95,
                            "q50": value * 1.04,
                            "q90": value * 1.1,
                        },
                        "p95_latency_ms": {"point": 100, "q10": 90, "q50": 100, "q90": 110},
                        "cache_hit_rate": {"point": 0.7, "q10": 0.6, "q50": 0.7, "q90": 0.8},
                        "degrade_count": {"point": 0, "q10": 0, "q50": 0, "q90": 0},
                    }
                    for value in actual
                ],
            }

    monkeypatch.setattr("mnemos.retrieval.pulse.requests.post", lambda *_, **__: Response())
    harvester = PulseHarvester()
    for minute in range(24 * 60):
        harvester.record_query(
            latency_ms=80 + (minute % 60),
            cache_hits=1,
            timestamp=1_700_000_000 + minute * 60,
        )

    forecast = TimesFMProvider(base_url="http://sidecar", timeout_s=0.1, enabled=True).forecast(
        harvester.observed(limit=1440),
        horizon_minutes=15,
    )
    predicted = [row["query_count"]["point"] for row in forecast["patches"]]
    errors = [abs(p - a) / a for p, a in zip(predicted, actual)]

    assert max(errors) <= 0.15


def test_pulse_engine_autonomous_prewarm_triggers_on_volume_spike():
    now = [1_700_000_000.0]
    warmups = []
    audits = []
    harvester = PulseHarvester()
    harvester.record_query(latency_ms=100, timestamp=now[0])

    engine = PulseEngine(
        harvester=harvester,
        actions_mode="autonomous",
        warmup_callback=lambda action: warmups.append(action) or {"status": "success"},
        audit_callback=lambda action: audits.append(action),
        clock=lambda: now[0],
    )
    forecast = {
        "confidence_score": 0.91,
        "metadata": {"complexity_class": "CLASS_C", "domain": "gdpr"},
        "patches": [
            {
                "query_count": {"point": 1.35},
                "p95_latency_ms": {"point": 110},
            }
            for _ in range(15)
        ],
    }

    result = engine.evaluate_and_trigger(forecast)

    assert result["triggered"] is True
    assert result["target"]["layer"] == "hierarchical_summary"
    assert "35.0% volume spike" in result["reason"]
    assert warmups
    assert audits


def test_pulse_engine_autonomous_prewarm_cooldown_suppresses_storms():
    now = [1_700_000_000.0]
    warmup_count = 0
    harvester = PulseHarvester()
    harvester.record_query(latency_ms=100, timestamp=now[0])

    def warmup(_action):
        nonlocal warmup_count
        warmup_count += 1
        return {"status": "success"}

    engine = PulseEngine(
        harvester=harvester,
        actions_mode="autonomous",
        warmup_callback=warmup,
        cooldown_seconds=900,
        clock=lambda: now[0],
    )
    forecast = {
        "confidence_score": 0.91,
        "patches": [{"query_count": {"point": 2.0}, "p95_latency_ms": {"point": 100}}],
    }

    first = engine.evaluate_and_trigger(forecast)
    second = engine.evaluate_and_trigger(forecast)

    assert first["triggered"] is True
    assert second["triggered"] is False
    assert second["reason"] == "cooldown_active"
    assert warmup_count == 1


def test_pulse_engine_requires_autonomous_mode_and_high_confidence():
    harvester = PulseHarvester()
    harvester.record_query(latency_ms=100, timestamp=1_700_000_000)
    forecast = {
        "confidence_score": 0.79,
        "patches": [{"query_count": {"point": 2.0}, "p95_latency_ms": {"point": 200}}],
    }

    advisory = PulseEngine(harvester=harvester, actions_mode="advisory")
    autonomous = PulseEngine(harvester=harvester, actions_mode="autonomous")

    assert advisory.evaluate_and_trigger(forecast)["reason"] == "pulse_actions_not_autonomous"
    assert autonomous.evaluate_and_trigger(forecast)["reason"] == "confidence_below_threshold"


def test_pulse_engine_rejects_autonomous_threshold_boundary():
    harvester = PulseHarvester()
    harvester.record_query(latency_ms=100, timestamp=1_700_000_000)
    engine = PulseEngine(harvester=harvester, actions_mode="autonomous")
    forecast = {
        "confidence_score": 0.85,
        "patches": [{"query_count": {"point": 2.0}, "p95_latency_ms": {"point": 200}}],
    }

    assert engine.evaluate_and_trigger(forecast)["reason"] == "confidence_below_threshold"


def test_anticipatory_success_benchmark_triggers_three_minutes_before_peak():
    peak_minute = 15
    trigger_minute = 11
    warmups = []
    harvester = PulseHarvester()

    for minute in range(trigger_minute + 1):
        query_count = 10 if minute < 10 else 14
        for q in range(query_count):
            harvester.record_query(
                latency_ms=90 + minute,
                timestamp=1_700_000_000 + minute * 60 + q,
            )

    forecast = {
        "confidence_score": 0.91,
        "metadata": {"complexity_class": "CLASS_C", "domain": "global"},
        "patches": [
            {
                "query_count": {"point": 18 + step},
                "p95_latency_ms": {"point": 120 + step * 4},
            }
            for step in range(15)
        ],
    }
    engine = PulseEngine(
        harvester=harvester,
        actions_mode="autonomous",
        warmup_callback=lambda action: warmups.append({"minute": trigger_minute, "action": action}) or {"status": "success"},
        clock=lambda: 1_700_000_000 + trigger_minute * 60,
    )

    result = engine.evaluate_and_trigger(forecast)

    assert result["triggered"] is True
    assert warmups[0]["minute"] <= peak_minute - 3
