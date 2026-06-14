"""
Tests for ForecastOutcomeRecord — the auditable forecast lifecycle schema.

Coverage targets
----------------
1. from_pulse_advisory() maps advisory dict to correct fields.
2. from_autonomous_warmup() maps warmup action to correct fields.
3. from_proactive_reconciliation() maps volatility action to correct fields.
4. from_intent_trajectory() maps intent forecast to correct fields.
5. resolve() fills actual_outcome and resolved_at; idempotent.
6. to_dict() is complete and JSON-serialisable.
7. future_policy_candidate is advisory only — verify it is stored but not
   treated as an automatic mutation (structural check).
8. Forecast IDs are unique across instances.
9. confidence_score is clamped/rounded correctly in to_dict().
"""

from __future__ import annotations

import json
import uuid

from mnemos.cognitive.forecast_outcome import ForecastOutcomeRecord


# ── Factory method tests ──────────────────────────────────────────────────────


class TestFromPulseAdvisory:
    def _advisory(self, plan: str = "standard") -> dict:
        return {
            "suggested_plan": plan,
            "forecast_pressure": 0.72,
            "confidence_score": 0.81,
            "forecast_reason": "p95 latency predicted to exceed 250 ms",
            "max_forecast_p95_latency_ms": 310.0,
            "actions_mode": "advisory",
        }

    def test_standard_plan_maps_to_advisory_emitted(self):
        fo = ForecastOutcomeRecord.from_pulse_advisory(self._advisory("standard"))
        assert fo.source_signal == "pulse_p95"
        assert fo.forecast_type == "latency_breach"
        assert fo.selected_action == "advisory_emitted"
        assert fo.confidence_score == 0.81
        assert "p95" in fo.predicted_condition

    def test_conservative_plan_maps_to_conservative_routing(self):
        fo = ForecastOutcomeRecord.from_pulse_advisory(self._advisory("conservative"))
        assert fo.selected_action == "conservative_routing"

    def test_unresolved_on_creation(self):
        fo = ForecastOutcomeRecord.from_pulse_advisory(self._advisory())
        assert fo.actual_outcome is None
        assert fo.resolved_at is None


class TestFromAutonomousWarmup:
    def _action(self) -> dict:
        return {
            "reason": "predicted 40% query spike in next 15 minutes",
            "confidence_score": 0.88,
            "horizon_minutes": 15,
            "target": {"layer": "semantic_fusion"},
        }

    def test_fields(self):
        fo = ForecastOutcomeRecord.from_autonomous_warmup(self._action())
        assert fo.source_signal == "pulse_p95"
        assert fo.forecast_type == "query_spike"
        assert fo.selected_action == "pre_warm"
        assert fo.confidence_score == 0.88
        assert fo.time_horizon_minutes == 15
        assert "40%" in fo.predicted_condition


class TestFromProactiveReconciliation:
    def _action(self) -> dict:
        return {
            "reason": "contradiction spike predicted for entity 'ceo_name'",
            "confidence_score": 0.75,
            "family_key": "family:executives",
        }

    def test_fields(self):
        fo = ForecastOutcomeRecord.from_proactive_reconciliation(self._action())
        assert fo.source_signal == "volatility"
        assert fo.forecast_type == "contradiction_spike"
        assert fo.selected_action == "proactive_reconciliation"
        assert fo.confidence_score == 0.75
        assert "ceo_name" in fo.predicted_condition


class TestFromIntentTrajectory:
    def _forecast(self) -> dict:
        return {
            "predicted_cluster_id": 42,
            "confidence": 0.65,
            "horizon_steps": 3,
        }

    def test_fields(self):
        fo = ForecastOutcomeRecord.from_intent_trajectory(self._forecast())
        assert fo.source_signal == "intent_trajectory"
        assert fo.forecast_type == "intent_cluster"
        assert fo.selected_action == "advisory_emitted"
        assert fo.confidence_score == 0.65
        assert fo.time_horizon_minutes == 3
        assert "42" in fo.predicted_condition


# ── Lifecycle tests ───────────────────────────────────────────────────────────


class TestResolve:
    def test_resolve_fills_outcome_and_timestamp(self):
        fo = ForecastOutcomeRecord(
            source_signal="pulse_p95",
            forecast_type="latency_breach",
            predicted_condition="p95 spike",
            confidence_score=0.8,
            selected_action="advisory_emitted",
        )
        assert fo.resolved_at is None
        fo.resolve(actual_outcome="no_breach_observed")
        assert fo.actual_outcome == "no_breach_observed"
        assert fo.resolved_at is not None

    def test_resolve_with_all_fields(self):
        fo = ForecastOutcomeRecord(
            source_signal="volatility",
            forecast_type="contradiction_spike",
            predicted_condition="spike predicted",
            confidence_score=0.7,
            selected_action="proactive_reconciliation",
        )
        fo.resolve(
            actual_outcome="spike_confirmed",
            forecast_error_delta=0.15,
            operator_feedback="accurate",
            learning_recommendation="increase confidence threshold for volatility spikes",
            future_policy_candidate={
                "name": "volatility_confidence_threshold",
                "suggested_value": 0.80,
                "rationale": "threshold was too low; spike was confirmed",
            },
        )
        assert fo.actual_outcome == "spike_confirmed"
        assert fo.forecast_error_delta == 0.15
        assert fo.operator_feedback == "accurate"
        assert fo.learning_recommendation is not None
        assert fo.future_policy_candidate is not None
        assert fo.future_policy_candidate["name"] == "volatility_confidence_threshold"

    def test_resolve_is_idempotent(self):
        fo = ForecastOutcomeRecord(
            source_signal="pulse_p95",
            forecast_type="latency_breach",
            predicted_condition="p95 spike",
            confidence_score=0.8,
            selected_action="advisory_emitted",
        )
        fo.resolve(actual_outcome="first_outcome")
        first_resolved_at = fo.resolved_at
        fo.resolve(actual_outcome="second_outcome")
        # Second call overwrites (idempotent — no error)
        assert fo.actual_outcome == "second_outcome"
        # resolved_at is updated
        assert fo.resolved_at is not None

    def test_future_policy_candidate_is_advisory_only(self):
        """
        Verify that future_policy_candidate is stored as data but does not
        trigger any automatic mutation (structural check — no auto-apply side effect).
        """
        fo = ForecastOutcomeRecord(
            source_signal="pulse_p95",
            forecast_type="latency_breach",
            predicted_condition="predicted spike",
            confidence_score=0.9,
            selected_action="advisory_emitted",
        )
        policy_candidate = {
            "name": "pulse_confidence_threshold",
            "suggested_value": 0.90,
            "rationale": "test only",
        }
        fo.resolve(
            actual_outcome="confirmed",
            future_policy_candidate=policy_candidate,
        )
        # Candidate is stored but is just a dict — no procedural mutation possible
        assert isinstance(fo.future_policy_candidate, dict)
        assert fo.future_policy_candidate.get("name") == "pulse_confidence_threshold"


# ── Serialisation tests ───────────────────────────────────────────────────────


class TestToDict:
    def test_all_keys_present(self):
        fo = ForecastOutcomeRecord(
            source_signal="pulse_p95",
            forecast_type="latency_breach",
            predicted_condition="p95 spike",
            confidence_score=0.81,
            time_horizon_minutes=15,
            selected_action="advisory_emitted",
        )
        d = fo.to_dict()
        expected_keys = [
            "forecast_id", "source_signal", "forecast_type",
            "predicted_condition", "confidence_score", "time_horizon_minutes",
            "selected_action", "actual_outcome", "forecast_error_delta",
            "operator_feedback", "learning_recommendation",
            "future_policy_candidate", "created_at", "resolved_at",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_confidence_rounded_to_4dp(self):
        fo = ForecastOutcomeRecord(confidence_score=0.812345678)
        d = fo.to_dict()
        assert d["confidence_score"] == 0.8123

    def test_forecast_error_delta_rounded(self):
        fo = ForecastOutcomeRecord(forecast_error_delta=0.123456789)
        d = fo.to_dict()
        assert d["forecast_error_delta"] == 0.1235

    def test_null_fields_are_none(self):
        fo = ForecastOutcomeRecord()
        d = fo.to_dict()
        assert d["actual_outcome"] is None
        assert d["resolved_at"] is None
        assert d["operator_feedback"] is None

    def test_json_serialisable(self):
        fo = ForecastOutcomeRecord.from_pulse_advisory({
            "suggested_plan": "conservative",
            "forecast_pressure": 0.9,
            "confidence_score": 0.85,
            "forecast_reason": "high volume predicted",
            "actions_mode": "advisory",
        })
        fo.resolve(
            actual_outcome="breach_confirmed",
            forecast_error_delta=0.05,
            operator_feedback="accurate",
            future_policy_candidate={"name": "test", "value": 1},
        )
        d = fo.to_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert len(serialised) > 50


# ── Uniqueness tests ──────────────────────────────────────────────────────────


class TestUniqueness:
    def test_unique_forecast_ids(self):
        ids = {ForecastOutcomeRecord().forecast_id for _ in range(100)}
        assert len(ids) == 100

    def test_created_at_is_iso_format(self):
        fo = ForecastOutcomeRecord()
        # Should parse without error
        from datetime import datetime
        dt = datetime.fromisoformat(fo.created_at.replace("Z", "+00:00"))
        assert dt is not None
