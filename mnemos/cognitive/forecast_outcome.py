"""
ForecastOutcomeRecord — auditable forecast lifecycle for MNEMOS-Thinking.

MNEMOS already forecasts pulse pressure, volatility, and intent trajectories
via TimesFM and the PulseEngine.  This module adds the missing second half:
recording whether those forecasts were useful, accurate, ignored, or harmful.

Each forecast-driven cycle should produce a ForecastOutcomeRecord.  The record
starts with the prediction at creation time, and is resolved (optionally) when
the predicted condition either occurs or does not occur.

This converts MNEMOS forecasting from an isolated prediction feature into an
auditable learning loop compatible with the CoALA cognitive cycle contract.

Safety note
-----------
ForecastOutcomeRecord never records raw engram content, PII, or credentials.
future_policy_candidate is advisory — procedural memory is never mutated
automatically based on forecast outcomes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class ForecastOutcomeRecord:
    """
    Tracks the full lifecycle of a MNEMOS forecast: prediction, action taken,
    and eventual observed outcome.

    Created at forecast time; resolved when the predicted condition is either
    confirmed, refuted, or expires.

    Fields
    ------
    forecast_id         : Unique ID.
    source_signal       : Which signal drove the forecast:
                          pulse_p95 | volume_spike | volatility | intent_trajectory
    forecast_type       : What kind of event was predicted:
                          latency_breach | query_spike | contradiction_spike | intent_cluster
    predicted_condition : Human-readable description of the predicted condition.
    confidence_score    : Forecast confidence (0.0 – 1.0).
    time_horizon_minutes: Forecast lookahead window in minutes.
    selected_action     : Action taken based on this forecast:
                          pre_warm | conservative_routing | proactive_reconciliation |
                          advisory_emitted | no_op
    actual_outcome      : What actually happened (filled in at resolution time).
    forecast_error_delta: Numeric delta between predicted and observed values.
    operator_feedback   : Optional operator annotation at resolution time.
    learning_recommendation: What MNEMOS suggests learning from this outcome.
    future_policy_candidate: Dict describing a candidate policy change (advisory only;
                             never applied automatically).
    created_at          : ISO timestamp when the forecast was made.
    resolved_at         : ISO timestamp when the outcome was observed.
    """

    forecast_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    source_signal: str = ""
    """pulse_p95 | volume_spike | volatility | intent_trajectory"""

    forecast_type: str = ""
    """latency_breach | query_spike | contradiction_spike | intent_cluster"""

    predicted_condition: str = ""
    """Human-readable description of what was predicted."""

    confidence_score: float = 0.0
    time_horizon_minutes: int = 15

    selected_action: str = "no_op"
    """
    Action taken based on this forecast:
      pre_warm | conservative_routing | proactive_reconciliation |
      advisory_emitted | no_op
    """

    # Outcome (null until resolved)
    actual_outcome: Optional[str] = None
    """What actually happened.  Filled in at resolution time."""

    forecast_error_delta: Optional[float] = None
    """Numeric delta between predicted and observed metric (e.g. p95_ms)."""

    operator_feedback: Optional[str] = None
    """Optional operator annotation: accurate | inaccurate | irrelevant | harmful"""

    learning_recommendation: Optional[str] = None
    """Advisory recommendation for future policy (not applied automatically)."""

    future_policy_candidate: Optional[Dict[str, Any]] = None
    """
    Candidate policy change derived from this outcome.
    Advisory only — procedural memory is never mutated automatically.
    """

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: Optional[str] = None

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_pulse_advisory(
        cls,
        advisory: Dict[str, Any],
    ) -> "ForecastOutcomeRecord":
        """
        Create a ForecastOutcomeRecord from a PulseEngine advisory dict.

        Parameters
        ----------
        advisory : Dict returned by PulseEngine.advisory() or _forecast_advisory().
        """
        plan = advisory.get("suggested_plan", "standard")
        action = "conservative_routing" if plan == "conservative" else "advisory_emitted"
        return cls(
            source_signal="pulse_p95",
            forecast_type="latency_breach",
            predicted_condition=str(advisory.get("forecast_reason", "")),
            confidence_score=float(advisory.get("confidence_score", 0.0)),
            time_horizon_minutes=15,
            selected_action=action,
        )

    @classmethod
    def from_autonomous_warmup(
        cls,
        action: Dict[str, Any],
    ) -> "ForecastOutcomeRecord":
        """
        Create a ForecastOutcomeRecord from an autonomous warmup action dict
        as emitted by PulseEngine.evaluate_and_trigger().
        """
        return cls(
            source_signal="pulse_p95",
            forecast_type="query_spike",
            predicted_condition=str(action.get("reason", "")),
            confidence_score=float(action.get("confidence_score", 0.0)),
            time_horizon_minutes=int(action.get("horizon_minutes", 15)),
            selected_action="pre_warm",
        )

    @classmethod
    def from_proactive_reconciliation(
        cls,
        action: Dict[str, Any],
    ) -> "ForecastOutcomeRecord":
        """
        Create a ForecastOutcomeRecord from a proactive reconciliation action
        dict as emitted by the VolatilityEngine callback.
        """
        return cls(
            source_signal="volatility",
            forecast_type="contradiction_spike",
            predicted_condition=str(action.get("reason", "")),
            confidence_score=float(action.get("confidence_score", 0.0)),
            time_horizon_minutes=15,
            selected_action="proactive_reconciliation",
        )

    @classmethod
    def from_intent_trajectory(
        cls,
        forecast: Dict[str, Any],
    ) -> "ForecastOutcomeRecord":
        """
        Create a ForecastOutcomeRecord from an IntentEngine trajectory forecast.
        """
        return cls(
            source_signal="intent_trajectory",
            forecast_type="intent_cluster",
            predicted_condition=(
                f"Predicted next cluster: {forecast.get('predicted_cluster_id', 'unknown')}, "
                f"confidence={round(float(forecast.get('confidence', 0.0)), 3)}"
            ),
            confidence_score=float(forecast.get("confidence", 0.0)),
            time_horizon_minutes=forecast.get("horizon_steps", 3),
            selected_action="advisory_emitted",
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def resolve(
        self,
        *,
        actual_outcome: str,
        forecast_error_delta: Optional[float] = None,
        operator_feedback: Optional[str] = None,
        learning_recommendation: Optional[str] = None,
        future_policy_candidate: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record the observed outcome for this forecast.

        Should be called when the predicted condition is confirmed, refuted,
        or expires.  Idempotent (subsequent calls overwrite).
        """
        self.actual_outcome = actual_outcome
        self.resolved_at = datetime.now(timezone.utc).isoformat()
        if forecast_error_delta is not None:
            self.forecast_error_delta = forecast_error_delta
        if operator_feedback is not None:
            self.operator_feedback = operator_feedback
        if learning_recommendation is not None:
            self.learning_recommendation = learning_recommendation
        if future_policy_candidate is not None:
            self.future_policy_candidate = future_policy_candidate

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "source_signal": self.source_signal,
            "forecast_type": self.forecast_type,
            "predicted_condition": self.predicted_condition,
            "confidence_score": round(self.confidence_score, 4),
            "time_horizon_minutes": self.time_horizon_minutes,
            "selected_action": self.selected_action,
            "actual_outcome": self.actual_outcome,
            "forecast_error_delta": (
                round(self.forecast_error_delta, 4)
                if self.forecast_error_delta is not None
                else None
            ),
            "operator_feedback": self.operator_feedback,
            "learning_recommendation": self.learning_recommendation,
            "future_policy_candidate": self.future_policy_candidate,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }
