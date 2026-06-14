"""
MNEMOS Pulse telemetry.

Buckets live operational events into one-minute patches and exposes a compact
forecast contract. The first provider is intentionally simple so the telemetry
shape can stabilize before TimesFM is attached as a sidecar.
"""

from __future__ import annotations

import math
import os
import statistics
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

import requests


PULSE_BUCKET_SECONDS = 60
DEFAULT_BUFFER_PATCHES = 1440
DEFAULT_HORIZON_MINUTES = 15


def _floor_bucket(ts: float, bucket_seconds: int = PULSE_BUCKET_SECONDS) -> int:
    return int(ts // bucket_seconds) * bucket_seconds


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class PulsePatch:
    """One minute of MNEMOS operational rhythm."""

    bucket_start: int
    query_count: int = 0
    p95_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    degrade_count: int = 0
    avg_candidate_envelope: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_start": self.bucket_start,
            "bucket_start_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.bucket_start)
            ),
            "query_count": int(self.query_count),
            "p95_latency_ms": round(float(self.p95_latency_ms), 3),
            "cache_hit_rate": round(float(self.cache_hit_rate), 4),
            "degrade_count": int(self.degrade_count),
            "avg_candidate_envelope": round(float(self.avg_candidate_envelope), 3),
        }


@dataclass
class _PulseAccumulator:
    bucket_start: int
    latencies_ms: List[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    degrade_count: int = 0
    candidate_envelopes: List[int] = field(default_factory=list)

    def add(
        self,
        *,
        latency_ms: float,
        cache_hits: int = 0,
        cache_misses: int = 0,
        degrade_count: int = 0,
        candidate_envelope: int = 0,
    ) -> None:
        self.latencies_ms.append(max(0.0, float(latency_ms)))
        self.cache_hits += max(0, int(cache_hits))
        self.cache_misses += max(0, int(cache_misses))
        self.degrade_count += max(0, int(degrade_count))
        if candidate_envelope > 0:
            self.candidate_envelopes.append(int(candidate_envelope))

    def to_patch(self) -> PulsePatch:
        query_count = len(self.latencies_ms)
        cache_total = self.cache_hits + self.cache_misses
        return PulsePatch(
            bucket_start=self.bucket_start,
            query_count=query_count,
            p95_latency_ms=_percentile(self.latencies_ms, 0.95),
            cache_hit_rate=(self.cache_hits / cache_total) if cache_total else 0.0,
            degrade_count=self.degrade_count,
            avg_candidate_envelope=(
                statistics.fmean(self.candidate_envelopes)
                if self.candidate_envelopes
                else 0.0
            ),
        )


class PulseBuffer:
    """Thread-safe circular buffer of PulsePatch values."""

    def __init__(self, capacity: int = DEFAULT_BUFFER_PATCHES) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._patches: "OrderedDict[int, PulsePatch]" = OrderedDict()
        self._lock = threading.RLock()

    def upsert(self, patch: PulsePatch) -> None:
        with self._lock:
            if patch.bucket_start in self._patches:
                del self._patches[patch.bucket_start]
            self._patches[patch.bucket_start] = patch
            while len(self._patches) > self.capacity:
                self._patches.popitem(last=False)

    def latest(self, limit: Optional[int] = None) -> List[PulsePatch]:
        with self._lock:
            values = list(self._patches.values())
        if limit is not None:
            values = values[-max(0, int(limit)) :]
        return values

    def __len__(self) -> int:
        with self._lock:
            return len(self._patches)


class PulseHarvester:
    """Collects live events into one-minute PulsePatches."""

    def __init__(
        self,
        *,
        buffer: Optional[PulseBuffer] = None,
        bucket_seconds: int = PULSE_BUCKET_SECONDS,
        clock: Any = time.time,
    ) -> None:
        self.buffer = buffer if buffer is not None else PulseBuffer()
        self.bucket_seconds = int(bucket_seconds)
        self._clock = clock
        self._accumulators: Dict[int, _PulseAccumulator] = {}
        self._lock = threading.RLock()

    def record_query(
        self,
        *,
        latency_ms: float,
        cache_hits: int = 0,
        cache_misses: int = 0,
        degraded: bool = False,
        candidate_envelope: int = 0,
        timestamp: Optional[float] = None,
    ) -> PulsePatch:
        ts = self._clock() if timestamp is None else float(timestamp)
        bucket = _floor_bucket(ts, self.bucket_seconds)
        with self._lock:
            acc = self._accumulators.get(bucket)
            if acc is None:
                acc = _PulseAccumulator(bucket_start=bucket)
                self._accumulators[bucket] = acc
            acc.add(
                latency_ms=latency_ms,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                degrade_count=1 if degraded else 0,
                candidate_envelope=candidate_envelope,
            )
            patch = acc.to_patch()
            self.buffer.upsert(patch)
            self._prune_locked(bucket)
            return patch

    def _prune_locked(self, newest_bucket: int) -> None:
        oldest_allowed = newest_bucket - (self.buffer.capacity - 1) * self.bucket_seconds
        stale = [bucket for bucket in self._accumulators if bucket < oldest_allowed]
        for bucket in stale:
            del self._accumulators[bucket]

    def observed(self, limit: int = 60) -> List[PulsePatch]:
        return self.buffer.latest(limit=limit)


class PulseForecastProvider(Protocol):
    def forecast(
        self, history: Iterable[PulsePatch], *, horizon_minutes: int
    ) -> Dict[str, Any]:
        ...


class LinearMockTimesFMProvider:
    """
    Stabilization provider for Sprint 0/1.

    It uses a simple least-squares line per metric and emits quantile bands so
    downstream routing can already learn to reason over confidence.
    """

    provider_name = "linear_mock_timesfm"

    def forecast(
        self, history: Iterable[PulsePatch], *, horizon_minutes: int
    ) -> Dict[str, Any]:
        patches = list(history)
        horizon = max(1, int(horizon_minutes))
        metrics = {
            "query_count": [float(p.query_count) for p in patches],
            "p95_latency_ms": [float(p.p95_latency_ms) for p in patches],
            "cache_hit_rate": [float(p.cache_hit_rate) for p in patches],
            "degrade_count": [float(p.degrade_count) for p in patches],
        }
        start = patches[-1].bucket_start + PULSE_BUCKET_SECONDS if patches else _floor_bucket(time.time())
        forecast_rows = []
        for step in range(1, horizon + 1):
            row: Dict[str, Any] = {
                "bucket_start": start + (step - 1) * PULSE_BUCKET_SECONDS,
            }
            row["bucket_start_iso"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(row["bucket_start"])
            )
            for name, values in metrics.items():
                point, spread = self._project(values, step)
                if name in {"query_count", "degrade_count"}:
                    point = max(0.0, point)
                if name == "cache_hit_rate":
                    point = min(1.0, max(0.0, point))
                    spread = min(spread, 0.5)
                row[name] = {
                    "point": round(point, 4),
                    "q10": round(max(0.0, point - spread), 4),
                    "q50": round(point, 4),
                    "q90": round(point + spread, 4),
                }
            forecast_rows.append(row)

        return {
            "provider": self.provider_name,
            "mode": "mock",
            "horizon_minutes": horizon,
            "confidence_score": self._confidence(patches),
            "targets": list(metrics.keys()),
            "patches": forecast_rows,
        }

    @staticmethod
    def _project(values: List[float], step: int) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        if len(values) == 1:
            return values[-1], max(abs(values[-1]) * 0.1, 0.1)
        xs = list(range(len(values)))
        x_mean = statistics.fmean(xs)
        y_mean = statistics.fmean(values)
        denom = sum((x - x_mean) ** 2 for x in xs) or 1.0
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom
        intercept = y_mean - slope * x_mean
        point = intercept + slope * (len(values) - 1 + step)
        residuals = [abs(y - (intercept + slope * x)) for x, y in zip(xs, values)]
        spread = _percentile(residuals, 0.9) if residuals else 0.0
        return point, max(spread, 0.05)

    @staticmethod
    def _confidence(patches: List[PulsePatch]) -> float:
        if len(patches) >= 30:
            return 0.72
        if len(patches) >= 15:
            return 0.58
        if len(patches) >= 5:
            return 0.42
        return 0.2

    def forecast_next_cluster(self, sequence: List[int], *, horizon_steps: int = 3) -> Dict[str, Any]:
        if len(sequence) < 2:
            predicted = sequence[-1] if sequence else 0
            confidence = 0.2
        else:
            deltas = [b - a for a, b in zip(sequence[:-1], sequence[1:])]
            step = round(sum(deltas[-3:]) / min(3, len(deltas)))
            predicted = int(sequence[-1] + step * max(1, int(horizon_steps)))
            confidence = 0.88 if len(set(deltas[-3:])) == 1 else 0.62
        return {
            "provider": self.provider_name,
            "mode": "mock",
            "history": list(sequence),
            "predicted_cluster_id": predicted,
            "horizon_steps": int(horizon_steps),
            "confidence_score": confidence,
        }


class TimesFMProvider:
    """HTTP provider for the mnemos-timesfm sidecar with mock fallback."""

    provider_name = "timesfm_sidecar"

    def __init__(
        self,
        *,
        base_url: str = "http://mnemos-timesfm:8711",
        timeout_s: float = 0.08,
        enabled: Optional[bool] = None,
        fallback: Optional[PulseForecastProvider] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = max(0.01, float(timeout_s))
        self.enabled = enabled
        self.fallback = fallback or LinearMockTimesFMProvider()

    def forecast(
        self, history: Iterable[PulsePatch], *, horizon_minutes: int
    ) -> Dict[str, Any]:
        patches = list(history)
        enabled = self.enabled
        if enabled is None:
            enabled = os.getenv("MNEMOS_TIMESFM_ENABLED", "true").strip().lower() in {
                "true",
                "1",
                "yes",
            }
        if not enabled:
            return self._fallback(
                patches,
                horizon_minutes=horizon_minutes,
                reason="timesfm_disabled",
            )

        try:
            started = time.perf_counter()
            response = requests.post(
                f"{self.base_url}/forecast",
                json={
                    "horizon_minutes": int(horizon_minutes),
                    "patches": [patch.to_dict() for patch in patches],
                },
                timeout=self.timeout_s,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or "patches" not in data:
                raise ValueError("TimesFM sidecar response missing patches")
            data.setdefault("provider", self.provider_name)
            data.setdefault("mode", "timesfm")
            data["round_trip_ms"] = round(elapsed_ms, 3)
            data["fallback_used"] = False
            return data
        except Exception as exc:
            return self._fallback(
                patches,
                horizon_minutes=horizon_minutes,
                reason=f"{type(exc).__name__}: {exc}",
            )

    def _fallback(
        self,
        patches: List[PulsePatch],
        *,
        horizon_minutes: int,
        reason: str,
    ) -> Dict[str, Any]:
        data = self.fallback.forecast(patches, horizon_minutes=horizon_minutes)
        data["provider"] = self.provider_name
        data["mode"] = "fallback"
        data["fallback_provider"] = getattr(self.fallback, "provider_name", "fallback")
        data["fallback_used"] = True
        data["fallback_reason"] = reason
        return data

    def forecast_next_cluster(self, sequence: List[int], *, horizon_steps: int = 3) -> Dict[str, Any]:
        enabled = self.enabled
        if enabled is None:
            enabled = os.getenv("MNEMOS_TIMESFM_ENABLED", "true").strip().lower() in {
                "true",
                "1",
                "yes",
            }
        if not enabled:
            return self._cluster_fallback(sequence, horizon_steps=horizon_steps, reason="timesfm_disabled")
        try:
            started = time.perf_counter()
            response = requests.post(
                f"{self.base_url}/forecast_intent",
                json={"sequence": list(sequence), "horizon_steps": int(horizon_steps)},
                timeout=self.timeout_s,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or "predicted_cluster_id" not in data:
                raise ValueError("TimesFM sidecar response missing predicted_cluster_id")
            data.setdefault("provider", self.provider_name)
            data.setdefault("mode", "timesfm")
            data["round_trip_ms"] = round(elapsed_ms, 3)
            data["fallback_used"] = False
            return data
        except Exception as exc:
            return self._cluster_fallback(
                sequence,
                horizon_steps=horizon_steps,
                reason=f"{type(exc).__name__}: {exc}",
            )

    def _cluster_fallback(self, sequence: List[int], *, horizon_steps: int, reason: str) -> Dict[str, Any]:
        fallback = self.fallback
        if hasattr(fallback, "forecast_next_cluster"):
            data = fallback.forecast_next_cluster(sequence, horizon_steps=horizon_steps)
        else:
            data = LinearMockTimesFMProvider().forecast_next_cluster(sequence, horizon_steps=horizon_steps)
        data["provider"] = self.provider_name
        data["mode"] = "fallback"
        data["fallback_used"] = True
        data["fallback_reason"] = reason
        return data


def forecast_pressure(forecast: Optional[Dict[str, Any]], *, p95_budget_ms: float = 250.0) -> Dict[str, Any]:
    """Convert a pulse forecast into an advisory routing pressure score."""
    if not forecast or not forecast.get("patches"):
        return {
            "forecast_pressure": 0.0,
            "suggested_plan": "standard",
            "forecast_reason": "no_forecast_available",
            "confidence_score": 0.0,
        }

    max_latency = 0.0
    max_degrade = 0.0
    for patch in forecast.get("patches", []):
        latency = patch.get("p95_latency_ms", {})
        degrade = patch.get("degrade_count", {})
        max_latency = max(max_latency, float(latency.get("q90", latency.get("point", 0.0)) or 0.0))
        max_degrade = max(max_degrade, float(degrade.get("q90", degrade.get("point", 0.0)) or 0.0))

    latency_pressure = min(1.0, max(0.0, max_latency / max(1.0, p95_budget_ms)))
    degrade_pressure = min(1.0, max(0.0, max_degrade / 1.0))
    pressure = round(max(latency_pressure, degrade_pressure), 4)
    suggested = "conservative" if pressure >= 0.8 else "standard"
    reason = (
        "Forecast predicts p95 breach; suggested shift to Conservative plan."
        if suggested == "conservative"
        else "Forecast within advisory budget; standard routing suggested."
    )
    return {
        "forecast_pressure": pressure,
        "suggested_plan": suggested,
        "forecast_reason": reason,
        "confidence_score": float(forecast.get("confidence_score", 0.0) or 0.0),
        "max_forecast_p95_latency_ms": round(max_latency, 3),
        "max_forecast_degrade_count": round(max_degrade, 3),
    }


def _forecast_metric_point(patch: Dict[str, Any], metric: str) -> float:
    raw = patch.get(metric, 0.0)
    if isinstance(raw, dict):
        raw = raw.get("point", raw.get("q50", 0.0))
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


class PulseEngine:
    """Coordinates live pulse harvesting, forecast refresh, and cached advice."""

    def __init__(
        self,
        *,
        harvester: Optional[PulseHarvester] = None,
        provider: Optional[PulseForecastProvider] = None,
        horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
        actions_mode: str = "advisory",
        warmup_callback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cooldown_seconds: int = 900,
        clock: Any = time.time,
    ) -> None:
        self.harvester = harvester if harvester is not None else PulseHarvester()
        self.provider = provider or LinearMockTimesFMProvider()
        self.horizon_minutes = int(horizon_minutes)
        self.actions_mode = actions_mode
        self.warmup_callback = warmup_callback
        self.audit_callback = audit_callback
        self.cooldown_seconds = int(cooldown_seconds)
        self._clock = clock
        self._forecast: Optional[Dict[str, Any]] = None
        self._forecast_ts = 0.0
        self._last_trigger_ts = 0.0
        self._last_action: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()

    def record_query(self, **kwargs: Any) -> PulsePatch:
        return self.harvester.record_query(**kwargs)

    def get_forecast(self, *, refresh: bool = False) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not refresh:
                return self._forecast
        return self.refresh_forecast()

    def refresh_forecast(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._forecast = self.provider.forecast(
                self.harvester.observed(limit=60),
                horizon_minutes=self.horizon_minutes,
            )
            self._forecast_ts = self._clock()
            return self._forecast

    def advisory(self, *, p95_budget_ms: float = 250.0) -> Dict[str, Any]:
        advice = forecast_pressure(self.get_forecast(refresh=False), p95_budget_ms=p95_budget_ms)
        advice["actions_mode"] = self.actions_mode
        return advice

    def forecast_next_cluster(self, sequence: List[int], *, horizon_steps: int = 3) -> Dict[str, Any]:
        if hasattr(self.provider, "forecast_next_cluster"):
            return self.provider.forecast_next_cluster(sequence, horizon_steps=horizon_steps)
        return LinearMockTimesFMProvider().forecast_next_cluster(sequence, horizon_steps=horizon_steps)

    def evaluate_and_trigger(self, forecast: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute autonomous predictive warmup when the pulse forecast is strong.

        Trigger conditions:
        - actions mode is autonomous
        - forecast confidence is > 0.85
        - next 15 minutes predict either >20% query volume spike or >50ms p95 rise
        - 15-minute cooldown has elapsed
        """
        if self.actions_mode != "autonomous":
            return {
                "triggered": False,
                "suppressed": True,
                "reason": "pulse_actions_not_autonomous",
                "actions_mode": self.actions_mode,
            }

        forecast = forecast or self.get_forecast(refresh=False)
        if not forecast or not forecast.get("patches"):
            return {
                "triggered": False,
                "suppressed": True,
                "reason": "no_forecast_available",
                "actions_mode": self.actions_mode,
            }

        confidence = float(forecast.get("confidence_score", 0.0) or 0.0)
        if confidence <= 0.85:
            return {
                "triggered": False,
                "suppressed": True,
                "reason": "confidence_below_threshold",
                "confidence_score": confidence,
                "actions_mode": self.actions_mode,
            }

        now = self._clock()
        cooldown_remaining = self.cooldown_seconds - (now - self._last_trigger_ts)
        if self._last_trigger_ts and cooldown_remaining > 0:
            return {
                "triggered": False,
                "suppressed": True,
                "reason": "cooldown_active",
                "cooldown_remaining_s": round(cooldown_remaining, 3),
                "confidence_score": confidence,
                "actions_mode": self.actions_mode,
            }

        observed = self.harvester.observed(limit=1)
        baseline_query = max(1.0, float(observed[-1].query_count) if observed else 1.0)
        baseline_p95 = float(observed[-1].p95_latency_ms) if observed else 0.0
        forecast_patches = forecast.get("patches", [])[: self.horizon_minutes]
        max_query = max(_forecast_metric_point(p, "query_count") for p in forecast_patches)
        max_p95 = max(_forecast_metric_point(p, "p95_latency_ms") for p in forecast_patches)
        query_spike_pct = ((max_query - baseline_query) / baseline_query) * 100.0
        p95_rise_ms = max_p95 - baseline_p95

        if query_spike_pct <= 20.0 and p95_rise_ms <= 50.0:
            return {
                "triggered": False,
                "suppressed": True,
                "reason": "forecast_below_action_threshold",
                "confidence_score": confidence,
                "query_spike_pct": round(query_spike_pct, 3),
                "p95_rise_ms": round(p95_rise_ms, 3),
                "actions_mode": self.actions_mode,
            }

        target = self._target_from_forecast(forecast)
        action = {
            "action": "predictive_warmup",
            "target": target,
            "reason": (
                f"Predicted {round(query_spike_pct, 1)}% volume spike in "
                f"{self.horizon_minutes} mins"
                if query_spike_pct > 20.0
                else f"Predicted {round(p95_rise_ms, 1)}ms p95 latency rise in {self.horizon_minutes} mins"
            ),
            "confidence_score": confidence,
            "query_spike_pct": round(query_spike_pct, 3),
            "p95_rise_ms": round(p95_rise_ms, 3),
            "actions_mode": self.actions_mode,
        }

        warmup_result: Dict[str, Any] = {}
        if self.warmup_callback is not None:
            warmup_result = self.warmup_callback(action) or {}
        action["warmup_result"] = warmup_result
        self._last_trigger_ts = now
        self._last_action = dict(action)
        if self.audit_callback is not None:
            self.audit_callback(action)
        return {"triggered": True, "suppressed": False, **action}

    @staticmethod
    def _target_from_forecast(forecast: Dict[str, Any]) -> Dict[str, Any]:
        metadata = forecast.get("metadata") if isinstance(forecast.get("metadata"), dict) else {}
        complexity_class = metadata.get("complexity_class") or forecast.get("complexity_class") or "general"
        domain = metadata.get("domain") or forecast.get("domain") or "global"
        layer = "hierarchical_summary" if complexity_class == "CLASS_C" else "retrieval_runtime"
        return {
            "complexity_class": complexity_class,
            "domain": domain,
            "layer": layer,
        }


def build_pulse_payload(
    *,
    contract_version: str,
    status: str,
    source: str,
    generated_at: str,
    harvester: PulseHarvester,
    provider: Optional[PulseForecastProvider] = None,
    observed_limit: int = 60,
    horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
    actions_mode: str = "advisory",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    observed = harvester.observed(limit=observed_limit)
    forecast = None
    if provider is not None:
        forecast = provider.forecast(observed, horizon_minutes=horizon_minutes)
    return {
        "contract_version": contract_version,
        "status": status,
        "source": source,
        "generated_at": generated_at,
        "error": error,
        "feature": "mnemos_pulse",
        "bucket_seconds": PULSE_BUCKET_SECONDS,
        "buffer_capacity_patches": harvester.buffer.capacity,
        "actions_mode": actions_mode,
        "observed": {
            "patch_count": len(observed),
            "targets": [
                "query_count",
                "p95_latency_ms",
                "cache_hit_rate",
                "degrade_count",
                "avg_candidate_envelope",
            ],
            "patches": [patch.to_dict() for patch in observed],
        },
        "forecast": forecast,
    }
