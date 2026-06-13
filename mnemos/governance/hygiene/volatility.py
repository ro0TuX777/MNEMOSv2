"""Semantic volatility telemetry for predictive hygiene.

Tracks memory event density by engram family so governance can shorten
freshness half-life and run reconciliation before users hit stale conflicts.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol


VOLATILITY_BUCKET_SECONDS = 60
DEFAULT_VOLATILITY_BUFFER_PATCHES = 1440
DEFAULT_VOLATILITY_HORIZON_MINUTES = 15


def _bucket(ts: float, bucket_seconds: int = VOLATILITY_BUCKET_SECONDS) -> int:
    return int(ts // bucket_seconds) * bucket_seconds


def family_key_from_engram(engram: Any) -> str:
    """Infer a stable family key from neuro-tags first, then source domain."""
    tags = getattr(engram, "neuro_tags", None) or []
    if tags:
        return f"tag:{str(tags[0]).strip().lower()}"
    source = str(getattr(engram, "source", "") or "").strip().lower()
    if source:
        domain = source.split("/")[0] if "/" in source else source
        return f"source:{domain}"
    metadata = getattr(engram, "metadata", None) or {}
    if isinstance(metadata, dict) and metadata.get("domain"):
        return f"domain:{str(metadata['domain']).strip().lower()}"
    return "family:unknown"


@dataclass
class VolatilityPatch:
    family_key: str
    bucket_start: int
    index_updates: int = 0
    contradiction_events: int = 0
    usage_frequency: int = 0
    entity_key: str = ""

    @property
    def event_density(self) -> int:
        return self.index_updates + self.contradiction_events + self.usage_frequency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_key": self.family_key,
            "bucket_start": self.bucket_start,
            "bucket_start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.bucket_start)),
            "index_updates": int(self.index_updates),
            "contradiction_events": int(self.contradiction_events),
            "usage_frequency": int(self.usage_frequency),
            "event_density": int(self.event_density),
            "entity_key": self.entity_key,
        }


@dataclass
class _Accumulator:
    family_key: str
    bucket_start: int
    index_updates: int = 0
    contradiction_events: int = 0
    usage_frequency: int = 0
    entity_keys: Dict[str, int] = field(default_factory=dict)

    def add(self, event_type: str, entity_key: str = "") -> None:
        if event_type == "index_update":
            self.index_updates += 1
        elif event_type == "contradiction_event":
            self.contradiction_events += 1
        elif event_type == "usage":
            self.usage_frequency += 1
        if entity_key:
            self.entity_keys[entity_key] = self.entity_keys.get(entity_key, 0) + 1

    def to_patch(self) -> VolatilityPatch:
        entity_key = ""
        if self.entity_keys:
            entity_key = max(self.entity_keys.items(), key=lambda item: item[1])[0]
        return VolatilityPatch(
            family_key=self.family_key,
            bucket_start=self.bucket_start,
            index_updates=self.index_updates,
            contradiction_events=self.contradiction_events,
            usage_frequency=self.usage_frequency,
            entity_key=entity_key,
        )


class VolatilityBuffer:
    def __init__(self, capacity: int = DEFAULT_VOLATILITY_BUFFER_PATCHES) -> None:
        self.capacity = int(capacity)
        self._patches: "OrderedDict[tuple[str, int], VolatilityPatch]" = OrderedDict()
        self._lock = threading.RLock()

    def upsert(self, patch: VolatilityPatch) -> None:
        key = (patch.family_key, patch.bucket_start)
        with self._lock:
            if key in self._patches:
                del self._patches[key]
            self._patches[key] = patch
            while len(self._patches) > self.capacity:
                self._patches.popitem(last=False)

    def latest(self, family_key: Optional[str] = None, limit: Optional[int] = None) -> List[VolatilityPatch]:
        with self._lock:
            patches = list(self._patches.values())
        if family_key is not None:
            patches = [p for p in patches if p.family_key == family_key]
        if limit is not None:
            patches = patches[-max(0, int(limit)) :]
        return patches


class VolatilityHarvester:
    def __init__(
        self,
        *,
        buffer: Optional[VolatilityBuffer] = None,
        bucket_seconds: int = VOLATILITY_BUCKET_SECONDS,
        clock: Any = time.time,
    ) -> None:
        self.buffer = buffer if buffer is not None else VolatilityBuffer()
        self.bucket_seconds = int(bucket_seconds)
        self._clock = clock
        self._accumulators: Dict[tuple[str, int], _Accumulator] = {}
        self._lock = threading.RLock()

    def record_event(
        self,
        family_key: str,
        event_type: str,
        *,
        entity_key: str = "",
        timestamp: Optional[float] = None,
    ) -> VolatilityPatch:
        if event_type not in {"index_update", "contradiction_event", "usage"}:
            raise ValueError("event_type must be index_update, contradiction_event, or usage")
        ts = self._clock() if timestamp is None else float(timestamp)
        bucket_start = _bucket(ts, self.bucket_seconds)
        key = (family_key, bucket_start)
        with self._lock:
            acc = self._accumulators.get(key)
            if acc is None:
                acc = _Accumulator(family_key=family_key, bucket_start=bucket_start)
                self._accumulators[key] = acc
            acc.add(event_type, entity_key=entity_key)
            patch = acc.to_patch()
            self.buffer.upsert(patch)
            return patch

    def record_index_update(self, engram: Any, *, timestamp: Optional[float] = None) -> VolatilityPatch:
        return self.record_event(
            family_key_from_engram(engram),
            "index_update",
            entity_key=getattr(getattr(engram, "governance", None), "entity_key", ""),
            timestamp=timestamp,
        )

    def record_usage(self, engram: Any, *, timestamp: Optional[float] = None) -> VolatilityPatch:
        return self.record_event(
            family_key_from_engram(engram),
            "usage",
            entity_key=getattr(getattr(engram, "governance", None), "entity_key", ""),
            timestamp=timestamp,
        )

    def record_contradiction(
        self,
        family_key: str,
        *,
        entity_key: str = "",
        timestamp: Optional[float] = None,
    ) -> VolatilityPatch:
        return self.record_event(
            family_key,
            "contradiction_event",
            entity_key=entity_key,
            timestamp=timestamp,
        )

    def patches(self, family_key: Optional[str] = None, limit: int = 60) -> List[VolatilityPatch]:
        return self.buffer.latest(family_key=family_key, limit=limit)


class VolatilityForecastProvider(Protocol):
    def forecast(self, patches: Iterable[VolatilityPatch], *, horizon_minutes: int) -> Dict[str, Any]:
        ...


class LinearVolatilityProvider:
    provider_name = "linear_volatility"

    def forecast(self, patches: Iterable[VolatilityPatch], *, horizon_minutes: int) -> Dict[str, Any]:
        history = list(patches)
        horizon = max(1, int(horizon_minutes))
        if not history:
            return {
                "provider": self.provider_name,
                "confidence_score": 0.0,
                "horizon_minutes": horizon,
                "families": {},
            }
        by_family: Dict[str, List[VolatilityPatch]] = {}
        for patch in history:
            by_family.setdefault(patch.family_key, []).append(patch)
        families: Dict[str, Any] = {}
        for family_key, rows in by_family.items():
            families[family_key] = self._forecast_family(rows, horizon)
        confidence = 0.86 if len(history) >= 5 else 0.55
        return {
            "provider": self.provider_name,
            "confidence_score": confidence,
            "horizon_minutes": horizon,
            "families": families,
        }

    def _forecast_family(self, rows: List[VolatilityPatch], horizon: int) -> Dict[str, Any]:
        values = [float(r.contradiction_events) for r in rows]
        update_values = [float(r.index_updates) for r in rows]
        usage_values = [float(r.usage_frequency) for r in rows]
        last_bucket = rows[-1].bucket_start
        entity_key = rows[-1].entity_key
        patches = []
        for step in range(1, horizon + 1):
            c = max(0.0, self._project(values, step))
            u = max(0.0, self._project(update_values, step))
            usage = max(0.0, self._project(usage_values, step))
            patches.append(
                {
                    "bucket_start": last_bucket + step * VOLATILITY_BUCKET_SECONDS,
                    "contradiction_events": {"point": round(c, 4), "q90": round(c * 1.1, 4)},
                    "index_updates": {"point": round(u, 4), "q90": round(u * 1.1, 4)},
                    "usage_frequency": {"point": round(usage, 4), "q90": round(usage * 1.1, 4)},
                }
            )
        max_c = max(p["contradiction_events"]["q90"] for p in patches)
        max_u = max(p["index_updates"]["q90"] for p in patches)
        return {
            "entity_key": entity_key,
            "volatility_level": "high" if max_c >= 2.0 or max_u >= 4.0 else "normal",
            "predicted_obsolescence": min(1.0, round(max(max_c / 3.0, max_u / 6.0), 4)),
            "patches": patches,
        }

    @staticmethod
    def _project(values: List[float], step: int) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[-1]
        xs = list(range(len(values)))
        x_mean = statistics.fmean(xs)
        y_mean = statistics.fmean(values)
        denom = sum((x - x_mean) ** 2 for x in xs) or 1.0
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom
        intercept = y_mean - slope * x_mean
        return intercept + slope * (len(values) - 1 + step)


class VolatilityEngine:
    def __init__(
        self,
        *,
        harvester: Optional[VolatilityHarvester] = None,
        provider: Optional[VolatilityForecastProvider] = None,
        horizon_minutes: int = DEFAULT_VOLATILITY_HORIZON_MINUTES,
        reconciliation_callback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.harvester = harvester if harvester is not None else VolatilityHarvester()
        self.provider = provider or LinearVolatilityProvider()
        self.horizon_minutes = int(horizon_minutes)
        self.reconciliation_callback = reconciliation_callback
        self.audit_callback = audit_callback
        self._forecast: Optional[Dict[str, Any]] = None

    def refresh_forecast(self, family_key: Optional[str] = None) -> Dict[str, Any]:
        self._forecast = self.provider.forecast(
            self.harvester.patches(family_key=family_key, limit=60),
            horizon_minutes=self.horizon_minutes,
        )
        return self._forecast

    def predicted_obsolescence(self, family_key: str) -> float:
        forecast = self._forecast or self.refresh_forecast()
        family = (forecast.get("families") or {}).get(family_key, {})
        return float(family.get("predicted_obsolescence", 0.0) or 0.0)

    def volatility_bias_for_family(self, family_key: str) -> float:
        return 2.0 if self.predicted_obsolescence(family_key) >= 0.7 else 1.0

    def evaluate_and_reconcile(self, family_key: Optional[str] = None) -> Dict[str, Any]:
        forecast = self.refresh_forecast(family_key=family_key)
        if float(forecast.get("confidence_score", 0.0) or 0.0) <= 0.80:
            return {"triggered": False, "reason": "confidence_below_threshold", "forecast": forecast}
        families = forecast.get("families") or {}
        triggered = []
        for key, data in families.items():
            max_contradictions = max(
                (float(p.get("contradiction_events", {}).get("q90", 0.0) or 0.0) for p in data.get("patches", [])),
                default=0.0,
            )
            if data.get("volatility_level") == "high" and max_contradictions >= 2.0:
                action = {
                    "action": "proactive_reconciliation",
                    "family_key": key,
                    "entity_key": data.get("entity_key", ""),
                    "reason": "Predicted contradiction spike; proactive reconciliation sweep triggered.",
                    "confidence_score": forecast.get("confidence_score", 0.0),
                    "predicted_obsolescence": data.get("predicted_obsolescence", 0.0),
                }
                result = self.reconciliation_callback(action) if self.reconciliation_callback else {"status": "not_configured"}
                action["reconciliation_result"] = result
                if self.audit_callback:
                    self.audit_callback(action)
                triggered.append(action)
        return {
            "triggered": bool(triggered),
            "actions": triggered,
            "forecast": forecast,
        }


__all__ = [
    "VolatilityPatch",
    "VolatilityBuffer",
    "VolatilityHarvester",
    "LinearVolatilityProvider",
    "VolatilityEngine",
    "family_key_from_engram",
]
