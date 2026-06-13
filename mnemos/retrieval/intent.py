"""Intent trajectory harvesting for pre-cognitive retrieval."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional


_ARTICLE_RE = re.compile(r"\b(article|art\.?)\s+(\d+)\b", re.IGNORECASE)


def _hash_cluster(query: str, modulo: int = 256) -> int:
    digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


class SimpleClusterMapper:
    """Fallback mapper until the Phase 9 hierarchy object is attached."""

    def __init__(self, *, cluster_count: int = 256) -> None:
        self.cluster_count = max(1, int(cluster_count))

    def map_query(self, query: str) -> int:
        match = _ARTICLE_RE.search(query)
        if match:
            return int(match.group(2))
        return _hash_cluster(query, self.cluster_count)

    def centroid_query(self, cluster_id: int, *, template_hint: str = "") -> str:
        if "{cluster_id}" in template_hint:
            return template_hint.format(cluster_id=cluster_id)
        if "gdpr" in template_hint.lower() or not template_hint:
            return f"GDPR Article {cluster_id}"
        return f"{template_hint.strip()} {cluster_id}".strip()


@dataclass
class IntentEvent:
    session_id: str
    query: str
    cluster_id: int
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "cluster_id": self.cluster_id,
            "timestamp": self.timestamp,
        }


class IntentHarvester:
    """Tracks semantic cluster-id trajectories per session."""

    def __init__(
        self,
        *,
        cluster_mapper: Optional[Any] = None,
        max_events_per_session: int = 32,
        clock: Any = time.time,
    ) -> None:
        self._mapper = cluster_mapper or SimpleClusterMapper()
        self._max_events = int(max_events_per_session)
        self._clock = clock
        self._events: Dict[str, Deque[IntentEvent]] = defaultdict(
            lambda: deque(maxlen=self._max_events)
        )
        self._template_hints: Dict[str, str] = {}
        self._lock = threading.RLock()

    def record_query(
        self,
        *,
        session_id: str,
        query: str,
        timestamp: Optional[float] = None,
    ) -> IntentEvent:
        cluster_id = int(self._mapper.map_query(query))
        event = IntentEvent(
            session_id=session_id,
            query=query,
            cluster_id=cluster_id,
            timestamp=self._clock() if timestamp is None else float(timestamp),
        )
        with self._lock:
            self._events[session_id].append(event)
            hint = self._template_hint(query)
            if hint:
                self._template_hints[session_id] = hint
        return event

    def map_query(self, query: str) -> int:
        return int(self._mapper.map_query(query))

    def sequence(self, session_id: str, *, limit: Optional[int] = None) -> List[int]:
        with self._lock:
            rows = list(self._events.get(session_id, []))
        if limit is not None:
            rows = rows[-max(0, int(limit)) :]
        return [row.cluster_id for row in rows]

    def forecast_cluster(self, session_id: str, *, horizon_steps: int = 3) -> Optional[Dict[str, Any]]:
        seq = self.sequence(session_id, limit=8)
        if len(seq) < 3:
            return None
        deltas = [b - a for a, b in zip(seq[-3:-1], seq[-2:])]
        step = round(sum(deltas) / len(deltas)) if deltas else 0
        predicted = int(seq[-1] + step * max(1, int(horizon_steps)))
        confidence = 0.88 if len(set(deltas)) == 1 else 0.62
        template_hint = self._template_hints.get(session_id, "")
        return {
            "session_id": session_id,
            "history": seq,
            "predicted_cluster_id": predicted,
            "horizon_steps": int(horizon_steps),
            "confidence_score": confidence,
            "centroid_query": self._mapper.centroid_query(predicted, template_hint=template_hint),
        }

    @staticmethod
    def _template_hint(query: str) -> str:
        match = _ARTICLE_RE.search(query)
        if not match:
            return ""
        start, end = match.span(2)
        return f"{query[:start]}{{cluster_id}}{query[end:]}"


class IntentEngine:
    """Thin engine wrapper around IntentHarvester for runtime wiring."""

    def __init__(
        self,
        *,
        harvester: Optional[IntentHarvester] = None,
        horizon_steps: int = 3,
        shadow_callback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.harvester = harvester or IntentHarvester()
        self.horizon_steps = int(horizon_steps)
        self.shadow_callback = shadow_callback
        self.last_forecast: Optional[Dict[str, Any]] = None
        self.last_shadow_result: Optional[Dict[str, Any]] = None

    def record_and_forecast(self, *, session_id: str, query: str) -> Optional[Dict[str, Any]]:
        self.harvester.record_query(session_id=session_id, query=query)
        forecast = self.harvester.forecast_cluster(
            session_id,
            horizon_steps=self.horizon_steps,
        )
        self.last_forecast = forecast
        if forecast and self.shadow_callback and forecast.get("confidence_score", 0.0) >= 0.80:
            self.last_shadow_result = self.shadow_callback(forecast)
        return forecast


__all__ = [
    "IntentEvent",
    "IntentHarvester",
    "IntentEngine",
    "SimpleClusterMapper",
]
