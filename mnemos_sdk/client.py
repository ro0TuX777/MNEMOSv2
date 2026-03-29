"""
MNEMOS SDK — Boundary Client.

Typed HTTP client for all MNEMOS service endpoints with readiness polling,
retry logic, timeout management, and graceful degradation.

Based on the MSF BoundaryClient pattern from ``boundary_sdk/client.py``.
"""

from __future__ import annotations

import subprocess
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from mnemos_sdk.config import MnemosConfig

logger = logging.getLogger("mnemos_sdk")

ALLOWED_STATUS = {"healthy", "degraded", "unavailable"}


# ──────────────────── Response models ────────────────────


@dataclass
class SearchHit:
    """A single search result from MNEMOS."""
    engram: Dict[str, Any]
    score: float
    tier: str
    tiers: List[str]
    component_scores: Optional[Dict[str, float]] = None
    retrieval_sources: Optional[List[str]] = None
    fusion_policy: Optional[str] = None


@dataclass
class IndexResult:
    """Result of an index (ingest) operation."""
    indexed: int
    tiers: Dict[str, int]
    engram_ids: List[str]
    latency_s: float


@dataclass
class MnemosResponse:
    """Envelope wrapping every MNEMOS API response."""
    status: str
    source: str
    error: Optional[str]
    data: Dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == "healthy" and self.error is None

    @property
    def degraded(self) -> bool:
        return self.status == "degraded"

    @property
    def unavailable(self) -> bool:
        return self.status == "unavailable"


# ──────────────────── Client ────────────────────


class MnemosClient:
    """Boundary adapter client for the MNEMOS memory service.

    Provides typed methods for every MNEMOS endpoint, with built-in
    readiness polling, retry, timeout, and graceful degradation.

    Example::

        from mnemos_sdk import MnemosClient, MnemosConfig

        config = MnemosConfig.from_env()
        client = MnemosClient(config)

        if client.wait_until_ready():
            result = client.search("quantum entanglement", top_k=5)
            for hit in result:
                print(hit.score, hit.engram["content"][:80])
    """

    def __init__(self, config: MnemosConfig, *, source: str = "mnemos_boundary") -> None:
        self._cfg = config
        self._source = source

    @property
    def config(self) -> MnemosConfig:
        return self._cfg

    # ──────── HTTP primitives ────────

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._cfg.token:
            headers["Authorization"] = f"Bearer {self._cfg.token}"
        return headers

    def _envelope(
        self, *, status: str, error: Optional[str] = None, data: Optional[Dict] = None,
    ) -> MnemosResponse:
        normalized = status if status in ALLOWED_STATUS else "degraded"
        return MnemosResponse(
            status=normalized,
            source=self._source,
            error=error,
            data=data or {},
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> MnemosResponse:
        """Core HTTP caller with retry and timeout."""
        if not self._cfg.enabled:
            return self._envelope(status="degraded", error="feature_disabled")

        if not self._cfg.base_url:
            return self._envelope(status="unavailable", error="missing_base_url")

        url = f"{self._cfg.base_url}{path}"
        attempts = max(0, self._cfg.retries) + 1
        last_error: Optional[str] = None

        for attempt in range(attempts):
            try:
                if method.upper() == "POST":
                    resp = requests.post(
                        url,
                        headers={**self._headers(), "Content-Type": "application/json"},
                        json=payload or {},
                        timeout=max(0.1, self._cfg.timeout_s),
                    )
                elif method.upper() == "DELETE":
                    resp = requests.delete(
                        url,
                        headers=self._headers(),
                        timeout=max(0.1, self._cfg.timeout_s),
                    )
                else:
                    resp = requests.get(
                        url,
                        headers=self._headers(),
                        timeout=max(0.1, self._cfg.timeout_s),
                    )

                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise ValueError("MNEMOS response must be a JSON object")

                return MnemosResponse(
                    status=data.get("status", "healthy"),
                    source=data.get("source", self._source),
                    error=data.get("error"),
                    data=data,
                )

            except Exception as exc:
                last_error = str(exc)
                if attempt < attempts - 1:
                    time.sleep(max(0.0, self._cfg.retry_delay_s))
                    logger.debug(
                        "MNEMOS request attempt %d/%d failed: %s",
                        attempt + 1, attempts, last_error,
                    )

        return self._envelope(status="unavailable", error=last_error or "remote_error")

    # ──────── Readiness ────────

    def wait_until_ready(self) -> bool:
        """Poll ``/health`` until the service is ready or timeout expires.

        Returns True if the service became healthy, False otherwise.
        """
        if self._cfg.autostart_on_demand and self._cfg.autostart_cmd.strip():
            self._maybe_autostart()

        health_url = f"{self._cfg.base_url}/health" if self._cfg.base_url else ""
        if not health_url:
            return False

        deadline = time.time() + max(0.0, self._cfg.ready_wait_s)
        while time.time() < deadline:
            try:
                resp = requests.get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    logger.info("MNEMOS service is ready at %s", self._cfg.base_url)
                    return True
            except Exception:
                pass
            time.sleep(0.4)

        logger.warning(
            "MNEMOS service not ready after %.1fs at %s",
            self._cfg.ready_wait_s, self._cfg.base_url,
        )
        return False

    def _maybe_autostart(self) -> None:
        cmd = self._cfg.autostart_cmd.strip()
        if not cmd:
            return
        try:
            logger.info("Auto-starting MNEMOS: %s", cmd)
            subprocess.run(
                cmd,
                shell=True,
                timeout=max(1.0, self._cfg.autostart_timeout_s),
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            logger.warning("MNEMOS autostart failed: %s", exc)

    # ──────── Typed API methods ────────

    def health(self) -> MnemosResponse:
        """Check service health."""
        return self._request("GET", "/health")

    def capabilities(self) -> MnemosResponse:
        """Retrieve service capabilities and tier info."""
        return self._request("GET", "/v1/mnemos/capabilities")

    def index(
        self,
        documents: List[Dict[str, Any]],
        *,
        tiers: Optional[List[str]] = None,
    ) -> MnemosResponse:
        """Ingest documents into MNEMOS.

        Args:
            documents: List of dicts with ``content``, ``source``,
                ``neuro_tags``, ``confidence``, ``metadata``.
            tiers: Optional list of tier names to target.

        Returns:
            MnemosResponse with ``data["result"]`` containing index counts.
        """
        payload: Dict[str, Any] = {"documents": documents}
        if tiers:
            payload["options"] = {"tiers": tiers}
        return self._request("POST", "/v1/mnemos/index", payload=payload)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        tiers: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        retrieval_mode: Optional[str] = None,
        fusion_policy: Optional[str] = None,
        explain: Optional[bool] = None,
    ) -> List[SearchHit]:
        """Search across retrieval tiers and return fused results.

        Args:
            query: Natural language search query.
            top_k: Maximum results to return.
            tiers: Optional list of tier names to target.
            filters: Optional metadata filter dict.

        Returns:
            List of SearchHit objects ranked by score.
        """
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        if tiers:
            payload["tiers"] = tiers
        if filters:
            payload["filters"] = filters
        if retrieval_mode:
            payload["retrieval_mode"] = retrieval_mode
        if fusion_policy:
            payload["fusion_policy"] = fusion_policy
        if explain is not None:
            payload["explain"] = explain

        resp = self._request("POST", "/v1/mnemos/search", payload=payload)
        if not resp.ok:
            return []

        hits = []
        for r in resp.data.get("results", []):
            hits.append(SearchHit(
                engram=r.get("engram", {}),
                score=r.get("score", 0.0),
                tier=r.get("tier", ""),
                tiers=r.get("tiers", []),
                component_scores=r.get("component_scores"),
                retrieval_sources=r.get("retrieval_sources"),
                fusion_policy=r.get("fusion_policy"),
            ))
        return hits

    def search_raw(
        self,
        query: str,
        *,
        top_k: int = 10,
        tiers: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        retrieval_mode: Optional[str] = None,
        fusion_policy: Optional[str] = None,
        explain: Optional[bool] = None,
    ) -> MnemosResponse:
        """Search and return the full MnemosResponse envelope."""
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        if tiers:
            payload["tiers"] = tiers
        if filters:
            payload["filters"] = filters
        if retrieval_mode:
            payload["retrieval_mode"] = retrieval_mode
        if fusion_policy:
            payload["fusion_policy"] = fusion_policy
        if explain is not None:
            payload["explain"] = explain
        return self._request("POST", "/v1/mnemos/search", payload=payload)

    def get_engram(self, engram_id: str) -> MnemosResponse:
        """Retrieve a specific engram by ID."""
        return self._request("GET", f"/v1/mnemos/engrams/{engram_id}")

    def delete_engram(self, engram_id: str) -> MnemosResponse:
        """Delete an engram from all tiers."""
        return self._request("DELETE", f"/v1/mnemos/engrams/{engram_id}")

    def audit(
        self, *, limit: int = 50, query: Optional[str] = None,
    ) -> MnemosResponse:
        """Query the forensic audit ledger."""
        path = f"/v1/mnemos/audit?limit={limit}"
        if query:
            path += f"&q={query}"
        return self._request("GET", path)

    def stats(self) -> MnemosResponse:
        """Get system-wide statistics."""
        return self._request("GET", "/v1/mnemos/stats")
