"""Associative Routing E1 — opt-in shadow adapter.

Wraps the frozen E0 ``AssociativeRouter`` (unmodified) over the expanded E1
fixture corpus and exposes a single, isolated method, ``run(query)``, that
returns a non-authoritative shadow block. This module is never imported by
the default retrieval path; it is composed at the call boundary (the REST
search handler) only when the caller explicitly opts in. See
``docs/associative_routing_e1_design_note.md`` and the E1 task spec
authorization header for the full invariant list:

* fail-closed on the kill switch or any internal error (status
  ``"unavailable"``, never raises into the request path);
* never returns content, scores, or governance/authority fields;
* never reorders, suppresses, or injects into normal retrieval results.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from prototype.associative_routing_e0 import AssociativeRouter, build_projection

from .config import ASSOCIATIVE_SHADOW_DISABLE_ENV, E1_FIXTURES_DIR

logger = logging.getLogger(__name__)


def _empty_block(latency_ms: float, status: str, abstention_reason: Optional[str]) -> Dict[str, Any]:
    return {
        "status": status,
        "projection_snapshot": None,
        "matched_cues": [],
        "routing_paths": [],
        "candidate_source_ids": [],
        "candidate_count": 0,
        "abstention_reason": abstention_reason,
        "latency_ms": round(latency_ms, 3),
        "non_authoritative": True,
    }


class AssociativeShadowAdapter:
    """Read-only, opt-in shadow path over the E1 associative-routing projection."""

    def __init__(self, fixtures_dir: Path = E1_FIXTURES_DIR) -> None:
        self._fixtures_dir = fixtures_dir
        self._router: Optional[AssociativeRouter] = None
        self._source_uri_by_content_id: Dict[str, str] = {}

    def _ensure_built(self) -> None:
        if self._router is not None:
            return
        projection = build_projection(self._fixtures_dir)
        self._router = AssociativeRouter(projection=projection)
        self._source_uri_by_content_id = {
            content_id: ref.source_uri for content_id, ref in projection.content_index.items()
        }

    def run(self, query: str) -> Dict[str, Any]:
        """Return the E1 shadow block for ``query``. Never raises."""
        t0 = time.perf_counter()

        if os.environ.get(ASSOCIATIVE_SHADOW_DISABLE_ENV, "false").lower() == "true":
            return _empty_block((time.perf_counter() - t0) * 1000.0, "unavailable", "kill_switch_enabled")

        try:
            self._ensure_built()
            router = self._router
            assert router is not None
            response = router.route(query)
        except Exception:
            logger.exception("Associative shadow adapter failed; reporting unavailable.")
            return _empty_block((time.perf_counter() - t0) * 1000.0, "unavailable", "adapter_error")

        payload = response.to_dict()
        candidate_source_ids = [
            self._source_uri_by_content_id.get(content_id, content_id)
            for content_id in payload["candidate_content_ids"]
        ]
        abstention = payload.get("abstention")

        return {
            "status": payload["routing_result"],
            "projection_snapshot": payload["projection_snapshot"],
            "matched_cues": payload["matched_cues"],
            "routing_paths": payload["routing_paths"],
            "candidate_source_ids": candidate_source_ids,
            "candidate_count": len(candidate_source_ids),
            "abstention_reason": abstention["reason_code"] if abstention else None,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "non_authoritative": True,
        }


#: Process-wide singleton reused by the REST search handler; the projection is
#: built once (lazily, on first use) and is immutable thereafter.
default_adapter = AssociativeShadowAdapter()
