"""
Governor — main entry point for the MNEMOS governance layer.

Instantiated once in the service runtime and wired into the search path.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.models.contradiction_record import ContradictionRecord
from mnemos.governance.policies.relevance_veto_policy import RelevanceVetoPolicy
from mnemos.governance.policies.utility_policy import UtilityPolicy
from mnemos.governance.policies.contradiction_policy import ContradictionPolicy
from mnemos.governance.policy_registry import PolicyRegistry
from mnemos.governance.read_path import ReadPath, GOVERNANCE_MODES

logger = logging.getLogger("mnemos.governance")


class Governor:
    """
    Governance layer for MNEMOS retrieval.

    Default policy chain (Wave 2):
      Per-candidate:
        1. RelevanceVetoPolicy  — score floor, deletion state, toxic flag, freshness
        2. UtilityPolicy        — trust and utility modifiers
      Cross-candidate:
        3. ContradictionPolicy  — entity-slot contradiction detection and resolution

    Future waves will add:
      4. DecayPolicy hygiene  (Wave 3 — Task 5.2)

    Configuration
    -------------
    min_score_threshold
        Veto candidates whose retrieval score is below this value.
        Default 0.0 (veto disabled — conservative for advisory mode).

    freshness_half_life_days
        Half-life in days for freshness decay.  A memory created
        ``half_life_days`` ago receives a 0.5 freshness modifier.
        Default 180 days.

    disabled_policies
        List of policy names to skip.
    """

    def __init__(
        self,
        min_score_threshold: float = 0.0,
        freshness_half_life_days: float = 180.0,
        disabled_policies: Optional[List[str]] = None,
    ):
        registry = PolicyRegistry(disabled_policies=disabled_policies)
        registry.register(
            RelevanceVetoPolicy(
                min_score_threshold=min_score_threshold,
                freshness_half_life_days=freshness_half_life_days,
            )
        )
        registry.register(UtilityPolicy())

        self._registry = registry
        self._contradiction_policy = ContradictionPolicy()
        self._read_path = ReadPath(registry, contradiction_policy=self._contradiction_policy)

        # ── In-memory aggregate stats (reset on service restart) ───────────
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "total_governed_queries": 0,
            "advisory_queries": 0,
            "enforced_queries": 0,
            "total_candidates_evaluated": 0,
            "total_vetoed": 0,
            "total_suppressed": 0,
            "total_contradictions_detected": 0,
            "total_contradiction_suppressed": 0,
        }

    # ── Public API ────────────────────────────────────────────────────────

    def govern(
        self,
        results: List[SearchResult],
        query: str,
        governance_mode: str = "advisory",
        top_k: int = 10,
    ) -> Tuple[List[SearchResult], List[GovernanceDecision], List[ContradictionRecord]]:
        """
        Apply governance to a list of search results.

        Args:
            results:         Raw SearchResult list from the retrieval tier.
            query:           Original query text.
            governance_mode: "off" | "advisory" | "enforced"
            top_k:           Result cap used in enforced mode.

        Returns:
            (governed_results, decisions, contradiction_records)
        """
        if governance_mode not in GOVERNANCE_MODES:
            raise ValueError(
                f"Invalid governance_mode: {governance_mode!r}. "
                f"Must be one of {sorted(GOVERNANCE_MODES)}"
            )

        if governance_mode == "off":
            return results, [], []

        governed, decisions, contradiction_records = self._read_path.apply(
            results=results,
            query=query,
            governance_mode=governance_mode,
            top_k=top_k,
        )

        self._record_stats(governance_mode, results, decisions, contradiction_records)
        return governed, decisions, contradiction_records

    def stats(self) -> Dict[str, Any]:
        """Return aggregate governance statistics."""
        with self._lock:
            s = dict(self._stats)
        total = s["total_candidates_evaluated"]
        s["veto_rate"] = round(s["total_vetoed"] / total, 4) if total else 0.0
        s["suppression_rate"] = (
            round(s["total_suppressed"] / total, 4) if total else 0.0
        )
        s["active_policies"] = self._registry.active_policy_names
        return s

    # ── Internals ─────────────────────────────────────────────────────────

    def _record_stats(
        self,
        mode: str,
        raw_results: List[SearchResult],
        decisions: List[GovernanceDecision],
        contradiction_records: List[ContradictionRecord],
    ) -> None:
        vetoed = sum(1 for d in decisions if not d.veto_pass)
        suppressed = sum(1 for d in decisions if d.suppressed)
        contradiction_suppressed = sum(
            1 for d in decisions if d.suppressed_by_contradiction
        )
        with self._lock:
            self._stats["total_governed_queries"] += 1
            key = f"{mode}_queries"
            if key in self._stats:
                self._stats[key] += 1
            self._stats["total_candidates_evaluated"] += len(raw_results)
            self._stats["total_vetoed"] += vetoed
            self._stats["total_suppressed"] += suppressed
            self._stats["total_contradictions_detected"] += len(contradiction_records)
            self._stats["total_contradiction_suppressed"] += contradiction_suppressed
