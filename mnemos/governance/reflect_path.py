"""
ReflectPath — post-generation feedback loop for memory reinforcement.

Wave 3: orchestrates the full reflect cycle:
  1. Detect which memories were used, ignored, contradicted, or vetoed
  2. Apply deterministic reinforcement rules (utility, trust, stability)
  3. Return a ReflectResult with all deltas and usage labels

Not a retrieval-time policy — called once per answer generation event
after the answer has been produced.

Rollout posture
---------------
- Advisory safe: reflect computes and applies in-memory updates.
  Persistence back to the storage backend (Qdrant/pgvector) is the
  caller's responsibility and is out of scope for Wave 3.
- The updates are always applied regardless of governance_mode so that
  the reflect loop can be benchmarked in isolation.  governance_mode is
  recorded in the result for observability only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.usage_detector import UsageDetector, UsageLabel
from mnemos.governance.reinforcement import Reinforcement, ReinforcementConfig

logger = logging.getLogger("mnemos.governance.reflect")


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class ReflectResult:
    """
    Output of one ReflectPath.reflect() call.

    All ID lists are parallel to the candidate set supplied to reflect().
    Deltas are the values actually applied — 0.0 means no change was made.
    """

    query: str

    # ── Usage classification ─────────────────────────────────────────────
    used_ids: List[str] = field(default_factory=list)
    ignored_ids: List[str] = field(default_factory=list)
    contradicted_ids: List[str] = field(default_factory=list)
    vetoed_ids: List[str] = field(default_factory=list)
    unknown_ids: List[str] = field(default_factory=list)

    # ── Applied deltas ───────────────────────────────────────────────────
    utility_deltas: Dict[str, float] = field(default_factory=dict)
    # {engram_id: delta_applied}  (positive = reinforced, negative = penalized)

    trust_deltas: Dict[str, float] = field(default_factory=dict)

    # ── Summary counts ───────────────────────────────────────────────────
    total_reinforced: int = 0
    # Candidates where utility_delta > 0

    total_penalized: int = 0
    # Candidates where utility_delta < 0

    # ── Context ──────────────────────────────────────────────────────────
    governance_mode: str = "advisory"
    ran_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "governance_mode": self.governance_mode,
            "ran_at": self.ran_at,
            "used_ids": self.used_ids,
            "ignored_ids": self.ignored_ids,
            "contradicted_ids": self.contradicted_ids,
            "vetoed_ids": self.vetoed_ids,
            "unknown_ids": self.unknown_ids,
            "utility_deltas": self.utility_deltas,
            "trust_deltas": self.trust_deltas,
            "total_reinforced": self.total_reinforced,
            "total_penalized": self.total_penalized,
        }


# ── ReflectPath ───────────────────────────────────────────────────────────────


class ReflectPath:
    """
    Orchestrates the Wave 3 post-generation reflect loop.

    Usage
    -----
    ::

        rp = ReflectPath()
        result = rp.reflect(
            query="What is the status of project X?",
            answer="Project X is currently active as of Q1 2026.",
            results=governed_results,
            decisions=all_decisions,
            cited_ids=["mem-001"],
        )
        # result.used_ids, result.utility_deltas, etc. are now populated
        # Engram.governance fields on the SearchResult objects are updated
        # in place.

    Advisory-safe
    -------------
    Wave 3 updates GovernanceMeta in-memory only.  Callers that want
    durability should persist updated engrams back to the storage backend
    after calling reflect().
    """

    def __init__(
        self,
        usage_detector: UsageDetector | None = None,
        reinforcement: Reinforcement | None = None,
    ) -> None:
        self._detector = usage_detector or UsageDetector()
        self._reinforcement = reinforcement or Reinforcement()

    # ── Public API ────────────────────────────────────────────────────────

    def reflect(
        self,
        query: str,
        answer: str,
        results: List[SearchResult],
        decisions: List[GovernanceDecision],
        cited_ids: Optional[List[str]] = None,
        governance_mode: str = "advisory",
    ) -> ReflectResult:
        """
        Run the full reflect cycle for one query/answer pair.

        Args:
            query:           Original query text.
            answer:          Generated answer to analyse for memory overlap.
            results:         SearchResult list (parallel to ``decisions``).
            decisions:       GovernanceDecision list produced by ReadPath.
            cited_ids:       Explicit memory IDs the answer cites (highest
                             confidence signal).  Optional.
            governance_mode: The governance mode that was active during
                             retrieval.  Recorded in the result only.

        Returns:
            ReflectResult with usage labels, applied deltas, and counts.
            GovernanceMeta on each result's Engram is updated in place.
        """
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        labels = self._detector.detect(
            query=query,
            answer=answer,
            results=results,
            decisions=decisions,
            cited_ids=cited_ids,
        )

        reflect_result = ReflectResult(
            query=query,
            governance_mode=governance_mode,
            ran_at=now_iso,
        )

        for result in results:
            eid = result.engram.id
            label = labels.get(eid, UsageLabel.UNKNOWN)

            # Bucket by label
            if label == UsageLabel.USED:
                reflect_result.used_ids.append(eid)
            elif label == UsageLabel.IGNORED:
                reflect_result.ignored_ids.append(eid)
            elif label == UsageLabel.CONTRADICTED:
                reflect_result.contradicted_ids.append(eid)
            elif label == UsageLabel.VETOED:
                reflect_result.vetoed_ids.append(eid)
            else:
                reflect_result.unknown_ids.append(eid)

            # Apply reinforcement (mutates GovernanceMeta in place)
            util_delta, trust_delta = self._reinforcement.apply(
                result, label, now_iso
            )
            reflect_result.utility_deltas[eid] = round(util_delta, 6)
            reflect_result.trust_deltas[eid] = round(trust_delta, 6)

            logger.debug(
                "reflect: id=%s label=%s util_delta=%+.4f trust_delta=%+.4f",
                eid,
                label.value,
                util_delta,
                trust_delta,
            )

        # Summary counts based on utility deltas (utility is the primary signal)
        reflect_result.total_reinforced = sum(
            1 for d in reflect_result.utility_deltas.values() if d > 0
        )
        reflect_result.total_penalized = sum(
            1 for d in reflect_result.utility_deltas.values() if d < 0
        )

        logger.info(
            "reflect complete: used=%d ignored=%d contradicted=%d vetoed=%d "
            "reinforced=%d penalized=%d",
            len(reflect_result.used_ids),
            len(reflect_result.ignored_ids),
            len(reflect_result.contradicted_ids),
            len(reflect_result.vetoed_ids),
            reflect_result.total_reinforced,
            reflect_result.total_penalized,
        )

        return reflect_result
