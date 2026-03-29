"""
ReadPath — applies governance policies to a candidate set at query time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.models.contradiction_record import ContradictionRecord
from mnemos.governance.policy_registry import PolicyRegistry

logger = logging.getLogger("mnemos.governance.read_path")

GOVERNANCE_MODES = frozenset({"off", "advisory", "enforced"})


class ReadPath:
    """
    Applies the governance policy pipeline to retrieval results.

    Modes
    -----
    off
        No governance.  Results returned unchanged with empty decisions list.

    advisory
        Every candidate is evaluated and a GovernanceDecision is produced.
        No candidate is suppressed — all are returned, re-ranked by
        governed_score.  This lets callers compare raw vs governed ordering
        without changing the visible result set.

    enforced
        Suppressed candidates (veto or contradiction losers) are removed.
        Survivors are re-ranked by governed_score and trimmed to top_k.
    """

    def __init__(
        self,
        registry: PolicyRegistry,
        contradiction_policy: Optional[object] = None,
    ):
        self._registry = registry
        self._contradiction_policy = contradiction_policy

    def apply(
        self,
        results: List[SearchResult],
        query: str,
        governance_mode: str = "advisory",
        top_k: int = 10,
    ) -> Tuple[List[SearchResult], List[GovernanceDecision], List[ContradictionRecord]]:
        """
        Run governance on a list of search candidates.

        Args:
            results:         Raw SearchResult list from the retrieval tier.
            query:           Original query text (passed to policies as context).
            governance_mode: "off" | "advisory" | "enforced"
            top_k:           Result cap for enforced mode.

        Returns:
            (governed_results, decisions, contradiction_records)
            governed_results      — list of SearchResult objects (filtered/reranked)
            decisions             — parallel list of GovernanceDecision objects
                                    (always in input order; empty when mode=="off")
            contradiction_records — list of ContradictionRecord objects detected
                                    during this call (empty when mode=="off")
        """
        if governance_mode == "off" or not results:
            return results, [], []

        context: Dict[str, Any] = {
            "query": query,
            "all_candidate_ids": [r.engram.id for r in results],
            "governance_mode": governance_mode,
        }

        # Step 1: per-candidate policy pipeline
        all_decisions: List[GovernanceDecision] = [
            self._registry.evaluate(result, context)
            for result in results
        ]

        # Step 2: cross-candidate contradiction detection (Wave 2)
        contradiction_records: List[ContradictionRecord] = []
        if self._contradiction_policy is not None:
            contradiction_records = self._contradiction_policy.detect_and_resolve(
                results, all_decisions
            )

        # Step 3: annotate would_be_suppressed_in_enforced_mode for all decisions
        for decision in all_decisions:
            decision.would_be_suppressed_in_enforced_mode = decision.suppressed

        # Step 4: apply mode-specific filtering and re-ranking
        if governance_mode == "enforced":
            # Drop suppressed, re-rank survivors by governed_score, cap at top_k
            survived = sorted(
                [(r, d) for r, d in zip(results, all_decisions) if not d.suppressed],
                key=lambda x: x[1].governed_score,
                reverse=True,
            )
            governed_results = [r for r, _ in survived[:top_k]]
        else:
            # Advisory: return all, re-ranked by governed_score
            paired = sorted(
                zip(results, all_decisions),
                key=lambda x: x[1].governed_score,
                reverse=True,
            )
            governed_results = [r for r, _ in paired]

        # Always return ALL decisions so callers can count vetoed/suppressed
        # candidates even when enforced mode has removed them from results.
        return governed_results, all_decisions, contradiction_records
