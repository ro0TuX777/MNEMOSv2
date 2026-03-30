"""
UsageDetector — classifies governed candidates by how they contributed to an answer.

Wave 3: called by ReflectPath after answer generation to determine which
memories were actually used, ignored, contradicted, or vetoed.

Detection signals (in order of confidence)
-------------------------------------------
1. Explicit veto / contradiction state on the GovernanceDecision
   (VETOED / CONTRADICTED are set before overlap analysis)
2. Explicit citation — memory ID appears in the ``cited_ids`` set
   supplied by the caller
3. Answer-text overlap — normalised word-overlap between answer and
   memory content (recall-oriented: fraction of memory words in answer)
4. Default — IGNORED when no signal fires

Output labels
-------------
    used          memory contributed to the answer
    ignored       memory was retrieved but not used
    contradicted  memory was a contradiction loser (may still have been used)
    vetoed        memory was hard-vetoed before scoring
    unknown       no decision available for this result
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional

from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision


class UsageLabel(str, Enum):
    USED = "used"
    IGNORED = "ignored"
    CONTRADICTED = "contradicted"
    VETOED = "vetoed"
    UNKNOWN = "unknown"


def _word_set(text: str) -> frozenset:
    """Lowercase alphanumeric tokens of length >= 3."""
    return frozenset(re.findall(r"\b[a-z0-9]{3,}\b", text.lower()))


class UsageDetector:
    """
    Heuristic used-memory detector.

    Args:
        overlap_threshold:
            Minimum fraction of memory-text words that must appear in the
            answer for the memory to be classified as ``used`` via overlap.
            Default 0.15 — 15 % of memory words present in answer.
            Prefer precision: raise this threshold to avoid false used labels.
    """

    def __init__(self, overlap_threshold: float = 0.15) -> None:
        self._threshold = overlap_threshold

    # ── Public API ────────────────────────────────────────────────────────

    def detect(
        self,
        query: str,
        answer: str,
        results: List[SearchResult],
        decisions: List[GovernanceDecision],
        cited_ids: Optional[List[str]] = None,
    ) -> Dict[str, UsageLabel]:
        """
        Classify each candidate in ``results``.

        Args:
            query:      Original query text (reserved for future signals).
            answer:     Generated answer text to test overlap against.
            results:    SearchResult list (parallel to ``decisions``).
            decisions:  GovernanceDecision list (parallel to ``results``).
            cited_ids:  Explicit memory IDs the answer is known to cite.

        Returns:
            ``{engram_id: UsageLabel}`` for every candidate in ``results``.
        """
        cited_set: frozenset = frozenset(cited_ids or [])
        answer_words = _word_set(answer)
        decision_map = {d.engram_id: d for d in decisions}

        labels: Dict[str, UsageLabel] = {}

        for result in results:
            eid = result.engram.id
            decision = decision_map.get(eid)

            if decision is None:
                labels[eid] = UsageLabel.UNKNOWN
                continue

            # Signal 0 — hard veto (highest priority, short-circuits everything)
            if not decision.veto_pass:
                labels[eid] = UsageLabel.VETOED
                continue

            # Signal 0b — contradiction loser
            # Note: a loser *could* still appear in advisory results.
            # We label it CONTRADICTED so reinforcement can apply the
            # appropriate penalty without treating it as IGNORED.
            if decision.suppressed_by_contradiction:
                labels[eid] = UsageLabel.CONTRADICTED
                continue

            # Signal 1 — explicit citation
            if eid in cited_set:
                labels[eid] = UsageLabel.USED
                continue

            # Signal 2 — answer-text overlap
            memory_words = _word_set(result.engram.content)
            if memory_words:
                overlap = len(answer_words & memory_words) / len(memory_words)
                if overlap >= self._threshold:
                    labels[eid] = UsageLabel.USED
                    continue

            # Default
            labels[eid] = UsageLabel.IGNORED

        return labels
