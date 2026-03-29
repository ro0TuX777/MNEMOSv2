"""
ContradictionPolicy — cross-candidate contradiction detection and resolution.

Wave 2: detects entity-slot conflicts across a candidate set and selects
a winner using a deterministic priority chain.

Unlike BasePolicy (which runs per-candidate), this policy operates across
the entire candidate set and is called once per query by ReadPath.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.models.contradiction_record import ContradictionRecord

logger = logging.getLogger("mnemos.governance.policies.contradiction")

# Contradiction modifiers
_WINNER_MOD = 1.0
_LOSER_MOD = 0.25


def _parse_ts(ts: str) -> float:
    """Return POSIX timestamp from ISO string; 0.0 on parse failure."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return 0.0


def _recompute_score(decision: GovernanceDecision) -> float:
    """Recompute governed_score from current modifier values."""
    return (
        decision.retrieval_score
        * decision.trust_modifier
        * decision.utility_modifier
        * decision.freshness_modifier
        * decision.contradiction_modifier
        * decision.veto_modifier
    )


def _resolution_reason(
    sorted_members: List[Tuple[int, SearchResult, GovernanceDecision]],
) -> str:
    """Describe why the winner beat the runner-up."""
    if len(sorted_members) < 2:
        return "unresolved"

    _, winner_result, _ = sorted_members[0]
    _, second_result, _ = sorted_members[1]
    wg = winner_result.engram.governance
    sg = second_result.engram.governance

    if wg.trust_score != sg.trust_score:
        return (
            f"higher trust_score ({wg.trust_score:.3f} vs {sg.trust_score:.3f})"
        )

    winner_ts = _parse_ts(winner_result.engram.created_at)
    second_ts = _parse_ts(second_result.engram.created_at)
    if winner_ts != second_ts:
        return "newer timestamp"

    if wg.utility_score != sg.utility_score:
        return (
            f"higher utility_score ({wg.utility_score:.3f} vs {sg.utility_score:.3f})"
        )

    if wg.source_authority != sg.source_authority:
        return (
            f"higher source_authority "
            f"({wg.source_authority:.3f} vs {sg.source_authority:.3f})"
        )

    return "deterministic tie-breaker (memory_id)"


class ContradictionPolicy:
    """
    Detects entity-slot contradictions across a candidate set and resolves
    them by selecting a winner using a deterministic priority chain.

    Priority (highest first):
        1. trust_score        (higher wins)
        2. created_at         (newer wins)
        3. utility_score      (higher wins)
        4. source_authority   (higher wins)
        5. engram id          (lexicographically lower wins — pure tiebreaker)

    Modifiers applied:
        winner  → contradiction_modifier = 1.0  (no penalty)
        loser   → contradiction_modifier = 0.25 (suppressed)

    A candidate is only considered when its GovernanceMeta has both
    ``entity_key`` and ``attribute_key`` set (non-empty).  Candidates
    without governance metadata or with empty slot keys are ignored.

    Groups with fewer than 2 members, or where all members share the same
    ``normalized_value``, are not contradictions and are left untouched.
    """

    def detect_and_resolve(
        self,
        results: List[SearchResult],
        decisions: List[GovernanceDecision],
    ) -> List[ContradictionRecord]:
        """
        Detect contradiction groups, select winners, and update decisions in place.

        Args:
            results:   Ordered list of SearchResult objects.
            decisions: Parallel list of GovernanceDecision objects (same order).

        Returns:
            List of ContradictionRecord — one per detected conflict group.
            Decisions are mutated in place: contradiction_modifier, conflict_status,
            conflict_group_id, contradiction_winner, contradiction_reason,
            suppressed, suppressed_by_contradiction, and governed_score are updated
            for all members of detected groups.
        """
        # Group candidates by (entity_key, attribute_key)
        groups: Dict[
            Tuple[str, str],
            List[Tuple[int, SearchResult, GovernanceDecision]],
        ] = defaultdict(list)

        for idx, (result, decision) in enumerate(zip(results, decisions)):
            gov = result.engram.governance
            if gov is None:
                continue
            if not gov.entity_key or not gov.attribute_key:
                continue
            groups[(gov.entity_key, gov.attribute_key)].append(
                (idx, result, decision)
            )

        records: List[ContradictionRecord] = []

        for (entity_key, attribute_key), members in groups.items():
            if len(members) < 2:
                continue

            # Collect distinct non-empty values
            values = {
                result.engram.governance.normalized_value
                for _, result, _ in members
                if result.engram.governance.normalized_value
            }

            if len(values) <= 1:
                # All members agree (or no values set) — not a contradiction
                continue

            conflict_group_id = f"conflict:{entity_key}:{attribute_key}"
            logger.debug(
                "Contradiction detected: group=%s members=%d values=%s",
                conflict_group_id,
                len(members),
                values,
            )

            # Sort by winner priority (all DESC except id which is ASC)
            sorted_members = sorted(
                members,
                key=lambda item: (
                    -item[1].engram.governance.trust_score,
                    -_parse_ts(item[1].engram.created_at),
                    -item[1].engram.governance.utility_score,
                    -item[1].engram.governance.source_authority,
                    item[1].engram.id,  # lexicographic ASC as final tiebreaker
                ),
            )

            _, winner_result, winner_decision = sorted_members[0]
            losers = sorted_members[1:]

            reason = _resolution_reason(sorted_members)

            record = ContradictionRecord(
                conflict_group_id=conflict_group_id,
                entity_key=entity_key,
                attribute_key=attribute_key,
                candidate_memory_ids=[r.engram.id for _, r, _ in members],
                candidate_values={
                    r.engram.id: r.engram.governance.normalized_value
                    for _, r, _ in members
                },
                winner_memory_id=winner_result.engram.id,
                loser_memory_ids=[r.engram.id for _, r, _ in losers],
                resolution_reason=reason,
                status="resolved",
            )
            records.append(record)

            # Update winner
            winner_decision.contradiction_modifier = _WINNER_MOD
            winner_decision.conflict_status = "winner"
            winner_decision.conflict_group_id = conflict_group_id
            winner_decision.contradiction_reason = reason
            winner_decision.governed_score = _recompute_score(winner_decision)

            # Update losers
            for _, loser_result, loser_decision in losers:
                loser_decision.contradiction_modifier = _LOSER_MOD
                loser_decision.conflict_status = "suppressed"
                loser_decision.conflict_group_id = conflict_group_id
                loser_decision.contradiction_winner = winner_result.engram.id
                loser_decision.contradiction_reason = reason
                loser_decision.suppressed_by_contradiction = True
                loser_decision.suppressed = True
                loser_decision.suppressed_reason = (
                    f"contradiction loser (group: {conflict_group_id})"
                )
                loser_decision.governed_score = _recompute_score(loser_decision)

        return records
