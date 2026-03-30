"""
ContradictionSweepRunner — offline contradiction detection across the corpus.

Wave 4: background hygiene pass that runs ContradictionPolicy over all
entity-slot clusters in the Engram corpus.  This catches contradictions
between memories that were never co-retrieved in the same query context.

Design principles
-----------------
- Reuses ContradictionPolicy.detect_and_resolve() directly; no duplicate
  resolution logic.
- Only processes Engrams with non-empty entity_key AND attribute_key.
- Groups with fewer than 2 members, or where all members share the same
  normalized_value, are not contradictions and are left untouched.
- Writes conflict_group_id, conflict_status, and superseded_by back to
  GovernanceMeta in place.  Does NOT apply score penalties (that is the
  reflect path's job when co-retrieved memories are assessed post-generation).
- ``dry_run=True`` computes but does not mutate — returns an accurate report.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.policies.contradiction_policy import ContradictionPolicy


def _make_fake_result(engram: Engram) -> SearchResult:
    """Wrap an Engram in a minimal SearchResult for ContradictionPolicy."""
    return SearchResult(
        engram=engram,
        score=engram.governance.utility_score if engram.governance else 0.0,
    )


def _make_fake_decision(result: SearchResult) -> GovernanceDecision:
    """Minimal GovernanceDecision for ContradictionPolicy input."""
    return GovernanceDecision(
        engram_id=result.engram.id,
        retrieval_score=result.score,
        governed_score=result.score,
    )


@dataclass
class ContradictionSweepRecord:
    """Outcome of resolving one contradiction cluster."""

    cluster_key: str
    # "entity_key:attribute_key"

    conflict_group_id: str
    winner_id: str
    loser_ids: List[str]
    resolution_reason: str


@dataclass
class ContradictionSweepReport:
    """Summary of one contradiction sweep pass."""

    clusters_scanned: int = 0
    # Entity-slot groups with 2+ members evaluated.

    contradictions_found: int = 0
    # Groups where members had conflicting normalized_value.

    winners_set: int = 0
    # Engrams whose conflict_status was set to "winner".

    losers_set: int = 0
    # Engrams whose conflict_status was set to "suppressed".

    skipped: int = 0
    # Engrams without governance metadata or empty entity/attribute keys.

    records: List[ContradictionSweepRecord] = field(default_factory=list)


class ContradictionSweepRunner:
    """
    Offline contradiction detection over a full Engram corpus.

    Usage
    -----
    ::

        runner = ContradictionSweepRunner()
        report = runner.run(engrams)
        # report.contradictions_found -> number of conflict clusters resolved
        # report.records -> per-cluster detail

    Dry-run mode
    ------------
    Pass ``dry_run=True`` to compute the report without mutating anything.
    """

    def __init__(self) -> None:
        self._policy = ContradictionPolicy()

    def run(
        self,
        engrams: List[Engram],
        dry_run: bool = False,
    ) -> ContradictionSweepReport:
        """
        Sweep all entity-slot clusters and resolve contradictions.

        Args:
            engrams:  List of Engram objects to sweep.
            dry_run:  If True, compute without mutating GovernanceMeta.

        Returns:
            ContradictionSweepReport with cluster-level resolution detail.
        """
        report = ContradictionSweepReport()

        # Group eligible Engrams by (entity_key, attribute_key).
        clusters: Dict[Tuple[str, str], List[Engram]] = defaultdict(list)

        for engram in engrams:
            gov = engram.governance
            if gov is None:
                report.skipped += 1
                continue
            if not gov.entity_key or not gov.attribute_key:
                report.skipped += 1
                continue
            clusters[(gov.entity_key, gov.attribute_key)].append(engram)

        for (entity_key, attribute_key), cluster_engrams in clusters.items():
            if len(cluster_engrams) < 2:
                continue

            report.clusters_scanned += 1

            fake_results = [_make_fake_result(e) for e in cluster_engrams]
            fake_decisions = [_make_fake_decision(r) for r in fake_results]

            contradiction_records = self._policy.detect_and_resolve(
                fake_results, fake_decisions
            )

            if not contradiction_records:
                continue

            # One record per cluster (ContradictionPolicy yields one per group).
            for rec in contradiction_records:
                report.contradictions_found += 1
                report.winners_set += 1
                report.losers_set += len(rec.loser_memory_ids)

                sweep_record = ContradictionSweepRecord(
                    cluster_key=f"{entity_key}:{attribute_key}",
                    conflict_group_id=rec.conflict_group_id,
                    winner_id=rec.winner_memory_id,
                    loser_ids=list(rec.loser_memory_ids),
                    resolution_reason=rec.resolution_reason,
                )
                report.records.append(sweep_record)

                if not dry_run:
                    # Write conflict state back to GovernanceMeta.
                    # Build a map from engram_id → GovernanceDecision for lookup.
                    decision_map = {
                        d.engram_id: d for d in fake_decisions
                    }
                    for engram in cluster_engrams:
                        dec = decision_map.get(engram.id)
                        if dec is None or engram.governance is None:
                            continue
                        gov = engram.governance
                        gov.conflict_group_id = dec.conflict_group_id
                        gov.conflict_status = dec.conflict_status
                        if dec.suppressed_by_contradiction:
                            gov.superseded_by = dec.contradiction_winner

        return report
