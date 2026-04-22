"""
PrunePromoter — composite-score-based promotion to prune_candidate state.

Wave 4: background hygiene pass that identifies memories whose composite
governance score has fallen below a floor and marks them as
``lifecycle_state = "prune_candidate"``.

Nothing is deleted.  Promotion is a low-risk signal that a memory is eligible
for operator review; the irreversible deletion decision is explicitly out of
scope for this wave.

Design principles
-----------------
- Composite score = utility_score × trust_score × contradiction_factor.
- Contradiction factor = 0.25 when conflict_status == "suppressed", else 1.0.
  This mirrors the 0.25 loser modifier applied at query time.
- Stale memories (lifecycle_state == "stale") are always eligible when
  ``respect_stale_state=True`` (default).
- Already-processed states ("prune_candidate", "archived") and hard-deleted
  states ("soft_deleted", "tombstone") are skipped.
- ``dry_run=True`` computes but does not mutate — returns an accurate report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from mnemos.engram.model import Engram

_LOSER_CONTRADICTION_FACTOR = 0.25
# Mirror the contradiction_modifier applied to losers in ContradictionPolicy.

_SKIP_LIFECYCLE_STATES = {"prune_candidate", "archived"}
_SKIP_DELETION_STATES = {"soft_deleted", "tombstone"}


@dataclass
class PruneConfig:
    """Configuration for the prune promoter."""

    composite_floor: float = 0.05
    # Composite score below this threshold triggers prune_candidate promotion.

    respect_stale_state: bool = True
    # When True, memories already in lifecycle_state="stale" are always promoted,
    # regardless of their composite score.


@dataclass
class PruneReport:
    """Summary of one prune promotion pass."""

    scanned: int = 0
    # Engrams examined (those with governance metadata and eligible states).

    promoted: int = 0
    # Engrams promoted to prune_candidate.

    skipped: int = 0
    # Engrams skipped (no governance, already pruned/archived/deleted).


class PrunePromoter:
    """
    Promotes eligible Engrams to ``lifecycle_state = "prune_candidate"``
    in place.

    Usage
    -----
    ::

        promoter = PrunePromoter()
        report = promoter.run(engrams)
        # report.promoted -> count of newly promoted memories

    Dry-run mode
    ------------
    Pass ``dry_run=True`` to compute the report without mutating anything.
    """

    def __init__(self, config: PruneConfig | None = None) -> None:
        self._cfg = config or PruneConfig()

    def run(
        self,
        engrams: List[Engram],
        dry_run: bool = False,
    ) -> PruneReport:
        """
        Evaluate each Engram and promote eligible ones to prune_candidate.

        Args:
            engrams:  List of Engram objects to sweep.
            dry_run:  If True, compute the report without mutating anything.

        Returns:
            PruneReport with counts of what was (or would be) changed.
        """
        cfg = self._cfg
        report = PruneReport()

        for engram in engrams:
            gov = engram.governance
            if gov is None:
                report.skipped += 1
                continue

            if gov.deletion_state in _SKIP_DELETION_STATES:
                report.skipped += 1
                continue

            if gov.lifecycle_state in _SKIP_LIFECYCLE_STATES:
                report.skipped += 1
                continue

            report.scanned += 1

            # Determine whether this memory should be promoted.
            should_promote = False

            if cfg.respect_stale_state and gov.lifecycle_state == "stale":
                should_promote = True
            else:
                contradiction_factor = (
                    _LOSER_CONTRADICTION_FACTOR
                    if gov.conflict_status == "suppressed"
                    else 1.0
                )
                composite = gov.utility_score * gov.trust_score * contradiction_factor
                if composite < cfg.composite_floor:
                    should_promote = True

            if should_promote:
                report.promoted += 1
                if not dry_run:
                    gov.lifecycle_state = "prune_candidate"

        return report
