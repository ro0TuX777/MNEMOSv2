"""
Hygiene path — Wave 4 background memory health management.

Three runners that can be chained via HygienePipeline:

    DecayRunner              — time-based utility decay; lifecycle_state -> stale
    PrunePromoter            — composite score floor; lifecycle_state -> prune_candidate
    ContradictionSweepRunner — offline entity-slot contradiction detection

All runners follow the same contract:
    - accept a List[Engram] and optional dry_run=True flag
    - return a typed report dataclass with scanned/changed/skipped counts
    - mutate GovernanceMeta in place (unless dry_run=True)

HygienePipeline chains all three in the correct order:
    decay -> prune promotion -> contradiction sweep

Usage
-----
::

    from mnemos.governance.hygiene import HygienePipeline, DecayConfig

    pipeline = HygienePipeline(decay_config=DecayConfig(horizon_days=30))
    report = pipeline.run(engrams, dry_run=True)   # preview
    report = pipeline.run(engrams)                  # apply
"""

from __future__ import annotations

from dataclasses import dataclass

from mnemos.governance.hygiene.decay_runner import (
    DecayConfig,
    DecayReport,
    DecayRunner,
)
from mnemos.governance.hygiene.prune_promoter import (
    PruneConfig,
    PruneReport,
    PrunePromoter,
)
from mnemos.governance.hygiene.contradiction_sweep import (
    ContradictionSweepRecord,
    ContradictionSweepReport,
    ContradictionSweepRunner,
)
from mnemos.engram.model import Engram
from typing import List, Optional


@dataclass
class HygienePipelineReport:
    """Combined report from all three hygiene runners."""

    decay: DecayReport
    prune: PruneReport
    sweep: ContradictionSweepReport

    @property
    def total_mutations(self) -> int:
        """Total number of Engrams mutated (decayed + promoted + sweep conflicts)."""
        return (
            self.decay.decayed
            + self.prune.promoted
            + self.sweep.winners_set
            + self.sweep.losers_set
        )


class HygienePipeline:
    """
    Chains DecayRunner, PrunePromoter, and ContradictionSweepRunner in order.

    Order matters:
        1. Decay first — may set lifecycle_state to "stale".
        2. Prune promotion second — respects "stale" state set by decay.
        3. Contradiction sweep last — independent of lifecycle state.

    Usage
    -----
    ::

        pipeline = HygienePipeline()
        report = pipeline.run(engrams, dry_run=True)  # preview
        report = pipeline.run(engrams)                 # apply
        print(report.total_mutations)
    """

    def __init__(
        self,
        decay_config: Optional[DecayConfig] = None,
        prune_config: Optional[PruneConfig] = None,
    ) -> None:
        self._decay = DecayRunner(decay_config)
        self._prune = PrunePromoter(prune_config)
        self._sweep = ContradictionSweepRunner()

    def run(
        self,
        engrams: List[Engram],
        now_iso: Optional[str] = None,
        dry_run: bool = False,
    ) -> HygienePipelineReport:
        """
        Run all three hygiene passes over the engram list.

        Args:
            engrams:  List of Engram objects to sweep.
            now_iso:  ISO UTC timestamp for decay calculation.
                      Defaults to current UTC time.
            dry_run:  If True, no Engram is mutated.

        Returns:
            HygienePipelineReport with individual sub-reports.
        """
        decay_report = self._decay.run(engrams, now_iso=now_iso, dry_run=dry_run)
        prune_report = self._prune.run(engrams, dry_run=dry_run)
        sweep_report = self._sweep.run(engrams, dry_run=dry_run)
        return HygienePipelineReport(
            decay=decay_report,
            prune=prune_report,
            sweep=sweep_report,
        )


__all__ = [
    "DecayConfig",
    "DecayReport",
    "DecayRunner",
    "PruneConfig",
    "PruneReport",
    "PrunePromoter",
    "ContradictionSweepRecord",
    "ContradictionSweepReport",
    "ContradictionSweepRunner",
    "HygienePipelineReport",
    "HygienePipeline",
]
