"""
HygieneMetrics — thread-safe accumulator for Wave 4 hygiene telemetry.

Merged into Governor.stats() so /governance/stats surfaces hygiene activity
alongside query-time and reflect-path counters.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from mnemos.governance.hygiene import (
        DecayReport,
        PruneReport,
        ContradictionSweepReport,
    )


class HygieneMetrics:
    """
    Accumulated hygiene-pass statistics.

    All mutation is protected by a lock so Governor can safely call
    record_*() from concurrent threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_decay_runs: int = 0
        self._total_stale_promoted: int = 0
        self._total_prune_candidates: int = 0
        self._total_contradiction_sweep_clusters: int = 0

    def record_decay(self, report: "DecayReport") -> None:
        with self._lock:
            self._total_decay_runs += 1
            self._total_stale_promoted += report.stale_promoted

    def record_prune(self, report: "PruneReport") -> None:
        with self._lock:
            self._total_prune_candidates += report.promoted

    def record_sweep(self, report: "ContradictionSweepReport") -> None:
        with self._lock:
            self._total_contradiction_sweep_clusters += report.contradictions_found

    def to_dict(self) -> Dict[str, Any]:
        """Snapshot of current counters (safe to call from any thread)."""
        with self._lock:
            return {
                "total_decay_runs": self._total_decay_runs,
                "total_stale_promoted": self._total_stale_promoted,
                "total_prune_candidates": self._total_prune_candidates,
                "total_contradiction_sweep_clusters": (
                    self._total_contradiction_sweep_clusters
                ),
            }
