"""
ReflectMetrics — thread-safe accumulator for reflect-path telemetry.

Extended into Governor.stats() so the /governance/stats endpoint
surfaces reflect activity alongside query-time veto/suppression counts.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from mnemos.governance.reflect_path import ReflectResult


class ReflectMetrics:
    """
    Accumulated reflect-path statistics.

    All mutation is protected by a reentrant lock so Governor can safely
    call ``record()`` from concurrent request threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_reflect_runs: int = 0
        self._total_used_memories: int = 0
        self._total_ignored_memories: int = 0
        self._total_utility_reinforcements: int = 0
        self._total_utility_penalties: int = 0
        self._total_trust_reinforcements: int = 0
        self._total_trust_penalties: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def record(self, result: "ReflectResult") -> None:
        """Accumulate stats from one ReflectResult."""
        trust_reinforced = sum(
            1 for d in result.trust_deltas.values() if d > 0
        )
        trust_penalized = sum(
            1 for d in result.trust_deltas.values() if d < 0
        )
        with self._lock:
            self._total_reflect_runs += 1
            self._total_used_memories += len(result.used_ids)
            self._total_ignored_memories += len(result.ignored_ids)
            self._total_utility_reinforcements += result.total_reinforced
            self._total_utility_penalties += result.total_penalized
            self._total_trust_reinforcements += trust_reinforced
            self._total_trust_penalties += trust_penalized

    def to_dict(self) -> Dict[str, Any]:
        """Snapshot of current counters (safe to call from any thread)."""
        with self._lock:
            return {
                "total_reflect_runs": self._total_reflect_runs,
                "total_used_memories": self._total_used_memories,
                "total_ignored_memories": self._total_ignored_memories,
                "total_utility_reinforcements": self._total_utility_reinforcements,
                "total_utility_penalties": self._total_utility_penalties,
                "total_trust_reinforcements": self._total_trust_reinforcements,
                "total_trust_penalties": self._total_trust_penalties,
            }
