"""
Reinforcement — deterministic trust/utility/stability update rules.

Wave 3: called by ReflectPath once per candidate after usage classification.
All updates are small, configurable, and clamped to safe ranges.

Design principles
-----------------
- Prefer precision over recall: use small deltas, never overshoot.
- Updates are idempotent in direction — calling reflect twice gives
  the right direction, just a larger cumulative shift.
- Trust moves more slowly than utility (smaller default deltas).
- Vetoed and unknown candidates receive no updates.
- All bounds are enforced with clamp — no score ever leaves [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple

from mnemos.retrieval.base import SearchResult
from mnemos.governance.usage_detector import UsageLabel


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class ReinforcementConfig:
    """
    All update deltas and clamping bounds for the reinforcement rules.

    Keep values small and change them via config, not code.
    """

    # ── Utility deltas ──────────────────────────────────────────────────
    utility_used: float = 0.05
    # Used memory gets a moderate positive boost.

    utility_ignored: float = -0.01
    # Ignored memory loses a small amount of priority over time.

    utility_contradiction_loser: float = -0.03
    # Contradiction loser gets a slightly larger penalty than plain ignore.

    # ── Trust deltas ────────────────────────────────────────────────────
    trust_used: float = 0.02
    # Trust moves more slowly; small positive for confirmed useful memory.

    trust_ignored: float = 0.0
    # No trust change for ignored by default.

    trust_contradiction_loser: float = -0.02
    # Loser status reflects weak/stale information; mild trust decay.

    # ── Stability delta (used only) ─────────────────────────────────────
    stability_used: float = 0.02
    # Stability increases for every confirmed use.

    # ── Clamp bounds ───────────────────────────────────────────────────
    utility_min: float = 0.0
    utility_max: float = 1.0
    trust_min: float = 0.0
    trust_max: float = 1.0
    stability_min: float = 0.0
    stability_max: float = 1.0


class Reinforcement:
    """
    Applies update rules to a single candidate's GovernanceMeta in place.

    Usage
    -----
    ::

        reinf = Reinforcement()
        util_delta, trust_delta = reinf.apply(result, UsageLabel.USED, now_iso)

    Returns
    -------
    (utility_delta, trust_delta) — the deltas actually applied (0.0 if no
    governance metadata is present on the engram).
    """

    def __init__(self, config: ReinforcementConfig | None = None) -> None:
        self._cfg = config or ReinforcementConfig()

    # ── Public API ────────────────────────────────────────────────────────

    def apply(
        self,
        result: SearchResult,
        label: UsageLabel,
        now_iso: str,
    ) -> Tuple[float, float]:
        """
        Update ``result.engram.governance`` in place.

        Args:
            result:   SearchResult whose GovernanceMeta will be mutated.
            label:    UsageLabel for this candidate.
            now_iso:  ISO-format UTC timestamp for ``last_used_at``.

        Returns:
            ``(utility_delta, trust_delta)`` — 0.0 for each field that
            was not changed (VETOED, UNKNOWN, or no governance metadata).
        """
        gov = result.engram.governance
        if gov is None:
            return 0.0, 0.0

        cfg = self._cfg

        if label == UsageLabel.USED:
            return self._apply_used(gov, cfg, now_iso)

        if label == UsageLabel.IGNORED:
            return self._apply_ignored(gov, cfg)

        if label == UsageLabel.CONTRADICTED:
            return self._apply_contradicted(gov, cfg)

        # VETOED, UNKNOWN — no updates
        return 0.0, 0.0

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _apply_used(gov, cfg: ReinforcementConfig, now_iso: str) -> Tuple[float, float]:
        util_delta = cfg.utility_used
        trust_delta = cfg.trust_used

        gov.utility_score = _clamp(
            gov.utility_score + util_delta, cfg.utility_min, cfg.utility_max
        )
        gov.trust_score = _clamp(
            gov.trust_score + trust_delta, cfg.trust_min, cfg.trust_max
        )
        gov.stability_score = _clamp(
            gov.stability_score + cfg.stability_used,
            cfg.stability_min,
            cfg.stability_max,
        )
        gov.last_used_at = now_iso
        gov.usage_count = getattr(gov, "usage_count", 0) + 1

        return util_delta, trust_delta

    @staticmethod
    def _apply_ignored(gov, cfg: ReinforcementConfig) -> Tuple[float, float]:
        util_delta = cfg.utility_ignored
        trust_delta = cfg.trust_ignored

        gov.utility_score = _clamp(
            gov.utility_score + util_delta, cfg.utility_min, cfg.utility_max
        )
        if trust_delta != 0.0:
            gov.trust_score = _clamp(
                gov.trust_score + trust_delta, cfg.trust_min, cfg.trust_max
            )
        # last_used_at intentionally not updated for ignored memories
        return util_delta, trust_delta

    @staticmethod
    def _apply_contradicted(gov, cfg: ReinforcementConfig) -> Tuple[float, float]:
        util_delta = cfg.utility_contradiction_loser
        trust_delta = cfg.trust_contradiction_loser

        gov.utility_score = _clamp(
            gov.utility_score + util_delta, cfg.utility_min, cfg.utility_max
        )
        gov.trust_score = _clamp(
            gov.trust_score + trust_delta, cfg.trust_min, cfg.trust_max
        )
        return util_delta, trust_delta
