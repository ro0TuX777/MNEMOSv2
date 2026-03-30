"""
DecayRunner — calendar-based utility decay for inactive memories.

Wave 4: background hygiene pass that applies time-based utility decay to
memories whose ``last_used_at`` (or ``created_at`` fallback) is older than a
configurable inactivity horizon.  When utility drops below the stale
threshold, ``lifecycle_state`` is set to ``"stale"``.

Design principles
-----------------
- Linear decay per elapsed day past the horizon.  Predictable and auditable.
- ``last_used_at`` takes priority over ``created_at`` as the reference.
- ``lifecycle_state`` is only set to ``"stale"``; the runner never deletes or
  hard-suppresses anything.
- ``dry_run=True`` computes but does not mutate — returns an accurate report.
- Engrams without governance metadata are counted as skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from mnemos.engram.model import Engram


def _parse_iso(ts: str) -> datetime:
    """Parse ISO timestamp to aware datetime; epoch on failure."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _elapsed_days(ref_iso: str, now_dt: datetime) -> float:
    """Return days elapsed between ref_iso and now_dt.  Never negative."""
    ref_dt = _parse_iso(ref_iso)
    delta = now_dt - ref_dt
    return max(0.0, delta.total_seconds() / 86400.0)


@dataclass
class DecayConfig:
    """Configuration for the decay runner."""

    horizon_days: int = 60
    # Inactivity horizon.  Decay begins once elapsed days exceed this value.

    decay_per_day: float = 0.005
    # Utility lost per day past the horizon.

    stale_threshold: float = 0.20
    # lifecycle_state is set to "stale" when utility_score drops below this.

    min_utility: float = 0.0
    # Hard floor — utility_score is never reduced below this value.


@dataclass
class DecayReport:
    """Summary of one decay pass."""

    scanned: int = 0
    # Engrams examined (those with governance metadata).

    decayed: int = 0
    # Engrams whose utility_score was reduced.

    stale_promoted: int = 0
    # Engrams whose lifecycle_state was set to "stale" this pass.

    skipped: int = 0
    # Engrams without governance metadata.


class DecayRunner:
    """
    Applies time-based utility decay to a list of Engrams in place.

    Usage
    -----
    ::

        runner = DecayRunner()
        report = runner.run(engrams, now_iso="2026-03-30T00:00:00Z")
        # report.decayed  -> number of engrams that lost utility
        # report.stale_promoted -> number promoted to lifecycle_state="stale"

    Dry-run mode
    ------------
    Pass ``dry_run=True`` to compute the report without mutating any Engram.
    Useful for audit/preview before applying hygiene.
    """

    def __init__(self, config: DecayConfig | None = None) -> None:
        self._cfg = config or DecayConfig()

    def run(
        self,
        engrams: List[Engram],
        now_iso: str | None = None,
        dry_run: bool = False,
    ) -> DecayReport:
        """
        Apply decay to each Engram in the list.

        Args:
            engrams:  List of Engram objects to sweep.
            now_iso:  ISO UTC timestamp representing "now".
                      Defaults to the current UTC time.
            dry_run:  If True, compute the report without mutating anything.

        Returns:
            DecayReport with counts of what was (or would be) changed.
        """
        cfg = self._cfg
        now_dt = _parse_iso(now_iso) if now_iso else datetime.now(timezone.utc)
        report = DecayReport()

        for engram in engrams:
            gov = engram.governance
            if gov is None:
                report.skipped += 1
                continue

            report.scanned += 1

            # Use last_used_at if set; fall back to created_at.
            ref_iso = gov.last_used_at or engram.created_at
            elapsed = _elapsed_days(ref_iso, now_dt)

            if elapsed <= cfg.horizon_days:
                continue  # Within horizon — no decay.

            decay_days = elapsed - cfg.horizon_days
            delta = cfg.decay_per_day * decay_days
            new_utility = max(cfg.min_utility, gov.utility_score - delta)

            if new_utility >= gov.utility_score:
                continue  # No change (already at floor).

            report.decayed += 1

            will_go_stale = (
                new_utility < cfg.stale_threshold
                and gov.lifecycle_state not in ("stale", "prune_candidate", "archived")
            )
            if will_go_stale:
                report.stale_promoted += 1

            if not dry_run:
                gov.utility_score = new_utility
                if will_go_stale:
                    gov.lifecycle_state = "stale"

        return report
