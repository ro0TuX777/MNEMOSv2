"""Plain data containers for the session-context-assembler prototype.

No behavior, no I/O, no MNEMOS runtime imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Turn:
    turn_id: str
    speaker: str
    content: str
    episode_hint: Optional[str] = None
    eligible: bool = True
    # Additive, optional structured source-link pool (Phase 2R). Lets a
    # turn declare source IDs it is grounded in without requiring the ID to
    # be regex-extractable from `content`. Defaults to empty so r0 cases
    # (which carry no such field) are unaffected.
    linked_source_ids: Tuple[str, ...] = field(default_factory=tuple)


def turn_from_dict(d: dict) -> Turn:
    return Turn(
        turn_id=d["turn_id"],
        speaker=d["speaker"],
        content=d["content"],
        episode_hint=d.get("episode_hint"),
        eligible=d.get("eligible", True),
        linked_source_ids=tuple(d.get("linked_source_ids", ())),
    )
