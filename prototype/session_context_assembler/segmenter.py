"""Deterministic episode segmentation from conversation turns.

This module MUST NOT read `Turn.episode_hint`. Segmentation is derived
purely from turn content using character-shingle lexical continuity, so
that the prototype's clustering is independently testable against the
corpus's human-authored episode_hint ground truth (see diagnostics.py)
rather than trained or tuned on it.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Sequence

from .models import Turn

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

DEFAULT_WINDOW = 3
DEFAULT_THRESHOLD = 0.03
SHINGLE_SIZE = 4


def _normalize(text: str) -> str:
    return _NON_ALNUM.sub("", text.lower())


def shingles(text: str, k: int = SHINGLE_SIZE) -> FrozenSet[str]:
    norm = _normalize(text)
    if not norm:
        return frozenset()
    if len(norm) < k:
        return frozenset({norm})
    return frozenset(norm[i : i + k] for i in range(len(norm) - k + 1))


def jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def segment_turns(
    turns: Sequence[Turn],
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Dict]:
    """Group turns into episodes by lexical continuity. Deterministic,
    content-only; never reads turn.episode_hint."""
    episodes: List[List[str]] = []
    current_turn_ids: List[str] = []
    recent_shingles: List[FrozenSet[str]] = []

    for turn in turns:
        turn_shingles = shingles(turn.content)
        if current_turn_ids:
            window_union: FrozenSet[str] = frozenset().union(*recent_shingles)
            score = jaccard(turn_shingles, window_union)
            if score < threshold:
                episodes.append(current_turn_ids)
                current_turn_ids = []
                recent_shingles = []
        current_turn_ids.append(turn.turn_id)
        recent_shingles.append(turn_shingles)
        if len(recent_shingles) > window:
            recent_shingles.pop(0)

    if current_turn_ids:
        episodes.append(current_turn_ids)

    return [
        {"episode_id": f"ep{idx}", "turn_ids": tuple(turn_ids)}
        for idx, turn_ids in enumerate(episodes)
    ]
