"""Offline-only diagnostics comparing predicted episodes to ground truth.

This is the ONLY module in this package permitted to read `Turn.episode_hint`.
It exists purely to score segmenter.py's output against the corpus's
human-authored episode boundaries for Phase 1/Phase 5 review. Nothing here
may be imported by segmenter.py, selector.py, or assembler.py - the
selection path must never see episode_hint, even indirectly.

No call in this module produces a benchmark claim. Phase 3 baseline replay
(full history vs. sliding window vs. governed selection) has not run.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .models import Turn


def ground_truth_groups(turns: Sequence[Turn]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for turn in turns:
        groups.setdefault(turn.episode_hint or "", []).append(turn.turn_id)
    return groups


def pairwise_boundary_agreement(predicted_episodes: Sequence[Dict], turns: Sequence[Turn]) -> float:
    """Rand-index-style pairwise agreement between predicted episode
    grouping and the corpus's ground-truth episode_hint grouping.

    Diagnostic only. 1.0 means every pair of turns agrees on whether they
    belong to the same episode; this is not an accuracy claim against any
    Phase 3 baseline.
    """
    gt = ground_truth_groups(turns)
    turn_to_gt: Dict[str, str] = {}
    for hint, turn_ids in gt.items():
        for turn_id in turn_ids:
            turn_to_gt[turn_id] = hint

    turn_to_pred: Dict[str, str] = {}
    for episode in predicted_episodes:
        for turn_id in episode["turn_ids"]:
            turn_to_pred[turn_id] = episode["episode_id"]

    ordered_turn_ids = [t.turn_id for t in turns]
    agree = 0
    total = 0
    for i in range(len(ordered_turn_ids)):
        for j in range(i + 1, len(ordered_turn_ids)):
            a, b = ordered_turn_ids[i], ordered_turn_ids[j]
            gt_same = turn_to_gt.get(a) == turn_to_gt.get(b)
            pred_same = turn_to_pred.get(a) == turn_to_pred.get(b)
            if gt_same == pred_same:
                agree += 1
            total += 1
    return (agree / total) if total else 1.0
