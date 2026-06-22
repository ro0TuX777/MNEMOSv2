"""Deterministic, seeded episode selection against a current task.

Relevance scoring reuses the segmenter's character-shingle Jaccard measure.
The seed only breaks ties among episodes with identical relevance scores -
the same seed always yields the same selection for the same input, which is
the determinism property this prototype phase must demonstrate.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from .models import Turn
from .segmenter import jaccard, shingles


def _episode_text(episode: Dict, turns_by_id: Dict[str, Turn]) -> str:
    return " ".join(turns_by_id[tid].content for tid in episode["turn_ids"])


def _episode_token_estimate(episode: Dict, turns_by_id: Dict[str, Turn]) -> int:
    return len(_episode_text(episode, turns_by_id).split())


def score_episodes(
    episodes: Sequence[Dict], current_task: str, turns_by_id: Dict[str, Turn]
) -> List[Tuple[Dict, float]]:
    task_shingles = shingles(current_task)
    return [
        (ep, jaccard(task_shingles, shingles(_episode_text(ep, turns_by_id))))
        for ep in episodes
    ]


def select_episodes(
    episodes: Sequence[Dict],
    current_task: str,
    turns_by_id: Dict[str, Turn],
    token_budget: int,
    seed: int = 0,
) -> Tuple[List[Dict], int, List[str]]:
    scored = score_episodes(episodes, current_task, turns_by_id)

    buckets: Dict[float, List[Dict]] = {}
    for ep, score in scored:
        buckets.setdefault(score, []).append(ep)

    rng = random.Random(seed)
    ordered: List[Tuple[Dict, float]] = []
    for score in sorted(buckets.keys(), reverse=True):
        group = sorted(buckets[score], key=lambda e: e["episode_id"])
        rng.shuffle(group)
        ordered.extend((ep, score) for ep in group)

    selected: List[Dict] = []
    used_tokens = 0
    rationale: List[str] = []
    for ep, score in ordered:
        ep_tokens = _episode_token_estimate(ep, turns_by_id)
        if selected and used_tokens + ep_tokens > token_budget:
            rationale.append(
                f"{ep['episode_id']} skipped: relevance={score:.4f}, "
                f"would exceed token_budget={token_budget}"
            )
            continue
        selected.append(ep)
        used_tokens += ep_tokens
        rationale.append(
            f"{ep['episode_id']} selected: relevance={score:.4f}, "
            f"running_token_estimate={used_tokens}"
        )
        if used_tokens >= token_budget:
            break

    return selected, used_tokens, rationale
