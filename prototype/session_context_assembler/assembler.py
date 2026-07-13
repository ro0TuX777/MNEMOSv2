"""Orchestrates segmentation, extraction, and selection into a bounded,
labeled context package.

Output is the required Phase 1 contract:
session_id, task_id, prototype_version, seed, selected_episode_ids,
selected_turn_ids, selected_parent_engram_ids, selected_source_ids,
synthetic_context_labels, selection_rationale, token_estimate,
corpus_manifest_hash, case_hash.

This module performs no I/O: it never opens, writes, or deletes a file, and
it never calls into mnemos/, service/, or mnemos_sdk/. It cannot mutate
Engrams, governance state, authority, retrieval ranking, or promotion
state because it has no access to anything that could perform such a
mutation - there is no import path from this package to runtime MNEMOS
code.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import PROTOTYPE_VERSION
from .corpus import compute_case_hash
from .extractor import extract_ids_from_turn
from .models import Turn, turn_from_dict
from .segmenter import segment_turns
from .selector import select_episodes

DEFAULT_TOKEN_BUDGET = 300


def assemble_context_package(
    case: dict,
    corpus_manifest_hash: str,
    seed: int = 0,
    token_budget: Optional[int] = None,
) -> Dict:
    turns: List[Turn] = [turn_from_dict(t) for t in case["conversation_history"]]
    turns_by_id = {t.turn_id: t for t in turns}
    turn_position = {t.turn_id: idx for idx, t in enumerate(turns)}

    episodes = segment_turns(turns)

    budget = (
        token_budget
        if token_budget is not None
        else case.get("expected_context_budget", DEFAULT_TOKEN_BUDGET)
    )

    selected_episodes, token_estimate, rationale = select_episodes(
        episodes, case["current_task"], turns_by_id, budget, seed=seed
    )

    selected_turn_ids: List[str] = []
    selected_parent_engram_ids = set()
    selected_source_ids = set()
    synthetic_context_labels = []

    for episode in selected_episodes:
        episode_decisions = set()
        episode_sources = set()
        for turn_id in episode["turn_ids"]:
            decisions, sources = extract_ids_from_turn(turns_by_id[turn_id])
            episode_decisions |= decisions
            episode_sources |= sources

        selected_turn_ids.extend(episode["turn_ids"])
        selected_parent_engram_ids |= episode_decisions
        selected_source_ids |= episode_sources

        synthetic_context_labels.append(
            {
                "episode_id": episode["episode_id"],
                "label": "synthetic_context",
                "non_authoritative": True,
                "non_promotable": True,
                "parent_turn_ids": list(episode["turn_ids"]),
                "parent_engram_ids": sorted(episode_decisions),
                "parent_source_ids": sorted(episode_sources),
            }
        )

    selected_turn_ids = sorted(set(selected_turn_ids), key=lambda tid: turn_position[tid])

    return {
        "session_id": case["session_id"],
        "task_id": case["task_id"],
        "prototype_version": PROTOTYPE_VERSION,
        "seed": seed,
        "selected_episode_ids": [ep["episode_id"] for ep in selected_episodes],
        "selected_turn_ids": selected_turn_ids,
        "selected_parent_engram_ids": sorted(selected_parent_engram_ids),
        "selected_source_ids": sorted(selected_source_ids),
        "synthetic_context_labels": synthetic_context_labels,
        "selection_rationale": rationale,
        "token_estimate": token_estimate,
        "corpus_manifest_hash": corpus_manifest_hash,
        "case_hash": compute_case_hash(case),
    }
