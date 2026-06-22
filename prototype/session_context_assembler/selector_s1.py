"""Governed S1 selector for the bounded Phase 4R offline experiment.

S1 uses only consumer-available session inputs: task text, eligible turns,
decision/source artifacts extracted from those turns, contradiction language,
and deterministic session order. It intentionally has no access to benchmark
gold or scoring annotations.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Sequence, Tuple

from . import PROTOTYPE_VERSION
from .corpus import compute_case_hash
from .extractor import extract_ids_from_turn
from .models import Turn, turn_from_dict
from .segmenter import jaccard, segment_turns, shingles

_CONTRADICTION_SIGNAL = re.compile(
    r"contradict|supersed|unresolved|not yet resolved|no decision|"
    r"no agreement|still (?:open|under debate)|remains open|pending|"
    r"provisional|more authoritative|original .*policy|update raised|"
    r"replaces|stale",
    re.IGNORECASE,
)


def _episode_text(episode: Dict, turns_by_id: Dict[str, Turn]) -> str:
    return " ".join(turns_by_id[tid].content for tid in episode["turn_ids"])


def _episode_tokens(episode: Dict, turns_by_id: Dict[str, Turn]) -> int:
    return len(_episode_text(episode, turns_by_id).split())


def _episode_artifacts(
    episode: Dict, turns_by_id: Dict[str, Turn]
) -> Tuple[set[str], set[str], List[Turn]]:
    decisions: set[str] = set()
    sources: set[str] = set()
    artifact_turns: List[Turn] = []
    for tid in episode["turn_ids"]:
        turn = turns_by_id[tid]
        turn_decisions, turn_sources = extract_ids_from_turn(turn)
        if turn_decisions or turn_sources:
            artifact_turns.append(turn)
        decisions.update(turn_decisions)
        sources.update(turn_sources)
    return decisions, sources, artifact_turns


def _artifact_relevance(
    episode: Dict, current_task: str, turns_by_id: Dict[str, Turn]
) -> float:
    """Score artifact-bearing turns when present, avoiding nearby decoys."""
    _, _, artifact_turns = _episode_artifacts(episode, turns_by_id)
    task = shingles(current_task)
    candidates = artifact_turns or [turns_by_id[tid] for tid in episode["turn_ids"]]
    return max(jaccard(task, shingles(turn.content)) for turn in candidates)


def _tier(episode: Dict, turns_by_id: Dict[str, Turn]) -> Tuple[int, Tuple[str, ...]]:
    decisions, sources, _ = _episode_artifacts(episode, turns_by_id)
    text = _episode_text(episode, turns_by_id)
    if decisions:
        types = ["prior_decision_artifact"]
        if _CONTRADICTION_SIGNAL.search(text):
            types.append("contradiction_artifact")
        if sources:
            types.append("source_linked_evidence")
        return 1, tuple(types)
    if _CONTRADICTION_SIGNAL.search(text):
        types = ["contradiction_artifact"]
        if sources:
            types.append("source_linked_evidence")
        return 2, tuple(types)
    if sources:
        return 3, ("source_linked_evidence",)
    return 4, ()


def select_episodes_s1(
    episodes: Sequence[Dict],
    current_task: str,
    turns_by_id: Dict[str, Turn],
    token_budget: int,
    seed: int = 0,
) -> Tuple[List[Dict], int, List[str], Dict]:
    """Select mandatory artifact tiers before semantic context fill.

    Episodes are atomic. If a mandatory episode cannot fit, selection returns
    a visible partial-abstention contract naming the omitted artifact types.
    """
    position = {ep["episode_id"]: index for index, ep in enumerate(episodes)}
    rng = random.Random(seed)
    tie_break = {ep["episode_id"]: rng.random() for ep in episodes}
    ranked = []
    for episode in episodes:
        tier, artifact_types = _tier(episode, turns_by_id)
        relevance = _artifact_relevance(episode, current_task, turns_by_id)
        if tier == 4 and relevance == 0.0:
            tier = 5
        ranked.append((episode, tier, artifact_types, relevance))

    ranked.sort(
        key=lambda item: (
            item[1],
            -item[3],
            -position[item[0]["episode_id"]],
            tie_break[item[0]["episode_id"]],
        )
    )

    selected: List[Dict] = []
    used_tokens = 0
    rationale: List[str] = []
    omitted_types: set[str] = set()
    mandatory_skipped: List[str] = []

    for episode, tier, artifact_types, relevance in ranked:
        if tier > 3 and mandatory_skipped:
            rationale.append(
                f"{episode['episode_id']} not considered: semantic fill blocked "
                "after mandatory-artifact budget abstention"
            )
            continue
        tokens = _episode_tokens(episode, turns_by_id)
        if used_tokens + tokens > token_budget:
            rationale.append(
                f"{episode['episode_id']} skipped: tier={tier}, "
                f"artifact_relevance={relevance:.4f}, would exceed "
                f"token_budget={token_budget}"
            )
            if tier <= 3:
                mandatory_skipped.append(episode["episode_id"])
                omitted_types.update(artifact_types)
            continue
        selected.append(episode)
        used_tokens += tokens
        rationale.append(
            f"{episode['episode_id']} selected: tier={tier}, "
            f"artifact_relevance={relevance:.4f}, "
            f"running_token_estimate={used_tokens}"
        )

    insufficient = bool(mandatory_skipped)
    status = {
        "context_budget_insufficient": insufficient,
        "omitted_required_artifact_types": sorted(omitted_types),
        "selection_abstention_reason": (
            "mandatory eligible artifacts exceed the context budget; "
            f"omitted episodes: {', '.join(mandatory_skipped)}"
            if insufficient
            else None
        ),
    }
    return selected, used_tokens, rationale, status


def assemble_context_package_s1(
    case: Dict, corpus_manifest_hash: str, seed: int, token_budget: int
) -> Dict:
    """Build the standard governed package plus the S1 abstention contract."""
    turns = [turn_from_dict(item) for item in case["conversation_history"]]
    turns_by_id = {turn.turn_id: turn for turn in turns}
    turn_position = {turn.turn_id: index for index, turn in enumerate(turns)}
    episodes = segment_turns(turns)
    selected, token_estimate, rationale, status = select_episodes_s1(
        episodes, case["current_task"], turns_by_id, token_budget, seed
    )

    selected_turn_ids: List[str] = []
    selected_decisions: set[str] = set()
    selected_sources: set[str] = set()
    labels = []
    for episode in selected:
        decisions, sources, _ = _episode_artifacts(episode, turns_by_id)
        selected_turn_ids.extend(episode["turn_ids"])
        selected_decisions.update(decisions)
        selected_sources.update(sources)
        labels.append(
            {
                "episode_id": episode["episode_id"],
                "label": "synthetic_context",
                "non_authoritative": True,
                "non_promotable": True,
                "parent_turn_ids": list(episode["turn_ids"]),
                "parent_engram_ids": sorted(decisions),
                "parent_source_ids": sorted(sources),
            }
        )

    selected_turn_ids = sorted(set(selected_turn_ids), key=turn_position.__getitem__)
    return {
        "session_id": case["session_id"],
        "task_id": case["task_id"],
        "prototype_version": f"{PROTOTYPE_VERSION}-s1",
        "seed": seed,
        "selected_episode_ids": [ep["episode_id"] for ep in selected],
        "selected_turn_ids": selected_turn_ids,
        "selected_parent_engram_ids": sorted(selected_decisions),
        "selected_source_ids": sorted(selected_sources),
        "synthetic_context_labels": labels,
        "selection_rationale": rationale,
        "token_estimate": token_estimate,
        "corpus_manifest_hash": corpus_manifest_hash,
        "case_hash": compute_case_hash(case),
        **status,
    }
