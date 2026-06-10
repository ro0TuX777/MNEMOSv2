"""
Counterfactual explainability for governed search results.

W5 Explainability: "what would it take" traces computed by inverting one
factor of the deterministic governed-score product while holding the others
fixed:

    governed_score = retrieval_score × trust × utility × freshness
                     × contradiction × veto

Pure arithmetic — no models, no retrieval calls, no stored state. For each
analysed candidate the output answers, per modifier: what value would tie the
current rank-1 governed score, whether that value is reachable within the
modifier's valid range, and (when it is not) the best rank reachable at the
range maximum. Vetoed candidates get policy hints instead — a score-floor
veto is the only veto reversible by a threshold change.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mnemos.governance.models.governance_decision import GovernanceDecision

# Valid output ranges of the modifier-producing policies.
# trust/utility: UtilityPolicy maps scores into [0.75, 1.25].
# freshness: exponential decay, 1.0 for brand-new memories.
# contradiction: 0.25 for group losers, 1.0 otherwise.
MODIFIER_RANGES: Dict[str, tuple] = {
    "trust": (0.75, 1.25),
    "utility": (0.75, 1.25),
    "freshness": (0.0, 1.0),
    "contradiction": (0.25, 1.0),
}

_SCORE_FLOOR_MARKER = "below threshold"


def compute_counterfactuals(
    decisions: List[GovernanceDecision],
    top_n: int = 3,
    min_score_threshold: float = 0.0,
    created_at_by_id: Optional[Dict[str, str]] = None,
    freshness_half_life_days: float = 180.0,
) -> List[Dict[str, Any]]:
    """
    Compute distance-to-rank-1 counterfactuals for the strongest non-rank-1
    candidates (suppressed candidates included).

    Args:
        decisions:           GovernanceDecision list from the read path.
        top_n:               How many candidates to analyse.
        min_score_threshold: Effective score-floor veto threshold for the
                             active policy profile (used in policy hints).
        created_at_by_id:    Optional ``{engram_id: created_at}`` map. When a
                             candidate's timestamp is present, the freshness
                             counterfactual is inverted to age terms
                             ("would tie rank 1 if younger than N days").
        freshness_half_life_days:
                             Effective freshness half-life for the active
                             policy profile (used for the age inversion).

    Returns:
        One entry per analysed candidate: current rank (None when the
        candidate is excluded from ranked results), the gap to rank 1,
        per-modifier counterfactual statements, and policy hints.
    """
    included = [d for d in decisions if d.veto_pass and not d.suppressed]
    if not included:
        return []

    ranked = sorted(included, key=lambda d: (-d.governed_score, d.engram_id))
    rank_of = {d.engram_id: i + 1 for i, d in enumerate(ranked)}
    rank1_score = ranked[0].governed_score
    included_scores = [d.governed_score for d in ranked]

    targets = sorted(
        (d for d in decisions if rank_of.get(d.engram_id) != 1),
        key=lambda d: (-d.governed_score, d.engram_id),
    )[:top_n]

    entries: List[Dict[str, Any]] = []
    for decision in targets:
        entry: Dict[str, Any] = {
            "engram_id": decision.engram_id,
            "current_rank": rank_of.get(decision.engram_id),
            "governed_score": round(decision.governed_score, 4),
            "rank1_gap": round(rank1_score - decision.governed_score, 4),
            "counterfactuals": [],
            "policy_hints": [],
        }

        if not decision.veto_pass:
            entry["policy_hints"].append(_veto_hint(decision, min_score_threshold))
            entries.append(entry)
            continue

        modifiers = {
            "trust": decision.trust_modifier,
            "utility": decision.utility_modifier,
            "freshness": decision.freshness_modifier,
            "contradiction": decision.contradiction_modifier,
        }
        for name, current in modifiers.items():
            counterfactual = _modifier_counterfactual(
                name, current, decision.governed_score, rank1_score, included_scores
            )
            if counterfactual is None:
                continue
            if name == "freshness" and created_at_by_id:
                _invert_freshness_to_age(
                    counterfactual,
                    created_at_by_id.get(decision.engram_id),
                    freshness_half_life_days,
                )
            entry["counterfactuals"].append(counterfactual)

        if decision.suppressed_by_contradiction:
            entry["policy_hints"].append(
                f"Suppressed as contradiction loser to {decision.contradiction_winner}; "
                "the contradiction counterfactual above shows the outcome had this "
                "candidate won its group."
            )

        entries.append(entry)

    return entries


def _modifier_counterfactual(
    name: str,
    current: float,
    governed_score: float,
    rank1_score: float,
    included_scores: List[float],
) -> Optional[Dict[str, Any]]:
    if current <= 0.0 or governed_score <= 0.0:
        return None

    base = governed_score / current  # product of every other factor
    required = rank1_score / base
    if required <= current:
        return None  # already at or above rank 1 via this factor

    _, range_max = MODIFIER_RANGES[name]
    achievable = required <= range_max
    counterfactual: Dict[str, Any] = {
        "modifier": name,
        "current": round(current, 4),
        "required_for_rank1": round(required, 4),
        "achievable": achievable,
    }
    if achievable:
        counterfactual["statement"] = (
            f"If {name}_modifier were {required:.4f} instead of {current:.4f}, "
            f"governed_score would be {rank1_score:.4f} and this candidate would tie rank 1."
        )
    else:
        best_score = base * range_max
        best_rank = 1 + sum(1 for s in included_scores if s > best_score)
        counterfactual["best_achievable_rank"] = best_rank
        counterfactual["statement"] = (
            f"{name}_modifier cannot reach {required:.4f} (range max {range_max}); "
            f"at its maximum this candidate would rank {best_rank}."
        )
    return counterfactual


def _invert_freshness_to_age(
    counterfactual: Dict[str, Any],
    created_at: Optional[str],
    half_life_days: float,
) -> None:
    """
    Restate an achievable freshness counterfactual in age terms.

    Inverts the read-path decay ``freshness = 0.5 ** (age / half_life)``:
    the maximum age that still yields the required modifier is
    ``half_life × log2(1 / required)``. Falls back silently to the
    modifier-level statement when the timestamp is missing/unparseable or
    the required modifier exceeds 1.0 (no age is fresh enough).
    """
    age_days = _age_days(created_at)
    if age_days is None:
        return
    counterfactual["current_age_days"] = round(age_days, 1)

    required = counterfactual["required_for_rank1"]
    if not counterfactual["achievable"] or required > 1.0 or required <= 0.0:
        return
    max_age_days = max(half_life_days, 1.0) * math.log2(1.0 / required)
    counterfactual["max_age_days_for_rank1"] = round(max_age_days, 1)
    counterfactual["statement"] = (
        f"This result lost ground to freshness decay ({age_days:.0f} days old); "
        f"it would tie rank 1 if it were younger than {max_age_days:.0f} days "
        f"(freshness_modifier {required:.4f} vs current {counterfactual['current']:.4f})."
    )


def _age_days(created_at: Optional[str]) -> Optional[float]:
    if not created_at:
        return None
    try:
        parsed = datetime.fromisoformat(created_at.rstrip("Z"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = (datetime.now(tz=timezone.utc) - parsed).total_seconds() / 86400.0
        return max(age, 0.0)
    except (ValueError, AttributeError):
        return None


def _veto_hint(decision: GovernanceDecision, min_score_threshold: float) -> str:
    reason = decision.veto_reason or "unspecified veto"
    if _SCORE_FLOOR_MARKER in reason:
        return (
            f"Current min_score veto threshold ({min_score_threshold:.4f}) suppressed "
            f"this result; a threshold of {decision.retrieval_score:.4f} or lower "
            "would have included it."
        )
    # deletion_state / toxic vetoes are state-based, not score-based
    return f"Vetoed ({reason}); no modifier or threshold change can include this candidate."
