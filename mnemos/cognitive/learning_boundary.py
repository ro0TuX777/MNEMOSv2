"""
Semantic/procedural learning boundary helpers for MNEMOS cognitive cycles.

The boundary is intentionally explicit: cycle records may describe learning
activity, but they must not smuggle procedural mutation into semantic memory
or audit trails.  These helpers keep write classes small, serialisable, and
easy for validation harnesses to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet


EPISODIC_WRITE = "episodic_write"
SEMANTIC_WRITE = "semantic_write"
SEMANTIC_CANDIDATE_WRITE = "semantic_candidate_write"
GOVERNANCE_SCORE_UPDATE = "governance_score_update"
FORECAST_OUTCOME_WRITE = "forecast_outcome_write"
CACHE_WRITE = "cache_write"
AUDIT_WRITE = "audit_write"
PROCEDURAL_CHANGE_CANDIDATE = "procedural_change_candidate"
BLOCKED_PROCEDURAL_MUTATION = "blocked_procedural_mutation"


LEARNING_WRITE_CLASSES: FrozenSet[str] = frozenset({
    EPISODIC_WRITE,
    SEMANTIC_WRITE,
    SEMANTIC_CANDIDATE_WRITE,
    GOVERNANCE_SCORE_UPDATE,
    FORECAST_OUTCOME_WRITE,
    CACHE_WRITE,
    AUDIT_WRITE,
    PROCEDURAL_CHANGE_CANDIDATE,
    BLOCKED_PROCEDURAL_MUTATION,
})

AUTHORITATIVE_WRITE_CLASSES: FrozenSet[str] = frozenset({
    EPISODIC_WRITE,
    SEMANTIC_WRITE,
    GOVERNANCE_SCORE_UPDATE,
    FORECAST_OUTCOME_WRITE,
    CACHE_WRITE,
    AUDIT_WRITE,
})

ADVISORY_ONLY_WRITE_CLASSES: FrozenSet[str] = frozenset({
    SEMANTIC_CANDIDATE_WRITE,
    PROCEDURAL_CHANGE_CANDIDATE,
    BLOCKED_PROCEDURAL_MUTATION,
})

PATTERN_TYPE_TO_WRITE_CLASS: Dict[str, str] = {
    "descriptive": SEMANTIC_CANDIDATE_WRITE,
    "operational_recommendation": PROCEDURAL_CHANGE_CANDIDATE,
    "policy_mutation": BLOCKED_PROCEDURAL_MUTATION,
    "routing_mutation": BLOCKED_PROCEDURAL_MUTATION,
    "template_mutation": BLOCKED_PROCEDURAL_MUTATION,
}


@dataclass(frozen=True)
class LearningBoundaryDecision:
    """Classification of a proposed learning/write event."""

    write_class: str
    authoritative_for_retrieval: bool
    advisory_only: bool
    mutation_allowed: bool
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "write_class": self.write_class,
            "authoritative_for_retrieval": self.authoritative_for_retrieval,
            "advisory_only": self.advisory_only,
            "mutation_allowed": self.mutation_allowed,
            "reason": self.reason,
        }


def classify_learning_write(write_class: str) -> LearningBoundaryDecision:
    """Return the boundary decision for a declared write class."""
    if write_class not in LEARNING_WRITE_CLASSES:
        raise ValueError(f"Unknown learning write class: {write_class}")

    if write_class == AUDIT_WRITE:
        return LearningBoundaryDecision(
            write_class=write_class,
            authoritative_for_retrieval=False,
            advisory_only=False,
            mutation_allowed=True,
            reason="Audit writes are append-only and non-authoritative for retrieval.",
        )

    if write_class == BLOCKED_PROCEDURAL_MUTATION:
        return LearningBoundaryDecision(
            write_class=write_class,
            authoritative_for_retrieval=False,
            advisory_only=True,
            mutation_allowed=False,
            reason="Policy, routing, or template mutation requires explicit governance approval.",
        )

    if write_class in ADVISORY_ONLY_WRITE_CLASSES:
        return LearningBoundaryDecision(
            write_class=write_class,
            authoritative_for_retrieval=False,
            advisory_only=True,
            mutation_allowed=False,
            reason="Candidate writes are advisory until separately promoted by governance.",
        )

    return LearningBoundaryDecision(
        write_class=write_class,
        authoritative_for_retrieval=True,
        advisory_only=False,
        mutation_allowed=True,
        reason="Class is allowed as an authoritative non-procedural write.",
    )


def classify_pattern_candidate(pattern_type: str) -> LearningBoundaryDecision:
    """
    Classify a pattern candidate according to its behavioral risk.

    Descriptive candidates are semantic candidate writes.  Candidates that
    recommend operations are procedural change candidates.  Any policy,
    routing, or template mutation is blocked until explicit governance review.
    """
    write_class = PATTERN_TYPE_TO_WRITE_CLASS.get(pattern_type)
    if write_class is None:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    return classify_learning_write(write_class)
