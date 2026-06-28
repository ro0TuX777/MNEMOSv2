"""Injected, offline-only retrieval/disclosure shadow observation boundary."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from .models import (
    CleanInputProjection,
    DisclosureResult,
    RetrievedArtifact,
    ShadowObservation,
)

_VISIBLE_DELETE_QUERY = re.compile(
    r"\b(?:delete|deleted|deletion|forget|forgotten|erased|removed\s+from\s+(?:memory|record))\b",
    re.IGNORECASE,
)
_VISIBLE_HISTORICAL_QUERY = re.compile(
    r"\b(?:old|earlier|previous|previously|before|used\s+to|replaced|first\s+one|"
    r"temporary\s+(?:safe|confidential|callback)|updated\s+on|switched\s+to)\b",
    re.IGNORECASE,
)
_VISIBLE_DELETE_HISTORY = re.compile(
    r"\b(?:delete|deleted|deletion|forget|forgotten|erase|erased|remove|removed|do\s+not\s+retain)\b",
    re.IGNORECASE,
)

RetrievalProbe = Callable[[CleanInputProjection], Sequence[RetrievedArtifact]]
DisclosureProbe = Callable[
    [CleanInputProjection, tuple[RetrievedArtifact, ...]], DisclosureResult
]


def visible_deletion_relevant(projection: CleanInputProjection) -> bool:
    """Conservative visible-input signal; never uses a benchmark query label."""

    if not (
        _VISIBLE_DELETE_QUERY.search(projection.query_text)
        or _VISIBLE_HISTORICAL_QUERY.search(projection.query_text)
    ):
        return False
    return any(_VISIBLE_DELETE_HISTORY.search(turn.text) for turn in projection.ordered_turns)


def _observation(
    projection: CleanInputProjection,
    *,
    retrieval_outcome: str,
    disclosure_outcome: str,
    response_text: str = "",
    used_record_ids: tuple[str, ...] = (),
    retrieved_count: int = 0,
    disclosed_count: int = 0,
    deletion_evaluation_status: str = "not_applicable",
    policy_reference: str = "offline-shadow-policy",
) -> ShadowObservation:
    return ShadowObservation(
        checkpoint_id=projection.checkpoint_id,
        projection_digest=projection.projection_digest,
        retrieval_outcome=retrieval_outcome,
        disclosure_outcome=disclosure_outcome,
        response_text=response_text,
        used_record_ids=used_record_ids,
        retrieved_count=retrieved_count,
        disclosed_count=disclosed_count,
        deletion_evaluation_status=deletion_evaluation_status,
        policy_reference=policy_reference,
    )


def observe_shadow(
    projection: CleanInputProjection,
    retrieval_probe: RetrievalProbe,
    disclosure_probe: DisclosureProbe,
    *,
    deletion_mode: str = "unsupported",
) -> ShadowObservation:
    """Observe retrieval/disclosure without connecting to the MNEMOS runtime.

    ``deletion_mode`` may be ``unsupported`` or ``simulated_shadow``. The
    unsupported mode skips retrieval entirely. The simulated mode may inspect
    what an injected offline probe would expose, but its response content is
    discarded and the normalizer is required to emit a refusal.
    """

    if deletion_mode not in {"unsupported", "simulated_shadow"}:
        raise ValueError("deletion_mode must be unsupported or simulated_shadow")

    deletion_relevant = visible_deletion_relevant(projection)
    if deletion_relevant and deletion_mode == "unsupported":
        return _observation(
            projection,
            retrieval_outcome="not_run",
            disclosure_outcome="not_evaluated",
            deletion_evaluation_status="unsupported",
            policy_reference="gatemem-g1-no-deletion-capability",
        )

    artifacts = tuple(retrieval_probe(projection))
    record_ids = [artifact.record_id for artifact in artifacts]
    if any(not record_id for record_id in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("Retrieved artifact IDs must be non-empty and unique.")
    if not artifacts:
        return _observation(
            projection,
            retrieval_outcome="empty",
            disclosure_outcome="not_evaluated",
            deletion_evaluation_status=(
                "simulated_shadow" if deletion_relevant else "not_applicable"
            ),
        )

    decision = disclosure_probe(projection, artifacts)
    if decision.outcome not in {"allowed", "redacted", "denied"}:
        raise ValueError("Disclosure outcome must be allowed, redacted, or denied.")
    retrieved_ids = set(record_ids)
    disclosed_ids = tuple(dict.fromkeys(decision.disclosed_record_ids))
    if not set(disclosed_ids).issubset(retrieved_ids):
        raise ValueError("Disclosure returned an artifact that was not retrieved.")
    if decision.outcome == "denied" and disclosed_ids:
        raise ValueError("A denied disclosure cannot identify disclosed records.")

    if deletion_relevant:
        return _observation(
            projection,
            retrieval_outcome="available",
            disclosure_outcome=decision.outcome,
            retrieved_count=len(artifacts),
            disclosed_count=len(disclosed_ids),
            deletion_evaluation_status="simulated_shadow",
            policy_reference=decision.policy_reference,
        )

    return _observation(
        projection,
        retrieval_outcome="available",
        disclosure_outcome=decision.outcome,
        response_text=decision.response_text if decision.outcome != "denied" else "",
        used_record_ids=disclosed_ids if decision.outcome != "denied" else (),
        retrieved_count=len(artifacts),
        disclosed_count=len(disclosed_ids),
        policy_reference=decision.policy_reference,
    )
