"""GateMem-compatible prediction normalization with explicit G1 limits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io import write_json_rows_external
from .models import ShadowObservation
from .projector import PROHIBITED_SCORING_FIELDS

VALID_ACTIONS = frozenset({"answer", "answer_redacted", "refuse", "no_memory"})


def _contains_prohibited_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key) in PROHIBITED_SCORING_FIELDS or _contains_prohibited_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_prohibited_field(item) for item in value)
    return False


def normalize_prediction(observation: ShadowObservation) -> dict[str, Any]:
    if not observation.shadow_only:
        raise ValueError("G1 accepts shadow-only observations.")

    deletion_status = observation.deletion_evaluation_status
    if deletion_status not in {"not_applicable", "unsupported", "simulated_shadow"}:
        raise ValueError("Unknown deletion evaluation status.")

    used_record_ids = list(observation.used_record_ids)
    if deletion_status != "not_applicable":
        action = "refuse"
        answer = (
            "Deletion capability is unsupported in GateMem G1."
            if deletion_status == "unsupported"
            else "Deletion behavior was observed in simulation-only shadow mode; no forgetting claim is made."
        )
        used_record_ids = []
    elif observation.disclosure_outcome == "denied":
        action = "refuse"
        answer = "The offline disclosure policy denied this request."
        used_record_ids = []
    elif observation.retrieval_outcome == "empty":
        action = "no_memory"
        answer = "No eligible memory was available."
        used_record_ids = []
    elif observation.retrieval_outcome != "available":
        action = "refuse"
        answer = "The offline shadow observation did not produce an eligible result."
        used_record_ids = []
    elif observation.disclosure_outcome == "redacted":
        action = "answer_redacted"
        answer = observation.response_text
    elif observation.disclosure_outcome == "allowed":
        action = "answer"
        answer = observation.response_text
    else:
        action = "refuse"
        answer = "The offline disclosure state was not eligible for an answer."
        used_record_ids = []

    if action in {"answer", "answer_redacted"} and not answer.strip():
        raise ValueError("Answering observations must provide non-empty response text.")
    prediction = {
        "checkpoint_id": observation.checkpoint_id,
        "action": action,
        "answer": answer,
        "answer_structured": {
            "gatemem_g1": {
                "shadow_only": True,
                "projection_digest": observation.projection_digest,
                "retrieval_outcome": observation.retrieval_outcome,
                "disclosure_outcome": observation.disclosure_outcome,
                "deletion_evaluation_status": deletion_status,
                "deletion_capability_claim": False,
                "policy_reference": observation.policy_reference,
            }
        },
        "used_record_ids": used_record_ids,
    }
    if prediction["action"] not in VALID_ACTIONS:
        raise AssertionError("Normalizer emitted an invalid GateMem action.")
    if _contains_prohibited_field(prediction):
        raise AssertionError("Normalizer emitted a scoring-only field.")
    if deletion_status != "not_applicable" and prediction["action"] == "no_memory":
        raise AssertionError("G1 must never present deletion as successful forgetting.")
    return prediction


def write_predictions_jsonl(
    observations: Iterable[ShadowObservation], output_path: str | Path
) -> int:
    """Write normalized predictions; the caller owns output-path isolation."""

    rows = [normalize_prediction(observation) for observation in observations]
    seen: set[str] = set()
    for row in rows:
        checkpoint_id = row["checkpoint_id"]
        if checkpoint_id in seen:
            raise ValueError(f"Duplicate checkpoint_id: {checkpoint_id}")
        seen.add(checkpoint_id)
    return write_json_rows_external(rows, output_path)
