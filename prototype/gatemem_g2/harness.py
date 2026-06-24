"""External-artifact harness for the clean-projection-only G2 adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from prototype.gatemem_g1 import clean_projection_from_dict
from prototype.gatemem_g1.io import write_json_rows_external
from prototype.gatemem_g1.models import CleanInputProjection

from .adapter import OfflineGovernedAdapter


def load_clean_projections_jsonl(path: str | Path) -> list[CleanInputProjection]:
    projections: list[CleanInputProjection] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Projection row {line_number} is not an object.")
            projection = clean_projection_from_dict(value)
            if projection.checkpoint_id in seen:
                raise ValueError(f"Duplicate checkpoint_id: {projection.checkpoint_id}")
            seen.add(projection.checkpoint_id)
            projections.append(projection)
    return projections


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_offline_adapter(
    projections: list[CleanInputProjection],
    adapter: OfflineGovernedAdapter,
    *,
    predictions_path: str | Path,
    diagnostics_path: str | Path,
) -> dict[str, Any]:
    results = [adapter.evaluate(projection) for projection in projections]
    write_json_rows_external(
        (result.prediction for result in results), predictions_path
    )
    write_json_rows_external(
        (result.diagnostic.to_dict() for result in results), diagnostics_path
    )
    actions: dict[str, int] = {}
    for result in results:
        action = result.diagnostic.normalized_action
        actions[action] = actions.get(action, 0) + 1
    deletion_rows = [
        result
        for result in results
        if result.diagnostic.deletion_evaluation_status != "not_applicable"
    ]
    provenance_valid = sum(
        1 for result in results if result.diagnostic.provenance_integrity
    )
    cross_principal = sum(
        result.diagnostic.cross_principal_candidate_count for result in results
    )
    blocked_cross_principal = sum(
        result.diagnostic.blocked_cross_principal_count for result in results
    )
    return {
        "schema_version": "gatemem-g2-run-summary-v1",
        "adapter_version": "gatemem-g2-offline-v1",
        "projection_count": len(projections),
        "prediction_count": len(results),
        "action_counts": actions,
        "redaction_count": sum(result.diagnostic.redaction_applied for result in results),
        "denial_count": sum(result.diagnostic.denial_applied for result in results),
        "visible_deletion_count": len(deletion_rows),
        "visible_deletion_refusal_count": sum(
            result.diagnostic.normalized_action == "refuse" for result in deletion_rows
        ),
        "provenance_integrity_rate": (
            provenance_valid / len(results) if results else 0.0
        ),
        "cross_principal_candidate_count": cross_principal,
        "blocked_cross_principal_count": blocked_cross_principal,
        "blocked_cross_principal_rate": (
            blocked_cross_principal / cross_principal if cross_principal else 1.0
        ),
        "predictions_sha256": _sha256(predictions_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "offline_only": True,
        "deletion_capability_claim": False,
    }

