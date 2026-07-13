"""Verify G5 handoff readiness without accessing or simulating sealed data."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CANDIDATE = ROOT / "benchmarks" / "evaluation" / "gatemem_g5_candidate_nomination.json"
STATE = ROOT / "benchmarks" / "evaluation" / "gatemem_g5_handoff_state.json"
G4_MANIFEST = ROOT / "benchmarks" / "results" / "gatemem_g4_frozen_reference_manifest.json"


def verify_readiness() -> dict[str, Any]:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    state = json.loads(STATE.read_text(encoding="utf-8"))
    g4 = json.loads(G4_MANIFEST.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(G4_MANIFEST.read_bytes()).hexdigest()

    from tools.verify_gatemem_g4_frozen import verify_frozen_reference

    frozen = verify_frozen_reference()
    required_paths = [state["packet_index"], *state["required_documents"]]
    checks = {
        "g4_frozen_reference_verified": frozen["verified"] is True,
        "candidate_composite_matches_g4": candidate["candidate_composite_sha256"] == g4["composite_sha256"],
        "candidate_manifest_hash_matches": candidate["candidate_manifest_sha256"] == manifest_sha,
        "candidate_not_misrepresented_as_held_out": candidate["development_evidence"]["held_out_eligible"] is False,
        "candidate_external_acceptance_pending": candidate["custodian_acceptance"]["accepted"] is False,
        "packet_files_present": all((ROOT / path).is_file() for path in required_paths),
        "internal_preparation_complete": all(state["completed_internal_preparation"].values()),
        "external_requirements_unfulfilled": not any(state["external_requirements"].values()),
        "sealed_evaluation_still_blocked": state["evaluation_state"] == "SEALED_EVALUATION_BLOCKED_EXTERNAL_INPUTS_REQUIRED",
        "no_performance_or_generalization_claim": state["performance_claim_authorized"] is False and state["generalization_claim_authorized"] is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"G5 readiness verification failed: {failed}")
    return {
        "status": state["status"],
        "evaluation_state": state["evaluation_state"],
        "checks": checks,
        "all_checks_passed": True,
    }


def main() -> None:
    result = verify_readiness()
    print(result["status"])
    print(result["evaluation_state"])
    print(f"Checks passed: {len(result['checks'])}/{len(result['checks'])}")


if __name__ == "__main__":
    main()
