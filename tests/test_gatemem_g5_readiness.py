import json
import subprocess
import sys
from pathlib import Path

from tools.verify_gatemem_g5_readiness import CANDIDATE, STATE, verify_readiness


ROOT = Path(__file__).resolve().parents[1]


def test_g5_packet_is_ready_for_external_handoff_but_not_execution():
    result = verify_readiness()
    assert result["all_checks_passed"] is True
    assert result["status"] == "GATEMEM_G5_PACKET_READY_FOR_EXTERNAL_HANDOFF"
    assert result["evaluation_state"] == "SEALED_EVALUATION_BLOCKED_EXTERNAL_INPUTS_REQUIRED"


def test_candidate_is_frozen_g4_and_requires_external_acceptance():
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert candidate["candidate_composite_sha256"] == "ed3b5c7672e591b039183eaa2d8c7c7630a655575bfc7866558d53f2eb874c52"
    assert candidate["custodian_acceptance"]["accepted"] is False
    assert candidate["development_evidence"]["held_out_eligible"] is False


def test_handoff_state_does_not_claim_external_work_is_complete():
    state = json.loads(STATE.read_text(encoding="utf-8"))
    assert all(state["completed_internal_preparation"].values())
    assert not any(state["external_requirements"].values())
    assert state["performance_claim_authorized"] is False
    assert state["generalization_claim_authorized"] is False


def test_packet_contains_no_sealed_corpus_or_labels():
    packet = ROOT / "benchmarks" / "evaluation"
    names = {path.name for path in packet.rglob("*") if path.is_file()}
    assert names == {
        "gatemem_g5_candidate_nomination.json",
        "gatemem_g5_handoff_state.json",
    }


def test_documented_readiness_command_runs_from_repository_root():
    completed = subprocess.run(
        [sys.executable, "tools/verify_gatemem_g5_readiness.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "GATEMEM_G5_PACKET_READY_FOR_EXTERNAL_HANDOFF" in completed.stdout
    assert "SEALED_EVALUATION_BLOCKED_EXTERNAL_INPUTS_REQUIRED" in completed.stdout
