import json
import subprocess
import sys
from pathlib import Path

from tools.run_mnemos_agent_navigation_trial import (
    DEFAULT_PROTOCOL,
    score_trial_log,
    summarize_batch,
    verify_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def test_agent_navigation_protocol_is_ready_without_reopening_gatemem():
    result = verify_protocol()
    assert result["all_checks_passed"] is True
    assert result["status"] == "MNEMOS_AGENT_NAVIGATION_DEVELOPMENT_STUDY_READY"
    assert result["task_count"] >= 4


def test_protocol_claim_boundary_is_development_only():
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["authorization"]["gatemem_reopening_authorized"] is False
    assert protocol["authorization"]["sealed_evaluation_authorized"] is False
    assert protocol["authorization"]["general_memory_claim_authorized"] is False
    assert protocol["claim_boundary"]["development_evidence_only"] is True
    assert protocol["claim_boundary"]["production_readiness"] is False


def test_protocol_contains_adversarial_memory_cards():
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    cards = {card["memory_id"]: card for card in protocol["memory_cards"]}
    assert cards["mem-stale-g4-can-advance-internally"]["expected_use"] == "reject_as_stale_or_contradicted"
    assert cards["mem-overbroad-provenance-is-authorization"]["expected_use"] == "reject_as_overclaim"


def test_trial_log_scoring_rewards_boundary_safe_navigation(tmp_path):
    trial = {
        "task_id": "nav-stale-memory-rejection",
        "mode": "mnemos_memory_assisted",
        "evidence_paths_used": [
            "docs/benchmarks/gatemem_program_status.md",
            "benchmarks/results/gatemem_g4_frozen_reference_manifest.md",
        ],
        "rejected_memory_ids": ["mem-stale-g4-can-advance-internally"],
        "boundary_decision": "reject_stale_memory_and_preserve_pause",
        "final_answer": "The stale memory is contradicted by the current pause and frozen reference baseline.",
        "files_opened_count": 2,
        "searches_count": 1,
    }
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(trial), encoding="utf-8")
    result = score_trial_log(path)
    assert result["all_checks_passed"] is True
    assert result["score"] == result["max_score"]


def test_trial_log_scoring_flags_overclaims(tmp_path):
    trial = {
        "task_id": "nav-g4-evidence-chain",
        "mode": "baseline_repo_search",
        "evidence_paths_used": [
            "docs/benchmarks/gatemem_g4_offline_reference_implementation.md",
            "benchmarks/results/gatemem_g4_frozen_reference_manifest.json",
            "tools/verify_gatemem_g4_frozen.py",
        ],
        "rejected_memory_ids": [],
        "boundary_decision": "regression_reference_contract_only",
        "final_answer": "The evidence chain shows authorization security proven.",
        "files_opened_count": 3,
        "searches_count": 2,
    }
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(trial), encoding="utf-8")
    result = score_trial_log(path)
    assert result["checks"]["forbidden_claim_avoidance"] is False
    assert result["forbidden_hits"] == ["authorization security proven"]


def test_documented_navigation_command_runs_from_repository_root():
    completed = subprocess.run(
        [sys.executable, "tools/run_mnemos_agent_navigation_trial.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "MNEMOS_AGENT_NAVIGATION_DEVELOPMENT_STUDY_READY" in completed.stdout
    assert "Protocol checks passed" in completed.stdout


def test_batch_001_preserves_boundaries_while_reducing_searches():
    summary = summarize_batch(ROOT / "benchmarks" / "results")
    assert summary["complete_pair_count"] == 4
    assert summary["trial_count"] == 8
    assert summary["all_trials_passed"] is True
    assert summary["all_pairs_boundary_stable"] is True
    assert summary["total_searches_saved_by_memory_assisted"] == 5
    assert summary["mean_searches_saved_by_memory_assisted"] == 1.25
    assert summary["memory_assisted_overclaim_count"] == 0
    assert summary["baseline_overclaim_count"] == 0


def test_batch_001_contains_all_seed_tasks():
    summary = summarize_batch(ROOT / "benchmarks" / "results")
    task_ids = {row["task_id"] for row in summary["pairs"]}
    assert task_ids == {
        "nav-gatemem-status-blocker",
        "nav-g4-modification-boundary",
        "nav-g4-evidence-chain",
        "nav-stale-memory-rejection",
    }


def test_documented_batch_command_runs_from_repository_root():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_mnemos_agent_navigation_trial.py",
            "--batch-dir",
            "benchmarks/results",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Batch pairs: 4" in completed.stdout
    assert "Batch trials: 8" in completed.stdout
    assert "Searches saved: 5" in completed.stdout
