import json
import subprocess
import sys
from pathlib import Path

from tools.verify_ai_dev_memory_trial import verify_trial_folder


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_mnemos_enabled_trial_requires_memory_calls_and_nonempty_logs(tmp_path):
    trial = tmp_path / "mnemos_enabled"
    trial.mkdir()
    manifest = {
        "run_label": "AI_DEV_MEMORY_TRIAL_MNEMOS_ENABLED",
        "memory_condition": "mnemos_enabled",
        "agent": "codex",
        "model": "unknown",
        "started_at": "2026-06-26T00:00:00Z",
        "completed_at": "2026-06-26T01:00:00Z",
        "final_status": "completed",
        "app_task": "Local Issue Tracker",
        "token_counts_available": False,
        "estimated_input_tokens": 1,
        "estimated_output_tokens": 1,
        "claim_boundary": {"local_development_evidence_only": True},
    }
    (trial / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_jsonl(
        trial / "memory_calls.jsonl",
        [
            {"tool": "health_check"},
            {"tool": "get_capabilities"},
            {"tool": "search_memory"},
            {"tool": "record_decision"},
            {"tool": "summarize_session_handoff"},
        ],
    )
    _write_jsonl(trial / "agent_route_log.jsonl", [{"phase": "orientation"}])
    _write_jsonl(trial / "repo_activity.jsonl", [{"activity": "create_file"}])
    _write_jsonl(trial / "test_runs.jsonl", [{"command": "npm test", "passed": True}])
    _write_jsonl(trial / "user_interventions.jsonl", [])
    _write_jsonl(trial / "wrong_turns.jsonl", [])
    (trial / "final_report.md").write_text("# Report\ncompleted\n", encoding="utf-8")
    (trial / "blockers.md").write_text("None\n", encoding="utf-8")

    result = verify_trial_folder(trial, condition="mnemos_enabled")
    assert result["valid"] is True


def test_empty_mnemos_trial_is_invalid(tmp_path):
    trial = tmp_path / "mnemos_enabled"
    trial.mkdir()
    manifest = {
        "run_label": "AI_DEV_MEMORY_TRIAL_MNEMOS_ENABLED",
        "memory_condition": "mnemos_enabled",
        "agent": "unknown",
        "model": "unknown",
        "started_at": "2026-06-26T00:00:00Z",
        "app_task": "Local Issue Tracker",
        "token_counts_available": False,
        "claim_boundary": {"local_development_evidence_only": True},
    }
    (trial / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in [
        "memory_calls.jsonl",
        "agent_route_log.jsonl",
        "repo_activity.jsonl",
        "test_runs.jsonl",
        "user_interventions.jsonl",
        "wrong_turns.jsonl",
        "final_report.md",
        "blockers.md",
    ]:
        (trial / name).write_text("", encoding="utf-8")

    result = verify_trial_folder(trial, condition="mnemos_enabled")
    assert result["valid"] is False
    assert "memory_calls_nonempty" in result["failed_checks"]
    assert "manifest_completion_fields_present" in result["failed_checks"]


def test_trial_verifier_command_reports_invalid_folder(tmp_path):
    trial = tmp_path / "mnemos_enabled"
    trial.mkdir()
    (trial / "run_manifest.json").write_text("{}", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "tools/verify_ai_dev_memory_trial.py",
            str(trial),
            "--condition",
            "mnemos_enabled",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert '"valid": false' in completed.stdout
