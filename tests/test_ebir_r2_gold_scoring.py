"""Tests for EBIR-R2 admin-only gold-label scoring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.test_ebir_r2_pilot_markdown import _complete_form, _run_preflight, _run_renderer


def _prepare_completed_full_responses(output_dir: Path) -> None:
    _run_preflight(output_dir)
    assert _run_renderer(output_dir).returncode == 0
    responses = output_dir / "full_responses"
    responses.mkdir()
    for form in (output_dir / "pilot_review_forms").glob("reviewer_R*.md"):
        pseudo = form.stem.replace("reviewer_", "")
        (responses / f"reviewer_{pseudo}_completed.md").write_text(
            _complete_form(form.read_text(encoding="utf-8")),
            encoding="utf-8",
        )


def test_gold_scoring_outputs_required_metrics(tmp_path):
    output_dir = tmp_path / "ebir_r2"
    _prepare_completed_full_responses(output_dir)
    output_json = output_dir / "gold_scoring.json"
    output_md = output_dir / "gold_scoring.md"

    proc = subprocess.run(
        [
            sys.executable,
            "tools/score_ebir_r2_gold_report.py",
            "--manifest",
            str(output_dir / "assignment_manifest.json"),
            "--responses-dir",
            str(output_dir / "full_responses"),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--synthetic-dry-run",
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["admin_only"] is True
    assert payload["synthetic_dry_run"] is True
    assert payload["response_count"] == 9
    for key in (
        "correct_resolution_rate_by_condition",
        "correct_escalation_or_abstention_rate_by_condition",
        "unsupported_claim_detection_rate_by_condition",
        "mean_quality_score_by_condition",
        "confidence_calibration_by_condition",
        "condition_recognition_rate",
        "reviewer_time_by_condition",
        "case_family_breakdown",
    ):
        assert key in payload
    assert set(payload["correct_resolution_rate_by_condition"]) == {
        "raw_evidence",
        "one_pass_reconciliation",
        "ebir_refinement",
    }
    assert "ADMIN-ONLY" in output_md.read_text(encoding="utf-8")


def test_gold_scoring_rejects_missing_responses(tmp_path):
    output_dir = tmp_path / "ebir_r2"
    _run_preflight(output_dir)
    responses = output_dir / "full_responses"
    responses.mkdir()

    proc = subprocess.run(
        [
            sys.executable,
            "tools/score_ebir_r2_gold_report.py",
            "--manifest",
            str(output_dir / "assignment_manifest.json"),
            "--responses-dir",
            str(responses),
            "--output-json",
            str(output_dir / "gold_scoring.json"),
            "--output-md",
            str(output_dir / "gold_scoring.md"),
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "missing response file" in (proc.stdout + proc.stderr)
