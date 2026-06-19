"""Tests for EBIR-R2 end-to-end evaluation orchestration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.test_ebir_r2_pilot_markdown import _complete_form


def test_ebir_r2_e2e_blocks_when_responses_missing(tmp_path):
    output_dir = tmp_path / "ebir_r2_full"

    proc = subprocess.run(
        [
            sys.executable,
            "tools/run_ebir_r2_e2e.py",
            "--output-dir",
            str(output_dir),
            "--synthetic-dry-run",
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    status = json.loads((output_dir / "ebir_r2_e2e_status.json").read_text(encoding="utf-8"))
    assert status["preflight_pass"] is True
    assert status["render_pass"] is True
    assert status["compile_pass"] is False
    assert status["scoring_pass"] is False
    assert status["blocked_until_responses"] is True


def test_ebir_r2_e2e_scores_when_responses_present(tmp_path):
    output_dir = tmp_path / "ebir_r2_full"
    initial = subprocess.run(
        [
            sys.executable,
            "tools/run_ebir_r2_e2e.py",
            "--output-dir",
            str(output_dir),
            "--synthetic-dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert initial.returncode == 0

    responses = output_dir / "full_responses"
    responses.mkdir(exist_ok=True)
    for form in (output_dir / "full_review_forms").glob("reviewer_R*.md"):
        pseudo = form.stem.replace("reviewer_", "")
        (responses / f"reviewer_{pseudo}_completed.md").write_text(
            _complete_form(form.read_text(encoding="utf-8")),
            encoding="utf-8",
        )

    proc = subprocess.run(
        [
            sys.executable,
            "tools/run_ebir_r2_e2e.py",
            "--output-dir",
            str(output_dir),
            "--responses-dir",
            str(responses),
            "--synthetic-dry-run",
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    status = json.loads((output_dir / "ebir_r2_e2e_status.json").read_text(encoding="utf-8"))
    assert status["overall_pass"] is True
    assert status["scoring_pass"] is True
    assert status["value_claims_blocked"] is True
    scoring = json.loads((output_dir / "ebir_r2_full_gold_scoring.json").read_text(encoding="utf-8"))
    assert scoring["status"] == "PASS"
    assert scoring["response_count"] == 27
