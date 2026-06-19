"""Tests for the external EBIR-R2 reviewer trial kit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_external_trial_prepare_builds_separated_bundles(tmp_path):
    output_dir = tmp_path / "external_trial"

    proc = subprocess.run(
        [
            sys.executable,
            "tools/ebir_r2_trial.py",
            "prepare",
            "--pilot",
            "--output-dir",
            str(output_dir),
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    reviewer_bundle = output_dir / "reviewer_bundle"
    admin_bundle = output_dir / "admin_bundle"
    forms = sorted((reviewer_bundle / "packets").glob("reviewer_R*.md"))

    assert reviewer_bundle.exists()
    assert admin_bundle.exists()
    assert (reviewer_bundle / "README_FOR_REVIEWERS.md").exists()
    assert (reviewer_bundle / "RETURN_CHECKLIST.md").exists()
    assert (reviewer_bundle / "manifest_public.json").exists()
    assert (admin_bundle / "assignment_manifest.json").exists()
    assert (admin_bundle / "preflight_report.json").exists()
    assert len(forms) == 3

    public_manifest = json.loads(
        (reviewer_bundle / "manifest_public.json").read_text(encoding="utf-8")
    )
    assert public_manifest["reviewer_file_count"] == 3
    assert public_manifest["reviewer_packet_count"] == 9
    assert public_manifest["human_value_claim"].startswith("blocked")

    reviewer_text = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in reviewer_bundle.rglob("*")
        if path.is_file()
    )
    assert "raw_evidence" not in reviewer_text
    assert "one_pass_reconciliation" not in reviewer_text
    assert "ebir_refinement" not in reviewer_text
    assert "gold_label" not in reviewer_text
    assert "assignment_manifest" not in reviewer_text


def test_external_trial_validate_rejects_admin_marker_in_reviewer_bundle(tmp_path):
    output_dir = tmp_path / "external_trial"
    subprocess.run(
        [
            sys.executable,
            "tools/ebir_r2_trial.py",
            "prepare",
            "--pilot",
            "--output-dir",
            str(output_dir),
            "--fail-on-gate",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    packet = next((output_dir / "reviewer_bundle" / "packets").glob("reviewer_R*.md"))
    packet.write_text(
        packet.read_text(encoding="utf-8") + "\ncondition_key: raw_evidence\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "tools/ebir_r2_trial.py",
            "validate",
            "--output-dir",
            str(output_dir),
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "condition_key" in (proc.stdout + proc.stderr)

