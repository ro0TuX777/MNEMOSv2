"""Tests for tools/run_slo_reliability_gate.py."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_slo_reliability_gate.py"
TMP_BASE = ROOT / "tests" / "_tmp"
TMP_BASE.mkdir(parents=True, exist_ok=True)


def _run(output_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--output-dir",
        str(output_dir),
        *args,
    ]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)


def test_slo_reliability_gate_generates_artifacts_and_passes():
    out_dir = TMP_BASE / f"slo_gate_pass_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _run(out_dir, "--stage", "canary_25", "--fail-on-breach")
    assert result.returncode == 0, result.stderr or result.stdout

    raw_files = sorted((out_dir / "raw").glob("slo_reliability_*_raw.json"))
    report_files = sorted((out_dir / "summaries").glob("slo_reliability_*_report.md"))
    assert raw_files, "Expected raw SLO artifact"
    assert report_files, "Expected markdown SLO report"

    payload = json.loads(raw_files[-1].read_text(encoding="utf-8"))
    assert payload["stage"] == "canary_25"
    assert payload["pass"] is True
    assert payload["results"]["stale_cache_survival_rate"] <= 0.0005
    assert payload["results"]["suppression_drift_rate"] <= 0.005
    assert payload["results"]["contradictions_checked"] >= 1


def test_slo_reliability_gate_fails_on_threshold_breach_when_enforced():
    out_dir = TMP_BASE / f"slo_gate_fail_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Impossible budget to force a deterministic breach.
    result = _run(out_dir, "--stage", "canary_25", "--max-p95-ms", "-1", "--fail-on-breach")
    assert result.returncode == 1, result.stderr or result.stdout

    raw_files = sorted((out_dir / "raw").glob("slo_reliability_*_raw.json"))
    assert raw_files, "Expected raw SLO artifact on failure path"
    payload = json.loads(raw_files[-1].read_text(encoding="utf-8"))
    assert payload["pass"] is False
    assert payload["evaluation"]["checks"]["latency_p95_within_budget"] is False
