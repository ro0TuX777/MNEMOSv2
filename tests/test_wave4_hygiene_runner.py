"""Tests for tools/run_wave4_hygiene.py control-loop runner."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_wave4_hygiene.py"
SEED = ROOT / "benchmarks" / "truthsets" / "wave4_hygiene_seed.json"
TMP_BASE = ROOT / "tests" / "_tmp"
TMP_BASE.mkdir(parents=True, exist_ok=True)


def _run(output_dir: Path, mode: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--mode",
        mode,
        "--input",
        str(SEED),
        "--output-dir",
        str(output_dir),
        "--fail-on-gate",
    ]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)


def test_wave4_hygiene_dry_run_passes_and_emits_artifacts():
    out_dir = TMP_BASE / f"wave4_dry_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _run(out_dir, "dry-run")
    assert result.returncode == 0, result.stderr or result.stdout

    raw_files = sorted((out_dir / "raw").glob("wave4_hygiene_*_raw.json"))
    report_files = sorted((out_dir / "summaries").glob("wave4_hygiene_*_report.md"))
    assert raw_files, "Expected raw JSON artifact"
    assert report_files, "Expected markdown report artifact"

    payload = json.loads(raw_files[-1].read_text(encoding="utf-8"))
    assert payload["mode"] == "dry-run"
    assert payload["pass"] is True
    assert payload["report"]["decay"]["stale_promoted"] >= 1
    assert payload["report"]["prune"]["promoted"] >= 1
    assert payload["report"]["sweep"]["contradictions_found"] >= 1


def test_wave4_hygiene_apply_mode_writes_mutated_snapshot():
    out_dir = TMP_BASE / f"wave4_apply_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _run(out_dir, "apply")
    assert result.returncode == 0, result.stderr or result.stdout

    raw_files = sorted((out_dir / "raw").glob("wave4_hygiene_*_raw.json"))
    assert raw_files, "Expected raw JSON artifact"
    payload = json.loads(raw_files[-1].read_text(encoding="utf-8"))
    mutated_path = payload.get("mutated_output_path")
    assert mutated_path, "Expected mutated output path in apply mode"
    assert Path(mutated_path).exists()
