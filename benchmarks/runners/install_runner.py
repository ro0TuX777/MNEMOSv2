"""
MNEMOS Benchmark - Installer Overhead Runner (Track 3)
========================================================

Product question: Does the installer add value or just ceremony?

Compares installer-generated deployment vs manual baseline.
Does NOT require Docker - uses --dry-run mode for file generation.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _run_installer(profile: str, output_dir: Path, dry_run: bool = True) -> Dict[str, Any]:
    """Run the installer for a given profile and measure results."""
    cmd = [
        sys.executable, "-m", "installer",
        "--profile", profile,
        "--output-dir", str(output_dir),
    ]
    if dry_run:
        cmd.append("--dry-run")

    t0 = time.perf_counter()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
    )
    elapsed = time.perf_counter() - t0

    return {
        "exit_code": result.returncode,
        "elapsed_s": round(elapsed, 3),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _validate_generated_files(output_dir: Path, profile: str) -> Dict[str, Any]:
    """Validate that generated files exist and are well-formed."""
    checks = {}

    # Check compose file
    compose_path = output_dir / "docker-compose.generated.yml"
    checks["compose_exists"] = compose_path.exists()
    if compose_path.exists():
        content = compose_path.read_text()
        checks["compose_has_services"] = "services:" in content or "# MNEMOS" in content
        checks["compose_bytes"] = len(content)

    # Check env file
    env_path = output_dir / ".env.mnemos"
    checks["env_exists"] = env_path.exists()
    if env_path.exists():
        content = env_path.read_text()
        checks["env_has_profile"] = f"MNEMOS_PROFILE={profile}" in content
        checks["env_bytes"] = len(content)

    # Check manifest
    manifest_path = output_dir / "mnemos_profile.yaml"
    checks["manifest_exists"] = manifest_path.exists()
    if manifest_path.exists():
        content = manifest_path.read_text()
        checks["manifest_has_profile"] = profile in content
        checks["manifest_bytes"] = len(content)

    checks["all_valid"] = all(
        checks.get(k, False) for k in
        ["compose_exists", "env_exists", "manifest_exists"]
    )

    return checks


def _manual_baseline(profile: str, output_dir: Path) -> Dict[str, Any]:
    """Simulate manual deployment setup (copy template, create env)."""
    templates_dir = Path(__file__).resolve().parents[2] / "installer" / "templates"

    t0 = time.perf_counter()

    # Step 1: Find and copy the right template
    template_file = templates_dir / f"{profile}.yml"
    compose_dest = output_dir / "docker-compose.yml"
    errors = []

    if template_file.exists():
        shutil.copy2(template_file, compose_dest)
    else:
        errors.append(f"Template {profile}.yml not found")

    # Step 2: Create env file manually
    env_dest = output_dir / ".env.mnemos"
    env_content = f"MNEMOS_PROFILE={profile}\nMNEMOS_GPU_DEVICE=cuda\n"
    env_dest.write_text(env_content)

    elapsed = time.perf_counter() - t0

    return {
        "elapsed_s": round(elapsed, 3),
        "errors": errors,
        "files_created": [str(compose_dest), str(env_dest)],
        "has_manifest": False,  # Manual path doesn't generate manifest
    }


def run_installer_track(
    profiles: List[str] = None,
    n_runs: int = 3,
) -> Dict[str, Any]:
    """
    Run full Track 3: Installer Overhead Benchmarks.

    Compares installer vs manual for each profile across multiple runs.
    """
    if profiles is None:
        profiles = ["core_memory_appliance", "governance_native"]

    print("\n" + "=" * 70)
    print("  TRACK 3: Installer Overhead Benchmarks")
    print("=" * 70)

    results = {"track": "installer", "status": "success", "profiles": {}}

    for profile in profiles:
        print(f"\n  Profile: {profile}")
        results["profiles"][profile] = {"installer": [], "manual": [], "validation": []}

        for run_idx in range(n_runs):
            # Installer path
            with tempfile.TemporaryDirectory() as tmpdir:
                installer_result = _run_installer(profile, Path(tmpdir))
                results["profiles"][profile]["installer"].append(installer_result)

            # Manual path
            with tempfile.TemporaryDirectory() as tmpdir:
                manual_result = _manual_baseline(profile, Path(tmpdir))
                results["profiles"][profile]["manual"].append(manual_result)

            # Validation (on a full installer run, not dry-run)
            with tempfile.TemporaryDirectory() as tmpdir:
                _run_installer(profile, Path(tmpdir), dry_run=False)
                validation = _validate_generated_files(Path(tmpdir), profile)
                results["profiles"][profile]["validation"].append(validation)

        # Summary
        installer_times = [r["elapsed_s"] for r in results["profiles"][profile]["installer"]]
        manual_times = [r["elapsed_s"] for r in results["profiles"][profile]["manual"]]
        validations = results["profiles"][profile]["validation"]

        import numpy as np
        summary = {
            "installer_median_s": round(float(np.median(installer_times)), 3),
            "manual_median_s": round(float(np.median(manual_times)), 3),
            "installer_errors": sum(1 for r in results["profiles"][profile]["installer"] if r["exit_code"] != 0),
            "manual_errors": sum(1 for r in results["profiles"][profile]["manual"] if r.get("errors")),
            "validation_pass_rate": sum(1 for v in validations if v.get("all_valid")) / len(validations),
            "installer_generates_manifest": any(v.get("manifest_exists") for v in validations),
            "manual_generates_manifest": False,
        }
        results["profiles"][profile]["summary"] = summary

        print(f"    Installer: {summary['installer_median_s']:.3f}s (median)  "
              f"Errors: {summary['installer_errors']}")
        print(f"    Manual:    {summary['manual_median_s']:.3f}s (median)  "
              f"Errors: {summary['manual_errors']}")
        print(f"    Validation pass rate: {summary['validation_pass_rate']:.0%}")
        print(f"    Installer generates manifest: {summary['installer_generates_manifest']}")

    return results
