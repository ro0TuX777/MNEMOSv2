#!/usr/bin/env python3
"""MNEMOS CI/CD gates — run health, contract, and build checks as CI pipeline steps.

Usage:
  python tools/mnemos_ci_gates.py --run-health-audit
  python tools/mnemos_ci_gates.py --run-health-audit --run-container-build
  python tools/mnemos_ci_gates.py --run-health-audit --contract-dir service
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _find_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "registry" / "services.json").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing registry/services.json")


def _run_health_audit(root: Path) -> bool:
    """Run mnemos_health_audit.py and return True if audit passes."""
    script = root / "tools" / "mnemos_health_audit.py"
    if not script.exists():
        print(f"[SKIP] health audit: {script} not found")
        return True

    print("\n── Health & Contract Audit ──")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        capture_output=False,
    )
    return result.returncode == 0


def _run_container_build(root: Path) -> bool:
    """Attempt to build the Docker image and return True if build succeeds."""
    dockerfile = root / "Dockerfile"
    if not dockerfile.exists():
        print(f"[SKIP] container build: {dockerfile} not found")
        return True

    print("\n── Container Build ──")
    result = subprocess.run(
        ["docker", "build", "-t", "mnemos-service:ci-test", "."],
        cwd=str(root),
        capture_output=False,
    )
    if result.returncode == 0:
        print("[OK]   container build succeeded")
    else:
        print("[FAIL] container build failed")
    return result.returncode == 0


def _run_contract_validation(root: Path, contract_dir: str) -> bool:
    """Validate that all contract JSON files are well-formed."""
    cdir = root / contract_dir
    if not cdir.exists():
        print(f"[SKIP] contract validation: {cdir} not found")
        return True

    print("\n── Contract Validation ──")
    contracts = list(cdir.glob("*.json"))
    if not contracts:
        print(f"  [SKIP] no .json files in {cdir}")
        return True

    ok = True
    for cpath in sorted(contracts):
        try:
            with cpath.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("must be a JSON object")
            required_fields = data.get("required_fields")
            if required_fields is not None and not isinstance(required_fields, dict):
                raise ValueError("required_fields must be a dict")
            print(f"  [OK]   {cpath.name}")
        except Exception as exc:
            print(f"  [FAIL] {cpath.name}: {exc}")
            ok = False
    return ok


def _run_smoke_spec(root: Path, spec_path: Optional[str]) -> bool:
    """Run smoke spec checks against running MNEMOS instance."""
    if not spec_path:
        return True

    import os

    try:
        import requests
    except ImportError:
        print("[SKIP] smoke spec: 'requests' package not installed")
        return True

    spec_file = Path(spec_path).resolve()
    if not spec_file.exists():
        print(f"[SKIP] smoke spec: {spec_file} not found")
        return True

    with spec_file.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    base_url_env = spec.get("base_url_env", "MNEMOS_BASE_URL")
    base_url = os.getenv(base_url_env, spec.get("default_base_url", "http://localhost:8700")).rstrip("/")

    print(f"\n── Smoke Spec: {spec_file.name} ──")
    ok = True
    for check in spec.get("checks", []):
        name = check.get("name", "unnamed")
        method = check.get("method", "GET").upper()
        path = check.get("path", "/")
        expect_status = check.get("expect_status", 200)
        expect_fields = check.get("expect_json_fields", [])

        url = f"{base_url}{path}"
        try:
            if method == "POST":
                resp = requests.post(url, json=check.get("body", {}), timeout=5)
            else:
                resp = requests.get(url, timeout=5)

            if resp.status_code != expect_status:
                print(f"  [FAIL] {name}: expected status {expect_status}, got {resp.status_code}")
                ok = False
                continue

            if expect_fields:
                data = resp.json()
                missing = [f for f in expect_fields if f not in data]
                if missing:
                    print(f"  [FAIL] {name}: missing fields: {', '.join(missing)}")
                    ok = False
                    continue

            print(f"  [OK]   {name}")
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            ok = False

    return ok


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MNEMOS CI/CD gate runner")
    parser.add_argument("--run-health-audit", action="store_true", help="Run health & contract audit")
    parser.add_argument("--run-container-build", action="store_true", help="Build Docker image as smoke test")
    parser.add_argument("--contract-dir", default="service", help="Directory containing contract JSON files (default: service)")
    parser.add_argument("--smoke-spec", default="", help="Path to smoke spec JSON for live checks")
    args = parser.parse_args(argv)

    root = _find_root(Path(__file__).resolve().parent)
    gates_passed = True

    # Always run contract validation
    if not _run_contract_validation(root, args.contract_dir):
        gates_passed = False

    if args.run_health_audit:
        if not _run_health_audit(root):
            gates_passed = False

    if args.run_container_build:
        if not _run_container_build(root):
            gates_passed = False

    if args.smoke_spec:
        if not _run_smoke_spec(root, args.smoke_spec):
            gates_passed = False

    print()
    if gates_passed:
        print("All CI gates passed. ✅")
        return 0
    else:
        print("CI gates FAILED. ❌")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
