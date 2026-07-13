"""Prove the MCP bridge uses its isolated venv and root MNEMOS still tests."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCP_DIR = ROOT / "mcp_servers" / "mnemos"
MCP_VENV = MCP_DIR / ".venv"
MCP_PYTHON = MCP_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
ROOT_REQUIREMENTS = ROOT / "requirements.txt"


def run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    completed.check_returncode()
    return completed


def assert_isolated_inputs() -> None:
    if not MCP_PYTHON.is_file():
        raise SystemExit(
            f"Missing isolated MCP Python: {MCP_PYTHON}\n"
            "Run: python tools/setup_mnemos_mcp_env.py"
        )
    root_requirements = ROOT_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    root_requirement_names = {
        line.strip().lower().split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0]
        for line in root_requirements
        if line.strip() and not line.strip().startswith("#")
    }
    if "mcp" in root_requirement_names:
        raise SystemExit("Root requirements.txt includes MCP dependencies; expected MCP deps only in mcp_servers/mnemos.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pytest", action="store_true", help="Only run the MCP venv smoke and static verifier.")
    args = parser.parse_args()

    assert_isolated_inputs()
    print(f"Root Python: {sys.executable}")
    print(f"MCP Python: {MCP_PYTHON}")
    print(f"MCP venv: {MCP_VENV}")

    run([str(MCP_PYTHON), "tools/smoke_mnemos_mcp_stdio.py"], timeout=60)
    run([sys.executable, "tools/verify_mnemos_msf_mcp.py"], timeout=60)
    if not args.skip_pytest:
        run([sys.executable, "-m", "pytest", "tests/test_mnemos_msf_mcp.py", "-q"], timeout=120)

    print("MCP isolation check passed.")


if __name__ == "__main__":
    main()
