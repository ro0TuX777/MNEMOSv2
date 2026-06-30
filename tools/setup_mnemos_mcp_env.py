"""Create or refresh the isolated MNEMOS MCP bridge virtual environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCP_DIR = ROOT / "mcp_servers" / "mnemos"
VENV_DIR = MCP_DIR / ".venv"
LOCKFILE = MCP_DIR / "requirements.lock.txt"


def venv_python(venv_dir: Path = VENV_DIR) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def create_env(*, recreate: bool, upgrade_pip: bool) -> Path:
    if recreate and VENV_DIR.exists():
        shutil.rmtree(VENV_DIR)
    if not VENV_DIR.exists():
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    python = venv_python()
    if upgrade_pip:
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    return python


def install_locked(python: Path) -> None:
    if not LOCKFILE.is_file():
        raise FileNotFoundError(f"Missing MCP lockfile: {LOCKFILE}")
    run([str(python), "-m", "pip", "install", "-r", str(LOCKFILE)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the MCP venv first.")
    parser.add_argument("--upgrade-pip", action="store_true", help="Upgrade pip inside the MCP venv.")
    args = parser.parse_args()

    python = create_env(recreate=args.recreate, upgrade_pip=args.upgrade_pip)
    install_locked(python)
    print(f"MCP venv ready: {VENV_DIR}")
    print(f"MCP Python: {python}")
    print("Claude Desktop should use this Python as the mnemos command.")


if __name__ == "__main__":
    main()
