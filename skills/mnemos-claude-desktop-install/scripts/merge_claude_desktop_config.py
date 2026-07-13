#!/usr/bin/env python3
"""Merge the MNEMOS MCP server entry into Claude Desktop config."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_claude_config_path() -> Path:
    appdata = os.environ.get("APPDATA", "").strip()
    if not appdata:
        raise RuntimeError("APPDATA is not set; this helper currently targets Windows Claude Desktop installs.")
    return Path(appdata) / "Claude" / "claude_desktop_config.json"


def _resolve_repo_root(raw: str) -> Path:
    root = Path(raw).resolve()
    server_path = root / "mcp_servers" / "mnemos" / "server.py"
    if not server_path.exists():
        raise FileNotFoundError(f"MNEMOS MCP server not found at {server_path}")
    return root


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.stem}.{stamp}.bak.json")
    shutil.copy2(path, backup_path)
    return backup_path


def _mnemos_entry(repo_root: Path, python_command: str, base_url: str, timeout_s: int) -> dict[str, Any]:
    return {
        "command": python_command,
        "args": [str((repo_root / "mcp_servers" / "mnemos" / "server.py").resolve())],
        "env": {
            "MNEMOS_BASE_URL": base_url,
            "MNEMOS_TIMEOUT_S": str(timeout_s),
        },
    }


def _merge_config(
    existing: dict[str, Any],
    *,
    repo_root: Path,
    python_command: str,
    base_url: str,
    timeout_s: int,
) -> dict[str, Any]:
    merged = dict(existing)
    mcp_servers = dict(merged.get("mcpServers") or {})
    mcp_servers["mnemos"] = _mnemos_entry(repo_root, python_command, base_url, timeout_s)
    merged["mcpServers"] = mcp_servers
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Path to the MNEMOS repo root.")
    parser.add_argument(
        "--config-path",
        default=str(_default_claude_config_path()),
        help="Path to claude_desktop_config.json.",
    )
    parser.add_argument(
        "--python-command",
        default=sys.executable,
        help="Absolute Python interpreter path for Claude Desktop to run.",
    )
    parser.add_argument("--base-url", default="http://localhost:8700")
    parser.add_argument("--timeout-s", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = _resolve_repo_root(args.repo_root)
    config_path = Path(args.config_path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_json(config_path)
    merged = _merge_config(
        existing,
        repo_root=repo_root,
        python_command=args.python_command,
        base_url=args.base_url,
        timeout_s=args.timeout_s,
    )

    if args.dry_run:
        print(json.dumps(merged, indent=2))
        return

    backup_path = _backup(config_path)
    config_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Claude Desktop config: {config_path}")
    if backup_path is not None:
        print(f"Backup created: {backup_path}")


if __name__ == "__main__":
    main()
