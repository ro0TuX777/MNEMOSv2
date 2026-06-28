"""Prepare a dedicated MNEMOS collection workflow for AI dev E1 task_01."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    ROOT
    / "benchmarks"
    / "evaluation"
    / "ai_dev_memory_quality_e1_task_01_starter_repo"
    / "task_control_manifest.json"
)
DEFAULT_OVERRIDE_PATH = ROOT / "docker-compose.ai_dev_task_01.override.yml"
DEFAULT_COLLECTION = "mnemos_ai_dev_e1_task_01"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--override-path", type=Path, default=DEFAULT_OVERRIDE_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    manifest = _read_json(args.manifest_path)
    manifest["mnemos_preferred_collection_name"] = args.collection
    manifest["mnemos_collection_override_file"] = str(args.override_path.relative_to(ROOT)).replace("\\", "/")
    manifest["mnemos_collection_switch_commands"] = [
        f"docker compose -f docker-compose.yml -f {args.override_path.name} up -d --build mnemos",
        "python tools/seed_mnemos_ai_dev_task_01.py",
    ]
    manifest["mnemos_collection_switch_note"] = (
        "Use the override compose file to restart MNEMOS on an isolated task collection before the MNEMOS-enabled leg."
    )
    _write_json(args.manifest_path, manifest)

    print(f"Preferred collection: {args.collection}")
    print(f"Override file: {args.override_path}")
    print("Restart command:")
    print(f"  docker compose -f docker-compose.yml -f {args.override_path.name} up -d --build mnemos")
    print("Then seed task docs:")
    print("  python tools/seed_mnemos_ai_dev_task_01.py")


if __name__ == "__main__":
    main()
