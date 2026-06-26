"""Verify external AI-developer memory trial result folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = {
    "run_manifest.json",
    "agent_route_log.jsonl",
    "repo_activity.jsonl",
    "user_interventions.jsonl",
    "wrong_turns.jsonl",
    "test_runs.jsonl",
    "final_report.md",
    "blockers.md",
}

MNEMOS_REQUIRED_FILES = REQUIRED_FILES | {"memory_calls.jsonl"}

REQUIRED_MANIFEST_START_FIELDS = {
    "run_label",
    "memory_condition",
    "agent",
    "model",
    "started_at",
    "app_task",
    "token_counts_available",
    "claim_boundary",
}

REQUIRED_MANIFEST_COMPLETION_FIELDS = {
    "completed_at",
    "final_status",
    "estimated_input_tokens",
    "estimated_output_tokens",
}

REQUIRED_MNEMOS_TOOLS = {
    "health_check",
    "get_capabilities",
    "find_related_context",
    "search_memory",
    "record_decision",
    "write_observation",
    "summarize_session_handoff",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} invalid JSONL: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(row)
    return rows


def verify_trial_folder(path: str | Path, *, condition: str) -> dict[str, Any]:
    root = Path(path)
    required_files = MNEMOS_REQUIRED_FILES if condition == "mnemos_enabled" else REQUIRED_FILES
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    memory_calls = _read_jsonl(root / "memory_calls.jsonl") if condition == "mnemos_enabled" else []
    route_rows = _read_jsonl(root / "agent_route_log.jsonl")
    repo_rows = _read_jsonl(root / "repo_activity.jsonl")
    test_rows = _read_jsonl(root / "test_runs.jsonl")
    final_report_text = (root / "final_report.md").read_text(encoding="utf-8") if (root / "final_report.md").exists() else ""

    memory_tools = {str(row.get("tool")) for row in memory_calls if row.get("tool")}
    checks = {
        "folder_exists": root.is_dir(),
        "required_files_present": all((root / name).is_file() for name in required_files),
        "manifest_start_fields_present": REQUIRED_MANIFEST_START_FIELDS.issubset(manifest),
        "manifest_completion_fields_present": REQUIRED_MANIFEST_COMPLETION_FIELDS.issubset(manifest),
        "manifest_condition_matches": manifest.get("memory_condition") == condition,
        "route_log_nonempty": len(route_rows) > 0,
        "repo_activity_nonempty": len(repo_rows) > 0,
        "test_runs_nonempty": len(test_rows) > 0,
        "final_report_nonempty": bool(final_report_text.strip()),
    }
    if condition == "mnemos_enabled":
        checks.update(
            {
                "memory_calls_nonempty": len(memory_calls) > 0,
                "required_startup_memory_calls_present": {"health_check", "get_capabilities"}.issubset(memory_tools),
                "required_workflow_memory_calls_present": bool(
                    memory_tools & {"find_related_context", "search_memory"}
                )
                and bool(memory_tools & {"record_decision", "write_observation"})
                and "summarize_session_handoff" in memory_tools,
            }
        )

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "path": str(root),
        "condition": condition,
        "valid": not failed,
        "checks": checks,
        "failed_checks": failed,
        "counts": {
            "memory_calls": len(memory_calls),
            "route_log_rows": len(route_rows),
            "repo_activity_rows": len(repo_rows),
            "test_run_rows": len(test_rows),
            "final_report_chars": len(final_report_text.strip()),
        },
        "memory_tools_observed": sorted(memory_tools),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--condition", choices=["mnemos_enabled", "no_memory"], required=True)
    args = parser.parse_args()
    result = verify_trial_folder(args.path, condition=args.condition)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
