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

E1_MANIFEST_START_FIELDS = {
    "trial_id",
    "task_id",
    "task_spec_hash",
    "client_version",
    "repo_root",
    "initial_repo_commit_or_hash",
    "tool_configuration",
    "execution_path",
    "token_accounting_method",
}

E1_MANIFEST_COMPLETION_FIELDS = {
    "acceptance_test_command",
    "acceptance_test_result",
    "cache_state_at_end",
}

MNEMOS_REQUIRED_MANIFEST_FIELDS = {
    "mnemos_base_url",
    "mcp_server_name",
    "mnemos_service_revision",
    "mcp_revision",
    "collection_name",
    "seed_snapshot",
    "configured_retrieval_profile",
    "cache_state_at_start",
    "cache_policy_version",
}

BASELINE_REQUIRED_MANIFEST_FIELDS = {
    "mnemos_or_memory_tools_allowed",
    "mnemos_base_url",
    "mcp_server_name",
    "mnemos_service_revision",
    "mcp_revision",
    "collection_name",
    "seed_snapshot",
    "configured_retrieval_profile",
    "cache_state_at_start",
    "cache_policy_version",
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

REQUIRED_MEMORY_CALL_FIELDS = {
    "timestamp",
    "tool",
    "purpose",
    "query_or_summary",
    "query_text",
    "retrieval_fingerprint",
    "execution_path",
    "cache_state_observed",
    "cache_hit",
    "configured_retrieval_profile",
    "duplicate_suppression_count",
    "result_count",
    "returned_source_ids",
    "returned_source_labels",
    "returned_source_types",
    "returned_scores",
    "used_result_ids",
    "rejected_result_ids",
    "abstention_reason",
    "verified_against_primary_evidence",
    "used_in_next_action",
    "next_action_summary",
    "helpfulness",
    "notes",
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
    condition_manifest_fields = MNEMOS_REQUIRED_MANIFEST_FIELDS if condition == "mnemos_enabled" else BASELINE_REQUIRED_MANIFEST_FIELDS
    e1_manifest_present = (
        REQUIRED_MANIFEST_START_FIELDS.issubset(manifest)
        and REQUIRED_MANIFEST_COMPLETION_FIELDS.issubset(manifest)
        and E1_MANIFEST_START_FIELDS.issubset(manifest)
        and E1_MANIFEST_COMPLETION_FIELDS.issubset(manifest)
        and condition_manifest_fields.issubset(manifest)
    )
    e1_memory_rows_complete = condition != "mnemos_enabled" or (
        len(memory_calls) > 0 and all(REQUIRED_MEMORY_CALL_FIELDS.issubset(row) for row in memory_calls)
    )
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
        "e1_manifest_fields_present": e1_manifest_present,
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
                "e1_memory_call_fields_present": e1_memory_rows_complete,
                "execution_path_declared": manifest.get("execution_path") in {"mcp", "rest_fallback"},
            }
        )
    else:
        checks.update(
            {
                "baseline_memory_calls_absent": not (root / "memory_calls.jsonl").exists(),
                "execution_path_declared": manifest.get("execution_path") == "no_memory_control",
            }
        )

    validity_checks = {
        name: passed
        for name, passed in checks.items()
        if name
        not in {
            "e1_manifest_fields_present",
            "e1_memory_call_fields_present",
            "execution_path_declared",
        }
    }
    failed = [name for name, passed in validity_checks.items() if not passed]
    e1_ready = (
        checks["e1_manifest_fields_present"]
        and checks.get("e1_memory_call_fields_present", True)
        and checks.get("execution_path_declared", True)
    )
    return {
        "path": str(root),
        "condition": condition,
        "valid": not failed,
        "e1_ready": e1_ready,
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
    parser.add_argument("--require-e1", action="store_true")
    args = parser.parse_args()
    result = verify_trial_folder(args.path, condition=args.condition)
    print(json.dumps(result, indent=2))
    if not result["valid"] or (args.require_e1 and not result["e1_ready"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
