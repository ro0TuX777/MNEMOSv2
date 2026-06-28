"""Compare validated MNEMOS-enabled and no-memory AI developer trials."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.verify_ai_dev_memory_trial import verify_trial_folder


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _md_cell(value: Any) -> str:
    if value is None:
        return "None"
    return str(value).replace("|", "\\|")


def _parse_timestamp(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return max((end - start).total_seconds(), 0.0)


def _safe_divide(numerator: float | int, denominator: float | int) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


def _none_if_zero_token_estimate(value: Any) -> int | None:
    if value in (None, "", "unknown"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return None if parsed == 0 else parsed


def _test_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed_runs = 0
    passed_runs = 0
    tests_passed = None
    tests_failed = None
    failure_total = 0
    first_passing_timestamp = None
    for row in rows:
        result = str(row.get("result", "")).lower()
        passed = row.get("passed")
        if passed is True or result == "pass":
            passed_runs += 1
            if first_passing_timestamp is None:
                first_passing_timestamp = row.get("timestamp")
        if passed is False or result == "fail":
            failed_runs += 1
        if row.get("tests_passed") is not None:
            tests_passed = row.get("tests_passed")
        if row.get("tests_failed") is not None:
            tests_failed = row.get("tests_failed")
        failure_count = row.get("failure_count")
        if failure_count is not None:
            failure_total += int(failure_count)
            tests_failed = failure_count
    return {
        "runs": len(rows),
        "passed_runs": passed_runs,
        "failed_runs": failed_runs,
        "final_tests_passed": tests_passed,
        "final_tests_failed": tests_failed,
        "failure_total": failure_total,
        "first_passing_timestamp": first_passing_timestamp,
    }


def _memory_helpfulness(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("helpfulness", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _repo_activity_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_activity: dict[str, int] = {}
    touched_targets: set[str] = set()
    changed_targets: set[str] = set()
    changed_rows = 0
    search_rows = 0
    command_rows = 0
    for row in rows:
        activity = str(row.get("activity", "unknown"))
        by_activity[activity] = by_activity.get(activity, 0) + 1
        target = row.get("target")
        if isinstance(target, str) and target:
            touched_targets.add(target)
            if activity in {"edit_file", "create_file"}:
                changed_targets.add(target)
        if activity in {"edit_file", "create_file"}:
            changed_rows += 1
        if activity == "search":
            search_rows += 1
        if activity == "command":
            command_rows += 1
    return {
        "by_activity": by_activity,
        "unique_targets_touched": len(touched_targets),
        "unique_files_changed": len(changed_targets),
        "changed_rows": changed_rows,
        "search_rows": search_rows,
        "command_rows": command_rows,
    }


def _rework_after_first_pass(repo_rows: list[dict[str, Any]], first_passing_timestamp: str | None) -> int | None:
    first_pass = _parse_timestamp(first_passing_timestamp)
    if first_pass is None:
        return None
    count = 0
    for row in repo_rows:
        when = _parse_timestamp(row.get("timestamp"))
        if when is None or when <= first_pass:
            continue
        if row.get("activity") in {"edit_file", "create_file"}:
            count += 1
    return count


def _observed_retrieval_path(memory_rows: list[dict[str, Any]], manifest: dict[str, Any], condition: str) -> str:
    if condition != "mnemos_enabled":
        return "no_memory_control"
    notes = " ".join(str(row.get("notes", "")) for row in memory_rows).lower()
    purposes = " ".join(str(row.get("purpose", "")) for row in memory_rows).lower()
    report_hints = " ".join(
        str(manifest.get(key, "")) for key in ("mcp_server_name", "configured_retrieval_profile")
    ).lower()
    combined = " ".join((notes, purposes, report_hints))
    if "rest" in combined or "/index" in combined or "/health" in combined or "/capabilities" in combined:
        return "rest_fallback"
    return "mcp"


def _observed_route_fingerprint(memory_rows: list[dict[str, Any]]) -> str | None:
    fingerprints: list[str] = []
    for row in memory_rows:
        fp = row.get("retrieval_fingerprint")
        if not isinstance(fp, str):
            continue
        normalized = fp.strip()
        if not normalized or normalized in {"not_applicable", "not_available", "unknown"}:
            continue
        if normalized not in fingerprints:
            fingerprints.append(normalized)
    if not fingerprints:
        return None
    return " | ".join(fingerprints)


def _observed_cache_state(memory_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str | None:
    states: list[str] = []
    for row in memory_rows:
        state = row.get("cache_state_observed")
        if not isinstance(state, str):
            continue
        normalized = state.strip()
        if not normalized or normalized in {"not_applicable", "unknown"}:
            continue
        if normalized not in states:
            states.append(normalized)
    if states:
        return " | ".join(states)
    manifest_state = manifest.get("cache_state_at_end") or manifest.get("cache_state_at_start")
    if manifest_state in {None, "", "unknown", "not_applicable"}:
        return None
    return str(manifest_state)


def _duplicate_suppression_total(memory_rows: list[dict[str, Any]]) -> int | None:
    counts = []
    for row in memory_rows:
        value = row.get("duplicate_suppression_count")
        if isinstance(value, int):
            counts.append(value)
    if not counts:
        return None
    return sum(counts)


def _timing_integrity(
    manifest: dict[str, Any],
    test_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    repo_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    started_at = _parse_timestamp(manifest.get("started_at"))
    completed_at = _parse_timestamp(manifest.get("completed_at"))
    if started_at is None or completed_at is None:
        return {
            "comparable": False,
            "reason": "missing_manifest_timestamps",
        }
    out_of_window = False
    observed_rows = [*test_rows, *route_rows, *repo_rows, *memory_rows]
    for row in observed_rows:
        row_time = _parse_timestamp(row.get("timestamp"))
        if row_time is None:
            continue
        if row_time < started_at or row_time > completed_at:
            out_of_window = True
            break
    if out_of_window:
        return {
            "comparable": False,
            "reason": "row_timestamps_outside_manifest_window",
        }
    return {
        "comparable": True,
        "reason": "manifest_window_consistent",
    }


def _condition_summary(
    *,
    condition: str,
    validation: dict[str, Any],
    manifest: dict[str, Any],
    memory_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    wrong_turn_rows: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    repo_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    test_summary = _test_summary(test_rows)
    activity = _repo_activity_counts(repo_rows)
    started_at = _parse_timestamp(manifest.get("started_at"))
    completed_at = _parse_timestamp(manifest.get("completed_at"))
    timing_integrity = _timing_integrity(manifest, test_rows, route_rows, repo_rows, memory_rows)
    elapsed_seconds = (
        _seconds_between(started_at, completed_at) if timing_integrity["comparable"] else None
    )
    time_to_passing_tests_seconds = (
        _seconds_between(started_at, _parse_timestamp(test_summary.get("first_passing_timestamp")))
        if timing_integrity["comparable"]
        else None
    )

    final_tests_failed = test_summary.get("final_tests_failed")
    acceptance_test_pass_rate = None
    if final_tests_failed == 0 and test_summary["passed_runs"] > 0:
        acceptance_test_pass_rate = 1.0
    elif isinstance(final_tests_failed, int) and final_tests_failed > 0:
        acceptance_test_pass_rate = 0.0

    helpfulness = _memory_helpfulness(memory_rows)
    useful_memory = helpfulness.get("useful", 0)
    harmful_memory = helpfulness.get("harmful", 0)
    neutral_memory = helpfulness.get("neutral", 0)
    memory_call_count = len(memory_rows)
    verified_memory = sum(1 for row in memory_rows if row.get("verified_against_primary_evidence") is True)

    required_constraints_satisfied = validation["valid"]
    if condition == "mnemos_enabled":
        required_constraints_satisfied = required_constraints_satisfied and validation["checks"].get(
            "required_workflow_memory_calls_present",
            False,
        )
    if condition == "no_memory":
        required_constraints_satisfied = required_constraints_satisfied and memory_call_count == 0

    retrieval_path = _observed_retrieval_path(memory_rows, manifest, condition)
    retrieval_profile = "mnemos_memory_assisted" if condition == "mnemos_enabled" else "none"
    total_estimated_tokens = None
    estimated_input_tokens = _none_if_zero_token_estimate(manifest.get("estimated_input_tokens"))
    estimated_output_tokens = _none_if_zero_token_estimate(manifest.get("estimated_output_tokens"))
    if estimated_input_tokens is not None and estimated_output_tokens is not None:
        total_estimated_tokens = estimated_input_tokens + estimated_output_tokens

    return {
        "condition": condition,
        "task_outcome": {
            "final_status": manifest.get("final_status"),
            "task_completion_rate": 1.0 if manifest.get("final_status") == "completed" else 0.0,
            "acceptance_test_pass_rate": acceptance_test_pass_rate,
            "time_to_passing_tests_seconds": time_to_passing_tests_seconds,
            "elapsed_seconds": elapsed_seconds,
            "timing_comparable": timing_integrity["comparable"],
            "timing_integrity_reason": timing_integrity["reason"],
            "required_constraints_satisfied": required_constraints_satisfied,
            "regressions_introduced": int(final_tests_failed or 0),
        },
        "workflow_efficiency": {
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "total_estimated_tokens": total_estimated_tokens,
            "token_counts_comparable": total_estimated_tokens is not None,
            "tool_calls_logged": memory_call_count + activity["command_rows"] + activity["search_rows"],
            "memory_tool_calls": memory_call_count,
            "command_rows": activity["command_rows"],
            "search_rows": activity["search_rows"],
            "failed_test_count": test_summary["failure_total"],
            "wrong_turn_count": len(wrong_turn_rows),
            "files_changed": activity["unique_files_changed"],
            "change_churn_rows": activity["changed_rows"],
            "rework_after_first_implementation": _rework_after_first_pass(
                repo_rows,
                test_summary.get("first_passing_timestamp"),
            ),
            "route_log_rows": len(route_rows),
            "repo_activity_rows": len(repo_rows),
        },
        "memory_quality": {
            "correct_source_or_decision_neighborhood": (
                useful_memory > 0 if condition == "mnemos_enabled" else None
            ),
            "provenance_retained": (
                _safe_divide(verified_memory, memory_call_count)
                if condition == "mnemos_enabled"
                else None
            ),
            "irrelevant_context_rate": (
                _safe_divide(harmful_memory, memory_call_count)
                if condition == "mnemos_enabled"
                else None
            ),
            "retrieval_precision": (
                _safe_divide(useful_memory, memory_call_count)
                if condition == "mnemos_enabled"
                else None
            ),
            "abstentions_correct": None,
            "abstentions_missed": None,
            "false_abstention_rate": None,
            "retrieved_context_usefulness": (
                _safe_divide(useful_memory, memory_call_count)
                if condition == "mnemos_enabled"
                else None
            ),
            "memory_helpfulness_breakdown": helpfulness if condition == "mnemos_enabled" else {},
            "verified_against_primary_evidence_count": verified_memory if condition == "mnemos_enabled" else 0,
            "neutral_context_count": neutral_memory if condition == "mnemos_enabled" else 0,
        },
        "retrieval_integrity_controls": {
            "seed_snapshot": manifest.get("seed_snapshot"),
            "executed_route_fingerprint": _observed_route_fingerprint(memory_rows),
            "cache_state": _observed_cache_state(memory_rows, manifest),
            "duplicate_suppression_count": _duplicate_suppression_total(memory_rows),
            "retrieval_profile_or_path": retrieval_profile,
            "retrieval_path": retrieval_path,
        },
        "validation": validation,
    }


def _pairwise_summary(mnemos: dict[str, Any], no_memory: dict[str, Any]) -> dict[str, Any]:
    mnemos_tokens = mnemos["workflow_efficiency"]["total_estimated_tokens"]
    no_memory_tokens = no_memory["workflow_efficiency"]["total_estimated_tokens"]
    token_delta = None
    if mnemos_tokens is not None and no_memory_tokens is not None:
        token_delta = mnemos_tokens - no_memory_tokens
    return {
        "task_outcome": {
            "both_completed": (
                mnemos["task_outcome"]["final_status"] == "completed"
                and no_memory["task_outcome"]["final_status"] == "completed"
            ),
            "acceptance_parity": (
                mnemos["task_outcome"]["acceptance_test_pass_rate"]
                == no_memory["task_outcome"]["acceptance_test_pass_rate"]
            ),
            "constraint_satisfaction_parity": (
                mnemos["task_outcome"]["required_constraints_satisfied"]
                == no_memory["task_outcome"]["required_constraints_satisfied"]
            ),
            "timing_comparable_between_conditions": (
                mnemos["task_outcome"]["timing_comparable"] and no_memory["task_outcome"]["timing_comparable"]
            ),
        },
        "workflow_efficiency": {
            "token_estimate_delta_mnemos_minus_no_memory": token_delta,
            "token_counts_comparable_between_conditions": (
                mnemos["workflow_efficiency"]["token_counts_comparable"]
                and no_memory["workflow_efficiency"]["token_counts_comparable"]
            ),
            "wrong_turn_delta_mnemos_minus_no_memory": (
                mnemos["workflow_efficiency"]["wrong_turn_count"]
                - no_memory["workflow_efficiency"]["wrong_turn_count"]
            ),
            "repo_activity_delta_mnemos_minus_no_memory": (
                mnemos["workflow_efficiency"]["repo_activity_rows"]
                - no_memory["workflow_efficiency"]["repo_activity_rows"]
            ),
            "failed_test_delta_mnemos_minus_no_memory": (
                mnemos["workflow_efficiency"]["failed_test_count"]
                - no_memory["workflow_efficiency"]["failed_test_count"]
            ),
        },
        "memory_quality": {
            "mnemos_used_required_memory_tools": mnemos["validation"]["checks"].get(
                "required_workflow_memory_calls_present",
                False,
            ),
            "mnemos_had_irrelevant_context": (
                (mnemos["memory_quality"]["irrelevant_context_rate"] or 0.0) > 0.0
            ),
            "mnemos_retrieved_context_usefulness": mnemos["memory_quality"]["retrieved_context_usefulness"],
            "mnemos_provenance_retained": mnemos["memory_quality"]["provenance_retained"],
        },
        "retrieval_integrity_controls": {
            "mnemos_path": mnemos["retrieval_integrity_controls"]["retrieval_path"],
            "no_memory_path": no_memory["retrieval_integrity_controls"]["retrieval_path"],
            "seed_snapshot_present": bool(mnemos["retrieval_integrity_controls"]["seed_snapshot"]),
            "executed_route_fingerprint_present": bool(
                mnemos["retrieval_integrity_controls"]["executed_route_fingerprint"]
            ),
            "cache_state_present": bool(mnemos["retrieval_integrity_controls"]["cache_state"]),
            "duplicate_suppression_observed": mnemos["retrieval_integrity_controls"]["duplicate_suppression_count"],
        },
    }


def compare_trials(mnemos_path: str | Path, no_memory_path: str | Path) -> dict[str, Any]:
    mnemos_root = Path(mnemos_path)
    no_memory_root = Path(no_memory_path)
    mnemos_validation = verify_trial_folder(mnemos_root, condition="mnemos_enabled")
    no_memory_validation = verify_trial_folder(no_memory_root, condition="no_memory")
    if not mnemos_validation["valid"] or not no_memory_validation["valid"]:
        raise ValueError("Both trial folders must validate before comparison")

    mnemos_manifest = _read_json(mnemos_root / "run_manifest.json")
    no_memory_manifest = _read_json(no_memory_root / "run_manifest.json")
    mnemos_memory = _read_jsonl(mnemos_root / "memory_calls.jsonl")
    mnemos_tests = _read_jsonl(mnemos_root / "test_runs.jsonl")
    no_memory_tests = _read_jsonl(no_memory_root / "test_runs.jsonl")
    mnemos_wrong_turns = _read_jsonl(mnemos_root / "wrong_turns.jsonl")
    no_memory_wrong_turns = _read_jsonl(no_memory_root / "wrong_turns.jsonl")
    mnemos_route = _read_jsonl(mnemos_root / "agent_route_log.jsonl")
    no_memory_route = _read_jsonl(no_memory_root / "agent_route_log.jsonl")
    mnemos_repo = _read_jsonl(mnemos_root / "repo_activity.jsonl")
    no_memory_repo = _read_jsonl(no_memory_root / "repo_activity.jsonl")

    mnemos_summary = _condition_summary(
        condition="mnemos_enabled",
        validation=mnemos_validation,
        manifest=mnemos_manifest,
        memory_rows=mnemos_memory,
        test_rows=mnemos_tests,
        wrong_turn_rows=mnemos_wrong_turns,
        route_rows=mnemos_route,
        repo_rows=mnemos_repo,
    )
    no_memory_summary = _condition_summary(
        condition="no_memory",
        validation=no_memory_validation,
        manifest=no_memory_manifest,
        memory_rows=[],
        test_rows=no_memory_tests,
        wrong_turn_rows=no_memory_wrong_turns,
        route_rows=no_memory_route,
        repo_rows=no_memory_repo,
    )

    return {
        "schema_version": "ai-dev-memory-quality-lane-result-v1",
        "benchmark_lane": "ai_dev_memory_trial_pair",
        "claim_boundary": {
            "local_development_evidence_only": True,
            "general_memory_claim": False,
        },
        "conditions": {
            "mnemos_enabled": mnemos_summary,
            "no_memory": no_memory_summary,
        },
        "pairwise_summary": _pairwise_summary(mnemos_summary, no_memory_summary),
    }


def write_markdown(summary: dict[str, Any], path: str | Path) -> None:
    m = summary["conditions"]["mnemos_enabled"]
    n = summary["conditions"]["no_memory"]
    pairwise = summary["pairwise_summary"]
    lines = [
        "# AI Developer Memory Quality Lane Result",
        "",
        "```text",
        "LOCAL_DEVELOPMENT_EVIDENCE_ONLY",
        "NO_GENERAL_MEMORY_CLAIM",
        "```",
        "",
        "## Task Outcome",
        "",
        "| Metric | MNEMOS enabled | No memory |",
        "|---|---:|---:|",
        f"| Task completion rate | {_md_cell(m['task_outcome']['task_completion_rate'])} | {_md_cell(n['task_outcome']['task_completion_rate'])} |",
        f"| Acceptance-test pass rate | {_md_cell(m['task_outcome']['acceptance_test_pass_rate'])} | {_md_cell(n['task_outcome']['acceptance_test_pass_rate'])} |",
        f"| Time to passing tests (s) | {_md_cell(m['task_outcome']['time_to_passing_tests_seconds'])} | {_md_cell(n['task_outcome']['time_to_passing_tests_seconds'])} |",
        f"| Timing comparable | {_md_cell(m['task_outcome']['timing_comparable'])} | {_md_cell(n['task_outcome']['timing_comparable'])} |",
        f"| Required constraints satisfied | {_md_cell(m['task_outcome']['required_constraints_satisfied'])} | {_md_cell(n['task_outcome']['required_constraints_satisfied'])} |",
        f"| Regressions introduced | {_md_cell(m['task_outcome']['regressions_introduced'])} | {_md_cell(n['task_outcome']['regressions_introduced'])} |",
        "",
        "## Workflow Efficiency",
        "",
        "| Metric | MNEMOS enabled | No memory |",
        "|---|---:|---:|",
        f"| Total estimated tokens | {_md_cell(m['workflow_efficiency']['total_estimated_tokens'])} | {_md_cell(n['workflow_efficiency']['total_estimated_tokens'])} |",
        f"| Token counts comparable | {_md_cell(m['workflow_efficiency']['token_counts_comparable'])} | {_md_cell(n['workflow_efficiency']['token_counts_comparable'])} |",
        f"| Logged tool calls | {_md_cell(m['workflow_efficiency']['tool_calls_logged'])} | {_md_cell(n['workflow_efficiency']['tool_calls_logged'])} |",
        f"| Failed-test count | {_md_cell(m['workflow_efficiency']['failed_test_count'])} | {_md_cell(n['workflow_efficiency']['failed_test_count'])} |",
        f"| Wrong-turn count | {_md_cell(m['workflow_efficiency']['wrong_turn_count'])} | {_md_cell(n['workflow_efficiency']['wrong_turn_count'])} |",
        f"| Files changed | {_md_cell(m['workflow_efficiency']['files_changed'])} | {_md_cell(n['workflow_efficiency']['files_changed'])} |",
        f"| Rework after first implementation | {_md_cell(m['workflow_efficiency']['rework_after_first_implementation'])} | {_md_cell(n['workflow_efficiency']['rework_after_first_implementation'])} |",
        "",
        "## Memory Quality",
        "",
        "| Metric | MNEMOS enabled | No memory |",
        "|---|---:|---:|",
        f"| Correct source / decision neighborhood | {_md_cell(m['memory_quality']['correct_source_or_decision_neighborhood'])} | {_md_cell(n['memory_quality']['correct_source_or_decision_neighborhood'])} |",
        f"| Provenance retained | {_md_cell(m['memory_quality']['provenance_retained'])} | {_md_cell(n['memory_quality']['provenance_retained'])} |",
        f"| Irrelevant-context rate | {_md_cell(m['memory_quality']['irrelevant_context_rate'])} | {_md_cell(n['memory_quality']['irrelevant_context_rate'])} |",
        f"| Retrieval precision | {_md_cell(m['memory_quality']['retrieval_precision'])} | {_md_cell(n['memory_quality']['retrieval_precision'])} |",
        f"| False-abstention rate | {_md_cell(m['memory_quality']['false_abstention_rate'])} | {_md_cell(n['memory_quality']['false_abstention_rate'])} |",
        f"| Retrieved-context usefulness | {_md_cell(m['memory_quality']['retrieved_context_usefulness'])} | {_md_cell(n['memory_quality']['retrieved_context_usefulness'])} |",
        "",
        "## Retrieval-Integrity Controls",
        "",
        "| Control | MNEMOS enabled | No memory |",
        "|---|---|---|",
        f"| Seed snapshot | {_md_cell(m['retrieval_integrity_controls']['seed_snapshot'])} | {_md_cell(n['retrieval_integrity_controls']['seed_snapshot'])} |",
        f"| Executed-route fingerprint | {_md_cell(m['retrieval_integrity_controls']['executed_route_fingerprint'])} | {_md_cell(n['retrieval_integrity_controls']['executed_route_fingerprint'])} |",
        f"| Cache state | {_md_cell(m['retrieval_integrity_controls']['cache_state'])} | {_md_cell(n['retrieval_integrity_controls']['cache_state'])} |",
        f"| Duplicate suppression count | {_md_cell(m['retrieval_integrity_controls']['duplicate_suppression_count'])} | {_md_cell(n['retrieval_integrity_controls']['duplicate_suppression_count'])} |",
        f"| Retrieval profile / path | {_md_cell(m['retrieval_integrity_controls']['retrieval_profile_or_path'])} | {_md_cell(n['retrieval_integrity_controls']['retrieval_profile_or_path'])} |",
        f"| Observed execution path | {_md_cell(m['retrieval_integrity_controls']['retrieval_path'])} | {_md_cell(n['retrieval_integrity_controls']['retrieval_path'])} |",
        "",
        "## Pairwise Summary",
        "",
        f"- Both completed: {pairwise['task_outcome']['both_completed']}",
        f"- Acceptance parity: {pairwise['task_outcome']['acceptance_parity']}",
        f"- Constraint satisfaction parity: {pairwise['task_outcome']['constraint_satisfaction_parity']}",
        f"- Timing comparable between conditions: {pairwise['task_outcome']['timing_comparable_between_conditions']}",
        f"- Token estimate delta, MNEMOS minus no-memory: {pairwise['workflow_efficiency']['token_estimate_delta_mnemos_minus_no_memory']}",
        f"- Token counts comparable between conditions: {pairwise['workflow_efficiency']['token_counts_comparable_between_conditions']}",
        f"- Wrong-turn delta, MNEMOS minus no-memory: {pairwise['workflow_efficiency']['wrong_turn_delta_mnemos_minus_no_memory']}",
        f"- Repo-activity delta, MNEMOS minus no-memory: {pairwise['workflow_efficiency']['repo_activity_delta_mnemos_minus_no_memory']}",
        f"- MNEMOS required memory tools used: {pairwise['memory_quality']['mnemos_used_required_memory_tools']}",
        f"- MNEMOS retrieved-context usefulness: {pairwise['memory_quality']['mnemos_retrieved_context_usefulness']}",
        f"- Observed MNEMOS execution path: {pairwise['retrieval_integrity_controls']['mnemos_path']}",
        "",
        "## Interpretation",
        "",
        "This paired artifact now records task outcome, workflow efficiency, "
        "memory quality, and retrieval-integrity controls in one contract. "
        "That makes future comparisons more inspectable and reduces the risk "
        "of reading performance differences without the retrieval conditions "
        "that produced them.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnemos", type=Path, required=True)
    parser.add_argument("--no-memory", type=Path, required=True)
    parser.add_argument("--write-json", type=Path)
    parser.add_argument("--write-md", type=Path)
    args = parser.parse_args()
    summary = compare_trials(args.mnemos, args.no_memory)
    if args.write_json:
        args.write_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.write_md:
        write_markdown(summary, args.write_md)
    print(json.dumps(summary["pairwise_summary"], indent=2))


if __name__ == "__main__":
    main()
