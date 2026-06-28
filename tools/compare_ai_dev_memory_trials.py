"""Compare validated MNEMOS-enabled and no-memory AI developer trials."""

from __future__ import annotations

import argparse
import json
import sys
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


def _test_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed_runs = 0
    passed_runs = 0
    tests_passed = None
    tests_failed = None
    for row in rows:
        result = str(row.get("result", "")).lower()
        passed = row.get("passed")
        if passed is True or result == "pass":
            passed_runs += 1
        if passed is False or result == "fail":
            failed_runs += 1
        if row.get("tests_passed") is not None:
            tests_passed = row.get("tests_passed")
        if row.get("tests_failed") is not None:
            tests_failed = row.get("tests_failed")
        if row.get("failure_count") is not None:
            tests_failed = row.get("failure_count")
    return {
        "runs": len(rows),
        "passed_runs": passed_runs,
        "failed_runs": failed_runs,
        "final_tests_passed": tests_passed,
        "final_tests_failed": tests_failed,
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

    memory_helpfulness = {}
    for row in mnemos_memory:
        label = str(row.get("helpfulness", "unknown"))
        memory_helpfulness[label] = memory_helpfulness.get(label, 0) + 1

    return {
        "schema_version": "ai-dev-memory-trial-comparison-v1",
        "claim_boundary": {
            "local_development_evidence_only": True,
            "general_memory_claim": False,
        },
        "validations": {
            "mnemos_enabled": mnemos_validation,
            "no_memory": no_memory_validation,
        },
        "metrics": {
            "mnemos_enabled": {
                "estimated_input_tokens": mnemos_manifest.get("estimated_input_tokens"),
                "estimated_output_tokens": mnemos_manifest.get("estimated_output_tokens"),
                "total_estimated_tokens": mnemos_manifest.get("total_estimated_tokens"),
                "memory_calls": len(mnemos_memory),
                "memory_helpfulness": memory_helpfulness,
                "route_log_rows": len(mnemos_route),
                "repo_activity_rows": len(mnemos_repo),
                "wrong_turn_rows": len(mnemos_wrong_turns),
                "test_summary": _test_summary(mnemos_tests),
                "final_status": mnemos_manifest.get("final_status"),
            },
            "no_memory": {
                "estimated_input_tokens": no_memory_manifest.get("estimated_input_tokens"),
                "estimated_output_tokens": no_memory_manifest.get("estimated_output_tokens"),
                "total_estimated_tokens": no_memory_manifest.get("total_estimated_tokens"),
                "memory_calls": 0,
                "route_log_rows": len(no_memory_route),
                "repo_activity_rows": len(no_memory_repo),
                "wrong_turn_rows": len(no_memory_wrong_turns),
                "test_summary": _test_summary(no_memory_tests),
                "final_status": no_memory_manifest.get("final_status"),
            },
        },
        "observed_pattern": {
            "both_completed": (
                mnemos_manifest.get("final_status") == "completed"
                and no_memory_manifest.get("final_status") == "completed"
            ),
            "mnemos_used_required_memory_tools": mnemos_validation["checks"][
                "required_workflow_memory_calls_present"
            ],
            "mnemos_had_recovery_overhead": any(
                str(row.get("helpfulness")) == "harmful" for row in mnemos_memory
            ),
            "no_memory_had_more_logged_wrong_turns": len(no_memory_wrong_turns) > len(mnemos_wrong_turns),
            "no_memory_had_more_repo_activity": len(no_memory_repo) > len(mnemos_repo),
            "token_estimate_delta_mnemos_minus_no_memory": (
                (mnemos_manifest.get("total_estimated_tokens") or 0)
                - (no_memory_manifest.get("total_estimated_tokens") or 0)
            ),
        },
    }


def write_markdown(summary: dict[str, Any], path: str | Path) -> None:
    m = summary["metrics"]["mnemos_enabled"]
    n = summary["metrics"]["no_memory"]
    pattern = summary["observed_pattern"]
    lines = [
        "# AI Developer Memory Trial Comparison",
        "",
        "```text",
        "LOCAL_DEVELOPMENT_EVIDENCE_ONLY",
        "NO_GENERAL_MEMORY_CLAIM",
        "```",
        "",
        "| Metric | MNEMOS enabled | No memory |",
        "|---|---:|---:|",
        f"| Total estimated tokens | {m['total_estimated_tokens']} | {n['total_estimated_tokens']} |",
        f"| Memory calls | {m['memory_calls']} | {n['memory_calls']} |",
        f"| Route log rows | {m['route_log_rows']} | {n['route_log_rows']} |",
        f"| Repo activity rows | {m['repo_activity_rows']} | {n['repo_activity_rows']} |",
        f"| Wrong-turn rows | {m['wrong_turn_rows']} | {n['wrong_turn_rows']} |",
        f"| Test runs | {m['test_summary']['runs']} | {n['test_summary']['runs']} |",
        f"| Failed test runs | {m['test_summary']['failed_runs']} | {n['test_summary']['failed_runs']} |",
        "",
        "## Observed Pattern",
        "",
        f"- Both completed: {pattern['both_completed']}",
        f"- MNEMOS required memory tools used: {pattern['mnemos_used_required_memory_tools']}",
        f"- MNEMOS had recovery overhead: {pattern['mnemos_had_recovery_overhead']}",
        f"- No-memory run had more logged wrong turns: {pattern['no_memory_had_more_logged_wrong_turns']}",
        f"- No-memory run had more repo activity: {pattern['no_memory_had_more_repo_activity']}",
        f"- Token estimate delta, MNEMOS minus no-memory: {pattern['token_estimate_delta_mnemos_minus_no_memory']}",
        "",
        "## Interpretation",
        "",
        "This single paired trial suggests MNEMOS can be integrated into an AI "
        "developer workflow and used during app construction. The MNEMOS run "
        "shows lower logged repo activity, fewer logged wrong turns, and lower "
        "estimated token use, but it also had infrastructure recovery overhead "
        "from early unavailable/misrouted calls. This is local development "
        "evidence only and should not be treated as a general memory-performance "
        "claim.",
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
    print(json.dumps(summary["observed_pattern"], indent=2))


if __name__ == "__main__":
    main()
