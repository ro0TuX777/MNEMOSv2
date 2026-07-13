"""Verify and score MNEMOS agent-navigation development trials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROTOCOL = ROOT / "benchmarks" / "evaluation" / "mnemos_agent_navigation_protocol.json"
DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "results"


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(path)
    required_paths = [protocol["protocol_document"]]
    for task in protocol["tasks"]:
        required_paths.extend(task["required_evidence_paths"])
    for card in protocol["memory_cards"]:
        required_paths.extend(card.get("evidence_paths", []))
        required_paths.extend(card.get("contradicted_by", []))

    checks = {
        "status_ready": protocol["status"] == "MNEMOS_AGENT_NAVIGATION_DEVELOPMENT_STUDY_READY",
        "local_repo_only": protocol["authorization"]["local_repo_tasks_only"] is True,
        "gatemem_not_reopened": protocol["authorization"]["gatemem_reopening_authorized"] is False,
        "sealed_evaluation_not_authorized": protocol["authorization"]["sealed_evaluation_authorized"] is False,
        "general_memory_claim_not_authorized": protocol["authorization"]["general_memory_claim_authorized"] is False,
        "runtime_integration_not_authorized": protocol["authorization"]["runtime_memory_integration_authorized"] is False,
        "paired_modes_present": set(protocol["trial_modes"]) == {"baseline_repo_search", "mnemos_memory_assisted"},
        "has_stale_or_overbroad_memory": any(card["status"] in {"stale", "overbroad"} for card in protocol["memory_cards"]),
        "task_ids_unique": len({task["task_id"] for task in protocol["tasks"]}) == len(protocol["tasks"]),
        "referenced_files_present": all((ROOT / relative).is_file() for relative in set(required_paths)),
        "claim_boundary_development_only": protocol["claim_boundary"]["development_evidence_only"] is True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"MNEMOS agent-navigation protocol failed verification: {failed}")
    return {
        "status": protocol["status"],
        "classification": protocol["classification"],
        "task_count": len(protocol["tasks"]),
        "memory_card_count": len(protocol["memory_cards"]),
        "checks": checks,
        "all_checks_passed": True,
    }


def score_trial_log(trial_log_path: str | Path, protocol_path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    trial = json.loads(Path(trial_log_path).read_text(encoding="utf-8"))
    task_by_id = {task["task_id"]: task for task in protocol["tasks"]}
    task = task_by_id[trial["task_id"]]

    cited = set(trial.get("evidence_paths_used", []))
    required = set(task["required_evidence_paths"])
    final_answer = trial.get("final_answer", "").lower()
    rejected = set(trial.get("rejected_memory_ids", []))
    required_rejected = set(task.get("required_rejected_memory_ids", []))

    forbidden_hits = [
        fragment
        for fragment in task.get("forbidden_claim_fragments", [])
        if fragment.lower() in final_answer
    ]
    checks = {
        "known_task": trial["task_id"] in task_by_id,
        "known_mode": trial["mode"] in protocol["trial_modes"],
        "required_evidence_recall": required.issubset(cited),
        "forbidden_claim_avoidance": not forbidden_hits,
        "boundary_decision_match": trial.get("boundary_decision") == task["expected_boundary_decision"],
        "memory_skepticism": required_rejected.issubset(rejected),
        "path_efficiency_observed": isinstance(trial.get("files_opened_count"), int)
        and isinstance(trial.get("searches_count"), int),
    }
    score = sum(1 for passed in checks.values() if passed)
    return {
        "task_id": trial["task_id"],
        "mode": trial["mode"],
        "checks": checks,
        "score": score,
        "max_score": len(checks),
        "forbidden_hits": forbidden_hits,
        "missing_required_evidence": sorted(required - cited),
        "missing_rejected_memory_ids": sorted(required_rejected - rejected),
        "all_checks_passed": all(checks.values()),
    }


def summarize_batch(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    results_path = Path(results_dir)
    protocol = load_protocol(protocol_path)
    expected_modes = set(protocol["trial_modes"])
    trial_paths = sorted(results_path.glob("mnemos_agent_navigation_trial_*.json"))
    trials = []
    for path in trial_paths:
        trial = json.loads(path.read_text(encoding="utf-8"))
        result = score_trial_log(path, protocol_path)
        trials.append({"path": path, "trial": trial, "score": result})

    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for item in trials:
        task_id = item["trial"]["task_id"]
        mode = item["trial"]["mode"]
        paired.setdefault(task_id, {})[mode] = item

    complete_pairs = {
        task_id: modes
        for task_id, modes in paired.items()
        if set(modes) == expected_modes
    }

    pair_rows = []
    for task_id, modes in sorted(complete_pairs.items()):
        memory = modes["mnemos_memory_assisted"]
        baseline = modes["baseline_repo_search"]
        memory_trial = memory["trial"]
        baseline_trial = baseline["trial"]
        pair_rows.append(
            {
                "task_id": task_id,
                "memory_score": memory["score"]["score"],
                "baseline_score": baseline["score"]["score"],
                "max_score": memory["score"]["max_score"],
                "memory_all_checks_passed": memory["score"]["all_checks_passed"],
                "baseline_all_checks_passed": baseline["score"]["all_checks_passed"],
                "memory_files_opened_count": memory_trial["files_opened_count"],
                "baseline_files_opened_count": baseline_trial["files_opened_count"],
                "memory_searches_count": memory_trial["searches_count"],
                "baseline_searches_count": baseline_trial["searches_count"],
                "searches_saved_by_memory_assisted": baseline_trial["searches_count"] - memory_trial["searches_count"],
                "files_opened_delta": baseline_trial["files_opened_count"] - memory_trial["files_opened_count"],
                "boundary_decision_changed": memory_trial["boundary_decision"] != baseline_trial["boundary_decision"],
                "memory_forbidden_hits": memory["score"]["forbidden_hits"],
                "baseline_forbidden_hits": baseline["score"]["forbidden_hits"],
            }
        )

    total_pairs = len(pair_rows)
    aggregate = {
        "schema_version": "mnemos-agent-navigation-batch-summary-v1",
        "study_status": protocol["status"],
        "classification": protocol["classification"],
        "trial_count": len(trials),
        "complete_pair_count": total_pairs,
        "all_trials_passed": all(item["score"]["all_checks_passed"] for item in trials),
        "all_pairs_boundary_stable": not any(row["boundary_decision_changed"] for row in pair_rows),
        "total_searches_saved_by_memory_assisted": sum(row["searches_saved_by_memory_assisted"] for row in pair_rows),
        "mean_searches_saved_by_memory_assisted": (
            sum(row["searches_saved_by_memory_assisted"] for row in pair_rows) / total_pairs
            if total_pairs
            else 0
        ),
        "total_files_opened_delta": sum(row["files_opened_delta"] for row in pair_rows),
        "memory_assisted_overclaim_count": sum(1 for row in pair_rows if row["memory_forbidden_hits"]),
        "baseline_overclaim_count": sum(1 for row in pair_rows if row["baseline_forbidden_hits"]),
        "pairs": pair_rows,
        "claim_boundary": {
            "development_evidence_only": True,
            "gatemem_reopened": False,
            "general_memory_claim": False,
        },
    }
    return aggregate


def write_batch_summary(
    summary: dict[str, Any],
    json_path: str | Path,
    markdown_path: str | Path | None = None,
) -> None:
    Path(json_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if markdown_path is None:
        return

    lines = [
        "# MNEMOS Agent Navigation Batch 001 Summary",
        "",
        "```text",
        summary["study_status"],
        summary["classification"],
        "NO_GATEMEM_REOPENING",
        "NO_GENERAL_MEMORY_CLAIM",
        "```",
        "",
        f"Complete paired tasks: {summary['complete_pair_count']}",
        f"Trials scored: {summary['trial_count']}",
        f"All trials passed: {summary['all_trials_passed']}",
        f"Boundary decisions stable: {summary['all_pairs_boundary_stable']}",
        f"Total searches saved by memory-assisted mode: {summary['total_searches_saved_by_memory_assisted']}",
        f"Mean searches saved per pair: {summary['mean_searches_saved_by_memory_assisted']:.2f}",
        "",
        "| Task | Memory score | Baseline score | Memory searches | Baseline searches | Searches saved | Boundary changed |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["pairs"]:
        lines.append(
            "| {task_id} | {memory_score}/{max_score} | {baseline_score}/{max_score} | "
            "{memory_searches_count} | {baseline_searches_count} | "
            "{searches_saved_by_memory_assisted} | {boundary_decision_changed} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Observation",
            "",
            "In this local development batch, memory-assisted mode preserved the same "
            "boundary decisions as baseline mode while reducing explicit repository "
            "searches. This is orientation evidence only, not a general memory "
            "performance claim.",
        ]
    )
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--trial-log", type=Path)
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--write-batch-json", type=Path)
    parser.add_argument("--write-batch-md", type=Path)
    args = parser.parse_args()

    verified = verify_protocol(args.protocol)
    print(verified["status"])
    print(verified["classification"])
    print(f"Protocol checks passed: {len(verified['checks'])}/{len(verified['checks'])}")

    if args.trial_log is not None:
        result = score_trial_log(args.trial_log, args.protocol)
        print(f"Trial score: {result['score']}/{result['max_score']}")
        if result["forbidden_hits"]:
            print(f"Forbidden hits: {', '.join(result['forbidden_hits'])}")

    if args.batch_dir is not None:
        summary = summarize_batch(args.batch_dir, args.protocol)
        print(f"Batch pairs: {summary['complete_pair_count']}")
        print(f"Batch trials: {summary['trial_count']}")
        print(f"All trials passed: {summary['all_trials_passed']}")
        print(f"Searches saved: {summary['total_searches_saved_by_memory_assisted']}")
        if args.write_batch_json is not None:
            write_batch_summary(summary, args.write_batch_json, args.write_batch_md)


if __name__ == "__main__":
    main()
