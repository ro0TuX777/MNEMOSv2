"""Run Phase 5A held-out technical verification for Selector S1.

Offline only. No model calls, human-review claims, MNEMOS runtime imports,
memory writes, retrieval changes, or governance/promotion mutations.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prototype.session_context_assembler.corpus import load_validated_corpus  # noqa: E402
from prototype.session_context_assembler.replay import (  # noqa: E402
    run_condition_a,
    run_condition_b,
    run_condition_c,
    run_condition_c1,
)

R1_PATH = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r1.json"
R1_MANIFEST_PATH = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r1.manifest.json"
R2_PATH = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r2.json"
R2_MANIFEST_PATH = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r2.manifest.json"
SELECTOR_PATH = REPO_ROOT / "prototype" / "session_context_assembler" / "selector_s1.py"
RESULT_JSON = REPO_ROOT / "benchmarks" / "results" / "session_context_assembler_r2_verification.json"
RESULT_MD = REPO_ROOT / "benchmarks" / "results" / "session_context_assembler_r2_verification.md"
OWNER_PACK = REPO_ROOT / "benchmarks" / "review_packets" / "session_context_assembler_phase_5a_owner_review.json"
OWNER_MANIFEST = REPO_ROOT / "benchmarks" / "review_packets" / "session_context_assembler_phase_5a_owner_manifest.json"

EXPECTED_R1_SHA256 = "9dc5682ec08ffad24a9c329ef8b581d3d68c3f83c92e078502f3d37c837e53dc"
SCORING_ONLY_FIELD = "verification_expectations"
CONDITION_NAMES = {
    "A_full_history": "A",
    "B_sliding_window": "B",
    "C1_selector_s1_mandatory_preservation": "C1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selector_boundary_violations(source: str | None = None) -> List[str]:
    source = SELECTOR_PATH.read_text(encoding="utf-8") if source is None else source
    tree = ast.parse(source)
    violations = []
    if SCORING_ONLY_FIELD in source:
        violations.append("scoring_only_field_access")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        if any(name.startswith(("mnemos", "service", "mnemos_sdk")) for name in names):
            violations.append("runtime_import")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {
                "write_text", "write_bytes", "unlink", "remove", "rename",
                "index", "upsert", "promote", "mutate",
            }:
                violations.append(f"forbidden_call:{node.func.attr}")
    return sorted(set(violations))


def _recall(required: Sequence[str], selected: Sequence[str]) -> float | None:
    if not required:
        return None
    return len(set(required) & set(selected)) / len(required)


def _lineage_loss(record: Dict) -> tuple[int, int]:
    labels = record.get("labels", [])
    decisions = {item for label in labels for item in label.get("parent_engram_ids", [])}
    sources = {item for label in labels for item in label.get("parent_source_ids", [])}
    return (
        len(set(record["selected_parent_engram_ids"]) - decisions),
        len(set(record["selected_source_ids"]) - sources),
    )


def _label_coverage(record: Dict) -> float:
    episodes = set(record["selected_episode_ids"])
    if not episodes:
        return 0.0
    labeled = {label["episode_id"] for label in record.get("labels", [])}
    return len(episodes & labeled) / len(episodes)


def _score_case(case: Dict, condition_record: Dict) -> Dict:
    expected = case[SCORING_ONLY_FIELD]
    decisions = condition_record["selected_parent_engram_ids"]
    sources = condition_record["selected_source_ids"]
    decision_recall = _recall(expected["required_decision_ids"], decisions)
    source_recall = _recall(expected["required_source_ids"], sources)
    missing = sorted(
        (set(expected["required_decision_ids"]) - set(decisions))
        | (set(expected["required_source_ids"]) - set(sources))
    )
    abstained = condition_record.get("context_budget_insufficient", False)
    absent_source_violations = sorted(
        set(expected["expected_absent_source_ids"]) & set(sources)
    )
    decision_loss, source_loss = _lineage_loss(condition_record)
    return {
        "case_id": case["id"],
        "verification_class": case["verification_class"],
        "condition": condition_record["condition"],
        "token_estimate": condition_record["token_estimate"],
        "context_budget": case["expected_context_budget"],
        "budget_feasible": expected["budget_feasible"],
        "decision_artifact_retention": decision_recall,
        "required_source_recall": source_recall,
        "missing_required_artifact_ids": missing,
        "absent_source_violations": absent_source_violations,
        "context_budget_insufficient": abstained,
        "omitted_required_artifact_types": condition_record.get(
            "omitted_required_artifact_types", []
        ),
        "selection_abstention_reason": condition_record.get(
            "selection_abstention_reason"
        ),
        "expected_context_budget_insufficient": expected[
            "expect_context_budget_insufficient"
        ],
        "expected_omitted_artifact_types": expected[
            "expected_omitted_artifact_types"
        ],
        "silent_required_artifact_omission": bool(missing) and not abstained,
        "decision_lineage_loss_count": decision_loss,
        "source_lineage_loss_count": source_loss,
        "provenance_loss_count": decision_loss + source_loss,
        "synthetic_context_label_coverage": _label_coverage(condition_record),
        "selected_turn_ids": condition_record["selected_turn_ids"],
        "selected_episode_ids": condition_record["selected_episode_ids"],
        "selected_parent_engram_ids": decisions,
        "selected_source_ids": sources,
        "synthetic_context_labels": condition_record.get("labels", []),
    }


def _run_conditions(case: Dict, manifest_hash: str, seed: int) -> List[Dict]:
    budget = case["expected_context_budget"]
    return [
        run_condition_a(case),
        run_condition_b(case, budget),
        run_condition_c1(case, manifest_hash, seed, budget),
    ]


def _aggregate(c1_rows: List[Dict]) -> Dict:
    feasible = [row for row in c1_rows if row["budget_feasible"]]
    decision_values = [
        row["decision_artifact_retention"] for row in feasible
        if row["decision_artifact_retention"] is not None
    ]
    source_values = [
        row["required_source_recall"] for row in feasible
        if row["required_source_recall"] is not None
    ]
    infeasible = [row for row in c1_rows if not row["budget_feasible"]]
    total_selected_ids = sum(
        len(row["selected_parent_engram_ids"]) + len(row["selected_source_ids"])
        for row in c1_rows
    )
    lineage_loss = sum(row["provenance_loss_count"] for row in c1_rows)
    return {
        "feasible_case_count": len(feasible),
        "infeasible_case_count": len(infeasible),
        "required_artifact_retention_feasible": (
            sum(decision_values + source_values) / len(decision_values + source_values)
            if decision_values or source_values else None
        ),
        "all_infeasible_sets_abstained": all(
            row["context_budget_insufficient"] for row in infeasible
        ),
        "silent_required_artifact_omission_count": sum(
            row["silent_required_artifact_omission"] for row in c1_rows
        ),
        "budget_compliance_rate": sum(
            row["token_estimate"] <= row["context_budget"] for row in c1_rows
        ) / len(c1_rows),
        "provenance_loss_count": lineage_loss,
        "source_and_decision_lineage_preservation_rate": (
            1.0 if total_selected_ids == 0
            else 1.0 - (lineage_loss / total_selected_ids)
        ),
        "synthetic_context_label_coverage": sum(
            row["synthetic_context_label_coverage"] for row in c1_rows
        ) / len(c1_rows),
        "ineligible_or_missing_source_violation_count": sum(
            len(row["absent_source_violations"]) for row in c1_rows
        ),
        "abstention_expectation_match_rate": sum(
            row["context_budget_insufficient"]
            == row["expected_context_budget_insufficient"]
            for row in c1_rows
        ) / len(c1_rows),
    }


def _critical_gates(aggregate: Dict, deterministic: bool, boundaries_clean: bool) -> Dict:
    gates = {
        "required_artifact_retention_feasible": (
            aggregate["required_artifact_retention_feasible"], 1.0
        ),
        "all_infeasible_sets_abstained": (
            aggregate["all_infeasible_sets_abstained"], True
        ),
        "silent_required_artifact_omission_count": (
            aggregate["silent_required_artifact_omission_count"], 0
        ),
        "budget_compliance_rate": (aggregate["budget_compliance_rate"], 1.0),
        "provenance_loss_count": (aggregate["provenance_loss_count"], 0),
        "lineage_preservation_rate": (
            aggregate["source_and_decision_lineage_preservation_rate"], 1.0
        ),
        "synthetic_context_label_coverage": (
            aggregate["synthetic_context_label_coverage"], 1.0
        ),
        "ineligible_source_violation_count": (
            aggregate["ineligible_or_missing_source_violation_count"], 0
        ),
        "abstention_expectation_match_rate": (
            aggregate["abstention_expectation_match_rate"], 1.0
        ),
        "determinism": (deterministic, True),
        "selector_boundary_clean": (boundaries_clean, True),
    }
    return {
        name: {"value": value, "required": required, "passed": value == required}
        for name, (value, required) in gates.items()
    }


def _mutation_self_checks(cases: List[Dict], manifest_hash: str, c1_rows: List[Dict]) -> Dict:
    checks = {}

    legacy_failures = 0
    for case in cases:
        if not case[SCORING_ONLY_FIELD]["budget_feasible"]:
            continue
        legacy = _score_case(
            case,
            run_condition_c(case, manifest_hash, 7, case["expected_context_budget"]),
        )
        if legacy["missing_required_artifact_ids"]:
            legacy_failures += 1
    checks["mandatory_ordering_bypass_detected"] = legacy_failures > 0

    source_row = next(row for row in c1_rows if row["selected_source_ids"])
    mutated_source = copy.deepcopy(source_row)
    target = mutated_source["selected_source_ids"][0]
    for label in mutated_source["synthetic_context_labels"]:
        if target in label["parent_source_ids"]:
            label["parent_source_ids"].remove(target)
            break
    label_sources = {
        source for label in mutated_source["synthetic_context_labels"]
        for source in label["parent_source_ids"]
    }
    checks["parent_source_removal_detected"] = target not in label_sources

    label_row = next(row for row in c1_rows if row["selected_episode_ids"])
    mutated_label = copy.deepcopy(label_row)
    mutated_label["synthetic_context_labels"].pop()
    covered = {
        label["episode_id"] for label in mutated_label["synthetic_context_labels"]
    }
    checks["synthetic_label_removal_detected"] = not set(
        mutated_label["selected_episode_ids"]
    ) <= covered

    overflow = next(row for row in c1_rows if not row["budget_feasible"])
    mutated_abstention = copy.deepcopy(overflow)
    mutated_abstention["context_budget_insufficient"] = False
    checks["abstention_suppression_detected"] = bool(
        mutated_abstention["missing_required_artifact_ids"]
        and not mutated_abstention["context_budget_insufficient"]
    )

    mutated_source_code = SELECTOR_PATH.read_text(encoding="utf-8") + (
        "\n# mutation probe\n_SCORING = case['verification_expectations']\n"
    )
    checks["scoring_field_access_detected"] = (
        "scoring_only_field_access" in selector_boundary_violations(mutated_source_code)
    )
    return {
        name: {"passed": passed} for name, passed in checks.items()
    }


def _owner_pack(cases: List[Dict], records: Dict[tuple[str, str], Dict]) -> tuple[Dict, Dict]:
    tasks = []
    key = []
    for index, case in enumerate(cases, 1):
        task_code = f"OWNER-TASK-{index:02d}"
        conditions = list(CONDITION_NAMES)
        random.Random(5100 + index).shuffle(conditions)
        turns = {turn["turn_id"]: turn for turn in case["conversation_history"]}
        packages = []
        mapping = {}
        for package_index, condition in enumerate(conditions, 1):
            code = f"PACKAGE-{package_index}"
            record = records[(case["id"], condition)]
            artifacts = [
                {
                    "turn_id": tid,
                    "speaker": turns[tid]["speaker"],
                    "content": turns[tid]["content"],
                }
                for tid in record["selected_turn_ids"]
            ]
            package = {"package_code": code, "artifacts": artifacts}
            if condition == "C1_selector_s1_mandatory_preservation":
                package["synthetic_context_labels"] = record["labels"]
                if record.get("context_budget_insufficient"):
                    package["warning"] = {
                        "context_budget_insufficient": True,
                        "omitted_required_artifact_types": record[
                            "omitted_required_artifact_types"
                        ],
                        "selection_abstention_reason": (
                            "Mandatory eligible artifacts exceeded the bounded context budget."
                        ),
                    }
            packages.append(package)
            mapping[code] = condition
        tasks.append(
            {"task_code": task_code, "task_prompt": case["current_task"], "packages": packages}
        )
        key.append({"task_code": task_code, "condition_key": mapping})
    pack = {
        "schema": "sca_phase5a_owner_review_pack_v1",
        "labels": ["PRODUCT_OWNER_REVIEW", "NOT_INDEPENDENT_HUMAN_STUDY", "NOT_GENERALIZABLE"],
        "review_not_run": True,
        "instructions": "First-pass review must not use the coordinator condition key.",
        "tasks": tasks,
    }
    manifest = {
        "schema": "sca_phase5a_owner_review_manifest_v1",
        "restricted": True,
        "review_not_run": True,
        "pack_sha256": hashlib.sha256(
            (json.dumps(pack, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest(),
        "coordinator_condition_key": key,
    }
    return pack, manifest


def run_verification() -> tuple[Dict, Dict, Dict]:
    r1_manifest = json.loads(R1_MANIFEST_PATH.read_text(encoding="utf-8"))
    r1_unchanged = _sha256(R1_PATH) == EXPECTED_R1_SHA256 == r1_manifest["file_sha256"]
    load_validated_corpus(R1_PATH, R1_MANIFEST_PATH)
    corpus = load_validated_corpus(R2_PATH, R2_MANIFEST_PATH)
    r2_manifest = json.loads(R2_MANIFEST_PATH.read_text(encoding="utf-8"))

    all_records = []
    raw_records = {}
    deterministic = True
    for case in corpus["cases"]:
        first = _run_conditions(case, r2_manifest["file_sha256"], 7)
        repeated = _run_conditions(case, r2_manifest["file_sha256"], 7)
        deterministic &= first == repeated
        for record in first:
            raw_records[(case["id"], record["condition"])] = record
            all_records.append(_score_case(case, record))

    c1_rows = [
        row for row in all_records
        if row["condition"] == "C1_selector_s1_mandatory_preservation"
    ]
    aggregate = _aggregate(c1_rows)
    boundary_violations = selector_boundary_violations()
    gates = _critical_gates(aggregate, deterministic, not boundary_violations)
    mutation_checks = _mutation_self_checks(
        corpus["cases"], r2_manifest["file_sha256"], c1_rows
    )
    all_pass = (
        r1_unchanged
        and all(gate["passed"] for gate in gates.values())
        and all(check["passed"] for check in mutation_checks.values())
    )
    result = {
        "schema": "session_context_assembler_phase5a_r2_verification_v1",
        "labels": [
            "TECHNICAL_VERIFICATION_ONLY", "NOT_HUMAN_VALUE_EVIDENCE",
            "NO_RUNTIME_INTEGRATION", "NO_PRODUCTION_READINESS_CLAIM",
        ],
        "r1_file_sha256": _sha256(R1_PATH),
        "r1_unchanged_and_hash_valid": r1_unchanged,
        "r2_file_sha256": r2_manifest["file_sha256"],
        "r2_case_count": len(corpus["cases"]),
        "model_assisted_surrogate_evaluation": "NOT_RUN_OPTIONAL_WORKSTREAM",
        "product_owner_review": "PACK_PREPARED_NOT_RUN",
        "selector_boundary_violations": boundary_violations,
        "aggregate": aggregate,
        "gates": gates,
        "mutation_sensitivity_self_checks": mutation_checks,
        "all_phase5a_advancement_gates_passed": all_pass,
        "records": all_records,
    }
    owner_pack, owner_manifest = _owner_pack(corpus["cases"], raw_records)
    return result, owner_pack, owner_manifest


def _render_markdown(result: Dict) -> str:
    lines = [
        "# Session Context Assembler — Phase 5A R2 Technical Verification",
        "",
        " ".join(f"`{label}`" for label in result["labels"]),
        "",
        "This is held-out technical verification, not human usability evidence, "
        "production validation, or authority/governance validation.",
        "",
        f"- R2 cases: {result['r2_case_count']}",
        f"- R1 unchanged and hash-valid: {result['r1_unchanged_and_hash_valid']}",
        f"- Optional model-assisted surrogate: {result['model_assisted_surrogate_evaluation']}",
        f"- Product-owner review: {result['product_owner_review']}",
        "",
        "## Advancement gates",
        "",
        "| Gate | Value | Required | Result |",
        "|---|---:|---:|---|",
    ]
    for name, gate in result["gates"].items():
        lines.append(
            f"| {name} | {gate['value']} | {gate['required']} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(["", "## Mutation sensitivity", "", "| Mutation | Detected |", "|---|---|"])
    for name, check in result["mutation_sensitivity_self_checks"].items():
        lines.append(f"| {name} | {'PASS' if check['passed'] else 'FAIL'} |")
    outcome = "PASS" if result["all_phase5a_advancement_gates_passed"] else "FAIL"
    lines.extend(
        [
            "", f"**Phase 5A technical outcome: {outcome}**", "",
            "A PASS authorizes a separate proposal for a read-only, consumer-neutral "
            "technical shadow adapter. It does not authorize live routing, production "
            "use, memory writes, governance mutation, promotion behavior, or a "
            "human-value claim.", "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    result, owner_pack, owner_manifest = run_verification()
    for path in (RESULT_JSON, RESULT_MD, OWNER_PACK, OWNER_MANIFEST):
        path.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    RESULT_MD.write_bytes(_render_markdown(result).encode("utf-8"))
    OWNER_PACK.write_bytes(
        (json.dumps(owner_pack, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    OWNER_MANIFEST.write_bytes(
        (json.dumps(owner_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(f"R2 cases verified: {result['r2_case_count']}")
    print(f"All Phase 5A gates passed: {result['all_phase5a_advancement_gates_passed']}")
    print(f"Wrote {RESULT_JSON}")
    print(f"Wrote {RESULT_MD}")
    return 0 if result["all_phase5a_advancement_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
