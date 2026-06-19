"""Score EBIR-R2 responses against protected gold labels after response freeze.

This is an admin-only unblinding tool. It consumes the restricted assignment
manifest, frozen truthset gold labels, and completed pseudonymous Markdown
responses. It does not generate reviewer-facing material and must not be
circulated to reviewers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.compile_ebir_r2_pilot_report import (  # noqa: E402
    assigned_packets,
    first_free_text,
    parse_response_file,
    pseudonym_map,
    section_for_packet,
    selected_in_question,
)


VALUE_ALIASES = {
    "mfa_required": ["mfa required", "requires mfa", "mfa for all active operators"],
    "active_2026": ["active_2026", "active 2026"],
    "30_days_eu": ["30_days_eu", "30 days eu", "30 days"],
    "disabled_by_default": ["disabled_by_default", "disabled by default"],
    "restricted": ["restricted"],
    "route_b": ["route_b", "route b"],
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def rate(values: Iterable[bool]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def round_or_none(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)


def expected_value_present(text: str, expected_value: Optional[str]) -> bool:
    if expected_value is None:
        return False
    lowered = text.lower()
    if expected_value.lower() in lowered:
        return True
    return any(alias in lowered for alias in VALUE_ALIASES.get(expected_value, []))


def correct_resolution(response: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    if gold.get("expected_abstention"):
        return response.get("handling") == "Escalate / abstain"
    answer_text = " ".join(
        [
            str(response.get("best_supported_resolution") or ""),
            str(response.get("decisive_evidence") or ""),
        ]
    )
    handling_ok = response.get("handling") in {"Resolve", "Partially resolve"}
    return handling_ok and expected_value_present(answer_text, gold.get("expected_resolved_value"))


def correct_escalation(response: Dict[str, Any], gold: Dict[str, Any]) -> Optional[bool]:
    if not gold.get("expected_abstention"):
        return None
    return response.get("handling") == "Escalate / abstain"


def assessment_has_prohibited_claim(assessment_text: str, gold: Dict[str, Any]) -> bool:
    lowered = assessment_text.lower()
    return any(
        claim.lower() in lowered
        for claim in gold.get("prohibited_unsupported_claims", [])
    )


def unsupported_detection(response: Dict[str, Any], assessment_text: str, gold: Dict[str, Any]) -> bool:
    unsupported_answer = response.get("unsupported_claim")
    has_prohibited = assessment_has_prohibited_claim(assessment_text, gold)
    if has_prohibited:
        return unsupported_answer in {"Yes", "Unsure"}
    return unsupported_answer in {"No", "Unsure"}


def confidence_calibration(response: Dict[str, Any], correct: bool) -> Dict[str, Any]:
    confidence = int(response["confidence"]) if response.get("confidence") else None
    if confidence is None:
        return {"confidence": None, "accuracy": int(correct), "absolute_error": None}
    expected_confidence_accuracy = confidence / 5.0
    accuracy = 1.0 if correct else 0.0
    return {
        "confidence": confidence,
        "accuracy": accuracy,
        "absolute_error": abs(expected_confidence_accuracy - accuracy),
    }


def parse_response_details(path: Path, expected_reviewer: str, assigned: List[str]) -> Dict[str, Any]:
    parsed = parse_response_file(path, expected_reviewer, assigned)
    text = path.read_text(encoding="utf-8")
    details: Dict[str, Dict[str, Any]] = {}
    for packet_id in assigned:
        section = section_for_packet(text, packet_id)
        if not section:
            continue
        handling = selected_in_question(section, 2)
        unsupported = selected_in_question(section, 5)
        quality = selected_in_question(section, 6)
        confidence = selected_in_question(section, 7)
        synthesized = selected_in_question(section, 8)
        synthesized_confidence = selected_in_question(section, 9)
        assessment = ""
        if "### Assessment Material" in section and "### Reviewer Response" in section:
            assessment = section.split("### Assessment Material", 1)[1].split(
                "### Reviewer Response",
                1,
            )[0].strip()
        details[packet_id] = {
            "best_supported_resolution": first_free_text(section, 1),
            "handling": handling[0] if handling else None,
            "decisive_evidence": first_free_text(section, 3),
            "remaining_uncertainty": first_free_text(section, 4),
            "unsupported_claim": unsupported[0] if unsupported else None,
            "unsupported_claim_detail": first_free_text(section, 5),
            "quality": int(quality[0][0:1]) if quality else None,
            "confidence": int(confidence[0]) if confidence else None,
            "synthesized_impression": synthesized[0] if synthesized else None,
            "synthesized_impression_confidence": (
                int(synthesized_confidence[0]) if synthesized_confidence else None
            ),
            "usability_notes": first_free_text(section, 10),
            "assessment_material": assessment,
        }
    parsed["details"] = details
    return parsed


def group_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    recognition_values = [
        row["condition_recognition_correct"]
        for row in rows
        if row["condition_recognition_correct"] is not None
    ]
    calibration_errors = [
        row["confidence_calibration"]["absolute_error"]
        for row in rows
        if row["confidence_calibration"]["absolute_error"] is not None
    ]
    times = [
        row["reviewer_time_seconds"]
        for row in rows
        if isinstance(row.get("reviewer_time_seconds"), (int, float))
    ]
    escalation_values = [
        row["correct_escalation_or_abstention"]
        for row in rows
        if row["correct_escalation_or_abstention"] is not None
    ]
    return {
        "n": len(rows),
        "correct_resolution_rate": round_or_none(rate(row["correct_resolution"] for row in rows)),
        "correct_escalation_or_abstention_rate": round_or_none(rate(escalation_values)),
        "unsupported_claim_detection_rate": round_or_none(
            rate(row["unsupported_claim_detection"] for row in rows)
        ),
        "mean_quality_score": round_or_none(mean(row["quality"] for row in rows if row["quality"] is not None)),
        "mean_confidence": round_or_none(mean(row["confidence"] for row in rows if row["confidence"] is not None)),
        "confidence_calibration": {
            "mean_absolute_error": round_or_none(mean(calibration_errors)),
            "mean_confidence": round_or_none(mean(row["confidence"] for row in rows if row["confidence"] is not None)),
            "accuracy": round_or_none(rate(row["correct_resolution"] for row in rows)),
        },
        "condition_recognition_rate": round_or_none(rate(recognition_values)),
        "reviewer_time": {
            "status": "not_collected" if not times else "collected",
            "mean_seconds": round_or_none(mean(times)),
        },
    }


def markdown_report(result: Dict[str, Any]) -> str:
    lines = [
        "# EBIR-R2 Gold-Label Scoring Report",
        "",
        "`ADMIN-ONLY - UNBLIND AFTER ALL RESPONSES FROZEN`",
        "",
        f"Status: {result['status']}",
        f"Truthset Version: {result['truthset_version']}",
        f"Assignment Seed: {result['assignment_seed']}",
        f"Responses Scored: {result['response_count']}",
        "",
        "This report unblinds condition mappings for scoring only. It must not be circulated with reviewer-facing material.",
        "",
        "## Metrics By Condition",
        "",
        "| Condition | n | Correct Resolution | Correct Escalation/Abstention | Unsupported Claim Detection | Mean Quality | Recognition Rate | Time |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for condition, metrics in sorted(result["metrics_by_condition"].items()):
        time_status = metrics["reviewer_time"]["status"]
        time_value = metrics["reviewer_time"]["mean_seconds"]
        lines.append(
            f"| `{condition}` | {metrics['n']} | {metrics['correct_resolution_rate']} | "
            f"{metrics['correct_escalation_or_abstention_rate']} | "
            f"{metrics['unsupported_claim_detection_rate']} | {metrics['mean_quality_score']} | "
            f"{metrics['condition_recognition_rate']} | {time_status if time_value is None else time_value} |"
        )

    lines.extend(["", "## Confidence Calibration By Condition", ""])
    for condition, metrics in sorted(result["metrics_by_condition"].items()):
        lines.append(
            f"- `{condition}`: {metrics['confidence_calibration']}"
        )

    lines.extend(["", "## Case Family Breakdown", ""])
    for family, metrics in sorted(result["case_family_breakdown"].items()):
        lines.append(
            f"- `{family}`: n={metrics['n']}, correct_resolution_rate={metrics['correct_resolution_rate']}, mean_quality={metrics['mean_quality_score']}"
        )

    lines.extend(
        [
            "",
            "## Condition Recognition",
            "",
            f"- Overall recognition rate: {result['condition_recognition_rate']}",
            f"- Distribution: {result['condition_recognition_distribution']}",
            "",
            "## Caveat",
            "",
            result["caveat"],
            "",
        ]
    )
    return "\n".join(lines)


def score(
    *,
    manifest_path: Path,
    responses_dir: Path,
    output_json: Path,
    output_md: Path,
    synthetic_dry_run: bool,
) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    truthset_path = Path(manifest["truthset"])
    truthset = load_json(truthset_path)
    gold_by_case = {case["id"]: case["gold_label"] for case in truthset["cases"]}
    family_by_case = {case["id"]: case["case_family"] for case in truthset["cases"]}
    assignment_by_packet = {row["packet_id"]: row for row in manifest["assignments"]}
    assigned = assigned_packets(manifest)
    reviewer_map = pseudonym_map(manifest)
    reverse_reviewer_map = {pseudo: raw for raw, pseudo in reviewer_map.items()}

    parse_errors: List[str] = []
    parsed_by_reviewer: Dict[str, Any] = {}
    for pseudo_id, packet_ids in sorted(assigned.items()):
        path = responses_dir / f"reviewer_{pseudo_id}_completed.md"
        if not path.exists():
            parse_errors.append(f"missing response file for {pseudo_id}: {path}")
            continue
        parsed = parse_response_details(path, pseudo_id, packet_ids)
        parsed_by_reviewer[pseudo_id] = parsed
        parse_errors.extend(f"{pseudo_id}: {error}" for error in parsed.get("errors", []))

    if parse_errors:
        result = {
            "status": "FAIL",
            "errors": parse_errors,
            "admin_only": True,
            "synthetic_dry_run": synthetic_dry_run,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        output_md.write_text("# EBIR-R2 Gold-Label Scoring Report\n\nFAIL\n", encoding="utf-8")
        return result

    rows: List[Dict[str, Any]] = []
    for pseudo_id, parsed in parsed_by_reviewer.items():
        for packet_id, response in parsed["details"].items():
            assignment = assignment_by_packet[packet_id]
            gold = gold_by_case[assignment["case_id"]]
            condition = assignment["condition_key"]
            candidate_provided = bool(assignment.get("candidate_provided"))
            recognition: Optional[bool]
            if response.get("synthesized_impression") == "Unsure":
                recognition = None
            else:
                recognition = (
                    response.get("synthesized_impression") == "Yes"
                    if candidate_provided
                    else response.get("synthesized_impression") == "No"
                )
            correct = correct_resolution(response, gold)
            row = {
                "packet_id": packet_id,
                "case_id": assignment["case_id"],
                "case_family": family_by_case[assignment["case_id"]],
                "condition": condition,
                "reviewer_id": pseudo_id,
                "correct_resolution": correct,
                "correct_escalation_or_abstention": correct_escalation(response, gold),
                "unsupported_claim_detection": unsupported_detection(
                    response,
                    response.get("assessment_material", ""),
                    gold,
                ),
                "quality": response.get("quality"),
                "confidence": response.get("confidence"),
                "confidence_calibration": confidence_calibration(response, correct),
                "condition_recognition_correct": recognition,
                "synthesized_impression": response.get("synthesized_impression"),
                "synthesized_impression_confidence": response.get("synthesized_impression_confidence"),
                "reviewer_time_seconds": None,
            }
            rows.append(row)

    by_condition = defaultdict(list)
    by_family = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_family[row["case_family"]].append(row)

    recognition_counter = Counter(row["synthesized_impression"] for row in rows)
    recognition_values = [
        row["condition_recognition_correct"]
        for row in rows
        if row["condition_recognition_correct"] is not None
    ]
    result = {
        "status": "PASS",
        "admin_only": True,
        "synthetic_dry_run": synthetic_dry_run,
        "caveat": (
            "Synthetic parser/scoring dry run only; do not treat these results as "
            "human-review evidence or EBIR value evidence."
            if synthetic_dry_run
            else "Human-review scoring artifact; interpret only under the frozen R2 protocol."
        ),
        "truthset": str(truthset_path),
        "truthset_version": truthset["version"],
        "assignment_seed": manifest.get("seed"),
        "response_count": len(rows),
        "correct_resolution_rate_by_condition": {
            condition: round_or_none(rate(row["correct_resolution"] for row in rows_))
            for condition, rows_ in sorted(by_condition.items())
        },
        "correct_escalation_or_abstention_rate_by_condition": {
            condition: round_or_none(
                rate(
                    row["correct_escalation_or_abstention"]
                    for row in rows_
                    if row["correct_escalation_or_abstention"] is not None
                )
            )
            for condition, rows_ in sorted(by_condition.items())
        },
        "unsupported_claim_detection_rate_by_condition": {
            condition: round_or_none(rate(row["unsupported_claim_detection"] for row in rows_))
            for condition, rows_ in sorted(by_condition.items())
        },
        "mean_quality_score_by_condition": {
            condition: round_or_none(mean(row["quality"] for row in rows_ if row["quality"] is not None))
            for condition, rows_ in sorted(by_condition.items())
        },
        "confidence_calibration_by_condition": {
            condition: group_metrics(rows_)["confidence_calibration"]
            for condition, rows_ in sorted(by_condition.items())
        },
        "condition_recognition_rate": round_or_none(rate(recognition_values)),
        "condition_recognition_distribution": dict(recognition_counter),
        "reviewer_time_by_condition": {
            condition: group_metrics(rows_)["reviewer_time"]
            for condition, rows_ in sorted(by_condition.items())
        },
        "case_family_breakdown": {
            family: group_metrics(rows_)
            for family, rows_ in sorted(by_family.items())
        },
        "metrics_by_condition": {
            condition: group_metrics(rows_)
            for condition, rows_ in sorted(by_condition.items())
        },
        "rows": rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    result = score(
        manifest_path=args.manifest,
        responses_dir=args.responses_dir,
        output_json=args.output_json,
        output_md=args.output_md,
        synthetic_dry_run=args.synthetic_dry_run,
    )
    if result["status"] == "PASS":
        print("[PASS] gold-label scoring completed")
        print(f"responses: {result['response_count']}")
        print(f"condition_recognition_rate: {result['condition_recognition_rate']}")
        print(f"output_json: {args.output_json}")
        print(f"output_md: {args.output_md}")
    else:
        for error in result.get("errors", []):
            print(f"[FAIL] {error}")
    if args.fail_on_gate and result["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
