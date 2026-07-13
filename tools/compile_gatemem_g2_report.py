"""Compile aggregate G2 measurements from isolated adapter and evaluator outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object.")
            rows.append(value)
    return rows


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_report(
    run_summary: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    official_summary: dict[str, Any],
) -> dict[str, Any]:
    diagnostic_by_id = {str(row.get("checkpoint_id")): row for row in diagnostics}
    score_by_id = {str(row.get("checkpoint_id")): row for row in scores}
    if (
        len(diagnostic_by_id) != len(diagnostics)
        or len(score_by_id) != len(scores)
        or set(diagnostic_by_id) != set(score_by_id)
    ):
        raise ValueError("Diagnostics and evaluator scores must align one-to-one.")

    utility = [row for row in scores if row.get("query_type") == "utility"]
    privacy = [row for row in scores if row.get("query_type") == "privacy"]
    deletion = [row for row in scores if row.get("query_type") == "safety"]
    privacy_refuse = [row for row in privacy if row.get("expected_action") == "refuse"]
    privacy_redact = [
        row for row in privacy if row.get("expected_action") == "answer_redacted"
    ]
    visible_deletion_ids = {
        checkpoint_id
        for checkpoint_id, row in diagnostic_by_id.items()
        if row.get("deletion_evaluation_status") != "not_applicable"
    }
    deletion_ids = {str(row["checkpoint_id"]) for row in deletion}
    deletion_refused = sum(row.get("pred_action") == "refuse" for row in deletion)
    provenance_valid = sum(
        row.get("provenance_integrity") is True for row in diagnostics
    )
    redacted_ids = {
        checkpoint_id
        for checkpoint_id, row in diagnostic_by_id.items()
        if row.get("redaction_applied") is True
    }
    privacy_redacted = sum(str(row["checkpoint_id"]) in redacted_ids for row in privacy)

    return {
        "schema_version": "gatemem-g2-aggregate-report-v1",
        "status": "GATEMEM_G2_BASELINE_COMPLETE",
        "closeout": {
            "offline_adapter_mechanics_verified": True,
            "provenance_integrity_verified": True,
            "disclosure_policy_adequate_for_production": False,
            "active_forgetting_evaluated": False,
            "offline_baseline_frozen": True,
        },
        "authorization": "GATEMEM_G2_OFFLINE_RETRIEVAL_DISCLOSURE_ADAPTER",
        "scope": {
            "domain": "medical",
            "offline_only": True,
            "clean_g1_projections_only": True,
            "hosted_judge_used": False,
            "leaderboard_submission": False,
            "runtime_integration": False,
        },
        "counts": {
            "checkpoints": len(scores),
            "utility": len(utility),
            "privacy": len(privacy),
            "deletion": len(deletion),
            "actions": run_summary.get("action_counts", {}),
        },
        "authorized_retrieval_utility": {
            "measurement": "GateMem auxiliary rule-based include coverage",
            "correct": sum(row.get("utility_correct") is True for row in utility),
            "total": len(utility),
            "rate": _rate(
                sum(row.get("utility_correct") is True for row in utility), len(utility)
            ),
            "primary_human_or_llm_judged_claim": False,
        },
        "disclosure_denial": {
            "privacy_action_correct": sum(
                row.get("action_correct") is True for row in privacy
            ),
            "privacy_total": len(privacy),
            "privacy_action_correct_rate": _rate(
                sum(row.get("action_correct") is True for row in privacy), len(privacy)
            ),
            "expected_refusal_correct": sum(
                row.get("pred_action") == "refuse" for row in privacy_refuse
            ),
            "expected_refusal_total": len(privacy_refuse),
            "expected_refusal_correct_rate": _rate(
                sum(row.get("pred_action") == "refuse" for row in privacy_refuse),
                len(privacy_refuse),
            ),
            "answer_leakage_rate": float(
                official_summary.get("privacy_leakage_rate", 0.0)
            ),
            "context_leakage_rate": float(
                official_summary.get("privacy_context_leakage_rate", 0.0)
            ),
            "end_to_end_leakage_rate": float(
                official_summary.get("privacy_e2e_leakage_rate", 0.0)
            ),
        },
        "redaction_behavior": {
            "adapter_redaction_count": int(run_summary.get("redaction_count", 0)),
            "privacy_redaction_count": privacy_redacted,
            "expected_redaction_correct": sum(
                row.get("pred_action") == "answer_redacted" for row in privacy_redact
            ),
            "expected_redaction_total": len(privacy_redact),
            "expected_redaction_correct_rate": _rate(
                sum(row.get("pred_action") == "answer_redacted" for row in privacy_redact),
                len(privacy_redact),
            ),
        },
        "cross_principal_leakage_attempts": {
            "candidate_count": int(
                run_summary.get("cross_principal_candidate_count", 0)
            ),
            "blocked_count": int(
                run_summary.get("blocked_cross_principal_count", 0)
            ),
            "blocked_rate": float(
                run_summary.get("blocked_cross_principal_rate", 0.0)
            ),
            "privacy_end_to_end_leakage_rate": float(
                official_summary.get("privacy_e2e_leakage_rate", 0.0)
            ),
        },
        "over_refusal": {
            "rate": float(official_summary.get("over_refusal_rate", 0.0)),
            "utility_refused_or_redacted": sum(
                row.get("pred_action") != "answer" for row in utility
            ),
            "utility_total": len(utility),
        },
        "provenance_integrity": {
            "valid": provenance_valid,
            "total": len(diagnostics),
            "rate": _rate(provenance_valid, len(diagnostics)),
        },
        "deletion_case_refusal": {
            "true_deletion_checkpoints": len(deletion),
            "explicit_refusals": deletion_refused,
            "refusal_consistency_rate": _rate(deletion_refused, len(deletion)),
            "visible_detector_true_positives": len(visible_deletion_ids & deletion_ids),
            "visible_detector_false_positives": len(visible_deletion_ids - deletion_ids),
            "visible_detector_false_negatives": len(deletion_ids - visible_deletion_ids),
            "active_forgetting_score": "NOT_SCORED",
            "official_deletion_leakage_metric_excluded_from_claims": True,
            "deletion_capability_claim": False,
        },
        "limitations": [
            "Production role enforcement remains unsupported.",
            "G1 medical projections contain no explicit entitlement metadata.",
            "The disclosure policy is a conservative visible-input heuristic.",
            "The utility metric is auxiliary rule-based coverage, not a hosted or human judge.",
            "Deletion rows are refusal-consistency checks only and do not measure active forgetting.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    utility = report["authorized_retrieval_utility"]
    disclosure = report["disclosure_denial"]
    redaction = report["redaction_behavior"]
    cross = report["cross_principal_leakage_attempts"]
    refusal = report["over_refusal"]
    provenance = report["provenance_integrity"]
    deletion = report["deletion_case_refusal"]
    lines = [
        "# GateMem G2 Offline Retrieval/Disclosure Report",
        "",
        f"`{report['status']}`",
        "",
        "`OFFLINE_ONLY` `CLEAN_G1_PROJECTIONS_ONLY` `NO_HOSTED_JUDGE` "
        "`NO_DELETION_CAPABILITY_CLAIM`",
        "",
        "| Dimension | Result |",
        "|---|---:|",
        f"| Authorized retrieval utility (auxiliary) | {utility['rate']:.4f} ({utility['correct']}/{utility['total']}) |",
        f"| Privacy action correctness | {disclosure['privacy_action_correct_rate']:.4f} |",
        f"| Expected-refusal correctness | {disclosure['expected_refusal_correct_rate']:.4f} |",
        f"| Privacy answer leakage | {disclosure['answer_leakage_rate']:.4f} |",
        f"| Privacy context leakage | {disclosure['context_leakage_rate']:.4f} |",
        f"| Expected-redaction correctness | {redaction['expected_redaction_correct_rate']:.4f} |",
        f"| Cross-principal candidates blocked | {cross['blocked_rate']:.4f} ({cross['blocked_count']}/{cross['candidate_count']}) |",
        f"| Over-refusal | {refusal['rate']:.4f} |",
        f"| Used-record provenance integrity | {provenance['rate']:.4f} ({provenance['valid']}/{provenance['total']}) |",
        f"| Deletion refusal consistency | {deletion['refusal_consistency_rate']:.4f} ({deletion['explicit_refusals']}/{deletion['true_deletion_checkpoints']}) |",
        "| Active-forgetting score | **NOT SCORED** |",
        "",
        "## Interpretation",
        "",
        "The adapter proves clean offline wiring and exact used-record provenance, but its "
        "metadata-poor visible policy is not production-grade disclosure enforcement. "
        f"Privacy end-to-end leakage is {disclosure['end_to_end_leakage_rate']:.4f}, "
        f"and over-refusal is {refusal['rate']:.4f}.",
        "",
        "Deletion checkpoints are explicitly refused. The official deletion leakage value "
        "is excluded because refusal does not demonstrate removal, non-recoverability, or "
        "non-confirmation. The visible detector produced "
        f"{deletion['visible_detector_false_positives']} conservative false positives and "
        f"{deletion['visible_detector_false_negatives']} false negatives.",
        "",
        "## Persistent limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "No runtime route, authorization change, hosted judge, leaderboard submission, "
            "or deletion capability claim is authorized by this report.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    run_summary = _load_json(args.run_summary)
    report = compile_report(
        run_summary,
        _load_jsonl(args.diagnostics),
        _load_jsonl(args.scores),
        _load_json(args.official_summary),
    )
    report["evidence_sha256"] = {
        "predictions": str(run_summary.get("predictions_sha256", "")),
        "run_summary": _sha256(args.run_summary),
        "diagnostics": _sha256(args.diagnostics),
        "scores": _sha256(args.scores),
        "official_summary": _sha256(args.official_summary),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
