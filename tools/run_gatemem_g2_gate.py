"""Verify the GateMem G2 offline adapter and frozen medical result boundary."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "benchmarks" / "results" / "gatemem_g2_offline_report.json"
DEFAULT_JSON = ROOT / "benchmarks" / "results" / "gatemem_g2_gate.json"
DEFAULT_MD = ROOT / "benchmarks" / "results" / "gatemem_g2_gate.md"
PROHIBITED_KEYS = {
    "query_type", "attack_type", "expected_action", "judge_spec", "leak_targets"
}


def _keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def _imports_clean() -> tuple[bool, bool, bool]:
    package = ROOT / "prototype" / "gatemem_g2"
    runtime = {"mnemos", "mnemos_sdk", "service", "bench"}
    network = {"requests", "httpx", "urllib", "socket"}
    providers = {"openai", "anthropic", "transformers", "torch"}
    seen_runtime: set[str] = set()
    seen_network: set[str] = set()
    seen_providers: set[str] = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            seen_runtime.update(roots & runtime)
            seen_network.update(roots & network)
            seen_providers.update(roots & providers)
    return not seen_runtime, not seen_network, not seen_providers


def evaluate_gate(
    report: dict[str, Any], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    no_runtime, no_network, no_providers = _imports_clean()
    deletion = report["deletion_case_refusal"]
    provenance = report["provenance_integrity"]
    disclosure = report["disclosure_denial"]
    utility = report["authorized_retrieval_utility"]
    refusal = report["over_refusal"]
    gates = {
        "offline_only": report["scope"]["offline_only"] is True,
        "clean_g1_projections_only": report["scope"]["clean_g1_projections_only"] is True,
        "prediction_count_complete": len(predictions) == report["counts"]["checkpoints"],
        "prediction_schema_compatible": all(
            set(row) == {"checkpoint_id", "output"} for row in predictions
        ),
        "no_scoring_fields_in_predictions": all(
            not (set(_keys(row)) & PROHIBITED_KEYS) for row in predictions
        ),
        "provenance_integrity_1_0": provenance["rate"] == 1.0,
        "deletion_refusal_consistency_1_0": deletion["refusal_consistency_rate"] == 1.0,
        "deletion_false_negatives_0": deletion["visible_detector_false_negatives"] == 0,
        "deletion_not_scored": deletion["active_forgetting_score"] == "NOT_SCORED",
        "no_deletion_capability_claim": deletion["deletion_capability_claim"] is False,
        "utility_measured": utility["total"] > 0,
        "disclosure_measured": disclosure["privacy_total"] > 0,
        "over_refusal_measured": refusal["utility_total"] > 0,
        "limitations_retained": len(report.get("limitations", [])) >= 5,
        "no_runtime_or_gatemem_imports": no_runtime,
        "no_network_imports": no_network,
        "no_hosted_provider_imports": no_providers,
        "no_hosted_judge": report["scope"]["hosted_judge_used"] is False,
        "no_leaderboard_submission": report["scope"]["leaderboard_submission"] is False,
    }
    return {
        "schema_version": "gatemem-g2-gate-v1",
        "authorization": "GATEMEM_G2_OFFLINE_RETRIEVAL_DISCLOSURE_ADAPTER",
        "status": "GATEMEM_G2_BASELINE_COMPLETE",
        "closeout": [
            "OFFLINE_ADAPTER_MECHANICS_VERIFIED",
            "PROVENANCE_INTEGRITY_VERIFIED",
            "DISCLOSURE_POLICY_INADEQUATE_FOR_PRODUCTION",
            "ACTIVE_FORGETTING_NOT_EVALUATED",
        ],
        "gates": gates,
        "all_passed": all(gates.values()),
        "observed_limitations": {
            "auxiliary_utility_rate": utility["rate"],
            "privacy_end_to_end_leakage_rate": disclosure["end_to_end_leakage_rate"],
            "over_refusal_rate": refusal["rate"],
        },
        "advancement_boundary": (
            "This pass validates offline adapter mechanics and measurement only. "
            "It authorizes no runtime policy claim or deletion lifecycle."
        ),
    }


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# GateMem G2 Offline Adapter Gate",
        "",
        "`GATEMEM_G2_BASELINE_COMPLETE` `DISCLOSURE_POLICY_INADEQUATE_FOR_PRODUCTION`",
        "",
        "This is a successful benchmark-baseline closeout, not a successful governance-performance result.",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["gates"].items()
    )
    limitations = report["observed_limitations"]
    lines.extend(
        [
            "",
            f"**Overall: {'PASS' if report['all_passed'] else 'FAIL'}**",
            "",
            "Measured limitations remain gate output, not hidden failures:",
            "",
            f"- auxiliary utility: `{limitations['auxiliary_utility_rate']:.4f}`",
            f"- privacy end-to-end leakage: `{limitations['privacy_end_to_end_leakage_rate']:.4f}`",
            f"- over-refusal: `{limitations['over_refusal_rate']:.4f}`",
            "",
            report["advancement_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gate = evaluate_gate(report, predictions)
    DEFAULT_JSON.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    DEFAULT_MD.write_text(_render(gate), encoding="utf-8")
    print(f"All G2 gates passed: {gate['all_passed']}")
    print(f"Wrote {DEFAULT_JSON}")
    print(f"Wrote {DEFAULT_MD}")
    raise SystemExit(0 if gate["all_passed"] else 1)


if __name__ == "__main__":
    main()
