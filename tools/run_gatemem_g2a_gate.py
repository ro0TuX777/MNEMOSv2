"""Verify frozen-core integrity and external evidence for GateMem G2A replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compile_gatemem_g2a_cross_domain import (  # noqa: E402
    DOMAINS,
    _paths,
    _sha256,
    verify_frozen_baseline,
)

PROHIBITED_KEYS = {
    "query_type", "attack_type", "expected_action", "judge_spec", "leak_targets"
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def run_gate(
    *,
    medical_root: Path,
    replay_root: Path,
    results_dir: Path,
    baseline_manifest: Path,
) -> dict[str, Any]:
    baseline = verify_frozen_baseline(baseline_manifest)
    aggregate = _json(results_dir / "gatemem_g2a_cross_domain_report.json")
    per_domain = {
        domain: _json(results_dir / f"gatemem_g2a_{domain}_report.json")
        for domain in DOMAINS
    }
    external_hashes_valid = True
    predictions_clean = True
    projection_counts_match = True
    judge_artifacts_absent = True
    for domain in DOMAINS:
        paths = _paths(domain, medical_root, replay_root)
        report = per_domain[domain]
        for name, path in paths.items():
            expected = report["evidence_sha256"][name]
            actual = _sha256(path)
            external_hashes_valid &= actual == expected
        predictions = _jsonl(paths["predictions"])
        predictions_clean &= all(
            not (set(_keys(row)) & PROHIBITED_KEYS) for row in predictions
        )
        projection_count = sum(
            1 for line in paths["projections"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        projection_counts_match &= (
            projection_count
            == len(predictions)
            == report["counts"]["checkpoints"]
        )
        judge_artifacts_absent &= not (paths["scores"].parent / "judge_scores.jsonl").exists()

    gates = {
        "frozen_core_hash_valid": baseline["composite_sha256"]
        == "4211bc91e7dbe53f588a8ecb00a04e6e44d2fb775ca1e888d058680acb5ad209",
        "all_four_domains_present": set(per_domain) == set(DOMAINS),
        "all_2218_checkpoints_processed": aggregate["counts"]["checkpoints"] == 2218,
        "projection_prediction_counts_match": projection_counts_match,
        "external_evidence_hashes_valid": external_hashes_valid,
        "predictions_contain_no_scoring_fields": predictions_clean,
        "provenance_integrity_1_0_all_domains": all(
            report["provenance_integrity"]["rate"] == 1.0
            for report in per_domain.values()
        ),
        "deletion_refusal_measured_all_domains": all(
            report["deletion_case_refusal"]["true_deletion_checkpoints"] > 0
            for report in per_domain.values()
        ),
        "deletion_false_positives_measured_all_domains": all(
            "visible_detector_false_positives" in report["deletion_case_refusal"]
            for report in per_domain.values()
        ),
        "active_forgetting_not_scored": all(
            report["deletion_case_refusal"]["active_forgetting_score"] == "NOT_SCORED"
            for report in per_domain.values()
        ),
        "no_hosted_judge_artifacts": judge_artifacts_absent,
        "no_policy_tuning": all(
            report["frozen_baseline_composite_sha256"] == baseline["composite_sha256"]
            for report in per_domain.values()
        ),
        "aggregate_is_weighted_from_all_domains": set(aggregate["per_domain"])
        == set(DOMAINS),
        "production_claims_remain_blocked": (
            aggregate["scope"]["runtime_integration"] is False
            and aggregate["deletion_case_refusal"]["deletion_capability_claim"] is False
        ),
    }
    return {
        "schema_version": "gatemem-g2a-gate-v1",
        "status": "GATEMEM_G2A_CROSS_DOMAIN_BASELINE_REPLAY_COMPLETE",
        "frozen_baseline_composite_sha256": baseline["composite_sha256"],
        "gates": gates,
        "all_passed": all(gates.values()),
        "aggregate_characterization": {
            "checkpoints": aggregate["counts"]["checkpoints"],
            "auxiliary_utility_rate": aggregate["authorized_retrieval_utility"]["rate"],
            "privacy_e2e_leakage_rate": aggregate["disclosure_denial"][
                "end_to_end_leakage_rate"
            ],
            "over_refusal_rate": aggregate["over_refusal"]["rate"],
            "provenance_integrity_rate": aggregate["provenance_integrity"]["rate"],
            "deletion_refusal_consistency_rate": aggregate["deletion_case_refusal"][
                "refusal_consistency_rate"
            ],
            "deletion_false_positives": aggregate["deletion_case_refusal"][
                "visible_detector_false_positives"
            ],
            "deletion_false_negatives": aggregate["deletion_case_refusal"][
                "visible_detector_false_negatives"
            ],
            "active_forgetting": "NOT_SCORED",
        },
        "advancement_boundary": (
            "G2A characterizes the frozen offline baseline only. It authorizes no "
            "policy tuning, runtime integration, role-enforcement claim, or deletion claim."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GateMem G2A Cross-Domain Replay Gate",
        "",
        "`FROZEN_ADAPTER` `NO_POLICY_TUNING` `ACTIVE_FORGETTING_NOT_SCORED`",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["gates"].items()
    )
    values = report["aggregate_characterization"]
    lines.extend(
        [
            "",
            f"**Overall: {'PASS' if report['all_passed'] else 'FAIL'}**",
            "",
            "Aggregate characterization:",
            "",
            f"- checkpoints: `{values['checkpoints']}`",
            f"- auxiliary utility: `{values['auxiliary_utility_rate']:.4f}`",
            f"- privacy leakage: `{values['privacy_e2e_leakage_rate']:.4f}`",
            f"- over-refusal: `{values['over_refusal_rate']:.4f}`",
            f"- provenance integrity: `{values['provenance_integrity_rate']:.4f}`",
            f"- deletion refusal consistency: `{values['deletion_refusal_consistency_rate']:.4f}`",
            f"- deletion detector false positives / negatives: `{values['deletion_false_positives']} / {values['deletion_false_negatives']}`",
            "- active forgetting: **NOT SCORED**",
            "",
            report["advancement_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medical-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "benchmarks" / "results"
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "gatemem_g2_baseline_manifest.json",
    )
    args = parser.parse_args()
    report = run_gate(
        medical_root=args.medical_root,
        replay_root=args.replay_root,
        results_dir=args.results_dir,
        baseline_manifest=args.baseline_manifest,
    )
    json_path = args.results_dir / "gatemem_g2a_gate.json"
    md_path = args.results_dir / "gatemem_g2a_gate.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"All G2A gates passed: {report['all_passed']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
