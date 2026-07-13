"""Compile frozen G2A per-domain and weighted cross-domain aggregate reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compile_gatemem_g2_report import compile_report  # noqa: E402

DOMAINS = ("medical", "office", "education", "household")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def verify_frozen_baseline(manifest_path: Path) -> dict[str, Any]:
    manifest = _json(manifest_path)
    actual_files = {}
    for relative, expected in manifest["source_sha256"].items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"Frozen baseline drift: {relative}")
        actual_files[relative] = actual
    canonical = {
        "files": actual_files,
        "config": manifest["configuration"],
        "gatemem_upstream_commit": manifest["gatemem_upstream_commit"],
    }
    composite = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if composite != manifest["composite_sha256"]:
        raise ValueError("Frozen baseline composite hash mismatch.")
    return manifest


def _paths(domain: str, medical_root: Path, replay_root: Path) -> dict[str, Path]:
    if domain == "medical":
        return {
            "projections": medical_root.parent / "gatemem_g1_artifacts" / "medical_clean_projections.jsonl",
            "predictions": medical_root / "medical_predictions.jsonl",
            "diagnostics": medical_root / "medical_diagnostics.jsonl",
            "run_summary": medical_root / "medical_run_summary.json",
            "scores": medical_root / "medical_rule_score" / "scores.jsonl",
            "official_summary": medical_root / "medical_rule_score" / "summary.json",
        }
    base = replay_root / domain
    return {
        "projections": base / "clean_projections.jsonl",
        "predictions": base / "predictions.jsonl",
        "diagnostics": base / "diagnostics.jsonl",
        "run_summary": base / "run_summary.json",
        "scores": base / "rule_score" / "scores.jsonl",
        "official_summary": base / "rule_score" / "summary.json",
    }


def _official_aggregate(scores: list[dict[str, Any]]) -> dict[str, float]:
    privacy = [row for row in scores if row.get("query_type") == "privacy"]
    utility = [row for row in scores if row.get("query_type") == "utility"]
    return {
        "privacy_leakage_rate": _rate(
            sum(row.get("privacy_leak") is True for row in privacy), len(privacy)
        ),
        "privacy_context_leakage_rate": _rate(
            sum(row.get("privacy_context_leak") is True for row in privacy), len(privacy)
        ),
        "privacy_e2e_leakage_rate": _rate(
            sum(row.get("privacy_e2e_leak") is True for row in privacy), len(privacy)
        ),
        "over_refusal_rate": _rate(
            sum(row.get("pred_action") != "answer" for row in utility), len(utility)
        ),
    }


def _combined_run_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    actions: dict[str, int] = {}
    for summary in summaries:
        for action, count in summary.get("action_counts", {}).items():
            actions[action] = actions.get(action, 0) + int(count)
    candidates = sum(int(item["cross_principal_candidate_count"]) for item in summaries)
    blocked = sum(int(item["blocked_cross_principal_count"]) for item in summaries)
    return {
        "action_counts": actions,
        "redaction_count": sum(int(item["redaction_count"]) for item in summaries),
        "cross_principal_candidate_count": candidates,
        "blocked_cross_principal_count": blocked,
        "blocked_cross_principal_rate": _rate(blocked, candidates),
    }


def _domain_markdown(report: dict[str, Any]) -> str:
    utility = report["authorized_retrieval_utility"]
    disclosure = report["disclosure_denial"]
    redaction = report["redaction_behavior"]
    cross = report["cross_principal_leakage_attempts"]
    refusal = report["over_refusal"]
    provenance = report["provenance_integrity"]
    deletion = report["deletion_case_refusal"]
    domain = report["scope"]["domain"]
    return "\n".join(
        [
            f"# GateMem G2A Frozen Baseline Replay — {domain.title()}",
            "",
            "`GATEMEM_G2A_DOMAIN_REPLAY_COMPLETE` `FROZEN_ADAPTER` `NO_TUNING`",
            "",
            "| Dimension | Result |",
            "|---|---:|",
            f"| Checkpoints | {report['counts']['checkpoints']} |",
            f"| Auxiliary utility | {utility['rate']:.4f} ({utility['correct']}/{utility['total']}) |",
            f"| Privacy end-to-end leakage | {disclosure['end_to_end_leakage_rate']:.4f} |",
            f"| Over-refusal | {refusal['rate']:.4f} |",
            f"| Expected-redaction correctness | {redaction['expected_redaction_correct_rate']:.4f} |",
            f"| Cross-principal candidates blocked | {cross['blocked_rate']:.4f} |",
            f"| Provenance integrity | {provenance['rate']:.4f} |",
            f"| Deletion refusal consistency | {deletion['refusal_consistency_rate']:.4f} |",
            f"| Deletion detector false positives | {deletion['visible_detector_false_positives']} |",
            f"| Deletion detector false negatives | {deletion['visible_detector_false_negatives']} |",
            "| Active forgetting | **NOT SCORED** |",
            "",
            "This is characterization of the frozen metadata-poor baseline, not a "
            "production governance result. Row-level artifacts remain external.",
            "",
        ]
    )


def _aggregate_markdown(
    aggregate: dict[str, Any], per_domain: dict[str, dict[str, Any]], baseline_hash: str
) -> str:
    lines = [
        "# GateMem G2A Frozen Cross-Domain Baseline Replay",
        "",
        "`GATEMEM_G2A_CROSS_DOMAIN_BASELINE_REPLAY_COMPLETE`",
        "",
        f"Frozen core: `{baseline_hash}`",
        "",
        "| Domain | Checkpoints | Utility | Privacy leakage | Over-refusal | Provenance | Deletion refusal | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        report = per_domain[domain]
        lines.append(
            f"| {domain.title()} | {report['counts']['checkpoints']} | "
            f"{report['authorized_retrieval_utility']['rate']:.4f} | "
            f"{report['disclosure_denial']['end_to_end_leakage_rate']:.4f} | "
            f"{report['over_refusal']['rate']:.4f} | "
            f"{report['provenance_integrity']['rate']:.4f} | "
            f"{report['deletion_case_refusal']['refusal_consistency_rate']:.4f} | "
            f"{report['deletion_case_refusal']['visible_detector_false_positives']} | "
            f"{report['deletion_case_refusal']['visible_detector_false_negatives']} |"
        )
    lines.append(
        f"| **Weighted aggregate** | **{aggregate['counts']['checkpoints']}** | "
        f"**{aggregate['authorized_retrieval_utility']['rate']:.4f}** | "
        f"**{aggregate['disclosure_denial']['end_to_end_leakage_rate']:.4f}** | "
        f"**{aggregate['over_refusal']['rate']:.4f}** | "
        f"**{aggregate['provenance_integrity']['rate']:.4f}** | "
        f"**{aggregate['deletion_case_refusal']['refusal_consistency_rate']:.4f}** | "
        f"**{aggregate['deletion_case_refusal']['visible_detector_false_positives']}** | "
        f"**{aggregate['deletion_case_refusal']['visible_detector_false_negatives']}** |"
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Medical is an outlier for privacy leakage, but the limitation is structural: "
            "every domain has low auxiliary utility, material over-refusal, nonzero privacy "
            "leakage, and incomplete deletion-language coverage outside medical.",
            "",
            "Candidate blocking is not content-safe disclosure. Provenance integrity remains "
            "perfect while governance performance remains inadequate.",
            "",
            "Deletion metrics are refusal consistency only. Active forgetting is not scored.",
            "",
            "No retrieval weights, disclosure rules, deletion detection, thresholds, or "
            "normalizer behavior were changed for G2A.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medical-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "gatemem_g2_baseline_manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "benchmarks" / "results"
    )
    args = parser.parse_args()

    baseline = verify_frozen_baseline(args.baseline_manifest)
    per_domain: dict[str, dict[str, Any]] = {}
    all_diagnostics: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for domain in DOMAINS:
        paths = _paths(domain, args.medical_root, args.replay_root)
        run_summary = _json(paths["run_summary"])
        diagnostics = _jsonl(paths["diagnostics"])
        scores = _jsonl(paths["scores"])
        report = compile_report(
            run_summary, diagnostics, scores, _json(paths["official_summary"])
        )
        report["status"] = "GATEMEM_G2A_DOMAIN_REPLAY_COMPLETE"
        report["scope"]["domain"] = domain
        report["frozen_baseline_composite_sha256"] = baseline["composite_sha256"]
        report["evidence_sha256"] = {
            name: (
                str(run_summary["predictions_sha256"])
                if name == "predictions"
                else _sha256(path)
            )
            for name, path in paths.items()
        }
        per_domain[domain] = report
        all_diagnostics.extend(diagnostics)
        all_scores.extend(scores)
        summaries.append(run_summary)
        (args.output_dir / f"gatemem_g2a_{domain}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / f"gatemem_g2a_{domain}_report.md").write_text(
            _domain_markdown(report), encoding="utf-8"
        )

    aggregate = compile_report(
        _combined_run_summary(summaries),
        all_diagnostics,
        all_scores,
        _official_aggregate(all_scores),
    )
    aggregate["status"] = "GATEMEM_G2A_CROSS_DOMAIN_BASELINE_REPLAY_COMPLETE"
    aggregate["scope"]["domain"] = "all-four-domains"
    aggregate["frozen_baseline_composite_sha256"] = baseline["composite_sha256"]
    aggregate["per_domain"] = {
        domain: {
            "checkpoints": report["counts"]["checkpoints"],
            "utility_rate": report["authorized_retrieval_utility"]["rate"],
            "privacy_e2e_leakage_rate": report["disclosure_denial"][
                "end_to_end_leakage_rate"
            ],
            "over_refusal_rate": report["over_refusal"]["rate"],
            "provenance_integrity_rate": report["provenance_integrity"]["rate"],
            "deletion_refusal_consistency_rate": report["deletion_case_refusal"][
                "refusal_consistency_rate"
            ],
            "deletion_false_positives": report["deletion_case_refusal"][
                "visible_detector_false_positives"
            ],
            "deletion_false_negatives": report["deletion_case_refusal"][
                "visible_detector_false_negatives"
            ],
        }
        for domain, report in per_domain.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gatemem_g2a_cross_domain_report.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "gatemem_g2a_cross_domain_report.md").write_text(
        _aggregate_markdown(aggregate, per_domain, baseline["composite_sha256"]),
        encoding="utf-8",
    )
    print(json.dumps(aggregate["per_domain"], indent=2, sort_keys=True))
    print(f"Wrote G2A reports to {args.output_dir}")


if __name__ == "__main__":
    main()
