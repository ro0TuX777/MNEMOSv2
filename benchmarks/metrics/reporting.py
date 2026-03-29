"""
MNEMOS Benchmark - Report Generator
======================================

Generates JSON + markdown reports from benchmark results.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict

from benchmarks.metrics.system_metrics import capture_environment, snapshot_to_dict


def save_raw_results(results: Dict[str, Any], output_dir: Path) -> Path:
    """Save raw results as timestamped JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{timestamp}_profile_benchmarks.json"

    env = capture_environment()
    output = {
        "environment": snapshot_to_dict(env),
        "timestamp": env.timestamp,
        "results": results,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Raw results: {path}")
    return path


def generate_markdown_report(results: Dict[str, Any], output_dir: Path) -> Path:
    """Generate a human-readable markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{timestamp}_report.md"

    env = capture_environment()
    lines = [
        "# MNEMOS Profile Benchmark Report",
        "",
        f"*Generated: {env.timestamp}*",
        "",
        "## Environment",
        "",
        f"| Property | Value |",
        f"|---|---|",
        f"| OS | {env.os_name} {env.os_version} |",
        f"| CPU | {env.cpu_model} ({env.cpu_cores} cores) |",
        f"| RAM | {env.ram_gb} GB |",
        f"| GPU | {env.gpu_name or 'N/A'} |",
        f"| VRAM | {env.vram_mb} MB |",
        f"| Python | {env.python_version} |",
        f"| Docker | {env.docker_version or 'N/A'} |",
        "",
    ]

    # Track 1: Retrieval
    if "retrieval" in results:
        ret = results["retrieval"]
        lines.extend([
            "---",
            "",
            "## Track 1: Profile Retrieval",
            "",
        ])

        # Ingest results
        if "ingest" in ret:
            lines.extend(["### Ingest Performance", "", "| Tier | Docs | Time | Throughput |", "|---|---|---|---|"])
            for tier, data in ret["ingest"].items():
                if data.get("status") == "success":
                    lines.append(
                        f"| {tier} | {data['corpus_size']} | {data['elapsed_s']:.2f}s | "
                        f"{data['docs_per_sec']:.0f} docs/s |"
                    )
                else:
                    lines.append(f"| {tier} | - | - | {data.get('status', 'skipped')} |")
            lines.append("")

        # Search results
        if "search" in ret:
            lines.extend(["### Search Performance", ""])
            for tier, regimes in ret["search"].items():
                lines.extend([
                    f"#### {tier}",
                    "",
                    "| Regime | Recall@10 | MRR@10 | nDCG@10 | Compliance@10 | Violation Rate | p50 (ms) | p99 (ms) | QPS |",
                    "|---|---|---|---|---|---|---|---|---|",
                ])
                for regime, data in regimes.items():
                    if data.get("status") == "success":
                        compliance = data.get("filter_compliance_at_10")
                        violation = data.get("constraint_violation_rate_at_10")
                        compliance_str = f"{compliance:.4f}" if compliance is not None else "-"
                        violation_str = f"{violation:.4f}" if violation is not None else "-"
                        lines.append(
                            f"| {regime} | {data['recall_at_10']:.4f} | {data['mrr_at_10']:.4f} | "
                            f"{data['ndcg_at_10']:.4f} | {compliance_str} | {violation_str} | "
                            f"{data['latency_p50_ms']:.1f} | {data['latency_p99_ms']:.1f} | "
                            f"{data['throughput_qps']:.0f} |"
                        )
                    else:
                        lines.append(f"| {regime} | - | - | - | - | - | - | - | {data.get('status')} |")
                lines.append("")

    # Track 2: Reranking
    if "rerank" in results:
        rr = results["rerank"]
        lines.extend(["---", "", "## Track 2: ColBERT Reranking Uplift", ""])
        if rr.get("status") == "success":
            for tier_name, tier_data in rr.get("tiers", {}).items():
                lines.extend([f"### {tier_name}", ""])
                if tier_data.get("status") != "success":
                    lines.append(f"*Skipped: {tier_data.get('reason', 'unknown')}*")
                    lines.append("")
                    continue

                baseline = tier_data.get("baseline", {})
                lines.extend([
                    f"**Baseline:** MRR={baseline.get('mrr_at_10', 0):.4f}  "
                    f"nDCG={baseline.get('ndcg_at_10', 0):.4f}  "
                    f"p50={baseline.get('latency_p50_ms', 0):.1f}ms",
                    "",
                    "| Depth | MRR@10 | MRR Uplift | nDCG@10 | nDCG Uplift | p50 (ms) | Latency delta (ms) | VRAM delta (MB) | MRR Uplift/ms |",
                    "|---|---|---|---|---|---|---|---|---|",
                ])
                for _, data in tier_data.get("reranked", {}).items():
                    lines.append(
                        f"| top-{data['depth']} | {data['mrr_at_10']:.4f} | "
                        f"{data['mrr_uplift']:+.4f} | {data['ndcg_at_10']:.4f} | "
                        f"{data['ndcg_uplift']:+.4f} | {data['latency_p50_ms']:.1f} | "
                        f"{data['latency_increase_ms']:+.1f} | {data['vram_delta_mb']:.0f} | "
                        f"{data['mrr_uplift_per_ms']:+.6f} |"
                    )
                rec = tier_data.get("recommended_depth")
                lines.append("")
                lines.append(f"Recommended depth: top-{rec}" if rec else "Recommended depth: n/a")
                lines.append("")

                audit = tier_data.get("audit")
                if audit and audit.get("status") == "success":
                    lines.extend([
                        f"Audit traces: {audit.get('n_queries', 0)} queries @ top-{audit.get('depth', '-')}",
                        "Captured per query: baseline top-10, ColBERT top-10, lexical-overlap top-10, mean-cosine top-10, and gold-rank movement.",
                        "",
                    ])
                    summary = audit.get("summary", {})
                    if summary:
                        lines.extend([
                            "| Audit Ranker | MRR@10 | nDCG@10 | Recall@10 |",
                            "|---|---|---|---|",
                        ])
                        order = ["baseline", "colbert", "lexical_overlap", "mean_cosine"]
                        labels = {
                            "baseline": "Baseline ANN",
                            "colbert": "ColBERT",
                            "lexical_overlap": "Lexical overlap",
                            "mean_cosine": "Mean-cosine",
                        }
                        for k in order:
                            if k not in summary:
                                continue
                            s = summary[k]
                            lines.append(
                                f"| {labels[k]} | {s.get('mrr_at_10', 0):.4f} | "
                                f"{s.get('ndcg_at_10', 0):.4f} | {s.get('recall_at_10', 0):.4f} |"
                            )
                        lines.append("")
                    gate = audit.get("gate_a1_pass")
                    if gate is True:
                        lines.append("Gate A.1 vs trivial baselines: PASS")
                    elif gate is False:
                        lines.append("Gate A.1 vs trivial baselines: FAIL")
                    else:
                        lines.append("Gate A.1 vs trivial baselines: n/a")
                    lines.append("")
        else:
            lines.append(f"*Skipped: {rr.get('reason', 'unknown')}*")
            lines.append("")

    # Track 3: Installer
    if "installer" in results:
        inst = results["installer"]
        lines.extend(["---", "", "## Track 3: Installer Overhead", ""])
        if inst.get("status") == "success":
            lines.extend([
                "| Profile | Installer (median) | Manual (median) | Validation Pass | Manifest |",
                "|---|---|---|---|---|",
            ])
            for profile, data in inst.get("profiles", {}).items():
                s = data.get("summary", {})
                lines.append(
                    f"| {profile} | {s.get('installer_median_s', 0):.3f}s | "
                    f"{s.get('manual_median_s', 0):.3f}s | "
                    f"{s.get('validation_pass_rate', 0):.0%} | "
                    f"{'[OK]' if s.get('installer_generates_manifest') else '[NO]'} |"
                )
        else:
            lines.append(f"*Skipped: {inst.get('reason', 'unknown')}*")
        lines.append("")

    # Track 4: Migration
    if "migration" in results:
        mig = results["migration"]
        lines.extend(["---", "", "## Track 4: Migration & Recovery", ""])
        if mig.get("status") == "success":
            lines.extend([
                "| Migration | Time | Docs | Integrity | Data Loss |",
                "|---|---|---|---|---|",
            ])
            for direction, data in mig.get("migrations", {}).items():
                if data.get("status") == "success":
                    integrity = data.get("integrity_checks", {})
                    lines.append(
                        f"| {direction} | {data['migration_time_s']:.2f}s | "
                        f"{data['target_count']} | "
                        f"{integrity.get('found', 0)}/{integrity.get('checked', 0)} | "
                        f"{data.get('data_loss', 0)} |"
                    )
                else:
                    lines.append(f"| {direction} | - | - | - | {data.get('status')} |")
        else:
            lines.append(f"*Skipped: {mig.get('reason', 'unknown')}*")
        lines.append("")

    # Track 5: Hybrid Retrieval
    if "hybrid" in results:
        hyb = results["hybrid"]
        lines.extend(["---", "", "## Track 5: Hybrid Retrieval (Gate C)", ""])
        if hyb.get("status") == "success":
            lines.extend([
                f"Semantic backend: `{hyb.get('semantic_backend', 'unknown')}`",
                f"Query set: `{hyb.get('queryset', '')}`",
                f"Queries: {hyb.get('n_queries', 0)}",
                "",
                "| Mode | Recall@10 | MRR@10 | nDCG@10 | p50 (ms) | p95 (ms) | Lexical-only contrib | Semantic-only contrib | Overlap rate |",
                "|---|---|---|---|---|---|---|---|---|",
            ])
            for mode_name, row in hyb.get("modes", {}).items():
                lex = row.get("lexical_only_contribution_rate")
                sem = row.get("semantic_only_contribution_rate")
                ovl = row.get("overlap_rate")
                lines.append(
                    f"| {mode_name} | {row.get('recall_at_10', 0):.4f} | {row.get('mrr_at_10', 0):.4f} | "
                    f"{row.get('ndcg_at_10', 0):.4f} | {row.get('latency_p50_ms', 0):.1f} | "
                    f"{row.get('latency_p95_ms', 0):.1f} | "
                    f"{('-' if lex is None else f'{lex:.4f}')} | "
                    f"{('-' if sem is None else f'{sem:.4f}')} | "
                    f"{('-' if ovl is None else f'{ovl:.4f}')} |"
                )
            lines.append("")

            lines.extend([
                "### By Query Class",
                "",
            ])
            policy_winners = hyb.get("winning_policy_by_query_class", {})
            if policy_winners:
                lines.extend([
                    "### Winning Policy By Query Class",
                    "",
                    "| Query Class | Winning Fusion Policy | semantic_dominant MRR@10 | balanced MRR@10 | lexical_dominant MRR@10 |",
                    "|---|---|---|---|---|",
                ])
                for cls, row in policy_winners.items():
                    scores = row.get("scores", {})
                    lines.append(
                        f"| {cls} | {row.get('winning_policy', '-')} | "
                        f"{scores.get('semantic_dominant', 0):.4f} | "
                        f"{scores.get('balanced', 0):.4f} | "
                        f"{scores.get('lexical_dominant', 0):.4f} |"
                    )
                lines.append("")

            for mode_name, row in hyb.get("modes", {}).items():
                lines.extend([
                    f"#### {mode_name}",
                    "",
                    "| Query Class | Recall@10 | MRR@10 | nDCG@10 | p50 (ms) | p95 (ms) | Hybrid win rate vs semantic-only | Lexical-only rate | Semantic-only rate | Both rate |",
                    "|---|---|---|---|---|---|---|---|---|---|",
                ])
                wins = row.get("hybrid_win_rate_over_semantic_only", {})
                contrib_by_class = row.get("contribution_breakdown", {}).get("by_query_class", {})
                for cls, cls_row in row.get("by_query_class", {}).items():
                    win_str = "-"
                    if wins:
                        win_str = f"{wins.get(cls, {}).get('rate', 0):.4f}"
                    contrib = contrib_by_class.get(cls, {})
                    lex_rate = "-" if not contrib else f"{contrib.get('lexical_only_rate', 0):.4f}"
                    sem_rate = "-" if not contrib else f"{contrib.get('semantic_only_rate', 0):.4f}"
                    both_rate = "-" if not contrib else f"{contrib.get('both_rate', 0):.4f}"
                    lines.append(
                        f"| {cls} | {cls_row.get('recall_at_10', 0):.4f} | {cls_row.get('mrr_at_10', 0):.4f} | "
                        f"{cls_row.get('ndcg_at_10', 0):.4f} | {cls_row.get('latency_p50_ms', 0):.1f} | "
                        f"{cls_row.get('latency_p95_ms', 0):.1f} | {win_str} | "
                        f"{lex_rate} | {sem_rate} | {both_rate} |"
                    )
                if wins:
                    lines.append(
                        f"| overall | - | - | - | - | - | {wins.get('overall', {}).get('rate', 0):.4f} | - | - | - |"
                    )
                lines.append("")
        else:
            lines.append(f"*Skipped: {hyb.get('reason', 'unknown')}*")
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Report: {path}")
    return path
