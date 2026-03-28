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

    print(f"\n  📄 Raw results: {path}")
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
            baseline = rr.get("baseline", {})
            lines.extend([
                f"**Baseline ({rr['primary_tier']}):** MRR={baseline.get('mrr_at_10', 0):.4f}  "
                f"p50={baseline.get('latency_p50_ms', 0):.1f}ms",
                "",
                "| Depth | MRR@10 | MRR Uplift | nDCG@10 | Latency Δ (ms) | VRAM Δ (MB) |",
                "|---|---|---|---|---|---|",
            ])
            for depth_key, data in rr.get("reranked", {}).items():
                lines.append(
                    f"| top-{data['depth']} | {data['mrr_at_10']:.4f} | "
                    f"{data['mrr_uplift']:+.4f} | {data['ndcg_at_10']:.4f} | "
                    f"{data['latency_increase_ms']:+.1f} | {data['vram_delta_mb']:.0f} |"
                )
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

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  📊 Report: {path}")
    return path
