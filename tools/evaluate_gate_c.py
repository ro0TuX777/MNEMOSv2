"""Evaluate Gate C readiness from hybrid benchmark artifacts.

Usage:
  python tools/evaluate_gate_c.py
  python tools/evaluate_gate_c.py --raw benchmarks/outputs/raw/<timestamp>_profile_benchmarks.json
  python tools/evaluate_gate_c.py --require-pass
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "summaries"

REQUIRED_MODES = [
    "semantic_only",
    "lexical_only",
    "hybrid_semantic_dominant",
    "hybrid_balanced",
    "hybrid_lexical_dominant",
]


def _find_latest_hybrid_raw() -> Optional[Path]:
    if not RAW_DIR.exists():
        return None
    candidates = sorted(RAW_DIR.glob("*_profile_benchmarks.json"), reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "hybrid" in data.get("results", {}):
            return path
    return None


def _compute_best_policy(modes: Dict[str, Any]) -> Optional[str]:
    policy_map = {
        "hybrid_semantic_dominant": "semantic_dominant",
        "hybrid_balanced": "balanced",
        "hybrid_lexical_dominant": "lexical_dominant",
    }
    best_policy = None
    best_mrr = -1.0
    for mode, policy in policy_map.items():
        mrr = float(modes.get(mode, {}).get("mrr_at_10", 0.0) or 0.0)
        if mrr > best_mrr:
            best_mrr = mrr
            best_policy = policy
    return best_policy


def _collect_class_wins(modes: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for mode in ["hybrid_semantic_dominant", "hybrid_balanced", "hybrid_lexical_dominant"]:
        wins = modes.get(mode, {}).get("hybrid_win_rate_over_semantic_only", {})
        for query_class, row in wins.items():
            if query_class == "overall":
                continue
            rate = float(row.get("rate", 0.0) or 0.0)
            if rate > 0.0:
                out.setdefault(query_class, []).append(mode)
    return out


def evaluate_gate_c(payload: Dict[str, Any], latency_threshold: float = 1.5) -> Dict[str, Any]:
    results = payload.get("results", {})
    hybrid = results.get("hybrid", {})

    status = hybrid.get("status")
    modes = hybrid.get("modes", {}) if isinstance(hybrid.get("modes"), dict) else {}

    missing_modes = [m for m in REQUIRED_MODES if m not in modes]

    semantic_p50 = float(modes.get("semantic_only", {}).get("latency_p50_ms", 0.0) or 0.0)
    latency_checks: Dict[str, Any] = {}
    for mode in ["hybrid_semantic_dominant", "hybrid_balanced", "hybrid_lexical_dominant"]:
        p50 = float(modes.get(mode, {}).get("latency_p50_ms", 0.0) or 0.0)
        ratio = (p50 / semantic_p50) if semantic_p50 > 0 else None
        latency_checks[mode] = {
            "p50_ms": p50,
            "ratio_vs_semantic": round(ratio, 4) if ratio is not None else None,
            "within_threshold": (ratio is not None and ratio <= latency_threshold),
        }

    class_wins = _collect_class_wins(modes)
    has_class_win = len(class_wins) > 0

    best_policy = _compute_best_policy(modes)

    any_hybrid_within_latency = any(v.get("within_threshold") for v in latency_checks.values())

    pass_track = (status == "success") and (len(missing_modes) == 0)
    pass_quality = has_class_win
    pass_latency = any_hybrid_within_latency

    sprint_pass = pass_track and pass_quality and pass_latency

    decisions = {
        "does_hybrid_materially_improve_any_class": "yes" if has_class_win else "no_or_inconclusive",
        "recommended_default_fusion_policy": best_policy if has_class_win else "none_keep_semantic_default",
        "is_hybrid_ready_for_broad_exposure": "yes" if sprint_pass else "no_or_limited_pilot",
        "evidence_for_future_enterprise_search_profile": "yes" if sprint_pass else "not_yet",
    }

    return {
        "status": status,
        "missing_modes": missing_modes,
        "class_wins": class_wins,
        "latency_threshold": latency_threshold,
        "latency_checks": latency_checks,
        "pass_track": pass_track,
        "pass_quality": pass_quality,
        "pass_latency": pass_latency,
        "sprint_exit_pass": sprint_pass,
        "decisions": decisions,
    }


def _to_markdown(raw_path: Path, evaluation: Dict[str, Any]) -> str:
    lines = [
        "# Gate C Decision Report",
        "",
        f"- Source artifact: `{raw_path}`",
        f"- Generated at: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
        "",
        "## Exit Criteria",
        "",
        f"- Track execution complete: `{evaluation['pass_track']}`",
        f"- Quality class win found: `{evaluation['pass_quality']}`",
        f"- Latency threshold satisfied: `{evaluation['pass_latency']}`",
        f"- Sprint exit pass: `{evaluation['sprint_exit_pass']}`",
        "",
        "## Missing Modes",
        "",
        f"- {', '.join(evaluation['missing_modes']) if evaluation['missing_modes'] else 'none'}",
        "",
        "## Class Wins",
        "",
    ]

    class_wins = evaluation.get("class_wins", {})
    if not class_wins:
        lines.append("- none")
    else:
        for query_class, modes in sorted(class_wins.items()):
            lines.append(f"- `{query_class}`: {', '.join(modes)}")

    lines.extend([
        "",
        "## Latency Checks",
        "",
        "| Hybrid Mode | p50 (ms) | Ratio vs semantic | Within threshold |",
        "|---|---:|---:|---|",
    ])
    for mode, row in evaluation.get("latency_checks", {}).items():
        ratio = row.get("ratio_vs_semantic")
        lines.append(
            f"| {mode} | {row.get('p50_ms', 0):.2f} | {('-' if ratio is None else ratio)} | {row.get('within_threshold')} |"
        )

    d = evaluation.get("decisions", {})
    lines.extend([
        "",
        "## Product Decisions",
        "",
        f"1. Does hybrid improve enterprise-style retrieval for any class? `{d.get('does_hybrid_materially_improve_any_class')}`",
        f"2. Which fusion policy should be default? `{d.get('recommended_default_fusion_policy')}`",
        f"3. Is hybrid ready for broad exposure? `{d.get('is_hybrid_ready_for_broad_exposure')}`",
        f"4. Is there evidence to justify a future enterprise profile? `{d.get('evidence_for_future_enterprise_search_profile')}`",
        "",
    ])

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Gate C readiness from hybrid artifacts.")
    parser.add_argument("--raw", default="", help="Path to raw benchmark JSON (defaults to latest with results.hybrid)")
    parser.add_argument("--latency-threshold", type=float, default=1.5, help="Max p50 ratio vs semantic-only")
    parser.add_argument("--require-pass", action="store_true", help="Return exit code 1 if sprint exit fails")
    args = parser.parse_args()

    raw_path = Path(args.raw) if args.raw else _find_latest_hybrid_raw()
    if raw_path is None or not raw_path.exists():
        print("[ERROR] Could not locate a hybrid benchmark artifact.")
        return 2

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    evaluation = evaluate_gate_c(payload, latency_threshold=args.latency_threshold)

    ts = time.strftime("%Y%m%d_%H%M%S")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    md_path = SUMMARY_DIR / f"{ts}_gate_c_decision.md"
    md_path.write_text(_to_markdown(raw_path, evaluation), encoding="utf-8")

    print("[OK] Gate C decision report generated")
    print(f"  Raw:    {raw_path}")
    print(f"  Report: {md_path}")
    print(f"  Sprint exit pass: {evaluation['sprint_exit_pass']}")

    if args.require_pass and not evaluation["sprint_exit_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
