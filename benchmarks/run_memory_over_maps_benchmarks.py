"""
Memory Over Maps benchmark entry point.

Currently implements Phase 1 (M1 lineage integrity).
Produces:
  - benchmarks/outputs/raw/memory_over_maps_<timestamp>_raw.json
  - benchmarks/outputs/summaries/memory_over_maps_<timestamp>_report.md
  - benchmarks/outputs/summaries/memory_over_maps_<timestamp>_decision.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics.system_metrics import capture_environment, snapshot_to_dict
from benchmarks.runners.memory_over_maps_runner import (
    run_phase1_lineage_track,
    run_phase2_candidate_envelope_track,
    run_phase3_derived_views_track,
    run_phase4_cache_invalidation_track,
    run_phase5_reflect_bounded_track,
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _phase1_pass(result: Dict[str, Any]) -> bool:
    return (
        result["lineage_completeness_rate"] == 1.0
        and result["orphan_derived_views"] == 0
        and result["derived_view_input_completeness_rate"] == 1.0
    )


def run_phase1(sample_size: int = 12) -> Dict[str, Any]:
    env = capture_environment()
    track = run_phase1_lineage_track(sample_size=sample_size).to_dict()
    return {
        "phase": "phase1",
        "track": "M1_lineage_integrity",
        "timestamp": env.timestamp,
        "environment": snapshot_to_dict(env),
        "results": track,
        "pass": _phase1_pass(track),
        "criteria": {
            "lineage_completeness_rate": 1.0,
            "orphan_derived_views": 0,
            "derived_view_input_completeness_rate": 1.0,
        },
    }


def run_phase2() -> Dict[str, Any]:
    env = capture_environment()
    track = run_phase2_candidate_envelope_track().to_dict()
    return {
        "phase": "phase2",
        "track": "M2_candidate_envelope_efficiency",
        "timestamp": env.timestamp,
        "environment": snapshot_to_dict(env),
        "results": track,
        "pass": _phase2_pass(track),
        "criteria": {
            "compression_ratio_max": 0.75,
            "answer_support_retention_rate_min": 0.75,
            "deterministic_replay_match": True,
        },
    }


def run_phase3() -> Dict[str, Any]:
    env = capture_environment()
    track = run_phase3_derived_views_track().to_dict()
    return {
        "phase": "phase3",
        "track": "M3_on_demand_view_reproducibility",
        "timestamp": env.timestamp,
        "environment": snapshot_to_dict(env),
        "results": track,
        "pass": _phase3_pass(track),
        "criteria": {
            "reproducibility_success_rate": 1.0,
            "regeneration_mismatch_count": 0,
            "input_completeness_rate": 1.0,
        },
    }


def run_phase4() -> Dict[str, Any]:
    env = capture_environment()
    track = run_phase4_cache_invalidation_track().to_dict()
    return {
        "phase": "phase4",
        "track": "M4_cache_invalidation_correctness",
        "timestamp": env.timestamp,
        "environment": snapshot_to_dict(env),
        "results": track,
        "pass": _phase4_pass(track),
        "criteria": {
            "stale_cache_survival_rate": 0.0,
            "dry_run_real_run_parity": True,
            "invalidation_trigger_coverage_rate_min": 1.0,
        },
    }


def run_phase5() -> Dict[str, Any]:
    env = capture_environment()
    track = run_phase5_reflect_bounded_track().to_dict()
    return {
        "phase": "phase5",
        "track": "M5_bounded_semantic_reflect_evolution",
        "timestamp": env.timestamp,
        "environment": snapshot_to_dict(env),
        "results": track,
        "pass": _phase5_pass(track),
        "criteria": {
            "bounded_candidate_adherence_rate": 1.0,
            "proper_noun_sensitivity_rate_min": 0.9,
            "trust_recovery_delta_min": 0.05,
            "enforced_mode_drift_rate_max": 0.0,
            "concurrent_reflect_success_rate_min": 0.95,
        },
    }


def render_phase1_report(payload: Dict[str, Any]) -> str:
    r = payload["results"]
    lines = [
        "# Memory Over Maps Phase 1 Report",
        "",
        f"*Generated: {payload['timestamp']}*",
        "",
        "## Track",
        "",
        f"- Phase: `{payload['phase']}`",
        f"- Track: `{payload['track']}`",
        f"- Sample size: `{r['sample_size']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Target |",
        "|---|---:|---:|",
        f"| lineage_completeness_rate | {r['lineage_completeness_rate']:.4f} | 1.0000 |",
        f"| derived_view_input_completeness_rate | {r['derived_view_input_completeness_rate']:.4f} | 1.0000 |",
        f"| orphan_derived_views | {r['orphan_derived_views']} | 0 |",
        f"| responses_with_source_artifact_coverage_rate | {r['responses_with_source_artifact_coverage_rate']:.4f} | 1.0000 |",
        f"| source_trace_resolution_latency_ms | {r['source_trace_resolution_latency_ms']:.4f} | n/a |",
        f"| audit_log_derived_view_events | {r['audit_log_derived_view_events']} | >=1 |",
        "",
        "## Gate",
        "",
        f"- Phase 1 gate result: {'PASS' if payload['pass'] else 'HOLD'}",
    ]
    return "\n".join(lines) + "\n"


def render_phase2_report(payload: Dict[str, Any]) -> str:
    r = payload["results"]
    lines = [
        "# Memory Over Maps Phase 2 Report",
        "",
        f"*Generated: {payload['timestamp']}*",
        "",
        "## Track",
        "",
        f"- Phase: `{payload['phase']}`",
        f"- Track: `{payload['track']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Target |",
        "|---|---:|---:|",
        f"| initial_candidate_count | {r['initial_candidate_count']} | n/a |",
        f"| final_candidate_count | {r['final_candidate_count']} | n/a |",
        f"| compression_ratio | {r['compression_ratio']:.4f} | <= 0.7500 |",
        f"| answer_support_retention_rate | {r['answer_support_retention_rate']:.4f} | >= 0.7500 |",
        f"| duplicate_suppression_rate | {r['duplicate_suppression_rate']:.4f} | n/a |",
        f"| source_concentration_ratio | {r['source_concentration_ratio']:.4f} | n/a |",
        f"| deterministic_replay_match | {r['deterministic_replay_match']} | True |",
        "",
        "## Gate",
        "",
        f"- Phase 2 gate result: {'PASS' if payload['pass'] else 'HOLD'}",
    ]
    return "\n".join(lines) + "\n"


def render_phase3_report(payload: Dict[str, Any]) -> str:
    r = payload["results"]
    lines = [
        "# Memory Over Maps Phase 3 Report",
        "",
        f"*Generated: {payload['timestamp']}*",
        "",
        "## Track",
        "",
        f"- Phase: `{payload['phase']}`",
        f"- Track: `{payload['track']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Target |",
        "|---|---:|---:|",
        f"| reproducibility_success_rate | {r['reproducibility_success_rate']:.4f} | 1.0000 |",
        f"| regeneration_mismatch_count | {r['regeneration_mismatch_count']} | 0 |",
        f"| input_completeness_rate | {r['input_completeness_rate']:.4f} | 1.0000 |",
        f"| generated_view_count | {r['generated_view_count']} | n/a |",
        f"| mean_regeneration_ms | {r['mean_regeneration_ms']:.4f} | n/a |",
        "",
        "## Gate",
        "",
        f"- Phase 3 gate result: {'PASS' if payload['pass'] else 'HOLD'}",
    ]
    return "\n".join(lines) + "\n"


def render_phase4_report(payload: Dict[str, Any]) -> str:
    r = payload["results"]
    lines = [
        "# Memory Over Maps Phase 4 Report",
        "",
        f"*Generated: {payload['timestamp']}*",
        "",
        "## Track",
        "",
        f"- Phase: `{payload['phase']}`",
        f"- Track: `{payload['track']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Target |",
        "|---|---:|---:|",
        f"| invalidation_trigger_coverage_rate | {r['invalidation_trigger_coverage_rate']:.4f} | 1.0000 |",
        f"| stale_cache_survival_rate | {r['stale_cache_survival_rate']:.4f} | 0.0000 |",
        f"| dry_run_real_run_parity | {r['dry_run_real_run_parity']} | True |",
        f"| cache_hit_rate | {r['cache_hit_rate']:.4f} | n/a |",
        f"| false_invalidation_rate | {r['false_invalidation_rate']:.4f} | n/a |",
        f"| tested_events | {r['tested_events']} | n/a |",
        "",
        "## Gate",
        "",
        f"- Phase 4 gate result: {'PASS' if payload['pass'] else 'HOLD'}",
    ]
    return "\n".join(lines) + "\n"


def render_phase5_report(payload: Dict[str, Any]) -> str:
    r = payload["results"]
    lines = [
        "# Memory Over Maps Phase 5 Report",
        "",
        f"*Generated: {payload['timestamp']}*",
        "",
        "## Track",
        "",
        f"- Phase: `{payload['phase']}`",
        f"- Track: `{payload['track']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Target |",
        "|---|---:|---:|",
        f"| bounded_candidate_adherence_rate | {r['bounded_candidate_adherence_rate']:.4f} | 1.0000 |",
        f"| proper_noun_sensitivity_rate | {r['proper_noun_sensitivity_rate']:.4f} | >= 0.9000 |",
        f"| trust_recovery_delta | {r['trust_recovery_delta']:.4f} | >= 0.0500 |",
        f"| enforced_mode_drift_rate | {r['enforced_mode_drift_rate']:.4f} | <= 0.0000 |",
        f"| concurrent_reflect_success_rate | {r['concurrent_reflect_success_rate']:.4f} | >= 0.9500 |",
        f"| mean_reflect_latency_ms | {r['mean_reflect_latency_ms']:.4f} | n/a |",
        f"| scenario_count | {r['scenario_count']} | n/a |",
        "",
        "## Gate",
        "",
        f"- Phase 5 gate result: {'PASS' if payload['pass'] else 'HOLD'}",
    ]
    return "\n".join(lines) + "\n"


def render_phase1_decision(payload: Dict[str, Any]) -> str:
    decision = "pass" if payload["pass"] else "hold"
    rationale = (
        "Lineage completeness and derived-view input declaration meet Phase 1 acceptance targets."
        if payload["pass"]
        else "At least one Phase 1 acceptance target failed; do not begin Phase 2."
    )
    lines = [
        "# Memory Over Maps Phase 1 Decision",
        "",
        f"- Decision: **{decision.upper()}**",
        f"- Track: `{payload['track']}`",
        f"- Timestamp: `{payload['timestamp']}`",
        "",
        "## Rationale",
        "",
        f"- {rationale}",
        "",
        "## Promotion Recommendation",
        "",
        "- pass: proceed to Phase 2 planning implementation scope.",
        "- hold: remediate lineage/input contract gaps and re-run Phase 1 benchmark.",
    ]
    return "\n".join(lines) + "\n"


def render_phase2_decision(payload: Dict[str, Any]) -> str:
    decision = "pass" if payload["pass"] else "hold"
    rationale = (
        "Candidate narrowing shows useful compression, retained answer-supporting evidence, and deterministic behavior."
        if payload["pass"]
        else "Candidate-envelope acceptance targets failed; keep Phase 2 in hold and remediate."
    )
    lines = [
        "# Memory Over Maps Phase 2 Decision",
        "",
        f"- Decision: **{decision.upper()}**",
        f"- Track: `{payload['track']}`",
        f"- Timestamp: `{payload['timestamp']}`",
        "",
        "## Rationale",
        "",
        f"- {rationale}",
        "",
        "## Promotion Recommendation",
        "",
        "- pass: proceed to Phase 3 planning implementation scope.",
        "- hold: tune envelope policy and rerun Phase 2 benchmark.",
    ]
    return "\n".join(lines) + "\n"


def render_phase3_decision(payload: Dict[str, Any]) -> str:
    decision = "pass" if payload["pass"] else "hold"
    rationale = (
        "On-demand derived views are reproducible with complete input declarations."
        if payload["pass"]
        else "Derived view reproducibility/input completeness targets failed; keep Phase 3 in hold."
    )
    lines = [
        "# Memory Over Maps Phase 3 Decision",
        "",
        f"- Decision: **{decision.upper()}**",
        f"- Track: `{payload['track']}`",
        f"- Timestamp: `{payload['timestamp']}`",
        "",
        "## Rationale",
        "",
        f"- {rationale}",
        "",
        "## Promotion Recommendation",
        "",
        "- pass: proceed to Phase 4 planning implementation scope.",
        "- hold: remediate derived-view reproducibility gaps and rerun Phase 3 benchmark.",
    ]
    return "\n".join(lines) + "\n"


def render_phase4_decision(payload: Dict[str, Any]) -> str:
    decision = "pass" if payload["pass"] else "hold"
    rationale = (
        "Cache invalidation covers required triggers, preserves dry-run parity, and leaves no stale survivors."
        if payload["pass"]
        else "Cache invalidation gate failed; do not proceed until stale survivors/parity gaps are resolved."
    )
    lines = [
        "# Memory Over Maps Phase 4 Decision",
        "",
        f"- Decision: **{decision.upper()}**",
        f"- Track: `{payload['track']}`",
        f"- Timestamp: `{payload['timestamp']}`",
        "",
        "## Rationale",
        "",
        f"- {rationale}",
        "",
        "## Promotion Recommendation",
        "",
        "- pass: proceed to next-phase planning scope.",
        "- hold: fix invalidation correctness and rerun Phase 4 benchmark.",
    ]
    return "\n".join(lines) + "\n"


def render_phase5_decision(payload: Dict[str, Any]) -> str:
    decision = "pass" if payload["pass"] else "hold"
    rationale = (
        "Bounded semantic reflect scenarios show stable behavior with acceptable recovery and concurrency characteristics."
        if payload["pass"]
        else "Phase 5 semantic reflect gate failed; keep evolution work in hold and remediate target metrics."
    )
    lines = [
        "# Memory Over Maps Phase 5 Decision",
        "",
        f"- Decision: **{decision.upper()}**",
        f"- Track: `{payload['track']}`",
        f"- Timestamp: `{payload['timestamp']}`",
        "",
        "## Rationale",
        "",
        f"- {rationale}",
        "",
        "## Promotion Recommendation",
        "",
        "- pass: memory-over-maps phased lane can be marked implementation-complete.",
        "- hold: refine bounded semantic reflect behavior and rerun Phase 5 benchmark.",
    ]
    return "\n".join(lines) + "\n"


def _phase2_pass(result: Dict[str, Any]) -> bool:
    return (
        result["compression_ratio"] <= 0.75
        and result["answer_support_retention_rate"] >= 0.75
        and bool(result["deterministic_replay_match"])
    )


def _phase3_pass(result: Dict[str, Any]) -> bool:
    return (
        result["reproducibility_success_rate"] == 1.0
        and result["regeneration_mismatch_count"] == 0
        and result["input_completeness_rate"] == 1.0
    )


def _phase4_pass(result: Dict[str, Any]) -> bool:
    return (
        result["stale_cache_survival_rate"] == 0.0
        and bool(result["dry_run_real_run_parity"])
        and result["invalidation_trigger_coverage_rate"] >= 1.0
    )


def _phase5_pass(result: Dict[str, Any]) -> bool:
    return (
        result["bounded_candidate_adherence_rate"] == 1.0
        and result["proper_noun_sensitivity_rate"] >= 0.9
        and result["trust_recovery_delta"] >= 0.05
        and result["enforced_mode_drift_rate"] <= 0.0
        and result["concurrent_reflect_success_rate"] >= 0.95
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Memory Over Maps phase benchmarks")
    parser.add_argument("--phase", choices=["phase1", "phase2", "phase3", "phase4", "phase5"], default="phase1")
    parser.add_argument("--sample-size", type=int, default=12)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_dir = PROJECT_ROOT / "benchmarks" / "outputs" / "raw"
    summary_dir = PROJECT_ROOT / "benchmarks" / "outputs" / "summaries"

    if args.phase == "phase1":
        payload = run_phase1(sample_size=args.sample_size)
        report_text = render_phase1_report(payload)
        decision_text = render_phase1_decision(payload)
    elif args.phase == "phase2":
        payload = run_phase2()
        report_text = render_phase2_report(payload)
        decision_text = render_phase2_decision(payload)
    elif args.phase == "phase3":
        payload = run_phase3()
        report_text = render_phase3_report(payload)
        decision_text = render_phase3_decision(payload)
    elif args.phase == "phase4":
        payload = run_phase4()
        report_text = render_phase4_report(payload)
        decision_text = render_phase4_decision(payload)
    else:
        payload = run_phase5()
        report_text = render_phase5_report(payload)
        decision_text = render_phase5_decision(payload)

    raw_path = raw_dir / f"memory_over_maps_{timestamp}_raw.json"
    report_path = summary_dir / f"memory_over_maps_{timestamp}_report.md"
    decision_path = summary_dir / f"memory_over_maps_{timestamp}_decision.md"

    _write_json(raw_path, payload)
    _write_text(report_path, report_text)
    _write_text(decision_path, decision_text)

    print("\nMemory Over Maps benchmarks complete")
    print(f"  Raw:      {raw_path}")
    print(f"  Report:   {report_path}")
    print(f"  Decision: {decision_path}")


if __name__ == "__main__":
    main()
