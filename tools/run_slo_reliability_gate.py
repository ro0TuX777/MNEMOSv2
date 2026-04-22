#!/usr/bin/env python3
"""Run SLO reliability gate checks for promotion stages.

Usage:
  python tools/run_slo_reliability_gate.py
  python tools/run_slo_reliability_gate.py --stage canary_25 --fail-on-breach
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.runners.memory_over_maps_runner import (
    run_phase4_cache_invalidation_track,
    run_phase5_reflect_bounded_track,
)
from mnemos.engram.model import Engram
from mnemos.governance.governor import Governor
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.retrieval.base import BaseRetriever, SearchResult
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.retrieval_router import RetrievalRouter


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "outputs"

STAGE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "canary_5": {
        "max_p95_latency_ms": 50.0,
        "max_stale_cache_survival_rate": 0.0005,
        "max_suppression_drift_rate": 0.005,
        "min_contradiction_correctness_rate": 0.995,
    },
    "canary_25": {
        "max_p95_latency_ms": 45.0,
        "max_stale_cache_survival_rate": 0.0005,
        "max_suppression_drift_rate": 0.005,
        "min_contradiction_correctness_rate": 0.995,
    },
    "canary_50": {
        "max_p95_latency_ms": 40.0,
        "max_stale_cache_survival_rate": 0.0005,
        "max_suppression_drift_rate": 0.003,
        "min_contradiction_correctness_rate": 0.995,
    },
    "full": {
        "max_p95_latency_ms": 35.0,
        "max_stale_cache_survival_rate": 0.0005,
        "max_suppression_drift_rate": 0.002,
        "min_contradiction_correctness_rate": 0.995,
    },
}


class _DummyRetriever(BaseRetriever):
    def __init__(self, name: str, ids: list[str]):
        self._name = name
        self._ids = ids

    @property
    def tier_name(self) -> str:
        return self._name

    def index(self, engrams):
        return len(engrams)

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None):
        out = []
        for i, doc_id in enumerate(self._ids[:top_k]):
            out.append(
                SearchResult(
                    engram=Engram(id=doc_id, content=f"{self._name}-{doc_id}"),
                    score=float(top_k - i),
                    tier=self._name,
                )
            )
        return out

    def delete(self, engram_ids):
        return len(engram_ids)

    def stats(self):
        return {"tier": self._name, "document_count": len(self._ids)}


@dataclass
class SloReliabilityResult:
    hybrid_latency_p95_ms: float
    stale_cache_survival_rate: float
    suppression_drift_rate: float
    contradiction_correctness_rate: float
    contradictions_checked: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_latency_p95_ms() -> float:
    semantic = _DummyRetriever("qdrant", ["a", "b", "c", "d"])
    lexical = _DummyRetriever("lexical", ["a", "x", "y", "z"])
    router = RetrievalRouter(semantic_fusion=TierFusion([semantic]), lexical_tier=lexical)

    for idx in range(40):
        router.search(
            query=f"slo query {idx}",
            top_k=4,
            retrieval_mode="hybrid",
            fusion_policy="balanced",
            explain=False,
            lexical_top_k=4,
            semantic_top_k=4,
        )
    return float(router.stats().get("hybrid_latency_p95_ms", 0.0))


def _compute_contradiction_correctness_rate() -> tuple[float, int]:
    governor = Governor()
    scenarios = [
        (("a_old", 0.60, "2026-01-01T00:00:00Z"), ("a_new", 0.30, "2026-02-01T00:00:00Z")),
        (("b_new", 0.50, "2026-03-01T00:00:00Z"), ("b_old", 0.50, "2026-01-01T00:00:00Z")),
        (("c_alpha", 0.40, "2026-01-01T00:00:00Z"), ("c_beta", 0.40, "2026-01-01T00:00:00Z")),
    ]

    correct = 0
    for idx, row in enumerate(scenarios):
        entity = f"entity-{idx}"
        attribute = "status"
        winner_id, winner_trust, winner_created = row[0]
        loser_id, loser_trust, loser_created = row[1]

        winner = Engram(id=winner_id, content=f"{entity} status active")
        winner.governance = GovernanceMeta()
        winner.governance.entity_key = entity
        winner.governance.attribute_key = attribute
        winner.governance.normalized_value = "active"
        winner.governance.trust_score = winner_trust
        winner.governance.created_at = winner_created
        winner.governance.utility_score = 0.7
        winner.governance.source_authority = 0.7

        loser = Engram(id=loser_id, content=f"{entity} status inactive")
        loser.governance = GovernanceMeta()
        loser.governance.entity_key = entity
        loser.governance.attribute_key = attribute
        loser.governance.normalized_value = "inactive"
        loser.governance.trust_score = loser_trust
        loser.governance.created_at = loser_created
        loser.governance.utility_score = 0.7
        loser.governance.source_authority = 0.7

        results = [
            SearchResult(engram=winner, score=0.9, tier="hybrid"),
            SearchResult(engram=loser, score=0.8, tier="hybrid"),
        ]
        _, decisions, records = governor.govern(
            results=results,
            query=f"what is {entity} status",
            governance_mode="enforced",
            top_k=2,
        )
        if not records:
            continue
        record = records[0]
        decided_winner = record.winner_memory_id
        suppressed_ids = {d.engram_id for d in decisions if d.suppressed_by_contradiction}
        record_losers = set(record.loser_memory_ids)
        if (
            decided_winner in {winner_id, loser_id}
            and len(record_losers) == 1
            and len(suppressed_ids) == 1
            and suppressed_ids == record_losers
        ):
            correct += 1

    total = len(scenarios)
    return ((correct / total) if total else 0.0, total)


def run_slo_reliability_track() -> SloReliabilityResult:
    phase4 = run_phase4_cache_invalidation_track().to_dict()
    phase5 = run_phase5_reflect_bounded_track().to_dict()
    contradiction_rate, contradiction_total = _compute_contradiction_correctness_rate()
    return SloReliabilityResult(
        hybrid_latency_p95_ms=round(_compute_latency_p95_ms(), 4),
        stale_cache_survival_rate=float(phase4["stale_cache_survival_rate"]),
        suppression_drift_rate=float(phase5["enforced_mode_drift_rate"]),
        contradiction_correctness_rate=round(contradiction_rate, 4),
        contradictions_checked=contradiction_total,
    )


def _evaluate(result: SloReliabilityResult, thresholds: Dict[str, float]) -> Dict[str, Any]:
    checks = {
        "latency_p95_within_budget": result.hybrid_latency_p95_ms <= thresholds["max_p95_latency_ms"],
        "stale_cache_survival_within_budget": result.stale_cache_survival_rate <= thresholds["max_stale_cache_survival_rate"],
        "suppression_drift_within_budget": result.suppression_drift_rate <= thresholds["max_suppression_drift_rate"],
        "contradiction_correctness_within_budget": result.contradiction_correctness_rate >= thresholds["min_contradiction_correctness_rate"],
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
    }


def _to_markdown(*, stage: str, result: SloReliabilityResult, thresholds: Dict[str, float], evaluation: Dict[str, Any]) -> str:
    lines = [
        "# MNEMOS SLO Reliability Gate Report",
        "",
        f"- Stage: `{stage}`",
        f"- Generated at: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Target |",
        "|---|---:|---:|",
        f"| hybrid_latency_p95_ms | {result.hybrid_latency_p95_ms:.4f} | <= {thresholds['max_p95_latency_ms']:.4f} |",
        f"| stale_cache_survival_rate | {result.stale_cache_survival_rate:.4f} | <= {thresholds['max_stale_cache_survival_rate']:.4f} |",
        f"| suppression_drift_rate | {result.suppression_drift_rate:.4f} | <= {thresholds['max_suppression_drift_rate']:.4f} |",
        f"| contradiction_correctness_rate | {result.contradiction_correctness_rate:.4f} | >= {thresholds['min_contradiction_correctness_rate']:.4f} |",
        "",
        "## Gate Checks",
        "",
    ]
    for key, ok in evaluation["checks"].items():
        lines.append(f"- `{key}`: `{ok}`")
    lines.extend(
        [
            "",
            f"- Gate result: `{'PASS' if evaluation['pass'] else 'FAIL'}`",
            "",
            "## Rollback Discipline",
            "",
            "- On gate failure, stop promotion and rollback to previous stable canary stage.",
            "- Investigate failing metric dimension before re-attempting promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SLO reliability gate checks.")
    parser.add_argument("--stage", choices=sorted(STAGE_THRESHOLDS.keys()), default="canary_25")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-p95-ms", type=float, default=None, help="Override max p95 latency threshold")
    parser.add_argument("--fail-on-breach", action="store_true", help="Exit 1 when gate fails")
    args = parser.parse_args()

    thresholds = dict(STAGE_THRESHOLDS[args.stage])
    if args.max_p95_ms is not None:
        thresholds["max_p95_latency_ms"] = float(args.max_p95_ms)

    result = run_slo_reliability_track()
    evaluation = _evaluate(result, thresholds)

    out_root = Path(args.output_dir)
    raw_dir = out_root / "raw"
    summary_dir = out_root / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = raw_dir / f"slo_reliability_{ts}_raw.json"
    report_path = summary_dir / f"slo_reliability_{ts}_report.md"

    payload = {
        "stage": args.stage,
        "thresholds": thresholds,
        "results": result.to_dict(),
        "evaluation": evaluation,
        "pass": evaluation["pass"],
    }
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(
        _to_markdown(stage=args.stage, result=result, thresholds=thresholds, evaluation=evaluation),
        encoding="utf-8",
    )

    print("[OK] SLO reliability gate artifacts generated")
    print(f"  Raw:    {raw_path}")
    print(f"  Report: {report_path}")
    print(f"  Gate pass: {evaluation['pass']}")

    if args.fail_on_breach and not evaluation["pass"]:
        print("[FAIL] SLO reliability gate breached. Rollback to previous stable stage.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
