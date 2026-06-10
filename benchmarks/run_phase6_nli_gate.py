"""
Phase 6 gate: NLI Reflect Precision (W3 Reflect Precision + W5 Explainability).

Benchmarks the entailment-grounded NLIUsageDetector against the lexical
UsageDetector baseline on the labeled reflect truthset, and renders a
PASS/HOLD promotion decision in the established phase-gate artifact format.

Gate criteria
-------------
1. used_precision_uplift   — relative USED-precision improvement over the
                             lexical baseline >= 0.25
2. no_recall_regression    — NLI USED recall >= lexical USED recall
                             (roadmap mitigation: track precision and recall jointly)
3. adversarial_accuracy    — 1.0 on the gold-IGNORED judgments of the
                             short_token_generic + proper_noun_coincidence
                             categories (the documented precision boundaries —
                             generic/coincidental memories must not be labeled
                             used; gold-USED controls in those scenarios are
                             measured by recall instead)
4. latency_p95_ms_per_10   — p95 reflect-cycle overhead, normalised to a
                             10-candidate cycle, < 150 ms. Enforced only on
                             CUDA (the target hardware); reported informationally
                             on CPU.

Produces:
  - benchmarks/outputs/raw/phase6_nli_reflect_<timestamp>_raw.json
  - benchmarks/outputs/summaries/phase6_nli_reflect_<timestamp>_report.md
  - benchmarks/outputs/summaries/phase6_nli_reflect_<timestamp>_decision.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.usage_detector import UsageDetector, UsageLabel

DEFAULT_TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "reflect_usage_truthset_v1.json"
RAW_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "summaries"

ADVERSARIAL_CATEGORIES = {"short_token_generic", "proper_noun_coincidence"}

CRITERIA = {
    "used_precision_uplift_min": 0.25,
    "no_recall_regression": True,
    "adversarial_accuracy_min": 1.0,
    "latency_p95_ms_per_10_candidates_max": 150.0,
}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_cycle(scenario: Dict[str, Any]):
    """Construct SearchResult/GovernanceDecision pairs for one reflect cycle."""
    results: List[SearchResult] = []
    decisions: List[GovernanceDecision] = []
    for memory in scenario["memories"]:
        engram = Engram(id=memory["id"], content=memory["content"], source="truthset://reflect_usage_v1")
        results.append(SearchResult(engram=engram, score=0.9, tier="truthset"))
        decisions.append(GovernanceDecision(engram_id=memory["id"], retrieval_score=0.9, governed_score=0.9))
    return results, decisions


def _used_metrics(predictions: Dict[str, str], gold: Dict[str, str]) -> Dict[str, float]:
    tp = sum(1 for eid, label in predictions.items() if label == "used" and gold[eid] == "used")
    fp = sum(1 for eid, label in predictions.items() if label == "used" and gold[eid] != "used")
    fn = sum(1 for eid, label in predictions.items() if label != "used" and gold[eid] == "used")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "true_positive": tp, "false_positive": fp, "false_negative": fn}


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def run_detector(detector, truthset: Dict[str, Any], collect_latency: bool = False):
    """Run one detector over every scenario; returns predictions, per-category accuracy, latencies."""
    predictions: Dict[str, str] = {}
    category_hits: Dict[str, List[bool]] = {}
    latencies_per_10: List[float] = []

    for scenario in truthset["scenarios"]:
        results, decisions = _build_cycle(scenario)
        started = time.perf_counter()
        labels = detector.detect(
            query=scenario["query"],
            answer=scenario["answer"],
            results=results,
            decisions=decisions,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if collect_latency and results:
            latencies_per_10.append(elapsed_ms * 10.0 / len(results))

        for memory in scenario["memories"]:
            eid = memory["id"]
            predicted = labels.get(eid, UsageLabel.UNKNOWN).value
            predictions[eid] = predicted
            category_hits.setdefault(scenario["category"], []).append(predicted == memory["gold_label"])

    category_accuracy = {
        category: round(sum(hits) / len(hits), 4) for category, hits in sorted(category_hits.items())
    }
    return predictions, category_accuracy, latencies_per_10


def run_gate(truthset_path: Path, model_name: str, threshold: float, device: str | None) -> Dict[str, Any]:
    truthset = json.loads(truthset_path.read_text(encoding="utf-8"))
    gold = {m["id"]: m["gold_label"] for s in truthset["scenarios"] for m in s["memories"]}
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    lexical = UsageDetector()
    lexical_predictions, lexical_categories, _ = run_detector(lexical, truthset)
    lexical_metrics = _used_metrics(lexical_predictions, gold)

    payload: Dict[str, Any] = {
        "gate": "phase6_nli_reflect_precision",
        "timestamp": timestamp,
        "truthset": str(truthset_path.relative_to(PROJECT_ROOT)),
        "truthset_version": truthset.get("version"),
        "scenario_count": len(truthset["scenarios"]),
        "judgment_count": len(gold),
        "criteria": CRITERIA,
        "lexical_baseline": {"metrics": lexical_metrics, "category_accuracy": lexical_categories},
    }

    try:
        from mnemos.governance.nli_usage_detector import NLIUsageDetector

        nli = NLIUsageDetector(model_name=model_name, entailment_threshold=threshold, device=device)
        nli._ensure_model()  # load before timing so warmup is excluded
        resolved_device = nli.resolved_device
        # warmup inference pass
        run_detector(nli, {"scenarios": truthset["scenarios"][:1]})
        nli_predictions, nli_categories, latencies = run_detector(nli, truthset, collect_latency=True)
    except Exception as exc:
        payload["pass"] = False
        payload["error"] = f"NLI detector unavailable: {exc}"
        payload["results"] = {}
        return payload

    nli_metrics = _used_metrics(nli_predictions, gold)
    lex_precision = lexical_metrics["precision"]
    uplift = (nli_metrics["precision"] - lex_precision) / lex_precision if lex_precision > 0 else float("inf")

    category_of = {m["id"]: s["category"] for s in truthset["scenarios"] for m in s["memories"]}
    adversarial_ids = [
        eid for eid, label in gold.items()
        if label == "ignored" and category_of[eid] in ADVERSARIAL_CATEGORIES
    ]
    adversarial_hits = [nli_predictions[eid] == "ignored" for eid in adversarial_ids]
    adversarial_accuracy = round(sum(adversarial_hits) / len(adversarial_hits), 4) if adversarial_hits else 0.0
    latency_p95 = round(_percentile(latencies, 95.0), 2)
    latency_enforced = resolved_device == "cuda"

    checks = {
        "used_precision_uplift": round(uplift, 4) >= CRITERIA["used_precision_uplift_min"],
        "no_recall_regression": nli_metrics["recall"] >= lexical_metrics["recall"],
        "adversarial_accuracy": adversarial_accuracy >= CRITERIA["adversarial_accuracy_min"],
        "latency_p95": (latency_p95 < CRITERIA["latency_p95_ms_per_10_candidates_max"]) if latency_enforced else True,
    }

    payload["results"] = {
        "model": model_name,
        "entailment_threshold": threshold,
        "device": resolved_device,
        "nli_metrics": nli_metrics,
        "nli_category_accuracy": nli_categories,
        "used_precision_uplift": round(uplift, 4),
        "adversarial_accuracy": adversarial_accuracy,
        "latency_p95_ms_per_10_candidates": latency_p95,
        "latency_criterion_enforced": latency_enforced,
        "checks": checks,
        "disagreements": [
            {"engram_id": eid, "gold": gold[eid], "lexical": lexical_predictions[eid], "nli": nli_predictions[eid]}
            for eid in sorted(gold)
            if lexical_predictions[eid] != nli_predictions[eid]
        ],
    }
    payload["pass"] = all(checks.values())
    return payload


def render_report(payload: Dict[str, Any]) -> str:
    lines = [
        "# Phase 6 — NLI Reflect Precision Gate",
        "",
        f"- Timestamp: {payload['timestamp']}",
        f"- Truthset: `{payload['truthset']}` ({payload['scenario_count']} scenarios, {payload['judgment_count']} judgments)",
        f"- Gate result: {'PASS' if payload['pass'] else 'HOLD'}",
        "",
        "## Lexical baseline (March 30 detector, current defaults)",
        "",
        f"```json\n{json.dumps(payload['lexical_baseline'], indent=2)}\n```",
        "",
        "## NLI candidate",
        "",
        f"```json\n{json.dumps(payload.get('results', {}), indent=2)}\n```",
    ]
    if payload.get("error"):
        lines += ["", f"## Error", "", payload["error"]]
    return "\n".join(lines) + "\n"


def render_decision(payload: Dict[str, Any]) -> str:
    decision = "pass" if payload["pass"] else "hold"
    guidance = (
        "NLI reflect precision gate passed; NLIUsageDetector is eligible to wire into the reflect "
        "path behind `reflect_precision_mode`, and enforced-mode promotion review may proceed."
        if payload["pass"]
        else "Phase 6 NLI gate failed; keep the lexical detector as the reflect default and remediate "
        "the failing criteria before promotion review."
    )
    results = payload.get("results", {})
    lines = [
        "# Phase 6 Decision — NLI Reflect Precision",
        "",
        f"- Timestamp: {payload['timestamp']}",
        f"- Decision: **{decision.upper()}**",
        f"- Criteria: {json.dumps(payload['criteria'])}",
        f"- Checks: {json.dumps(results.get('checks', {}))}",
        "",
        guidance,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 6 NLI reflect precision gate")
    parser.add_argument("--truthset", type=Path, default=DEFAULT_TRUTHSET)
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-xsmall")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="Torch device override (cuda/cpu)")
    args = parser.parse_args()

    payload = run_gate(args.truthset, args.model, args.threshold, args.device)
    timestamp = payload["timestamp"]

    raw_path = RAW_DIR / f"phase6_nli_reflect_{timestamp}_raw.json"
    report_path = SUMMARY_DIR / f"phase6_nli_reflect_{timestamp}_report.md"
    decision_path = SUMMARY_DIR / f"phase6_nli_reflect_{timestamp}_decision.md"

    _write_json(raw_path, payload)
    _write_text(report_path, render_report(payload))
    _write_text(decision_path, render_decision(payload))

    print(f"Phase 6 gate: {'PASS' if payload['pass'] else 'HOLD'}")
    print(f"  Raw:      {raw_path}")
    print(f"  Report:   {report_path}")
    print(f"  Decision: {decision_path}")
    sys.exit(0 if payload["pass"] else 1)


if __name__ == "__main__":
    main()
