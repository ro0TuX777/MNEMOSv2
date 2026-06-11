"""
Phase 8 query-complexity gate runner.

Evaluates the zero-shot NLI complexity classifier against
benchmarks/truthsets/query_complexity_v1.json and emits auditable artifacts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.retrieval.complexity import ZeroShotComplexityClassifier

DEFAULT_TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "query_complexity_v1.json"
RAW_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "summaries"


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def evaluate(
    *,
    truthset_path: Path,
    model_name: str,
    device: str | None,
) -> Dict[str, Any]:
    truthset = json.loads(truthset_path.read_text(encoding="utf-8"))
    classifier = ZeroShotComplexityClassifier(model_name=model_name, device=device)

    rows: List[Dict[str, Any]] = []
    correct_by_class: Counter[str] = Counter()
    total_by_class: Counter[str] = Counter()

    for item in truthset["queries"]:
        expected = item["label"]
        result = classifier.classify(item["query"])
        ok = result.label == expected
        total_by_class[expected] += 1
        if ok:
            correct_by_class[expected] += 1
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "expected": expected,
                "predicted": result.label,
                "correct": ok,
                "confidence": round(result.confidence, 4),
                "scores": {k: round(v, 4) for k, v in result.scores.items()},
                "route_posture": result.route_posture,
                "latency_ms": round(result.latency_ms, 4),
                "source_family": item.get("source_family"),
                "rationale": item.get("rationale"),
            }
        )

    latencies = [float(r["latency_ms"]) for r in rows]
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    per_class = {
        label: {
            "correct": correct_by_class[label],
            "total": total_by_class[label],
            "accuracy": round(correct_by_class[label] / total_by_class[label], 4)
            if total_by_class[label]
            else 0.0,
        }
        for label in sorted(total_by_class)
    }
    metrics = {
        "query_count": total,
        "overall_accuracy": round(correct / total, 4) if total else 0.0,
        "per_class": per_class,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 4) if latencies else 0.0,
            "median": round(statistics.median(latencies), 4) if latencies else 0.0,
            "p95": round(percentile(latencies, 95), 4),
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "gates": {
            "accuracy": {
                "threshold": "> 0.85",
                "pass": (correct / total) > 0.85 if total else False,
            },
            "p95_latency_ms": {
                "threshold": "< 25",
                "pass": percentile(latencies, 95) < 25 if latencies else False,
            },
        },
    }
    metrics["overall_gate_pass"] = all(g["pass"] for g in metrics["gates"].values())
    return {
        "truthset": str(truthset_path.relative_to(PROJECT_ROOT)),
        "model_name": model_name,
        "rows": rows,
        "metrics": metrics,
    }


def write_artifacts(result: Dict[str, Any]) -> tuple[Path, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"phase8_complexity_gate_{timestamp}_raw.json"
    summary_path = SUMMARY_DIR / f"phase8_complexity_gate_{timestamp}_summary.md"

    raw_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics = result["metrics"]
    class_lines = "\n".join(
        f"- {label}: {row['correct']}/{row['total']} ({row['accuracy']:.4f})"
        for label, row in metrics["per_class"].items()
    )
    gate_lines = "\n".join(
        f"- {name}: {'PASS' if gate['pass'] else 'FAIL'} - {gate['threshold']}"
        for name, gate in metrics["gates"].items()
    )
    summary_path.write_text(
        "# Phase 8 Complexity Gate Summary\n\n"
        f"- Model: `{result['model_name']}`\n"
        f"- Truthset: `{result['truthset']}`\n"
        f"- Overall accuracy: `{metrics['overall_accuracy']:.4f}`\n"
        f"- P95 latency: `{metrics['latency_ms']['p95']:.4f}ms`\n"
        f"- Gate: **{'PASS' if metrics['overall_gate_pass'] else 'FAIL'}**\n\n"
        "## Per-Class Accuracy\n\n"
        f"{class_lines}\n\n"
        "## Gates\n\n"
        f"{gate_lines}\n\n"
        f"- Raw: `{raw_path.name}`\n",
        encoding="utf-8",
    )
    return raw_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 8 query complexity gate")
    parser.add_argument("--truthset", type=Path, default=DEFAULT_TRUTHSET)
    parser.add_argument("--model-name", default="cross-encoder/nli-deberta-v3-xsmall")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args()

    result = evaluate(
        truthset_path=args.truthset,
        model_name=args.model_name,
        device=args.device,
    )
    metrics = result["metrics"]
    print(f"queries: {metrics['query_count']}")
    print(f"overall accuracy: {metrics['overall_accuracy']:.4f}")
    for label, row in metrics["per_class"].items():
        print(f"{label}: {row['correct']}/{row['total']} ({row['accuracy']:.4f})")
    print(f"p95 latency: {metrics['latency_ms']['p95']:.4f}ms")
    print(f"gate: {'PASS' if metrics['overall_gate_pass'] else 'FAIL'}")

    if not args.no_artifacts:
        raw, summary = write_artifacts(result)
        print(f"raw: {raw}")
        print(f"summary: {summary}")
    return 0 if metrics["overall_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
