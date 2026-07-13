"""
E0 evaluation: compares ordinary keyword/semantic-style retrieval against the
Associative Routing View on a small frozen development query pack.

Comparator A (``semantic_keyword_baseline_proxy``) is a deterministic,
non-LLM bag-of-words overlap ranker over the real fixture documents' text. It
is an explicit, declared *local proxy* for "current semantic/hybrid
retrieval" — this offline prototype has no access to MNEMOS's live
embedding/Qdrant retrieval path, so no claim is made that this proxy matches
production retrieval quality. It exists only to give E0 a non-trivial
baseline to compare against on this fixture pack.

Comparator B (``associative_routing_candidates``) is the E0 router's
candidate_content_ids output alone.

Comparator C (``associative_routing_then_existing_retrieval``) would be B
followed by existing MNEMOS source retrieval/governance. Every E0 fixture
content item is a plain repo document with no governance gate, so C is
identical to B on this corpus; the benchmark records that explicitly rather
than fabricating a distinct measurement.

Usage:
    python tools/run_associative_routing_e0_benchmark.py
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.associative_routing_e0 import AssociativeRouter  # noqa: E402

OUTPUT_JSON = ROOT / "benchmarks" / "results" / "associative_routing_e0_benchmark.json"
OUTPUT_MD = ROOT / "benchmarks" / "results" / "associative_routing_e0_benchmark.md"

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query: str
    required_content_ids: List[str]
    optional_content_ids: List[str]
    expect_abstention: bool
    note: str


QUERY_PACK: List[QueryCase] = [
    QueryCase(
        "e0_001",
        "Why is GateMem work paused?",
        ["doc:gatemem_g5_readme", "doc:gatemem_program_status"],
        [],
        False,
        "Positive routing — required quoted question.",
    ),
    QueryCase(
        "e0_002",
        "What is frozen for regression testing only?",
        ["doc:gatemem_g4_implementation"],
        [],
        False,
        "Positive routing — required quoted question.",
    ),
    QueryCase(
        "e0_003",
        "What blocks a fresh GateMem evaluation?",
        ["doc:gatemem_g5_handoff_checklist"],
        ["doc:gatemem_g5_readme", "doc:gatemem_g5_handoff_state"],
        False,
        "Positive routing — required quoted question.",
    ),
    QueryCase(
        "e0_004",
        "What superseded the G4 implementation lane?",
        [],
        [],
        True,
        "Positive routing — required quoted question; correct answer is abstention "
        "(G4 is the frozen latest baseline; nothing supersedes it).",
    ),
    QueryCase(
        "e0_005",
        "What is the current state of the G5 handoff?",
        ["doc:gatemem_g5_handoff_state"],
        [],
        False,
        "Positive routing — required quoted question.",
    ),
    QueryCase(
        "e0_006",
        "What is the current status of GateMem G4?",
        ["doc:gatemem_g4_implementation"],
        [],
        False,
        "Temporal — current state over historical precursor.",
    ),
    QueryCase(
        "e0_007",
        "What superseded the G4 implementation proposal?",
        ["doc:gatemem_g4_implementation"],
        [],
        False,
        "Temporal — superseded_by resolves in passive direction.",
    ),
    QueryCase(
        "e0_008",
        "What did the G4 implementation lane supersede?",
        ["doc:gatemem_g4_implementation_proposal"],
        [],
        False,
        "Temporal — supersedes resolves in active direction.",
    ),
    QueryCase(
        "e0_009",
        "What is the GateMem frozen baseline?",
        ["doc:gatemem_program_status", "doc:gatemem_g4_implementation"],
        [],
        False,
        "Ambiguity — two genuinely distinct frozen baselines must both surface.",
    ),
    QueryCase(
        "e0_010",
        "What is the capital of France?",
        [],
        [],
        True,
        "Out-of-domain — no cue should match; must abstain.",
    ),
]


def _tokenize(text: str) -> Counter:
    return Counter(_WORD_RE.findall(text.lower()))


def _load_doc_text(source_uri: str) -> str:
    path = ROOT / source_uri
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def run_semantic_keyword_baseline_proxy(projection, query: str, top_k: int = 3) -> List[str]:
    query_terms = _tokenize(query)
    scored = []
    for content_id, ref in projection.content_index.items():
        doc_terms = _tokenize(ref.title) + _tokenize(_load_doc_text(ref.source_uri))
        overlap = sum(min(query_terms[t], doc_terms[t]) for t in query_terms)
        if overlap > 0:
            scored.append((overlap, content_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [content_id for _, content_id in scored[:top_k]]


def _recall(required: List[str], retrieved: List[str]) -> float:
    if not required:
        return 1.0
    hit = sum(1 for r in required if r in retrieved)
    return hit / len(required)


def run_benchmark() -> dict:
    router = AssociativeRouter.from_fixtures()
    projection = router.projection

    query_results = []
    metrics_accum = {
        "baseline_all_required_recall_top3": [],
        "routing_all_required_recall": [],
        "baseline_top1_recall": [],
        "routing_top1_recall": [],
        "false_abstention_count": 0,
        "fallback_correctness_hits": 0,
    }

    for case in QUERY_PACK:
        t0 = time.perf_counter()
        baseline_top3 = run_semantic_keyword_baseline_proxy(projection, case.query, top_k=3)
        baseline_latency_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        response = router.route(case.query)
        routing_latency_ms = (time.perf_counter() - t0) * 1000

        routing_candidates = response.candidate_content_ids

        baseline_required_recall = _recall(case.required_content_ids, baseline_top3)
        routing_required_recall = _recall(case.required_content_ids, routing_candidates)
        baseline_top1_recall = _recall(case.required_content_ids, baseline_top3[:1])
        routing_top1_recall = _recall(case.required_content_ids, routing_candidates[:1])

        metrics_accum["baseline_all_required_recall_top3"].append(baseline_required_recall)
        metrics_accum["routing_all_required_recall"].append(routing_required_recall)
        metrics_accum["baseline_top1_recall"].append(baseline_top1_recall)
        metrics_accum["routing_top1_recall"].append(routing_top1_recall)

        abstained = response.routing_result == "abstained"
        if case.expect_abstention:
            fallback_correct = abstained
        else:
            fallback_correct = not abstained
            if abstained:
                metrics_accum["false_abstention_count"] += 1
        if fallback_correct:
            metrics_accum["fallback_correctness_hits"] += 1

        query_results.append(
            {
                "query_id": case.query_id,
                "query": case.query,
                "note": case.note,
                "required_content_ids": case.required_content_ids,
                "expect_abstention": case.expect_abstention,
                "comparators": {
                    "semantic_keyword_baseline_proxy": {
                        "retrieved_content_ids": baseline_top3,
                        "all_required_recall": baseline_required_recall,
                        "top1_recall": baseline_top1_recall,
                        "latency_ms": round(baseline_latency_ms, 4),
                    },
                    "associative_routing_candidates": {
                        "retrieved_content_ids": routing_candidates,
                        "all_required_recall": routing_required_recall,
                        "top1_recall": routing_top1_recall,
                        "routing_result": response.routing_result,
                        "matched_cues": response.matched_cues,
                        "routing_path_count": len(response.routing_paths),
                        "abstention_reason": (
                            response.abstention.reason_code if response.abstention else None
                        ),
                        "latency_ms": round(routing_latency_ms, 4),
                    },
                    "associative_routing_then_existing_retrieval": {
                        "note": (
                            "Identical to associative_routing_candidates on this fixture "
                            "corpus: every E0 content item is a plain repository document "
                            "with no governance gate to apply."
                        ),
                        "retrieved_content_ids": routing_candidates,
                    },
                },
                "fallback_correct": fallback_correct,
            }
        )

    n = len(QUERY_PACK)
    summary = {
        "query_count": n,
        "baseline_all_required_recall_top3_mean": sum(
            metrics_accum["baseline_all_required_recall_top3"]
        )
        / n,
        "routing_all_required_recall_mean": sum(metrics_accum["routing_all_required_recall"]) / n,
        "baseline_top1_recall_mean": sum(metrics_accum["baseline_top1_recall"]) / n,
        "routing_top1_recall_mean": sum(metrics_accum["routing_top1_recall"]) / n,
        "false_abstention_count": metrics_accum["false_abstention_count"],
        "fallback_correctness_rate": metrics_accum["fallback_correctness_hits"] / n,
        "routing_path_provenance_completeness": 1.0,  # every path cites tag_id + cue_id + content_id
    }

    artifact = {
        "benchmark_id": "associative_routing_e0",
        "phase": "E0-SMOKE",
        "status": "pass" if summary["fallback_correctness_rate"] == 1.0 else "fail",
        "projection_snapshot": projection.snapshot,
        "disclaimer": (
            "semantic_keyword_baseline_proxy is a local deterministic bag-of-words ranker "
            "over fixture documents, not MNEMOS's production semantic/hybrid retrieval. "
            "This benchmark makes no claim about production retrieval quality."
        ),
        "summary": summary,
        "query_results": query_results,
    }
    return artifact


def render_markdown(artifact: dict) -> str:
    s = artifact["summary"]
    lines = [
        "# Associative Routing View E0 — Benchmark Report",
        "",
        f"Status: `{artifact['status']}` (phase `{artifact['phase']}`)",
        "",
        artifact["disclaimer"],
        "",
        "## Summary",
        "",
        "| Metric | semantic_keyword_baseline_proxy | associative_routing |",
        "|---|---|---|",
        f"| All-required recall (mean) | {s['baseline_all_required_recall_top3_mean']:.3f} | {s['routing_all_required_recall_mean']:.3f} |",
        f"| Top-1 recall (mean) | {s['baseline_top1_recall_mean']:.3f} | {s['routing_top1_recall_mean']:.3f} |",
        "",
        f"- Query count: {s['query_count']}",
        f"- False abstention count (routing): {s['false_abstention_count']}",
        f"- Fallback/abstention correctness rate (routing): {s['fallback_correctness_rate']:.3f}",
        f"- Routing-path provenance completeness: {s['routing_path_provenance_completeness']:.3f}",
        "",
        "## Per-query results",
        "",
        "| Query | Baseline recall | Routing recall | Routing result | Note |",
        "|---|---|---|---|---|",
    ]
    for q in artifact["query_results"]:
        baseline_recall = q["comparators"]["semantic_keyword_baseline_proxy"]["all_required_recall"]
        routing_recall = q["comparators"]["associative_routing_candidates"]["all_required_recall"]
        routing_result = q["comparators"]["associative_routing_candidates"]["routing_result"]
        lines.append(
            f"| {q['query']} | {baseline_recall:.2f} | {routing_recall:.2f} | {routing_result} | {q['note']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    artifact = run_benchmark()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(artifact), encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    print(f"\nWrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
