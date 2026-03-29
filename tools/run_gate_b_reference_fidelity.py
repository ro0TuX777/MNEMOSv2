"""
Run Gate B reference-fidelity comparison on a fixed sanity set.

Outputs:
  - benchmarks/outputs/raw/<timestamp>_gate_b_reference_fidelity.json
  - benchmarks/outputs/summaries/<timestamp>_gate_b_reference_fidelity.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.engram.model import Engram
from mnemos.retrieval.colbert_tier import ColBERTConfig, ColBERTTier


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_corpus(path: Path) -> Dict[str, Engram]:
    data = _load_json(path)
    out: Dict[str, Engram] = {}
    for row in data:
        e = Engram.from_dict(row)
        out[e.id] = e
    return out


def _tokenize_for_overlap(text: str) -> List[str]:
    tokens = []
    cur = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def _overlap_score(query_text: str, doc_text: str) -> float:
    q = set(_tokenize_for_overlap(query_text))
    if not q:
        return 0.0
    d = set(_tokenize_for_overlap(doc_text))
    return len(q & d) / len(q)


def _maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    sim = query_emb @ doc_emb.T
    return float(sim.max(axis=1).mean())


def _mean_cos(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    q = query_emb.mean(axis=0)
    d = doc_emb.mean(axis=0)
    qn = np.linalg.norm(q)
    dn = np.linalg.norm(d)
    if qn == 0 or dn == 0:
        return 0.0
    return float(np.dot(q, d) / (qn * dn))


def _rank_by_scores(candidates: List[str], score_fn) -> List[str]:
    scored = [(cid, score_fn(cid)) for cid in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored]


def _first_rel_rank(ranked: List[str], labels: Dict[str, int], k: int = 10) -> float:
    for i, did in enumerate(ranked[:k]):
        if labels.get(did, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(ranked: List[str], labels: Dict[str, int], k: int = 10) -> float:
    rel = {d for d, r in labels.items() if r > 0}
    if not rel:
        return 0.0
    hits = set(ranked[:k]) & rel
    return len(hits) / len(rel)


def _ndcg_at_k_graded(ranked: List[str], labels: Dict[str, int], k: int = 10) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked[:k]):
        rel = float(labels.get(did, 0))
        dcg += rel / math.log2(i + 2)

    ideal = sorted([float(v) for v in labels.values()], reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _aggregate(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {"mrr_at_10": 0.0, "ndcg_at_10": 0.0, "recall_at_10": 0.0}
    return {
        "mrr_at_10": round(float(np.mean([r["mrr_at_10"] for r in per_query])), 4),
        "ndcg_at_10": round(float(np.mean([r["ndcg_at_10"] for r in per_query])), 4),
        "recall_at_10": round(float(np.mean([r["recall_at_10"] for r in per_query])), 4),
    }


def _rank_positions(ranked: List[str], rel_docs: List[str], k: int = 10) -> Dict[str, int | None]:
    out: Dict[str, int | None] = {}
    top = ranked[:k]
    for d in rel_docs:
        try:
            out[d] = top.index(d) + 1
        except ValueError:
            out[d] = None
    return out


def run_gate_b(
    queries_path: Path,
    pool_path: Path,
    labels_path: Path,
    corpus_path: Path,
    reference_path: Path,
) -> Dict[str, Any]:
    queries_data = _load_json(queries_path)
    pool_data = _load_json(pool_path)
    labels_data = _load_json(labels_path)
    ref_data = _load_json(reference_path)
    corpus = _load_corpus(corpus_path)

    queries = queries_data["queries"]
    pools = {p["query_id"]: p["candidate_ids"] for p in pool_data["pools"]}
    label_rows = {r["query_id"]: r["graded_labels"] for r in labels_data["labels"]}
    ref_rows = {r["query_id"]: r.get("top10_ids", []) for r in ref_data.get("rankings", [])}

    colbert = ColBERTTier(ColBERTConfig(index_dir="benchmarks/outputs/tmp_colbert_gate_b"))
    q_cache: Dict[str, np.ndarray] = {}
    d_cache: Dict[str, np.ndarray] = {}

    per_ranker_metrics: Dict[str, List[Dict[str, float]]] = {
        "baseline_ann": [],
        "mnemos_colbert": [],
        "reference_colbert": [],
        "lexical_overlap": [],
        "mean_cosine": [],
    }
    query_rows: List[Dict[str, Any]] = []
    underperform_both = 0
    overlap10_values: List[float] = []

    for q in queries:
        qid = q["query_id"]
        qtext = q["query_text"]
        candidate_ids = [cid for cid in pools.get(qid, []) if cid in corpus]
        labels_map = {x["doc_id"]: int(x["relevance"]) for x in label_rows.get(qid, [])}
        rel_docs = [d for d, r in labels_map.items() if r > 0]

        if qtext not in q_cache:
            q_cache[qtext] = colbert._encode_multi_vector(qtext)
        q_emb = q_cache[qtext]

        def get_doc_emb(cid: str) -> np.ndarray:
            if cid not in d_cache:
                d_cache[cid] = colbert._encode_multi_vector(corpus[cid].content)
            return d_cache[cid]

        baseline_top10 = candidate_ids[:10]
        colbert_ranked = _rank_by_scores(candidate_ids, lambda cid: _maxsim(q_emb, get_doc_emb(cid)))
        overlap_ranked = _rank_by_scores(candidate_ids, lambda cid: _overlap_score(qtext, corpus[cid].content))
        mean_ranked = _rank_by_scores(candidate_ids, lambda cid: _mean_cos(q_emb, get_doc_emb(cid)))
        ref_top10 = [cid for cid in ref_rows.get(qid, []) if cid in corpus][:10]

        rankers = {
            "baseline_ann": baseline_top10,
            "mnemos_colbert": colbert_ranked[:10],
            "reference_colbert": ref_top10,
            "lexical_overlap": overlap_ranked[:10],
            "mean_cosine": mean_ranked[:10],
        }

        for name, ranked in rankers.items():
            m = {
                "mrr_at_10": _first_rel_rank(ranked, labels_map, 10),
                "ndcg_at_10": _ndcg_at_k_graded(ranked, labels_map, 10),
                "recall_at_10": _recall_at_k(ranked, labels_map, 10),
            }
            per_ranker_metrics[name].append(m)

        col_mrr = per_ranker_metrics["mnemos_colbert"][-1]["mrr_at_10"]
        ov_mrr = per_ranker_metrics["lexical_overlap"][-1]["mrr_at_10"]
        mc_mrr = per_ranker_metrics["mean_cosine"][-1]["mrr_at_10"]
        if col_mrr < ov_mrr and col_mrr < mc_mrr:
            underperform_both += 1

        cset = set(rankers["mnemos_colbert"])
        rset = set(rankers["reference_colbert"])
        overlap10 = (len(cset & rset) / 10.0) if rset else 0.0
        overlap10_values.append(overlap10)

        baseline = rankers["baseline_ann"]
        query_rows.append(
            {
                "query_id": qid,
                "query_text": qtext,
                "top10": rankers,
                "gold_rank_positions": {
                    k: _rank_positions(v, rel_docs, 10) for k, v in rankers.items()
                },
                "score_deltas_vs_baseline": {
                    "mnemos_colbert_mrr_delta": round(
                        per_ranker_metrics["mnemos_colbert"][-1]["mrr_at_10"]
                        - per_ranker_metrics["baseline_ann"][-1]["mrr_at_10"],
                        4,
                    ),
                    "lexical_overlap_mrr_delta": round(
                        per_ranker_metrics["lexical_overlap"][-1]["mrr_at_10"]
                        - per_ranker_metrics["baseline_ann"][-1]["mrr_at_10"],
                        4,
                    ),
                    "mean_cosine_mrr_delta": round(
                        per_ranker_metrics["mean_cosine"][-1]["mrr_at_10"]
                        - per_ranker_metrics["baseline_ann"][-1]["mrr_at_10"],
                        4,
                    ),
                },
                "mnemos_vs_reference_overlap_at_10": round(overlap10, 4),
                "candidate_count": len(candidate_ids),
                "baseline_candidate_order_top10": baseline,
            }
        )

    aggregate = {k: _aggregate(v) for k, v in per_ranker_metrics.items()}
    mean_overlap10 = float(np.mean(overlap10_values)) if overlap10_values else 0.0

    gate_b_pass = (
        aggregate["mnemos_colbert"]["mrr_at_10"] >= aggregate["lexical_overlap"]["mrr_at_10"]
        or aggregate["mnemos_colbert"]["mrr_at_10"] >= aggregate["mean_cosine"]["mrr_at_10"]
    ) and (mean_overlap10 > 0.0)

    return {
        "track": "gate_b_reference_fidelity",
        "status": "success",
        "reference_stack": ref_data.get("reference_stack", {}),
        "inputs": {
            "queries": str(queries_path),
            "candidate_pool": str(pool_path),
            "labels": str(labels_path),
            "corpus": str(corpus_path),
            "reference": str(reference_path),
        },
        "aggregate_metrics": aggregate,
        "mean_overlap_at_10_mnemos_vs_reference": round(mean_overlap10, 4),
        "underperform_both_trivial_count": underperform_both,
        "n_queries": len(query_rows),
        "gate_b_pass": gate_b_pass,
        "queries": query_rows,
    }


def save_outputs(results: Dict[str, Any], root: Path) -> Tuple[Path, Path]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_dir = root / "benchmarks" / "outputs" / "raw"
    sum_dir = root / "benchmarks" / "outputs" / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{ts}_gate_b_reference_fidelity.json"
    md_path = sum_dir / f"{ts}_gate_b_reference_fidelity.md"

    raw_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    agg = results["aggregate_metrics"]
    lines = [
        "# Gate B Reference-Fidelity Report",
        "",
        f"*Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}*",
        "",
        "## Aggregate Comparison",
        "",
        "| Ranker | MRR@10 | nDCG@10 | Recall@10 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [
        ("baseline_ann", "Baseline ANN"),
        ("mnemos_colbert", "MNEMOS ColBERT"),
        ("reference_colbert", "Reference ColBERT"),
        ("lexical_overlap", "Lexical overlap"),
        ("mean_cosine", "Mean-cosine"),
    ]:
        m = agg.get(key, {})
        lines.append(
            f"| {label} | {m.get('mrr_at_10', 0):.4f} | {m.get('ndcg_at_10', 0):.4f} | {m.get('recall_at_10', 0):.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Mean overlap@10 (MNEMOS vs reference): {results.get('mean_overlap_at_10_mnemos_vs_reference', 0):.4f}",
            f"- Queries where MNEMOS underperforms both trivial baselines: {results.get('underperform_both_trivial_count', 0)} / {results.get('n_queries', 0)}",
            f"- Gate B PASS: {'YES' if results.get('gate_b_pass') else 'NO'}",
            "",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gate B reference-fidelity check.")
    parser.add_argument(
        "--queries",
        default="benchmarks/truthsets/gate_b_sanity_queries.json",
        help="Path to Gate B sanity queries JSON.",
    )
    parser.add_argument(
        "--pool",
        default="benchmarks/truthsets/gate_b_candidate_pool.json",
        help="Path to Gate B candidate pool JSON.",
    )
    parser.add_argument(
        "--labels",
        default="benchmarks/truthsets/gate_b_labels.json",
        help="Path to Gate B labels JSON.",
    )
    parser.add_argument(
        "--corpus",
        default="benchmarks/outputs/datasets/corpus_real.json",
        help="Path to corpus JSON used for candidate content lookup.",
    )
    parser.add_argument(
        "--reference",
        default="benchmarks/truthsets/gate_b_reference_rankings_template.json",
        help="Path to reference ColBERT rankings JSON.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results = run_gate_b(
        queries_path=root / args.queries,
        pool_path=root / args.pool,
        labels_path=root / args.labels,
        corpus_path=root / args.corpus,
        reference_path=root / args.reference,
    )
    raw_path, md_path = save_outputs(results, root)
    print(f"[OK] Gate B complete")
    print(f"  Raw:    {raw_path}")
    print(f"  Report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
