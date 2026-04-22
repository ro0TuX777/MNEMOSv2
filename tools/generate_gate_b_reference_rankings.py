"""
Generate Gate B reference rankings using official Stanford ColBERT (colbert-ai).

This script scores each query's fixed candidate pool and writes:
  benchmarks/truthsets/gate_b_reference_rankings.json

The output is intended to replace surrogate/template reference rankings before
running tools/run_gate_b_reference_fidelity.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_corpus_texts(path: Path) -> Dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for row in rows:
        did = row.get("id")
        text = row.get("content")
        if isinstance(did, str) and isinstance(text, str):
            out[did] = text
    return out


def _resolve_device(device: str, torch_mod) -> Tuple[str, int]:
    if device == "cuda":
        if not torch_mod.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return "cuda", 1
    if device == "cpu":
        return "cpu", 0
    # auto
    if torch_mod.cuda.is_available():
        return "cuda", 1
    return "cpu", 0


def _score_candidates_with_colbert(
    query_text: str,
    candidate_ids: List[str],
    corpus_text: Dict[str, str],
    checkpoint,
    colbert_score_packed_fn,
    torch_mod,
    topk: int,
    doc_bsize: int,
) -> List[str]:
    kept_ids = [cid for cid in candidate_ids if cid in corpus_text]
    if not kept_ids:
        return []

    docs = [corpus_text[cid] for cid in kept_ids]

    use_gpu = bool(getattr(checkpoint.colbert_config, "total_visible_gpus", 0) > 0)
    q = checkpoint.queryFromText([query_text], to_cpu=not use_gpu)
    d_packed, d_lens = checkpoint.docFromText(
        docs,
        bsize=doc_bsize,
        keep_dims="flatten",
        to_cpu=not use_gpu,
        showprogress=False,
    )
    if not isinstance(d_lens, list):
        d_lens = list(d_lens)

    d_lens_t = torch_mod.tensor(d_lens)
    scores_t = colbert_score_packed_fn(q, d_packed, d_lens_t, config=checkpoint.colbert_config)
    scores = scores_t.detach().cpu().tolist()

    ranked = sorted(zip(kept_ids, scores), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:topk]]


def generate_reference_rankings(
    queries_path: Path,
    pool_path: Path,
    corpus_path: Path,
    output_path: Path,
    model_id: str,
    topk: int,
    device: str,
    query_maxlen: int,
    doc_maxlen: int,
    doc_bsize: int,
) -> Dict[str, Any]:
    # Lazy imports so file remains importable on systems without torch/colbert.
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for official ColBERT reference generation. "
            "Install torch in the active environment."
        ) from exc

    try:
        from colbert.infra import ColBERTConfig  # type: ignore
        from colbert.modeling.checkpoint import Checkpoint  # type: ignore
        from colbert.modeling.colbert import colbert_score_packed  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Official ColBERT package is required. Install colbert-ai in the active environment."
        ) from exc

    effective_device, gpus = _resolve_device(device, torch)

    queries_data = _load_json(queries_path)
    pool_data = _load_json(pool_path)
    corpus_text = _load_corpus_texts(corpus_path)

    queries = queries_data.get("queries", [])
    pool_map = {p["query_id"]: p.get("candidate_ids", []) for p in pool_data.get("pools", [])}

    config = ColBERTConfig(
        query_maxlen=query_maxlen,
        doc_maxlen=doc_maxlen,
        gpus=gpus,
    )
    checkpoint = Checkpoint(model_id, colbert_config=config, verbose=1)
    if effective_device == "cuda":
        checkpoint = checkpoint.cuda()

    rankings: List[Dict[str, Any]] = []
    skipped_missing_pool = 0

    for row in queries:
        qid = row["query_id"]
        qtext = row["query_text"]
        candidate_ids = pool_map.get(qid, [])
        if not candidate_ids:
            skipped_missing_pool += 1
            rankings.append({"query_id": qid, "top10_ids": []})
            continue

        top_ids = _score_candidates_with_colbert(
            query_text=qtext,
            candidate_ids=candidate_ids,
            corpus_text=corpus_text,
            checkpoint=checkpoint,
            colbert_score_packed_fn=colbert_score_packed,
            torch_mod=torch,
            topk=topk,
            doc_bsize=doc_bsize,
        )
        rankings.append({"query_id": qid, "top10_ids": top_ids})

    now = datetime.now(timezone.utc).isoformat()
    out = {
        "version": "gate_b_reference_v1_official_colbert",
        "reference_stack": {
            "name": "stanford_colbert_reference",
            "model_id": model_id,
            "library": "colbert-ai",
            "library_version": pkg_version("colbert-ai"),
            "runtime": effective_device,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "transformers_version": pkg_version("transformers"),
            "query_maxlen": query_maxlen,
            "doc_maxlen": doc_maxlen,
            "doc_batch_size": doc_bsize,
            "generated_at_utc": now,
            "notes": "Generated from fixed Gate B candidate pools with official ColBERT late-interaction scoring.",
        },
        "rankings": rankings,
        "generation_summary": {
            "queries_total": len(queries),
            "queries_missing_pool": skipped_missing_pool,
            "topk": topk,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gate B reference rankings with official ColBERT."
    )
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
        "--corpus",
        default="benchmarks/outputs/datasets/corpus_real.json",
        help="Path to corpus JSON used by candidate IDs.",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/truthsets/gate_b_reference_rankings.json",
        help="Output path for generated reference rankings JSON.",
    )
    parser.add_argument(
        "--model-id",
        default="colbert-ir/colbertv2.0",
        help="ColBERT model ID/checkpoint path.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top K candidate IDs to emit per query.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device for the reference stack.",
    )
    parser.add_argument(
        "--query-maxlen",
        type=int,
        default=64,
        help="ColBERT query max token length.",
    )
    parser.add_argument(
        "--doc-maxlen",
        type=int,
        default=256,
        help="ColBERT document max token length.",
    )
    parser.add_argument(
        "--doc-bsize",
        type=int,
        default=32,
        help="Document encoding batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = generate_reference_rankings(
        queries_path=Path(args.queries),
        pool_path=Path(args.pool),
        corpus_path=Path(args.corpus),
        output_path=Path(args.out),
        model_id=args.model_id,
        topk=args.topk,
        device=args.device,
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        doc_bsize=args.doc_bsize,
    )
    summary = out.get("generation_summary", {})
    print("[OK] Gate B reference rankings generated")
    print(f"  Output: {args.out}")
    print(
        f"  Queries: {summary.get('queries_total', 0)} "
        f"(missing pool: {summary.get('queries_missing_pool', 0)})"
    )


if __name__ == "__main__":
    main()
