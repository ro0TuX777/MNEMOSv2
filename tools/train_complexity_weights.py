"""
Train Phase 8 embedded query-complexity weights.

The runtime classifier is intentionally just a linear softmax layer over the
query embedding that MNEMOS already computes for retrieval. This trainer uses
scikit-learn LogisticRegression when available, otherwise falls back to a
closed-form ridge linear classifier with the same runtime representation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.retrieval.complexity import COMPLEXITY_LABELS, DEFAULT_COMPLEXITY_WEIGHTS
from mnemos.retrieval.qdrant_tier import (
    NOMIC_QUERY_PREFIX,
    NOMIC_V15_MODEL_MARKER,
    _ensure_transformer_runtime_compat,
)

DEFAULT_TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "query_complexity_v1.json"
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def _default_embedding_model() -> str:
    return (
        os.getenv("MNEMOS_EMBEDDING_MODEL_NAME")
        or os.getenv("MNEMOS_EMBEDDING_MODEL")
        or DEFAULT_MODEL
    )


def _load_truthset(path: Path) -> Tuple[List[str], List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = [str(item["query"]) for item in payload["queries"]]
    labels = [str(item["label"]) for item in payload["queries"]]
    invalid = sorted(set(labels) - set(COMPLEXITY_LABELS))
    if invalid:
        raise ValueError(f"Unsupported labels in truthset: {invalid}")
    return queries, labels


def _embed_queries(
    queries: List[str],
    *,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> np.ndarray:
    _ensure_transformer_runtime_compat()
    from sentence_transformers import SentenceTransformer

    kwargs: Dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    if NOMIC_V15_MODEL_MARKER in model_name:
        kwargs["trust_remote_code"] = True
        texts = [f"{NOMIC_QUERY_PREFIX}{query}" for query in queries]
    else:
        texts = list(queries)

    model = SentenceTransformer(model_name, **kwargs)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def _train_with_sklearn(
    vectors: np.ndarray,
    labels: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], str] | None:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None

    clf = LogisticRegression(
        C=100.0,
        class_weight="balanced",
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
    )
    clf.fit(vectors, labels)
    return (
        np.asarray(clf.coef_, dtype=np.float32),
        np.asarray(clf.intercept_, dtype=np.float32),
        [str(label) for label in clf.classes_],
        "sklearn-logistic-regression",
    )


def _train_with_ridge(
    vectors: np.ndarray,
    labels: List[str],
    *,
    ridge: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    label_order: List[str] = [str(label) for label in COMPLEXITY_LABELS]
    label_to_index = {label: i for i, label in enumerate(label_order)}
    targets = np.full((len(labels), len(label_order)), -1.0, dtype=np.float32)
    for row, label in enumerate(labels):
        targets[row, label_to_index[label]] = 1.0

    features = np.concatenate(
        [vectors, np.ones((vectors.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    penalty = np.eye(features.shape[1], dtype=np.float32) * ridge
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(features.T @ features + penalty, features.T @ targets)
    weights = beta[:-1].T.astype(np.float32)
    bias = beta[-1].astype(np.float32)
    return weights, bias, label_order, "ridge-linear-softmax"


def _softmax_scores(vectors: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    logits = vectors @ weights.T + bias
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _metrics(
    vectors: np.ndarray,
    labels: List[str],
    label_order: List[str],
    weights: np.ndarray,
    bias: np.ndarray,
) -> Dict[str, Any]:
    probs = _softmax_scores(vectors, weights, bias)
    predictions = [label_order[int(i)] for i in np.argmax(probs, axis=1)]
    correct = [pred == expected for pred, expected in zip(predictions, labels)]
    total_by_class: Counter[str] = Counter(labels)
    correct_by_class: Counter[str] = Counter(
        expected for expected, ok in zip(labels, correct) if ok
    )
    return {
        "query_count": len(labels),
        "overall_accuracy": round(sum(correct) / len(correct), 4) if correct else 0.0,
        "per_class": {
            label: {
                "correct": correct_by_class[label],
                "total": total_by_class[label],
                "accuracy": round(correct_by_class[label] / total_by_class[label], 4)
                if total_by_class[label]
                else 0.0,
            }
            for label in label_order
        },
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    truthset = args.truthset.resolve()
    output = args.output.resolve()
    queries, labels = _load_truthset(truthset)

    started = time.perf_counter()
    vectors = _embed_queries(
        queries,
        model_name=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    embed_ms = (time.perf_counter() - started) * 1000.0

    trained = None if args.force_ridge else _train_with_sklearn(vectors, labels)
    if trained is None:
        weights, bias, label_order, trainer = _train_with_ridge(vectors, labels, ridge=args.ridge)
    else:
        weights, bias, label_order, trainer = trained

    metrics = _metrics(vectors, labels, label_order, weights, bias)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        np.savez_compressed(
            fh,
            labels=np.asarray(label_order),
            weights=weights,
            bias=bias,
            embedding_model_name=np.asarray(args.embedding_model),
            embedding_dim=np.asarray(vectors.shape[1], dtype=np.int32),
            classifier_name=np.asarray("embedded-linear-softmax"),
            trainer=np.asarray(trainer),
            truthset=np.asarray(str(truthset.relative_to(PROJECT_ROOT))),
            trained_at=np.asarray(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )

    return {
        "output": str(output),
        "truthset": str(truthset.relative_to(PROJECT_ROOT)),
        "embedding_model": args.embedding_model,
        "embedding_dim": int(vectors.shape[1]),
        "trainer": trainer,
        "embedding_ms": round(embed_ms, 4),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Phase 8 embedded complexity weights")
    parser.add_argument("--truthset", type=Path, default=DEFAULT_TRUTHSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_COMPLEXITY_WEIGHTS)
    parser.add_argument("--embedding-model", default=_default_embedding_model())
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--force-ridge", action="store_true")
    args = parser.parse_args()

    result = train(args)
    metrics = result["metrics"]
    print(f"output: {result['output']}")
    print(f"embedding model: {result['embedding_model']}")
    print(f"embedding dim: {result['embedding_dim']}")
    print(f"trainer: {result['trainer']}")
    print(f"queries: {metrics['query_count']}")
    print(f"training accuracy: {metrics['overall_accuracy']:.4f}")
    for label, row in metrics["per_class"].items():
        print(f"{label}: {row['correct']}/{row['total']} ({row['accuracy']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
