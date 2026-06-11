"""
Query-complexity classification for Phase 8 Adaptive Branching.

The zero-shot implementation uses an NLI cross-encoder and scores each
candidate route description against the query. It is intentionally small and
lazy-loaded so production search can run it only in explicit shadow mode.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

COMPLEXITY_LABELS = ("CLASS_A", "CLASS_B", "CLASS_C")
DEFAULT_COMPLEXITY_WEIGHTS = Path(__file__).with_name("complexity_weights.bin")

_ENTAILMENT_INDEX = 1
_NOMIC_V15_MODEL_MARKER = "nomic-embed-text-v1.5"
_NOMIC_QUERY_PREFIX = "search_query: "


@dataclass(frozen=True)
class ComplexityResult:
    """Classifier output for one query."""

    label: str
    confidence: float
    scores: Dict[str, float]
    route_posture: Dict[str, Any]
    latency_ms: float
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in sorted(self.scores.items())},
            "route_posture": dict(self.route_posture),
            "latency_ms": round(self.latency_ms, 4),
            "model_name": self.model_name,
        }


class BaseComplexityClassifier(ABC):
    """Interface for query-complexity classifiers."""

    @abstractmethod
    def classify(self, query: str) -> ComplexityResult:
        """Classify a query into CLASS_A, CLASS_B, or CLASS_C."""


def route_posture_for_label(label: str) -> Dict[str, Any]:
    """Map Phase 8 labels to routing posture metadata."""
    if label == "CLASS_A":
        return {
            "retrieval_posture": "semantic_dominant",
            "fusion_policy": "semantic_dominant",
            "graph": "skip",
            "hierarchical": "skip",
        }
    if label == "CLASS_B":
        return {
            "retrieval_posture": "balanced",
            "fusion_policy": "balanced",
            "graph": "trigger_memgraph_rag",
            "hierarchical": "skip",
        }
    if label == "CLASS_C":
        return {
            "retrieval_posture": "global_hierarchical",
            "fusion_policy": "lexical_dominant",
            "graph": "optional",
            "hierarchical": "trigger_future_raptor",
        }
    return {
        "retrieval_posture": "unknown",
        "fusion_policy": "balanced",
        "graph": "skip",
        "hierarchical": "skip",
    }


class ZeroShotComplexityClassifier(BaseComplexityClassifier):
    """
    NLI-based zero-shot complexity classifier.

    The query is treated as the premise, and each class description is scored
    as a hypothesis. The class with the highest entailment probability wins.
    """

    DEFAULT_HYPOTHESES: Dict[str, str] = {
        "CLASS_A": (
            "This query asks for a direct factoid, named reference, acronym, "
            "section, or single document lookup."
        ),
        "CLASS_B": (
            "This query asks for relationships, conflicts, overlaps, lineage, "
            "or multi-hop links between facts or documents."
        ),
        "CLASS_C": (
            "This query asks for a broad summary, global theme, corpus-wide "
            "comparison, or hierarchical synthesis."
        ),
    }

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        hypotheses: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self._hypotheses = dict(hypotheses or self.DEFAULT_HYPOTHESES)
        self._device = device
        self._model = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            if self._device:
                self._model = CrossEncoder(self.model_name, device=self._device)
            else:
                self._model = CrossEncoder(self.model_name)
        return self._model

    def classify(self, query: str) -> ComplexityResult:
        started = time.perf_counter()
        labels = [label for label in COMPLEXITY_LABELS if label in self._hypotheses]
        pairs = [[query, self._hypotheses[label]] for label in labels]
        model = self._ensure_model()
        logits = np.asarray(model.predict(pairs, apply_softmax=False))
        if logits.ndim == 1:
            # Binary-style fallback: treat each scalar as an entailment logit.
            exp = np.exp(logits - logits.max())
            entailment = exp / np.clip(exp.sum(), 1e-12, None)
        else:
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)
            entailment = probs[:, _ENTAILMENT_INDEX]

        scores = {label: float(score) for label, score in zip(labels, entailment)}
        label = max(scores, key=lambda item: scores[item])
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ComplexityResult(
            label=label,
            confidence=scores[label],
            scores=scores,
            route_posture=route_posture_for_label(label),
            latency_ms=latency_ms,
            model_name=self.model_name,
        )


class EmbeddedComplexityClassifier(BaseComplexityClassifier):
    """
    Lightweight linear classifier over an existing retrieval query embedding.

    For the intended hot path, callers should use classify_vector() with the
    query vector they already produced for semantic retrieval. classify() is
    retained for shadow/evaluation use and performs its own embedding.
    """

    def __init__(
        self,
        *,
        weights_path: Path | str = DEFAULT_COMPLEXITY_WEIGHTS,
        embedding_model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        payload = np.load(self.weights_path, allow_pickle=True)
        def scalar(name: str, default: Any) -> Any:
            if name not in payload.files:
                return default
            value = payload[name]
            return value.item() if getattr(value, "shape", None) == () else value

        self.labels = [str(label) for label in payload["labels"].tolist()]
        self.weights = np.asarray(payload["weights"], dtype=np.float32)
        self.bias = np.asarray(payload["bias"], dtype=np.float32)
        self.embedding_model_name = str(scalar("embedding_model_name", embedding_model_name or "unknown"))
        if embedding_model_name:
            self.embedding_model_name = embedding_model_name
        self.model_name = str(scalar("classifier_name", "embedded-linear-softmax"))
        self.embedding_dim = int(scalar("embedding_dim", self.weights.shape[1]))
        self._device = device
        self._embedder = None

    @classmethod
    def weights_available(cls, weights_path: Path | str = DEFAULT_COMPLEXITY_WEIGHTS) -> bool:
        return Path(weights_path).exists()

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            from mnemos.retrieval.qdrant_tier import _ensure_transformer_runtime_compat

            _ensure_transformer_runtime_compat()
            from sentence_transformers import SentenceTransformer

            kwargs: Dict[str, Any] = {}
            if self._device:
                kwargs["device"] = self._device
            if _NOMIC_V15_MODEL_MARKER in self.embedding_model_name:
                kwargs["trust_remote_code"] = True
            self._embedder = SentenceTransformer(self.embedding_model_name, **kwargs)
        return self._embedder

    def embed_query(self, query: str) -> np.ndarray:
        model = self._ensure_embedder()
        text = f"{_NOMIC_QUERY_PREFIX}{query}" if _NOMIC_V15_MODEL_MARKER in self.embedding_model_name else query
        vector = np.asarray(model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
        return vector

    def classify_vector(self, query_vector: np.ndarray) -> ComplexityResult:
        started = time.perf_counter()
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected query vector dim {self.embedding_dim}, got {vector.shape[0]}")
        logits = self.weights @ vector + self.bias
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.clip(exp.sum(), 1e-12, None)
        index = int(np.argmax(probs))
        scores = {label: float(probs[i]) for i, label in enumerate(self.labels)}
        label = self.labels[index]
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ComplexityResult(
            label=label,
            confidence=float(probs[index]),
            scores=scores,
            route_posture=route_posture_for_label(label),
            latency_ms=latency_ms,
            model_name=self.model_name,
        )

    def classify(self, query: str) -> ComplexityResult:
        return self.classify_vector(self.embed_query(query))


def default_complexity_classifier() -> BaseComplexityClassifier:
    """Prefer the embedded reflex classifier when trained weights exist."""
    if EmbeddedComplexityClassifier.weights_available():
        return EmbeddedComplexityClassifier()
    return ZeroShotComplexityClassifier()
