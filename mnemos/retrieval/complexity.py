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
from typing import Any, Dict, Optional

import numpy as np

COMPLEXITY_LABELS = ("CLASS_A", "CLASS_B", "CLASS_C")

_ENTAILMENT_INDEX = 1


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
