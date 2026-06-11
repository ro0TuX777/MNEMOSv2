"""Phase 8 query-complexity classifier unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.retrieval.complexity import (
    EmbeddedComplexityClassifier,
    ZeroShotComplexityClassifier,
    route_posture_for_label,
)


class FakeNLIModel:
    def predict(self, pairs, apply_softmax=False):
        assert apply_softmax is False
        # contradiction, entailment, neutral; CLASS_B wins.
        return np.asarray(
            [
                [0.0, 0.1, 0.9],
                [0.0, 4.0, 0.1],
                [0.0, 0.2, 0.8],
            ],
            dtype=np.float32,
        )


class FakeZeroShotComplexityClassifier(ZeroShotComplexityClassifier):
    def _ensure_model(self) -> Any:
        return FakeNLIModel()


def test_zero_shot_classifier_picks_highest_entailment_label():
    classifier = FakeZeroShotComplexityClassifier()

    result = classifier.classify("Which policy overlaps with the SIGINT authority rule?")

    assert result.label == "CLASS_B"
    assert result.route_posture["graph"] == "trigger_memgraph_rag"
    assert result.scores["CLASS_B"] > result.scores["CLASS_A"]
    assert result.latency_ms >= 0


def test_route_posture_mapping():
    assert route_posture_for_label("CLASS_A")["fusion_policy"] == "semantic_dominant"
    assert route_posture_for_label("CLASS_B")["graph"] == "trigger_memgraph_rag"
    assert route_posture_for_label("CLASS_C")["hierarchical"] == "trigger_future_raptor"


def test_embedded_classifier_loads_weights_and_classifies_vector(tmp_path):
    weights_path = tmp_path / "complexity_weights.bin"
    with weights_path.open("wb") as fh:
        np.savez_compressed(
            fh,
            labels=np.asarray(["CLASS_A", "CLASS_B", "CLASS_C"]),
            weights=np.asarray(
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 3.0],
                ],
                dtype=np.float32,
            ),
            bias=np.zeros(3, dtype=np.float32),
            embedding_model_name=np.asarray("test-embedder"),
            embedding_dim=np.asarray(3, dtype=np.int32),
            classifier_name=np.asarray("embedded-linear-softmax"),
        )

    classifier = EmbeddedComplexityClassifier(weights_path=weights_path)
    result = classifier.classify_vector(np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

    assert result.label == "CLASS_B"
    assert result.route_posture["fusion_policy"] == "balanced"
    assert result.latency_ms < 2.0
