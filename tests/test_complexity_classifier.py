"""Phase 8 query-complexity classifier unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.retrieval.complexity import (
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
