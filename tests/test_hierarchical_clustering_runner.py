"""Phase 9 RAPTOR-lite hierarchy scaffold tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.engram.model import Engram
from mnemos.governance.hygiene.clustering_runner import HierarchicalClusteringRunner


class RecordingIndexer:
    def __init__(self):
        self.indexed = []

    def index(self, engrams):
        self.indexed.extend(engrams)
        return len(engrams)


def test_hierarchical_clustering_runner_emits_dry_run_report(tmp_path):
    engrams = [
        Engram(id="privacy_a", content="GDPR disclosure and erasure obligations."),
        Engram(id="privacy_b", content="Privacy controls require deletion and disclosure."),
        Engram(id="sigint_a", content="SIGINT tenant policy and bounded reflection."),
        Engram(id="sigint_b", content="Tenant boundaries constrain SIGINT reflection."),
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    output = tmp_path / "hierarchy_report.json"

    report = HierarchicalClusteringRunner(n_clusters=2, random_seed=1).run(
        engrams,
        vectors=vectors,
        output_path=output,
    )

    assert report.dry_run is True
    assert report.engrams_scanned == 4
    assert report.cluster_count == 2
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary_engram_writes"] == 0
    assert len(payload["clusters"]) == 2
    assert "summary engram" in payload["clusters"][0]["summary_prompt_preview"]


def test_hierarchical_clustering_runner_action_mode_indexes_summary_engrams():
    engrams = [
        Engram(id="privacy_a", content="GDPR disclosure and erasure obligations."),
        Engram(id="privacy_b", content="Privacy controls require deletion and disclosure."),
        Engram(id="sigint_a", content="SIGINT tenant policy and bounded reflection."),
        Engram(id="sigint_b", content="Tenant boundaries constrain SIGINT reflection."),
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    indexer = RecordingIndexer()

    report = HierarchicalClusteringRunner(n_clusters=2, random_seed=1).run(
        engrams,
        vectors=vectors,
        dry_run=False,
        indexer=indexer,
    )

    assert report.dry_run is False
    assert report.summary_engram_writes == 2
    assert len(indexer.indexed) == 2
    summary = indexer.indexed[0]
    assert summary.metadata["is_summary_engram"] is True
    assert summary.metadata["depth"] == 1
    assert summary.source.startswith("derived://hierarchy/cluster_")
    assert summary.edges
    assert "Thematic summary" in summary.content
