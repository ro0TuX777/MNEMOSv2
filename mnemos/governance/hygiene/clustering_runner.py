"""
HierarchicalClusteringRunner - RAPTOR-lite dry-run hierarchy scaffold.

Phase 9 uses this runner to preview summary-engram clusters without mutating
the corpus. The apply path that writes synthetic summary engrams is deliberately
left for a later promotion gate.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mnemos.engram.model import Engram
from mnemos.retrieval.qdrant_tier import (
    NOMIC_DOC_PREFIX,
    NOMIC_V15_MODEL_MARKER,
    _ensure_transformer_runtime_compat,
)


@dataclass
class HierarchicalClusterRecord:
    cluster_id: str
    member_ids: List[str]
    centroid_norm: float
    representative_ids: List[str]
    summary_prompt_preview: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "member_count": len(self.member_ids),
            "member_ids": list(self.member_ids),
            "centroid_norm": round(self.centroid_norm, 6),
            "representative_ids": list(self.representative_ids),
            "summary_prompt_preview": self.summary_prompt_preview,
        }


@dataclass
class HierarchyReport:
    dry_run: bool
    model_name: str
    engrams_scanned: int
    cluster_count: int
    elapsed_ms: float
    clusters: List[HierarchicalClusterRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "model_name": self.model_name,
            "engrams_scanned": self.engrams_scanned,
            "cluster_count": self.cluster_count,
            "elapsed_ms": round(self.elapsed_ms, 4),
            "summary_engram_writes": 0,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }


class HierarchicalClusteringRunner:
    """
    Dry-run K-Means clustering over engram embeddings.

    The runner accepts explicit vectors for tests/offline reports. If vectors
    are omitted, it embeds engram content with the configured Nomic/Sentence
    Transformer model and uses document prefixes when required.
    """

    def __init__(
        self,
        *,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        n_clusters: Optional[int] = None,
        max_iterations: int = 30,
        random_seed: int = 7,
    ) -> None:
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.max_iterations = max(1, max_iterations)
        self.random_seed = random_seed
        self._embedder = None

    def run(
        self,
        engrams: List[Engram],
        *,
        vectors: Optional[np.ndarray] = None,
        dry_run: bool = True,
        output_path: Optional[Path | str] = None,
    ) -> HierarchyReport:
        started = time.perf_counter()
        if not dry_run:
            raise NotImplementedError("Summary-engram writes are not enabled in the Phase 9 scaffold")
        if not engrams:
            report = HierarchyReport(
                dry_run=True,
                model_name=self.model_name,
                engrams_scanned=0,
                cluster_count=0,
                elapsed_ms=0.0,
                clusters=[],
            )
            self._write_report(report, output_path)
            return report

        matrix = self._resolve_vectors(engrams, vectors)
        k = self._choose_cluster_count(len(engrams))
        assignments, centroids = self._kmeans(matrix, k)
        clusters = self._build_cluster_records(engrams, matrix, assignments, centroids)
        report = HierarchyReport(
            dry_run=True,
            model_name=self.model_name,
            engrams_scanned=len(engrams),
            cluster_count=len(clusters),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            clusters=clusters,
        )
        self._write_report(report, output_path)
        return report

    def _resolve_vectors(self, engrams: List[Engram], vectors: Optional[np.ndarray]) -> np.ndarray:
        if vectors is not None:
            matrix = np.asarray(vectors, dtype=np.float32)
            if matrix.shape[0] != len(engrams):
                raise ValueError(f"Expected {len(engrams)} vectors, got {matrix.shape[0]}")
            return self._l2_normalize(matrix)

        embedded = []
        for engram in engrams:
            meta = getattr(engram, "metadata", {}) or {}
            vector = meta.get("dense_768") or meta.get("embedding") or meta.get("vector")
            if vector is not None:
                embedded.append(np.asarray(vector, dtype=np.float32))
        if len(embedded) == len(engrams):
            return self._l2_normalize(np.vstack(embedded))

        model = self._ensure_embedder()
        texts = [engram.content for engram in engrams]
        if NOMIC_V15_MODEL_MARKER in self.model_name:
            texts = [f"{NOMIC_DOC_PREFIX}{text}" for text in texts]
        matrix = np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return self._l2_normalize(matrix)

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            _ensure_transformer_runtime_compat()
            from sentence_transformers import SentenceTransformer

            kwargs: Dict[str, Any] = {}
            if NOMIC_V15_MODEL_MARKER in self.model_name:
                kwargs["trust_remote_code"] = True
            self._embedder = SentenceTransformer(self.model_name, **kwargs)
        return self._embedder

    def _choose_cluster_count(self, n: int) -> int:
        if self.n_clusters is not None:
            return max(1, min(self.n_clusters, n))
        return max(1, min(n, int(round(math.sqrt(n)))))

    def _kmeans(self, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_seed)
        if k == 1:
            centroid = matrix.mean(axis=0, keepdims=True)
            return np.zeros(matrix.shape[0], dtype=np.int32), self._l2_normalize(centroid)

        initial = rng.choice(matrix.shape[0], size=k, replace=False)
        centroids = matrix[initial].copy()
        assignments = np.zeros(matrix.shape[0], dtype=np.int32)
        for _ in range(self.max_iterations):
            similarities = matrix @ centroids.T
            next_assignments = np.argmax(similarities, axis=1).astype(np.int32)
            if np.array_equal(assignments, next_assignments):
                break
            assignments = next_assignments
            for idx in range(k):
                members = matrix[assignments == idx]
                if len(members) == 0:
                    centroids[idx] = matrix[rng.integers(0, matrix.shape[0])]
                else:
                    centroids[idx] = members.mean(axis=0)
            centroids = self._l2_normalize(centroids)
        return assignments, centroids

    def _build_cluster_records(
        self,
        engrams: List[Engram],
        matrix: np.ndarray,
        assignments: np.ndarray,
        centroids: np.ndarray,
    ) -> List[HierarchicalClusterRecord]:
        records = []
        for cluster_idx in range(centroids.shape[0]):
            member_indices = np.where(assignments == cluster_idx)[0].tolist()
            if not member_indices:
                continue
            scored = sorted(
                member_indices,
                key=lambda i: float(matrix[i] @ centroids[cluster_idx]),
                reverse=True,
            )
            representative_ids = [engrams[i].id for i in scored[:3]]
            member_ids = [engrams[i].id for i in member_indices]
            preview = self._summary_prompt_preview([engrams[i] for i in scored[:5]])
            records.append(
                HierarchicalClusterRecord(
                    cluster_id=f"cluster_{cluster_idx:03d}",
                    member_ids=member_ids,
                    centroid_norm=float(np.linalg.norm(centroids[cluster_idx])),
                    representative_ids=representative_ids,
                    summary_prompt_preview=preview,
                )
            )
        records.sort(key=lambda record: (-len(record.member_ids), record.cluster_id))
        return records

    @staticmethod
    def _summary_prompt_preview(engrams: List[Engram]) -> str:
        snippets = []
        for engram in engrams:
            text = " ".join(engram.content.split())
            snippets.append(f"- {engram.id}: {text[:220]}")
        return (
            "Summarize the shared theme across these engrams. "
            "Return a governed summary engram with source member ids and no new facts.\n"
            + "\n".join(snippets)
        )

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    @staticmethod
    def _write_report(report: HierarchyReport, output_path: Optional[Path | str]) -> None:
        if output_path is None:
            return
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
