"""
HierarchicalClusteringRunner - RAPTOR-lite dry-run hierarchy scaffold.

Phase 9 uses this runner to preview summary-engram clusters without mutating
the corpus. The apply path that writes synthetic summary engrams is deliberately
left for a later promotion gate.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

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
    summary_engram_id: Optional[str] = None
    summary_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "member_count": len(self.member_ids),
            "member_ids": list(self.member_ids),
            "centroid_norm": round(self.centroid_norm, 6),
            "representative_ids": list(self.representative_ids),
            "summary_prompt_preview": self.summary_prompt_preview,
            "summary_engram_id": self.summary_engram_id,
            "summary_content": self.summary_content,
        }


@dataclass
class HierarchyReport:
    dry_run: bool
    model_name: str
    engrams_scanned: int
    cluster_count: int
    elapsed_ms: float
    summary_engram_writes: int = 0
    clusters: List[HierarchicalClusterRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "model_name": self.model_name,
            "engrams_scanned": self.engrams_scanned,
            "cluster_count": self.cluster_count,
            "elapsed_ms": round(self.elapsed_ms, 4),
            "summary_engram_writes": self.summary_engram_writes,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }


class EngramIndexer(Protocol):
    def index(self, engrams: List[Engram]) -> Any:
        ...


class ClusterSynthesizer:
    """SMC-compatible cluster summarizer with deterministic fallback."""

    def __init__(self, *, max_chars: int = 1200) -> None:
        self.max_chars = max_chars

    def synthesize(self, *, cluster_id: str, engrams: List[Engram]) -> str:
        prompt = self._prompt(cluster_id, engrams)
        llm = self._call_smc_llm(prompt)
        if llm:
            return llm.strip()
        return self._join_and_truncate(cluster_id, engrams)

    def _call_smc_llm(self, prompt: str) -> Optional[str]:
        base_url = os.getenv("SMC_LLM_BASE_URL", "").strip()
        model = os.getenv("SMC_LLM_MODEL", "").strip()
        if not base_url or not model:
            return None
        try:
            import requests

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Summarize only the provided MNEMOS engrams. "
                            "Do not invent facts. Preserve governance-relevant themes."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": float(os.getenv("SMC_LLM_TEMPERATURE", "0")),
                "max_tokens": int(os.getenv("SMC_LLM_MAX_TOKENS", "512")),
            }
            headers = {"Content-Type": "application/json"}
            api_key = os.getenv("SMC_LLM_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=int(os.getenv("SMC_LLM_TIMEOUT_SECONDS", "60")),
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return str(content)
        except Exception:
            return None

    def _join_and_truncate(self, cluster_id: str, engrams: List[Engram]) -> str:
        snippets = []
        for engram in engrams[:8]:
            text = " ".join(engram.content.split())
            snippets.append(f"{engram.id}: {text[:260]}")
        joined = " ".join(snippets)
        return (
            f"Thematic summary for {cluster_id}: "
            f"{joined[: self.max_chars]} "
            f"(Derived from {len(engrams)} source engrams; no new facts added.)"
        ).strip()

    @staticmethod
    def _prompt(cluster_id: str, engrams: List[Engram]) -> str:
        lines = [
            f"CLUSTER: {cluster_id}",
            "TASK: Write one concise thematic summary. Include only facts supported by the listed engrams.",
            "SOURCE ENGRAMS:",
        ]
        for engram in engrams[:20]:
            text = " ".join(engram.content.split())
            lines.append(f"- {engram.id}: {text[:600]}")
        return "\n".join(lines)


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
        target_cluster_size: int = 35,
        synthesizer: Optional[ClusterSynthesizer] = None,
    ) -> None:
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.max_iterations = max(1, max_iterations)
        self.random_seed = random_seed
        self.target_cluster_size = max(1, target_cluster_size)
        self.synthesizer = synthesizer or ClusterSynthesizer()
        self._embedder = None

    def run(
        self,
        engrams: List[Engram],
        *,
        vectors: Optional[np.ndarray] = None,
        dry_run: bool = True,
        indexer: Optional[EngramIndexer] = None,
        output_path: Optional[Path | str] = None,
        depth: int = 1,
    ) -> HierarchyReport:
        """
        depth=1 (default): cluster leaf engrams into thematic summaries.
        depth=2 (RAPTOR recursion): cluster only depth-1 summary engrams
        into a single root summary whose edges point at the depth-1 ids.
        """
        if depth not in (1, 2):
            raise ValueError("depth must be 1 or 2")
        started = time.perf_counter()
        if depth == 2:
            source_engrams = [
                engram
                for engram in engrams
                if (getattr(engram, "metadata", {}) or {}).get("is_summary_engram", False)
                and int((engram.metadata or {}).get("depth", 0) or 0) == 1
            ]
        else:
            source_engrams = [
                engram
                for engram in engrams
                if not (getattr(engram, "metadata", {}) or {}).get("is_summary_engram", False)
            ]
        if not source_engrams:
            report = HierarchyReport(
                dry_run=dry_run,
                model_name=self.model_name,
                engrams_scanned=0,
                cluster_count=0,
                elapsed_ms=0.0,
                clusters=[],
            )
            self._write_report(report, output_path)
            return report

        if vectors is not None and len(source_engrams) != len(engrams):
            raise ValueError("Explicit vectors must align with the selected source engrams")
        matrix = self._resolve_vectors(source_engrams, vectors)
        k = 1 if depth == 2 else self._choose_cluster_count(len(source_engrams))
        assignments, centroids = self._kmeans(matrix, k)
        clusters = self._build_cluster_records(source_engrams, matrix, assignments, centroids)
        summary_writes = 0
        if not dry_run:
            if indexer is None:
                raise ValueError("indexer is required when dry_run=False")
            summary_engrams = self._build_summary_engrams(clusters, source_engrams, depth=depth)
            summary_writes = int(indexer.index(summary_engrams))
        report = HierarchyReport(
            dry_run=dry_run,
            model_name=self.model_name,
            engrams_scanned=len(source_engrams),
            cluster_count=len(clusters),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            summary_engram_writes=summary_writes,
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
        return max(1, min(n, int(math.ceil(n / self.target_cluster_size))))

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
            cluster_id = f"cluster_{cluster_idx:03d}"
            records.append(
                HierarchicalClusterRecord(
                    cluster_id=cluster_id,
                    member_ids=member_ids,
                    centroid_norm=float(np.linalg.norm(centroids[cluster_idx])),
                    representative_ids=representative_ids,
                    summary_prompt_preview=preview,
                )
            )
        records.sort(key=lambda record: (-len(record.member_ids), record.cluster_id))
        return records

    def _build_summary_engrams(
        self,
        clusters: List[HierarchicalClusterRecord],
        engrams: List[Engram],
        *,
        depth: int = 1,
    ) -> List[Engram]:
        by_id = {engram.id: engram for engram in engrams}
        summaries: List[Engram] = []
        for cluster in clusters:
            members = [by_id[eid] for eid in cluster.member_ids if eid in by_id]
            # depth=2 produces exactly one cluster: the corpus root.
            label = "root" if depth == 2 else cluster.cluster_id
            summary = self.synthesizer.synthesize(cluster_id=label, engrams=members)
            digest = sha256(("|".join(cluster.member_ids) + summary).encode("utf-8")).hexdigest()[:16]
            summary_id = f"summary_{label}_{digest}"
            cluster.summary_engram_id = summary_id
            cluster.summary_content = summary
            summaries.append(
                Engram(
                    id=summary_id,
                    content=summary,
                    source=f"derived://hierarchy/{label}",
                    confidence=1.0,
                    # Recursive lineage: depth-2 edges point at depth-1
                    # summary ids, whose own edges point at leaves.
                    edges=list(cluster.member_ids),
                    metadata={
                        "is_summary_engram": True,
                        "cluster_id": label,
                        "depth": depth,
                        "source_member_count": len(cluster.member_ids),
                        "representative_ids": list(cluster.representative_ids),
                        "synthesis_policy": "smc_or_join_truncate_v1",
                    },
                )
            )
        return summaries

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
