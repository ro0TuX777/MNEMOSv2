"""
LanceDB Retrieval Tier
=======================

Hybrid keyword + vector queries via LanceDB.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mnemos.engram.model import Engram
from mnemos.retrieval.base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


class LanceDBTier(BaseRetriever):
    """LanceDB-backed hybrid retrieval tier."""

    def __init__(self, db_dir: str = "data/lance",
                 table_name: str = "mnemos_engrams",
                 embedding_dim: int = 384):
        self._db_dir = db_dir
        self._table_name = table_name
        self._embedding_dim = embedding_dim
        self._db = None
        self._table = None
        self._model = None
        self._initialize()

    def _initialize(self):
        """Initialize LanceDB connection."""
        try:
            import lancedb

            Path(self._db_dir).mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(self._db_dir)

            # Check if table already exists
            existing = self._db.table_names()
            if self._table_name in existing:
                self._table = self._db.open_table(self._table_name)

            logger.info(f"✅ LanceDB tier initialized: {self._table_name} ({self._db_dir})")
        except Exception as e:
            logger.error(f"❌ LanceDB initialization failed: {e}")
            raise

    def _get_embedder(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise
        return self._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        model = self._get_embedder()
        return model.encode(texts, normalize_embeddings=True)

    @property
    def tier_name(self) -> str:
        return "lancedb"

    def index(self, engrams: List[Engram]) -> int:
        """Index engrams into LanceDB."""
        if not self._db or not engrams:
            return 0

        try:
            texts = [e.content for e in engrams]
            embeddings = self._embed(texts)

            records = []
            for i, e in enumerate(engrams):
                records.append({
                    "id": e.id,
                    "content": e.content,
                    "vector": embeddings[i].tolist(),
                    "source": e.source,
                    "confidence": e.confidence,
                    "neuro_tags": ",".join(e.neuro_tags),
                    "created_at": e.created_at,
                })

            import pyarrow as pa
            if self._table is None:
                self._table = self._db.create_table(self._table_name, records)
            else:
                self._table.add(records)

            logger.debug(f"Indexed {len(records)} engrams into LanceDB")
            return len(records)
        except Exception as e:
            logger.error(f"LanceDB indexing failed: {e}")
            return 0

    def search(self, query: str, top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Hybrid search in LanceDB."""
        if not self._table:
            return []

        try:
            query_vec = self._embed([query])[0]

            results = (
                self._table
                .search(query_vec)
                .limit(top_k)
                .to_list()
            )

            search_results = []
            for row in results:
                neuro_tags = row.get("neuro_tags", "").split(",") if row.get("neuro_tags") else []
                engram = Engram(
                    id=row["id"],
                    content=row.get("content", ""),
                    source=row.get("source", ""),
                    confidence=float(row.get("confidence", 1.0)),
                    neuro_tags=[t for t in neuro_tags if t],
                    created_at=row.get("created_at", ""),
                )
                # LanceDB returns _distance — convert to similarity
                distance = row.get("_distance", 0.0)
                score = max(0.0, 1.0 - distance)

                search_results.append(SearchResult(
                    engram=engram, score=score, tier="lancedb"
                ))

            return search_results

        except Exception as e:
            logger.error(f"LanceDB search failed: {e}")
            return []

    def delete(self, engram_ids: List[str]) -> int:
        """Delete engrams from LanceDB."""
        if not self._table or not engram_ids:
            return 0
        try:
            id_list = ", ".join(f"'{eid}'" for eid in engram_ids)
            self._table.delete(f"id IN ({id_list})")
            return len(engram_ids)
        except Exception as e:
            logger.error(f"LanceDB delete failed: {e}")
            return 0

    def stats(self) -> Dict[str, Any]:
        """Get LanceDB tier statistics."""
        count = len(self._table) if self._table else 0
        return {
            "tier": "lancedb",
            "document_count": count,
            "db_dir": self._db_dir,
            "table": self._table_name,
            "embedding_dim": self._embedding_dim,
        }
