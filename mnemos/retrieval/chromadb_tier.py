"""
ChromaDB Retrieval Tier
========================

Semantic chunk retrieval via ChromaDB.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mnemos.engram.model import Engram
from mnemos.retrieval.base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


class ChromaDBTier(BaseRetriever):
    """ChromaDB-backed semantic retrieval tier."""

    def __init__(self, persist_dir: str = "data/chroma",
                 collection_name: str = "mnemos_engrams",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._client = None
        self._collection = None
        self._ef = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._ef = SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=self._ef,
            )
            logger.info(f"✅ ChromaDB tier initialized: {self._collection_name} ({self._persist_dir})")
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {e}")
            raise

    @property
    def tier_name(self) -> str:
        return "chromadb"

    def index(self, engrams: List[Engram]) -> int:
        """Index engrams into ChromaDB."""
        if not self._collection or not engrams:
            return 0

        ids = [e.id for e in engrams]
        documents = [e.content for e in engrams]
        metadatas = []
        for e in engrams:
            meta = {
                "source": e.source,
                "confidence": e.confidence,
                "neuro_tags": ",".join(e.neuro_tags),
                "created_at": e.created_at,
            }
            # Flatten app metadata (ChromaDB requires flat string/int/float values)
            for k, v in e.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[f"app_{k}"] = v
            metadatas.append(meta)

        try:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.debug(f"Indexed {len(ids)} engrams into ChromaDB")
            return len(ids)
        except Exception as e:
            logger.error(f"ChromaDB indexing failed: {e}")
            return 0

    def search(self, query: str, top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search ChromaDB for relevant engrams."""
        if not self._collection:
            return []

        try:
            where = None
            if filters:
                # Convert to ChromaDB where clause
                where = {}
                for k, v in filters.items():
                    where[k] = v

            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count() or top_k),
                where=where if where else None,
            )

            search_results = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    content = results["documents"][0][i] if results["documents"] else ""

                    # Reconstruct neuro_tags from comma-separated string
                    neuro_tags = meta.pop("neuro_tags", "").split(",") if meta.get("neuro_tags") else []

                    engram = Engram(
                        id=doc_id,
                        content=content,
                        source=meta.pop("source", ""),
                        confidence=float(meta.pop("confidence", 1.0)),
                        neuro_tags=[t for t in neuro_tags if t],
                        created_at=meta.pop("created_at", ""),
                        metadata={k.removeprefix("app_"): v for k, v in meta.items() if k.startswith("app_")},
                    )

                    # ChromaDB returns distances — convert to similarity score
                    score = max(0.0, 1.0 - distance)

                    search_results.append(SearchResult(
                        engram=engram, score=score, tier="chromadb"
                    ))

            return search_results

        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    def delete(self, engram_ids: List[str]) -> int:
        """Delete engrams from ChromaDB."""
        if not self._collection or not engram_ids:
            return 0
        try:
            self._collection.delete(ids=engram_ids)
            return len(engram_ids)
        except Exception as e:
            logger.error(f"ChromaDB delete failed: {e}")
            return 0

    def get(self, engram_id: str) -> Optional[Engram]:
        """Direct ID lookup in ChromaDB."""
        if not self._collection:
            return None
        try:
            result = self._collection.get(ids=[engram_id], include=["documents", "metadatas"])
            if result["ids"]:
                meta = result["metadatas"][0] if result["metadatas"] else {}
                neuro_tags = meta.pop("neuro_tags", "").split(",") if meta.get("neuro_tags") else []
                return Engram(
                    id=engram_id,
                    content=result["documents"][0] if result["documents"] else "",
                    source=meta.pop("source", ""),
                    confidence=float(meta.pop("confidence", 1.0)),
                    neuro_tags=[t for t in neuro_tags if t],
                    created_at=meta.pop("created_at", ""),
                    metadata={k.removeprefix("app_"): v for k, v in meta.items() if k.startswith("app_")},
                )
        except Exception:
            pass
        return None

    def stats(self) -> Dict[str, Any]:
        """Get ChromaDB tier statistics."""
        count = self._collection.count() if self._collection else 0
        return {
            "tier": "chromadb",
            "document_count": count,
            "persist_dir": self._persist_dir,
            "collection": self._collection_name,
            "embedding_model": self._embedding_model,
        }
