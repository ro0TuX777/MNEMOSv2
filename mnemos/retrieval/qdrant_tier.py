"""
Qdrant Retrieval Tier
======================

Semantic vector retrieval via Qdrant.
"""

import logging
from typing import Any, Dict, List, Optional

from mnemos.engram.model import Engram
from mnemos.retrieval.base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


class QdrantTier(BaseRetriever):
    """Qdrant-backed semantic retrieval tier."""

    def __init__(self, url: str = "http://localhost:6333",
                 collection_name: str = "mnemos_engrams",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 384,
                 gpu_device: str = "cuda"):
        self._url = url
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._embedding_dim = embedding_dim
        self._gpu_device = gpu_device
        self._client = None
        self._model = None
        self._initialize()

    def _initialize(self):
        """Initialize Qdrant client and ensure collection exists."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = QdrantClient(url=self._url, timeout=30)

            # Create collection if it doesn't exist
            collections = [c.name for c in self._client.get_collections().collections]
            if self._collection_name not in collections:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self._collection_name}")

            logger.info(
                f"✅ Qdrant tier initialized: {self._collection_name} ({self._url})"
            )
        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed: {e}")
            raise

    def _get_embedder(self):
        """Lazy-load the sentence transformer model (GPU-accelerated)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._embedding_model_name, device=self._gpu_device
                )
                logger.info(
                    f"Loaded embedding model: {self._embedding_model_name} "
                    f"(device={self._gpu_device})"
                )
            except Exception as e:
                logger.warning(f"GPU load failed ({e}), falling back to CPU")
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._embedding_model_name, device="cpu"
                )
        return self._model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        model = self._get_embedder()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def tier_name(self) -> str:
        return "qdrant"

    def index(self, engrams: List[Engram]) -> int:
        """Index engrams into Qdrant."""
        if not self._client or not engrams:
            return 0

        try:
            from qdrant_client.models import PointStruct

            texts = [e.content for e in engrams]
            embeddings = self._embed(texts)

            points = []
            for i, e in enumerate(engrams):
                payload = {
                    "content": e.content,
                    "source": e.source,
                    "confidence": e.confidence,
                    "neuro_tags": e.neuro_tags,
                    "created_at": e.created_at,
                    "edges": e.edges,
                }
                # Flatten app metadata into payload
                for k, v in e.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        payload[f"app_{k}"] = v

                points.append(PointStruct(
                    id=e.id,
                    vector=embeddings[i],
                    payload=payload,
                ))

            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            logger.debug(f"Indexed {len(points)} engrams into Qdrant")
            return len(points)

        except Exception as e:
            logger.error(f"Qdrant indexing failed: {e}")
            return 0

    def search(self, query: str, top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search Qdrant for relevant engrams."""
        if not self._client:
            return []

        try:
            query_vec = self._embed([query])[0]

            # Build Qdrant filter if provided
            query_filter = None
            if filters:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                conditions = []
                for k, v in filters.items():
                    conditions.append(
                        FieldCondition(key=k, match=MatchValue(value=v))
                    )
                query_filter = Filter(must=conditions)

            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vec,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

            search_results = []
            for hit in results:
                payload = hit.payload or {}
                neuro_tags = payload.get("neuro_tags", [])
                if isinstance(neuro_tags, str):
                    neuro_tags = [t for t in neuro_tags.split(",") if t]

                # Extract app metadata
                app_meta = {
                    k.removeprefix("app_"): v
                    for k, v in payload.items()
                    if k.startswith("app_")
                }

                engram = Engram(
                    id=str(hit.id),
                    content=payload.get("content", ""),
                    source=payload.get("source", ""),
                    confidence=float(payload.get("confidence", 1.0)),
                    neuro_tags=neuro_tags if isinstance(neuro_tags, list) else [],
                    created_at=payload.get("created_at", ""),
                    metadata=app_meta,
                    edges=payload.get("edges", []),
                )

                search_results.append(SearchResult(
                    engram=engram, score=hit.score, tier="qdrant"
                ))

            return search_results

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def delete(self, engram_ids: List[str]) -> int:
        """Delete engrams from Qdrant."""
        if not self._client or not engram_ids:
            return 0
        try:
            from qdrant_client.models import PointIdsList
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=engram_ids),
            )
            return len(engram_ids)
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")
            return 0

    def get(self, engram_id: str) -> Optional[Engram]:
        """Direct ID lookup in Qdrant."""
        if not self._client:
            return None
        try:
            results = self._client.retrieve(
                collection_name=self._collection_name,
                ids=[engram_id],
                with_payload=True,
            )
            if results:
                payload = results[0].payload or {}
                neuro_tags = payload.get("neuro_tags", [])
                if isinstance(neuro_tags, str):
                    neuro_tags = [t for t in neuro_tags.split(",") if t]

                app_meta = {
                    k.removeprefix("app_"): v
                    for k, v in payload.items()
                    if k.startswith("app_")
                }

                return Engram(
                    id=str(results[0].id),
                    content=payload.get("content", ""),
                    source=payload.get("source", ""),
                    confidence=float(payload.get("confidence", 1.0)),
                    neuro_tags=neuro_tags if isinstance(neuro_tags, list) else [],
                    created_at=payload.get("created_at", ""),
                    metadata=app_meta,
                    edges=payload.get("edges", []),
                )
        except Exception:
            pass
        return None

    def stats(self) -> Dict[str, Any]:
        """Get Qdrant tier statistics."""
        count = 0
        if self._client:
            try:
                info = self._client.get_collection(self._collection_name)
                count = info.points_count or 0
            except Exception:
                pass

        return {
            "tier": "qdrant",
            "document_count": count,
            "url": self._url,
            "collection": self._collection_name,
            "embedding_model": self._embedding_model_name,
            "embedding_dim": self._embedding_dim,
            "gpu_device": self._gpu_device,
        }
