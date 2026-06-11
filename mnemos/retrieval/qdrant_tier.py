"""
Qdrant Retrieval Tier
======================

Semantic vector retrieval via Qdrant.

Targets Qdrant server >= v1.17.0 and qdrant-client >= 1.13.
Uses the query_points() API exclusively (search() is deprecated and
will be removed in Qdrant v1.18).
"""

import logging
import sys
import uuid as _uuid
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from mnemos.engram.model import Engram
from mnemos.retrieval.base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)

NOMIC_V15_MODEL_MARKER = "nomic-embed-text-v1.5"
NOMIC_DOC_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "
NOMIC_MRL_DIM = 64
NOMIC_FULL_DIM = 768


def _ensure_transformer_runtime_compat() -> None:
    """Patch slim Python builds for torch._dynamo import compatibility."""
    if not hasattr(sys, "get_int_max_str_digits"):
        def get_int_max_str_digits() -> int:
            return 4300

        sys.get_int_max_str_digits = get_int_max_str_digits  # type: ignore[attr-defined]

    if not hasattr(sys, "set_int_max_str_digits"):
        def set_int_max_str_digits(maxdigits: int) -> None:
            return None

        sys.set_int_max_str_digits = set_int_max_str_digits  # type: ignore[attr-defined]


def _mrl_slice(matrix: Any, dim: int = NOMIC_MRL_DIM) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = matrix.mean(axis=1, keepdims=True)
    var = matrix.var(axis=1, keepdims=True)
    normed = (matrix - mean) / np.sqrt(var + 1e-5)
    sliced = normed[:, :dim]
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    return sliced / np.clip(norms, 1e-12, None)


class QdrantTier(BaseRetriever):
    """Qdrant-backed semantic retrieval tier."""

    # Qdrant server version this tier targets.
    QDRANT_MIN_VERSION = "1.17.0"

    def __init__(self, url: str = "http://localhost:6333",
                 collection_name: str = "mnemos_engrams",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 384,
                 gpu_device: str = "cuda"):
        self._url = url
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._embedding_dim = NOMIC_FULL_DIM if self._uses_nomic_mrl_model(embedding_model) else embedding_dim
        self._gpu_device = gpu_device
        self._client = None
        self._model = None
        self._initialize()

    @staticmethod
    def _uses_nomic_mrl_model(model_name: str) -> bool:
        return NOMIC_V15_MODEL_MARKER in model_name

    @property
    def _uses_nomic_mrl(self) -> bool:
        return self._uses_nomic_mrl_model(self._embedding_model_name)

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

            # Ensure full-text index on 'content' for hybrid RRF fusion.
            self._text_index_ready = self._ensure_text_index()

            logger.info(
                f"✅ Qdrant tier initialized: {self._collection_name} ({self._url})"
                f" [text_index={'ready' if self._text_index_ready else 'unavailable'}]"
            )
        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed: {e}")
            raise

    def _ensure_text_index(self) -> bool:
        """Create a full-text payload index on 'content' (idempotent).

        Returns True if the index exists or was successfully created,
        False if creation failed (e.g. older Qdrant server).
        """
        try:
            from qdrant_client.models import (
                PayloadSchemaType,
                TextIndexParams,
                TextIndexType,
                TokenizerType,
            )

            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="content",
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
                ),
            )
            logger.info("Created text index on 'content' payload field")
            return True
        except Exception as e:
            # Already exists → fine. Unsupported server → fallback to Python hybrid.
            if "already exists" in str(e).lower() or "conflict" in str(e).lower():
                logger.debug("Text index on 'content' already exists")
                return True
            logger.warning(f"Text index creation failed (hybrid fusion unavailable): {e}")
            return False

    def _get_embedder(self):
        """Lazy-load the sentence transformer model (GPU-accelerated)."""
        if self._model is None:
            try:
                _ensure_transformer_runtime_compat()
                from sentence_transformers import SentenceTransformer
                kwargs = {"device": self._gpu_device}
                if self._uses_nomic_mrl:
                    kwargs["trust_remote_code"] = True
                self._model = SentenceTransformer(self._embedding_model_name, **kwargs)
                logger.info(
                    f"Loaded embedding model: {self._embedding_model_name} "
                    f"(device={self._gpu_device})"
                )
            except Exception as e:
                logger.warning(f"GPU load failed ({e}), falling back to CPU")
                _ensure_transformer_runtime_compat()
                from sentence_transformers import SentenceTransformer
                kwargs = {"device": "cpu"}
                if self._uses_nomic_mrl:
                    kwargs["trust_remote_code"] = True
                self._model = SentenceTransformer(self._embedding_model_name, **kwargs)
        return self._model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        return self._embed_query(texts).tolist()

    def _embed_documents(self, texts: List[str]) -> np.ndarray:
        model = self._get_embedder()
        if self._uses_nomic_mrl:
            texts = [NOMIC_DOC_PREFIX + text for text in texts]
        return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def _embed_query(self, texts: List[str]) -> np.ndarray:
        model = self._get_embedder()
        if self._uses_nomic_mrl:
            texts = [NOMIC_QUERY_PREFIX + text for text in texts]
        return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def _point_vector(self, embedding: np.ndarray):
        if not self._uses_nomic_mrl:
            return embedding.tolist()
        dense_64 = _mrl_slice(embedding.reshape(1, -1), NOMIC_MRL_DIM)[0]
        return {
            "dense_64": dense_64.tolist(),
            "dense_768": embedding.tolist(),
        }

    @staticmethod
    def _make_trace_id() -> str:
        """Generate a unique request tracing ID for audit log correlation."""
        return str(_uuid.uuid4())

    def _to_point_id(self, engram_id: str):
        """
        Convert arbitrary engram IDs into Qdrant-compatible point IDs.

        Qdrant accepts unsigned integers or UUIDs only. Benchmark corpus IDs are
        short hashes, so we deterministically map non-UUID/non-integer IDs to
        UUIDv5 and store the original ID in payload for round-trip fidelity.
        """
        sid = str(engram_id)

        # Accept canonical UUIDs directly.
        try:
            return str(_uuid.UUID(sid))
        except Exception:
            pass

        # Accept unsigned integer IDs.
        if sid.isdigit():
            try:
                return int(sid)
            except Exception:
                pass

        # Deterministic fallback for arbitrary string IDs.
        return str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"mnemos:{sid}"))

    @property
    def tier_name(self) -> str:
        return "qdrant"

    def index(
        self,
        engrams: List[Engram],
        update_mode: Literal["upsert", "update", "insert"] = "upsert",
    ) -> int:
        """
        Index engrams into Qdrant.

        Args:
            engrams: List of engrams to store.
            update_mode: Qdrant v1.17+ update mode:
                - ``upsert`` — insert or overwrite (default, backward-compat)
                - ``update`` — update existing points only
                - ``insert`` — insert new points only (skip existing)
        """
        if not self._client or not engrams:
            return 0

        try:
            from qdrant_client.models import PointStruct

            texts = [e.content for e in engrams]
            embeddings = self._embed_documents(texts)

            points = []
            for i, e in enumerate(engrams):
                payload = {
                    "_mnemos_id": e.id,
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

                if e.governance is not None:
                    governance_payload = e.governance.to_dict()
                    for k, v in governance_payload.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            payload[f"gov_{k}"] = v
                        elif isinstance(v, list) and all(
                            isinstance(item, str) for item in v
                        ):
                            payload[f"gov_{k}"] = v

                points.append(PointStruct(
                    id=self._to_point_id(e.id),
                    vector=self._point_vector(embeddings[i]),
                    payload=payload,
                ))

            trace_id = self._make_trace_id()
            logger.debug(
                f"Indexing {len(points)} engrams into Qdrant "
                f"(mode={update_mode}, trace={trace_id})"
            )

            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            return len(points)

        except Exception as e:
            logger.error(f"Qdrant indexing failed: {e}")
            return 0

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        query_vector: Optional[Any] = None,
    ) -> List[SearchResult]:
        """Search Qdrant for relevant engrams.

        Uses the ``query_points()`` API (Qdrant >= 1.17).  The legacy
        ``search()`` method is deprecated and will be removed in v1.18.
        """
        if not self._client:
            return []

        try:
            # W6/W7 budget-plan sentinels (router-injected reserved keys)
            filters = dict(filters) if filters else {}
            oversample = float(filters.pop("__mrl_oversample__", 3.0))
            hnsw_ef = filters.pop("__hnsw_ef__", None)
            prefetch_only = bool(filters.pop("__prefetch_only__", False))

            query_vec = (
                np.asarray(query_vector, dtype=np.float32).reshape(-1)
                if query_vector is not None and np.asarray(query_vector).size == self._embedding_dim
                else self._embed_query([query])[0]
            )

            # Build Qdrant filter if provided
            query_filter = self._build_filter(filters)

            search_params = None
            if hnsw_ef is not None:
                from qdrant_client.models import SearchParams

                search_params = SearchParams(hnsw_ef=int(hnsw_ef))

            trace_id = self._make_trace_id()
            logger.debug(
                f"Qdrant search: top_k={top_k}, has_filter={query_filter is not None}, "
                f"trace={trace_id}"
            )

            if self._uses_nomic_mrl and prefetch_only:
                # Budget floor: coarse 64-dim lane only, no full-dim rescore.
                coarse = _mrl_slice(query_vec.reshape(1, -1), NOMIC_MRL_DIM)[0]
                response = self._client.query_points(
                    collection_name=self._collection_name,
                    query=coarse.tolist(),
                    using="dense_64",
                    limit=top_k,
                    query_filter=query_filter,
                    search_params=search_params,
                    with_payload=True,
                )
            elif self._uses_nomic_mrl:
                from qdrant_client.models import Prefetch

                coarse = _mrl_slice(query_vec.reshape(1, -1), NOMIC_MRL_DIM)[0]
                response = self._client.query_points(
                    collection_name=self._collection_name,
                    prefetch=Prefetch(
                        query=coarse.tolist(),
                        using="dense_64",
                        limit=max(int(top_k * oversample), top_k),
                        filter=query_filter,
                        params=search_params,
                    ),
                    query=query_vec.tolist(),
                    using="dense_768",
                    limit=top_k,
                    with_payload=True,
                )
            else:
                # query_points() is the canonical API from qdrant-client >= 1.13.
                response = self._client.query_points(
                    collection_name=self._collection_name,
                    query=query_vec.tolist(),
                    limit=top_k,
                    query_filter=query_filter,
                    search_params=search_params,
                    with_payload=True,
                )
            results = getattr(response, "points", response)

            return [self._hit_to_result(hit) for hit in results]

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    # ──────────────────────── filter builder ───────────────────

    @staticmethod
    def _build_filter(filters: Optional[Dict[str, Any]]):
        """Translate MNEMOS metadata filters into a Qdrant Filter object.

        The reserved key ``__exclude_derived__`` (injected by the retrieval
        router for default search paths) becomes a server-side must_not
        condition on the derived-fact payload flag (PIT-1 primary filter).
        """
        filters = dict(filters) if filters else {}
        exclude_derived = bool(filters.pop("__exclude_derived__", False))
        exclude_summaries = bool(filters.pop("__exclude_summaries__", False))
        # Budget-plan sentinels are consumed in search(); drop them defensively
        # for callers that pass raw router filters (e.g. hybrid fusion).
        for reserved in ("__mrl_oversample__", "__hnsw_ef__", "__prefetch_only__"):
            filters.pop(reserved, None)
        if not filters and not exclude_derived and not exclude_summaries:
            return None

        from qdrant_client.models import (
            Filter,
            FieldCondition,
            MatchValue,
            Range,
        )

        must_not = []
        if exclude_derived:
            # Engram metadata is flattened into payload as app_<key>.
            must_not.append(
                FieldCondition(key="app_is_derived_fact", match=MatchValue(value=True))
            )
        if exclude_summaries:
            # Phase 9b: summary engrams are a protected tier — only the
            # CLASS_C summary layer may retrieve them.
            must_not.append(
                FieldCondition(key="app_is_summary_engram", match=MatchValue(value=True))
            )
        must_not = must_not or None

        conditions = []
        for k, v in filters.items():
            if k == "metadata.timestamp_epoch_min":
                conditions.append(
                    FieldCondition(key="app_timestamp_epoch", range=Range(gte=float(v)))
                )
                continue
            if k == "metadata.timestamp_epoch_max":
                conditions.append(
                    FieldCondition(key="app_timestamp_epoch", range=Range(lte=float(v)))
                )
                continue

            # Benchmark metadata filters are expressed as metadata.<key>,
            # while Qdrant payload stores metadata fields as app_<key>.
            if k.startswith("metadata."):
                k = f"app_{k.split('.', 1)[1]}"

            if k == "confidence_min":
                conditions.append(
                    FieldCondition(key="confidence", range=Range(gte=float(v)))
                )
                continue

            conditions.append(
                FieldCondition(key=k, match=MatchValue(value=v))
            )

        return Filter(must=conditions or None, must_not=must_not)

    # ──────────────────── result conversion ───────────────────

    @staticmethod
    def _hit_to_result(hit) -> SearchResult:
        """Convert a Qdrant ScoredPoint into a MNEMOS SearchResult."""
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
            id=str(payload.get("_mnemos_id", hit.id)),
            content=payload.get("content", ""),
            source=payload.get("source", ""),
            confidence=float(payload.get("confidence", 1.0)),
            neuro_tags=neuro_tags if isinstance(neuro_tags, list) else [],
            created_at=payload.get("created_at", ""),
            metadata=app_meta,
            edges=payload.get("edges", []),
        )
        governance_payload = {
            k.removeprefix("gov_"): v
            for k, v in payload.items()
            if k.startswith("gov_")
        }
        if governance_payload:
            from mnemos.governance.models.memory_state import GovernanceMeta

            engram.governance = GovernanceMeta.from_dict(governance_payload)

        return SearchResult(engram=engram, score=hit.score, tier="qdrant")

    def delete(self, engram_ids: List[str]) -> int:
        """Delete engrams from Qdrant."""
        if not self._client or not engram_ids:
            return 0
        try:
            from qdrant_client.models import PointIdsList
            point_ids = [self._to_point_id(eid) for eid in engram_ids]
            trace_id = self._make_trace_id()
            logger.debug(
                f"Deleting {len(point_ids)} points from Qdrant (trace={trace_id})"
            )
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=point_ids),
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
            point_id = self._to_point_id(engram_id)
            results = self._client.retrieve(
                collection_name=self._collection_name,
                ids=[point_id],
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

                governance_payload = {
                    k.removeprefix("gov_"): v
                    for k, v in payload.items()
                    if k.startswith("gov_")
                }
                governance = None
                if governance_payload:
                    from mnemos.governance.models.memory_state import GovernanceMeta

                    governance = GovernanceMeta.from_dict(governance_payload)

                return Engram(
                    id=str(payload.get("_mnemos_id", results[0].id)),
                    content=payload.get("content", ""),
                    source=payload.get("source", ""),
                    confidence=float(payload.get("confidence", 1.0)),
                    neuro_tags=neuro_tags if isinstance(neuro_tags, list) else [],
                    created_at=payload.get("created_at", ""),
                    metadata=app_meta,
                    edges=payload.get("edges", []),
                    governance=governance,
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
