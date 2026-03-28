"""
pgvector Retrieval Tier
========================

PostgreSQL-native vector retrieval via pgvector extension.
Used by the Governance Native deployment profile.

Vectors live inside the same PostgreSQL instance as the forensic ledger,
enabling ANN retrieval combined with SQL WHERE clauses on metadata
in a single query.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from mnemos.engram.model import Engram
from mnemos.retrieval.base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


class PgvectorTier(BaseRetriever):
    """pgvector-backed retrieval tier (Governance Native profile)."""

    def __init__(self, dsn: str,
                 table_name: str = "mnemos_vectors",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 384,
                 gpu_device: str = "cuda"):
        self._dsn = dsn
        self._table_name = table_name
        self._embedding_model_name = embedding_model
        self._embedding_dim = embedding_dim
        self._gpu_device = gpu_device
        self._pool = None
        self._model = None
        self._initialize()

    def _initialize(self):
        """Create connection pool, install pgvector extension, and ensure table exists."""
        try:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                self._dsn,
                min_size=2,
                max_size=10,
                open=True,
            )

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    # Enable pgvector extension
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                    # Create engrams table with vector column
                    cur.execute(f'''
                        CREATE TABLE IF NOT EXISTS {self._table_name} (
                            id TEXT PRIMARY KEY,
                            content TEXT NOT NULL,
                            embedding vector({self._embedding_dim}),
                            source TEXT DEFAULT '',
                            confidence DOUBLE PRECISION DEFAULT 1.0,
                            neuro_tags TEXT[] DEFAULT '{{}}',
                            created_at TEXT DEFAULT '',
                            edges TEXT[] DEFAULT '{{}}',
                            metadata JSONB DEFAULT '{{}}'::jsonb
                        )
                    ''')

                    # Create HNSW index for ANN queries
                    cur.execute(f'''
                        CREATE INDEX IF NOT EXISTS idx_{self._table_name}_hnsw
                        ON {self._table_name}
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 200)
                    ''')

                conn.commit()

            logger.info(
                f"✅ pgvector tier initialized: {self._table_name} "
                f"(dim={self._embedding_dim}, HNSW)"
            )
        except Exception as e:
            logger.error(f"❌ pgvector initialization failed: {e}")
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
        return "pgvector"

    def index(self, engrams: List[Engram]) -> int:
        """Index engrams into pgvector table."""
        if not self._pool or not engrams:
            return 0

        try:
            texts = [e.content for e in engrams]
            embeddings = self._embed(texts)
            indexed = 0

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    for i, e in enumerate(engrams):
                        embedding_str = "[" + ",".join(str(v) for v in embeddings[i]) + "]"
                        meta_json = json.dumps(e.metadata)

                        cur.execute(f'''
                            INSERT INTO {self._table_name}
                                (id, content, embedding, source, confidence,
                                 neuro_tags, created_at, edges, metadata)
                            VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s, %s::jsonb)
                            ON CONFLICT (id) DO UPDATE SET
                                content = EXCLUDED.content,
                                embedding = EXCLUDED.embedding,
                                source = EXCLUDED.source,
                                confidence = EXCLUDED.confidence,
                                neuro_tags = EXCLUDED.neuro_tags,
                                edges = EXCLUDED.edges,
                                metadata = EXCLUDED.metadata
                        ''', (
                            e.id, e.content, embedding_str, e.source,
                            e.confidence, e.neuro_tags, e.created_at,
                            e.edges, meta_json,
                        ))
                        indexed += 1

                conn.commit()

            logger.debug(f"Indexed {indexed} engrams into pgvector")
            return indexed

        except Exception as e:
            logger.error(f"pgvector indexing failed: {e}")
            return 0

    def search(self, query: str, top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search pgvector for relevant engrams.

        Supports metadata filtering via SQL WHERE clauses —
        the core value proposition of the Governance Native profile.
        Filters can target:
          - Top-level fields: {"source": "..."}, {"confidence_min": 0.8}
          - Metadata JSONB: {"metadata.department": "finance"}
          - Neuro-tags: {"neuro_tag": "compliance"}
        """
        if not self._pool:
            return []

        try:
            query_vec = self._embed([query])[0]
            embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

            # Build WHERE clauses from filters
            conditions = []
            params: list = [embedding_str]

            if filters:
                for key, value in filters.items():
                    if key == "confidence_min":
                        conditions.append("confidence >= %s")
                        params.append(float(value))
                    elif key == "neuro_tag":
                        conditions.append("%s = ANY(neuro_tags)")
                        params.append(str(value))
                    elif key.startswith("metadata."):
                        json_key = key.split(".", 1)[1]
                        conditions.append("metadata->>%s = %s")
                        params.append(json_key)
                        params.append(str(value))
                    elif key == "source":
                        conditions.append("source = %s")
                        params.append(str(value))
                    else:
                        # Generic metadata JSONB filter
                        conditions.append("metadata->>%s = %s")
                        params.append(key)
                        params.append(str(value))

            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

            params.append(top_k)

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f'''
                        SELECT id, content, source, confidence,
                               neuro_tags, created_at, edges, metadata,
                               1 - (embedding <=> %s::vector) AS score
                        FROM {self._table_name}
                        {where_clause}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    ''', params + [embedding_str])

                    results = []
                    for row in cur.fetchall():
                        engram = Engram(
                            id=row[0],
                            content=row[1],
                            source=row[2] or "",
                            confidence=float(row[3]) if row[3] else 1.0,
                            neuro_tags=list(row[4]) if row[4] else [],
                            created_at=row[5] or "",
                            edges=list(row[6]) if row[6] else [],
                            metadata=row[7] if isinstance(row[7], dict) else {},
                        )
                        results.append(SearchResult(
                            engram=engram,
                            score=float(row[8]) if row[8] else 0.0,
                            tier="pgvector",
                        ))

                    return results

        except Exception as e:
            logger.error(f"pgvector search failed: {e}")
            return []

    def delete(self, engram_ids: List[str]) -> int:
        """Delete engrams from pgvector table."""
        if not self._pool or not engram_ids:
            return 0
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self._table_name} WHERE id = ANY(%s)",
                        (engram_ids,),
                    )
                    deleted = cur.rowcount
                conn.commit()
            return deleted
        except Exception as e:
            logger.error(f"pgvector delete failed: {e}")
            return 0

    def get(self, engram_id: str) -> Optional[Engram]:
        """Direct ID lookup in pgvector table."""
        if not self._pool:
            return None
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f'''
                        SELECT id, content, source, confidence,
                               neuro_tags, created_at, edges, metadata
                        FROM {self._table_name}
                        WHERE id = %s
                    ''', (engram_id,))
                    row = cur.fetchone()
                    if row:
                        return Engram(
                            id=row[0],
                            content=row[1],
                            source=row[2] or "",
                            confidence=float(row[3]) if row[3] else 1.0,
                            neuro_tags=list(row[4]) if row[4] else [],
                            created_at=row[5] or "",
                            edges=list(row[6]) if row[6] else [],
                            metadata=row[7] if isinstance(row[7], dict) else {},
                        )
        except Exception:
            pass
        return None

    def stats(self) -> Dict[str, Any]:
        """Get pgvector tier statistics."""
        count = 0
        table_size_mb = 0.0
        index_size_mb = 0.0

        if self._pool:
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"SELECT COUNT(*) FROM {self._table_name}"
                        )
                        count = cur.fetchone()[0]

                        cur.execute(
                            f"SELECT pg_total_relation_size(%s)",
                            (self._table_name,),
                        )
                        total_bytes = cur.fetchone()[0]
                        table_size_mb = round(total_bytes / (1024 * 1024), 2)

                        cur.execute(
                            f"SELECT pg_indexes_size(%s)",
                            (self._table_name,),
                        )
                        idx_bytes = cur.fetchone()[0]
                        index_size_mb = round(idx_bytes / (1024 * 1024), 2)
            except Exception:
                pass

        return {
            "tier": "pgvector",
            "document_count": count,
            "table_size_mb": table_size_mb,
            "index_size_mb": index_size_mb,
            "index_type": "hnsw",
            "embedding_model": self._embedding_model_name,
            "embedding_dim": self._embedding_dim,
            "gpu_device": self._gpu_device,
        }
