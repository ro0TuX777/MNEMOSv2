"""
PostgreSQL full-text lexical retrieval tier.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from mnemos.engram.model import Engram
from mnemos.retrieval.base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)


class LexicalTier(BaseRetriever):
    """Postgres FTS lexical retrieval lane used by hybrid mode."""

    def __init__(self, dsn: str, table_name: str = "mnemos_lexical"):
        self._dsn = dsn
        self._table_name = table_name
        self._pool = None
        self._initialize()

    @property
    def tier_name(self) -> str:
        return "lexical"

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("\x00", "")

    def _clean_metadata(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                self._clean_text(k): self._clean_metadata(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._clean_metadata(v) for v in value]
        if isinstance(value, tuple):
            return [self._clean_metadata(v) for v in value]
        if isinstance(value, str):
            return self._clean_text(value)
        return value

    def _initialize(self):
        try:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                self._dsn,
                min_size=1,
                max_size=6,
                open=True,
            )

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

                    cur.execute(f'''
                        CREATE TABLE IF NOT EXISTS {self._table_name} (
                            id TEXT PRIMARY KEY,
                            content TEXT NOT NULL,
                            source TEXT DEFAULT '',
                            confidence DOUBLE PRECISION DEFAULT 1.0,
                            neuro_tags TEXT[] DEFAULT '{{}}',
                            created_at TEXT DEFAULT '',
                            edges TEXT[] DEFAULT '{{}}',
                            metadata JSONB DEFAULT '{{}}'::jsonb
                        )
                    ''')

                    cur.execute(f'''
                        CREATE INDEX IF NOT EXISTS idx_{self._table_name}_fts
                        ON {self._table_name}
                        USING GIN (to_tsvector('simple', coalesce(content, '')))
                    ''')

                conn.commit()

            logger.info("Lexical tier initialized on table %s", self._table_name)
        except Exception as e:
            logger.error("Lexical tier initialization failed: %s", e)
            raise

    def index(self, engrams: List[Engram]) -> int:
        if not self._pool or not engrams:
            return 0

        indexed = 0
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    for e in engrams:
                        clean_meta = self._clean_metadata(e.metadata)
                        cur.execute(f'''
                            INSERT INTO {self._table_name}
                                (id, content, source, confidence,
                                 neuro_tags, created_at, edges, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                            ON CONFLICT (id) DO UPDATE SET
                                content = EXCLUDED.content,
                                source = EXCLUDED.source,
                                confidence = EXCLUDED.confidence,
                                neuro_tags = EXCLUDED.neuro_tags,
                                created_at = EXCLUDED.created_at,
                                edges = EXCLUDED.edges,
                                metadata = EXCLUDED.metadata
                        ''', (
                            self._clean_text(e.id),
                            self._clean_text(e.content),
                            self._clean_text(e.source),
                            float(e.confidence),
                            [self._clean_text(t) for t in e.neuro_tags],
                            self._clean_text(e.created_at),
                            [self._clean_text(t) for t in e.edges],
                            json.dumps(clean_meta),
                        ))
                        indexed += 1
                conn.commit()

            return indexed
        except Exception as e:
            logger.error("Lexical indexing failed: %s", e)
            return 0

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if not self._pool:
            return []

        try:
            conditions = []
            params: List[Any] = []

            if filters:
                for key, value in filters.items():
                    if key == "source":
                        conditions.append("source = %s")
                        params.append(str(value))
                    elif key == "confidence_min":
                        conditions.append("confidence >= %s")
                        params.append(float(value))
                    elif key == "neuro_tag":
                        conditions.append("%s = ANY(neuro_tags)")
                        params.append(str(value))
                    elif key == "metadata.timestamp_epoch_min":
                        conditions.append("(metadata->>%s)::bigint >= %s")
                        params.append("timestamp_epoch")
                        params.append(int(value))
                    elif key == "metadata.timestamp_epoch_max":
                        conditions.append("(metadata->>%s)::bigint <= %s")
                        params.append("timestamp_epoch")
                        params.append(int(value))
                    elif key.startswith("metadata."):
                        meta_key = key.split(".", 1)[1]
                        conditions.append("metadata->>%s = %s")
                        params.append(meta_key)
                        params.append(str(value))
                    else:
                        conditions.append("metadata->>%s = %s")
                        params.append(key)
                        params.append(str(value))

            where_parts = list(conditions)
            where_parts.append("q.query @@ to_tsvector('simple', coalesce(content, ''))")
            where_clause = "WHERE " + " AND ".join(where_parts)

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f'''
                        WITH q AS (
                            SELECT websearch_to_tsquery('simple', %s) AS query
                        )
                        SELECT id, content, source, confidence,
                               neuro_tags, created_at, edges, metadata,
                               ts_rank_cd(to_tsvector('simple', coalesce(content, '')), q.query) AS lexical_score
                        FROM {self._table_name}, q
                        {where_clause}
                        ORDER BY lexical_score DESC, id ASC
                        LIMIT %s
                    ''', [query] + params + [int(top_k)])

                    rows = cur.fetchall()

            results: List[SearchResult] = []
            for row in rows:
                engram = Engram(
                    id=row[0],
                    content=row[1],
                    source=row[2] or "",
                    confidence=float(row[3]) if row[3] is not None else 1.0,
                    neuro_tags=list(row[4]) if row[4] else [],
                    created_at=row[5] or "",
                    edges=list(row[6]) if row[6] else [],
                    metadata=row[7] if isinstance(row[7], dict) else {},
                )
                results.append(
                    SearchResult(
                        engram=engram,
                        score=float(row[8]) if row[8] else 0.0,
                        tier=self.tier_name,
                    )
                )

            return results

        except Exception as e:
            logger.error("Lexical search failed: %s", e)
            return []

    def delete(self, engram_ids: List[str]) -> int:
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
            logger.error("Lexical delete failed: %s", e)
            return 0

    def get(self, engram_id: str) -> Optional[Engram]:
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
            if not row:
                return None
            return Engram(
                id=row[0],
                content=row[1],
                source=row[2] or "",
                confidence=float(row[3]) if row[3] is not None else 1.0,
                neuro_tags=list(row[4]) if row[4] else [],
                created_at=row[5] or "",
                edges=list(row[6]) if row[6] else [],
                metadata=row[7] if isinstance(row[7], dict) else {},
            )
        except Exception:
            return None

    def stats(self) -> Dict[str, Any]:
        if not self._pool:
            return {
                "tier": self.tier_name,
                "document_count": 0,
                "table": self._table_name,
                "available": False,
            }

        count = 0
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {self._table_name}")
                    count = int(cur.fetchone()[0])
        except Exception:
            pass

        return {
            "tier": self.tier_name,
            "document_count": count,
            "table": self._table_name,
            "available": True,
        }
