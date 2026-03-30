"""
PostgreSQL Forensic Ledger — Immutable Audit Trail
====================================================

Production-grade audit trail backed by PostgreSQL with
full-text search via tsvector/GIN indexes.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PostgresLedger:
    """
    PostgreSQL-backed immutable audit trail for MNEMOS operations.

    Replaces the SQLite-based ForensicLedger for production deployments
    requiring concurrent writes, connection pooling, and scalable
    full-text search.
    """

    def __init__(self, dsn: str,
                 pool_min: int = 2,
                 pool_max: int = 10,
                 on_critical_event: Optional[Callable] = None):
        """
        Args:
            dsn: PostgreSQL connection string (e.g. postgresql://user:pass@host:5432/db).
            pool_min: Minimum connections in the pool.
            pool_max: Maximum connections in the pool.
            on_critical_event: Optional callback invoked on critical events.
        """
        self.dsn = dsn
        self._on_critical = on_critical_event
        self._pool = None
        self._initialize(pool_min, pool_max)
        logger.info(f"⚖️ PostgreSQL Forensic Ledger initialized")

    def _initialize(self, pool_min: int, pool_max: int):
        """Create connection pool and build schema."""
        import psycopg
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(
            self.dsn,
            min_size=pool_min,
            max_size=pool_max,
            open=True,
        )

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # Main transactions table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS transactions (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        component TEXT NOT NULL,
                        action TEXT NOT NULL,
                        session_id TEXT,
                        status TEXT DEFAULT 'success',
                        latency DOUBLE PRECISION DEFAULT 0.0,
                        raw_data TEXT,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        search_vector tsvector
                            GENERATED ALWAYS AS (
                                to_tsvector('english', COALESCE(raw_data, '') || ' ' || COALESCE(component, ''))
                            ) STORED
                    )
                ''')

                # GIN index for full-text search
                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_transactions_fts
                    ON transactions USING GIN (search_vector)
                ''')

                # Index for common query patterns
                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_transactions_timestamp
                    ON transactions (timestamp DESC)
                ''')
                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_transactions_component
                    ON transactions (component)
                ''')
                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_transactions_status
                    ON transactions (status)
                ''')

                # High-level events table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS events (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        details TEXT,
                        source TEXT DEFAULT 'system',
                        importance DOUBLE PRECISION DEFAULT 0.5
                    )
                ''')

                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_events_timestamp
                    ON events (timestamp DESC)
                ''')

            conn.commit()

    # ──────────────────────── Logging ────────────────────────

    def log_event(self, event_type: str, description: str,
                  details: Optional[str] = None,
                  source: str = "system",
                  importance: float = 0.5) -> int:
        """Log a high-level narrative event."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO events (type, description, details, source, importance)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                ''', (event_type, description, details, source, importance))
                row = cur.fetchone()
                conn.commit()
                return row[0] if row else 0

    def log_transaction(self, component: str, action: str, content: str,
                        session_id: Optional[str] = None,
                        status: str = "success",
                        latency: float = 0.0,
                        metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Record a system transaction.

        Args:
            component: Which service component performed the action.
            action: What happened (e.g. "index", "search", "delete").
            content: Human-readable description.
            session_id: Optional session correlation ID.
            status: "success", "failure", or "warning".
            latency: Operation latency in seconds.
            metadata: Structured details.

        Returns:
            Transaction ID.
        """
        meta_json = json.dumps(metadata or {})

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO transactions
                        (component, action, session_id, status, latency, raw_data, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                ''', (component, action, session_id, status, latency, content, meta_json))

                row = cur.fetchone()
                transaction_id = row[0] if row else 0
                conn.commit()

        # Fire critical event callback if applicable
        if self._on_critical and transaction_id:
            meta_tags = (metadata or {}).get("tags", "")
            is_critical = (
                status == "failure"
                or "identity" in str(meta_tags)
                or "critical" in str(meta_tags)
            )
            if is_critical:
                try:
                    self._on_critical(component, action, content, status, metadata)
                except Exception as e:
                    logger.error(f"Critical event callback failed: {e}")

                self.log_event(
                    event_type="system_alert" if status == "failure" else "critical_event",
                    description=f"[{component}] {action}",
                    details=content[:200],
                    source="forensic_hook",
                    importance=0.8,
                )

        return transaction_id

    def log_derived_view_generation(
        self,
        *,
        view_type: str,
        view_id: str,
        inputs: Dict[str, Any],
        query_fingerprint: str,
        governance_state_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Log a Memory Over Maps derived-view generation event."""
        merged = dict(metadata or {})
        merged.update({
            "view_type": view_type,
            "view_id": view_id,
            "inputs": inputs,
            "query_fingerprint": query_fingerprint,
            "governance_state_hash": governance_state_hash,
            "tags": "memory_over_maps,derived_view",
        })
        return self.log_transaction(
            component="memory-over-maps",
            action="derived_view_generation",
            content=f"Generated {view_type} view {view_id}",
            metadata=merged,
        )

    # ──────────────────────── Querying ────────────────────────

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent high-level events."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT * FROM events ORDER BY timestamp DESC LIMIT %s',
                    (limit,),
                )
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def search_traces(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search across transaction traces using PostgreSQL tsvector."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT id, timestamp, component, raw_data AS content, status, latency
                    FROM transactions
                    WHERE search_vector @@ plainto_tsquery('english', %s)
                    ORDER BY ts_rank(search_vector, plainto_tsquery('english', %s)) DESC
                    LIMIT %s
                ''', (query, query, limit))
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_recent_transactions(self, limit: int = 50,
                                component: Optional[str] = None,
                                status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent transactions with optional filtering."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                conditions = []
                params: list = []
                if component:
                    conditions.append("component = %s")
                    params.append(component)
                if status:
                    conditions.append("status = %s")
                    params.append(status)

                where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                params.append(limit)

                cur.execute(
                    f'SELECT * FROM transactions {where} ORDER BY timestamp DESC LIMIT %s',
                    params,
                )
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_performance_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculate latency and throughput metrics."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute('''
                        SELECT
                            AVG(latency) as avg_latency,
                            MAX(latency) as max_latency,
                            COUNT(*) as total_count,
                            COUNT(*) FILTER (WHERE status='success') as success_count
                        FROM transactions
                        WHERE session_id = %s
                    ''', (session_id,))
                else:
                    cur.execute('''
                        SELECT
                            AVG(latency) as avg_latency,
                            MAX(latency) as max_latency,
                            COUNT(*) as total_count,
                            COUNT(*) FILTER (WHERE status='success') as success_count
                        FROM transactions
                    ''')

                row = cur.fetchone()
                if not row or row[2] == 0:
                    return {"success": False, "error": "No data found"}

                return {
                    "avg_latency_ms": round(float(row[0] or 0) * 1000, 2),
                    "max_latency_ms": round(float(row[1] or 0) * 1000, 2),
                    "total_transactions": row[2],
                    "success_rate": round(row[3] / row[2], 2),
                }

    def get_stats(self) -> Dict[str, Any]:
        """Get ledger statistics."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM transactions")
                tx_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM events")
                ev_count = cur.fetchone()[0]

                # Get database size
                cur.execute("SELECT pg_database_size(current_database())")
                db_size_bytes = cur.fetchone()[0]

                return {
                    "transaction_count": tx_count,
                    "event_count": ev_count,
                    "db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
                    "backend": "postgresql",
                }
