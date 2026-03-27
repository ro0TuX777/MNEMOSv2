"""
Forensic Ledger — Immutable Audit Trail
========================================

Searchable SQLite-backed ledger for system-wide auditing.
Uses FTS5 for high-speed transaction indexing.

Extracted from SAM's Forensic Ledger with all SAM-specific
dependencies (CEREBRO, SAGA) removed.
"""

import sqlite3
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ForensicLedger:
    """
    Immutable audit trail for all MNEMOS operations.

    Tracks every index, search, update, and delete operation with
    full-text search over reasoning traces.
    """

    def __init__(self, db_path: Optional[str] = None,
                 on_critical_event: Optional[Callable] = None):
        """
        Args:
            db_path: Path to the SQLite database file.
            on_critical_event: Optional callback invoked on critical events
                               (failures, identity-tagged transactions).
                               Signature: callback(component, action, content, status, metadata)
        """
        if db_path is None:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "audit.db")

        self.db_path = db_path
        self._on_critical = on_critical_event
        self._initialize_db()
        logger.info(f"⚖️ Forensic Ledger initialized at {db_path}")

    def _initialize_db(self):
        """Build the schema and FTS5 virtual tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Main transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL,
                    action TEXT NOT NULL,
                    session_id TEXT,
                    status TEXT,
                    latency REAL,
                    raw_data TEXT,
                    metadata TEXT
                )
            ''')

            # FTS5 virtual table for searchable traces
            try:
                cursor.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS traces_search
                    USING fts5(transaction_id UNINDEXED, content, component, tags)
                ''')
            except sqlite3.OperationalError:
                logger.warning("⚠️ FTS5 not supported — falling back to basic table.")
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS traces_search (
                        transaction_id INTEGER,
                        content TEXT,
                        component TEXT,
                        tags TEXT
                    )
                ''')

            # High-level events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    details TEXT,
                    source TEXT,
                    importance REAL DEFAULT 0.5
                )
            ''')

            conn.commit()

    # ──────────────────────── Logging ────────────────────────

    def log_event(self, event_type: str, description: str,
                  details: Optional[str] = None,
                  source: str = "system",
                  importance: float = 0.5) -> int:
        """Log a high-level narrative event."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO events (timestamp, type, description, details, source, importance)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, event_type, description, details, source, importance))
            conn.commit()
            return cursor.lastrowid

    def log_transaction(self, component: str, action: str, content: str,
                        session_id: Optional[str] = None,
                        status: str = "success",
                        latency: float = 0.0,
                        metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Record a system transaction and index its content for search.

        Args:
            component: Which service component performed the action.
            action: What happened (e.g. "index", "search", "delete").
            content: Human-readable description.
            session_id: Optional session correlation ID.
            status: "success", "failure", or "warning".
            latency: Operation latency in seconds.
            metadata: Structured details (tags, affected IDs, etc).

        Returns:
            Transaction ID.
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        meta_json = json.dumps(metadata or {})

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO transactions (timestamp, component, action, session_id, status, latency, raw_data, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, component, action, session_id, status, latency, content, meta_json))

            transaction_id = cursor.lastrowid

            # Index into FTS
            tags = metadata.get("tags", "") if metadata else ""
            cursor.execute('''
                INSERT INTO traces_search (transaction_id, content, component, tags)
                VALUES (?, ?, ?, ?)
            ''', (transaction_id, content, component, tags))

            conn.commit()

            # Fire critical event callback if applicable
            if self._on_critical:
                meta_tags = metadata.get("tags", "") if metadata else ""
                is_critical = (status == "failure") or ("identity" in meta_tags) or ("critical" in meta_tags)
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

    # ──────────────────────── Querying ────────────────────────

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent high-level events."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            try:
                cursor.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (limit,))
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                return []

    def search_traces(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search across reasoning traces."""
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            try:
                sanitized = query.replace('"', '""')
                cursor.execute('''
                    SELECT t.id, t.timestamp, ts.component, ts.content, t.status, t.latency
                    FROM traces_search ts
                    JOIN transactions t ON ts.transaction_id = t.id
                    WHERE traces_search MATCH ?
                    ORDER BY rank
                    LIMIT ?
                ''', (f'"{sanitized}"', limit))
                for row in cursor.fetchall():
                    results.append(dict(row))
            except sqlite3.OperationalError:
                # Fallback for non-FTS5 environments
                cursor.execute('''
                    SELECT t.id, t.timestamp, ts.component, ts.content, t.status, t.latency
                    FROM traces_search ts
                    JOIN transactions t ON ts.transaction_id = t.id
                    WHERE ts.content LIKE ? OR ts.tags LIKE ?
                    LIMIT ?
                ''', (f"%{query}%", f"%{query}%", limit))
                for row in cursor.fetchall():
                    results.append(dict(row))

        return results

    def get_recent_transactions(self, limit: int = 50,
                                component: Optional[str] = None,
                                status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent transactions with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            conditions = []
            params = []
            if component:
                conditions.append("component = ?")
                params.append(component)
            if status:
                conditions.append("status = ?")
                params.append(status)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)

            cursor.execute(f'''
                SELECT * FROM transactions {where} ORDER BY timestamp DESC LIMIT ?
            ''', params)
            return [dict(row) for row in cursor.fetchall()]

    def get_performance_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculate latency and throughput metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            where_clause = "WHERE session_id = ?" if session_id else ""
            params = (session_id,) if session_id else ()

            cursor.execute(f'''
                SELECT
                    AVG(latency) as avg_latency,
                    MAX(latency) as max_latency,
                    COUNT(*) as total_count,
                    COUNT(CASE WHEN status='success' THEN 1 END) as success_count
                FROM transactions
                {where_clause}
            ''', params)

            row = cursor.fetchone()
            if not row or row['total_count'] == 0:
                return {"success": False, "error": "No data found"}

            return {
                "avg_latency_ms": round((row['avg_latency'] or 0) * 1000, 2),
                "max_latency_ms": round((row['max_latency'] or 0) * 1000, 2),
                "total_transactions": row['total_count'],
                "success_rate": round(row['success_count'] / row['total_count'], 2),
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get ledger statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM transactions")
            tx_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM events")
            ev_count = cursor.fetchone()[0]

            import os
            db_size_mb = round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if Path(self.db_path).exists() else 0

            return {
                "transaction_count": tx_count,
                "event_count": ev_count,
                "db_size_mb": db_size_mb,
                "db_path": self.db_path,
            }


# Singleton
_instance: Optional[ForensicLedger] = None


def get_forensic_ledger(db_path: Optional[str] = None) -> ForensicLedger:
    """Get the global MNEMOS forensic ledger instance."""
    global _instance
    if _instance is None:
        _instance = ForensicLedger(db_path=db_path)
    return _instance
