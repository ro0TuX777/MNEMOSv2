from contextlib import closing
"""SQLite-based metadata sidecar for TurbovecTier."""
import sqlite3
import json
from typing import Dict, List, Optional, Any

class TurbovecMetadata:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with closing(self._get_connection()) as conn, conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS engram_metadata (
                  engram_uuid TEXT PRIMARY KEY,
                  turbovec_id INTEGER UNIQUE NOT NULL,
                  source_uri TEXT,
                  content TEXT,
                  metadata_json TEXT,
                  governance_json TEXT,
                  content_hash TEXT,
                  created_at TEXT,
                  updated_at TEXT,
                  deleted INTEGER DEFAULT 0
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS engram_fts
                USING fts5(engram_uuid UNINDEXED, content, source_uri, metadata_text);
            ''')
            
    def upsert_engram(self, engram_uuid: str, source_uri: str, content: str,
                      metadata: dict, governance: dict, content_hash: str,
                      created_at: str, updated_at: str) -> int:
        """Insert or update an engram, returning its turbovec_id (uint64 equivalent)."""
        with closing(self._get_connection()) as conn, conn:
            cursor = conn.cursor()
            
            # Check if it exists
            cursor.execute("SELECT turbovec_id, deleted FROM engram_metadata WHERE engram_uuid = ?", (engram_uuid,))
            row = cursor.fetchone()
            
            if row:
                turbovec_id = row['turbovec_id']
                # Update
                cursor.execute("""
                    UPDATE engram_metadata 
                    SET source_uri=?, content=?, metadata_json=?, governance_json=?,
                        content_hash=?, updated_at=?, deleted=0
                    WHERE engram_uuid=?
                """, (source_uri, content, json.dumps(metadata), json.dumps(governance),
                      content_hash, updated_at, engram_uuid))
                # Update FTS (delete old, re-insert below)
                cursor.execute("DELETE FROM engram_fts WHERE engram_uuid=?", (engram_uuid,))
            else:
                # Generate new turbovec_id
                cursor.execute("SELECT MAX(turbovec_id) FROM engram_metadata")
                max_id = cursor.fetchone()[0]
                turbovec_id = (max_id or 0) + 1
                
                cursor.execute("""
                    INSERT INTO engram_metadata 
                    (engram_uuid, turbovec_id, source_uri, content, metadata_json, governance_json, content_hash, created_at, updated_at, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (engram_uuid, turbovec_id, source_uri, content, json.dumps(metadata), json.dumps(governance), content_hash, created_at, updated_at))
            
            # Insert into FTS
            cursor.execute("""
                INSERT INTO engram_fts (engram_uuid, content, source_uri, metadata_text)
                VALUES (?, ?, ?, ?)
            """, (engram_uuid, content, source_uri, json.dumps(metadata)))
            
            return turbovec_id

    def delete_engram(self, engram_uuid: str):
        """Soft delete an engram and remove it from FTS index."""
        with closing(self._get_connection()) as conn, conn:
            conn.execute("UPDATE engram_metadata SET deleted=1 WHERE engram_uuid=?", (engram_uuid,))
            conn.execute("DELETE FROM engram_fts WHERE engram_uuid=?", (engram_uuid,))
            
    def get_engram(self, engram_uuid: str) -> Optional[Dict[str, Any]]:
        with closing(self._get_connection()) as conn, conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM engram_metadata WHERE engram_uuid=? AND deleted=0", (engram_uuid,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def get_by_turbovec_ids(self, turbovec_ids: List[int]) -> List[Dict[str, Any]]:
        if not turbovec_ids:
            return []
        placeholders = ','.join('?' * len(turbovec_ids))
        with closing(self._get_connection()) as conn, conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM engram_metadata WHERE turbovec_id IN ({placeholders}) AND deleted=0", turbovec_ids)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def lexical_search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        safe_query = ''.join(c if c.isalnum() else ' ' for c in query).strip()
        if not safe_query:
            return []
        with closing(self._get_connection()) as conn, conn:
            cursor = conn.cursor()
            # FTS5 matches
            cursor.execute("""
                SELECT e.* FROM engram_metadata e
                JOIN engram_fts f ON e.engram_uuid = f.engram_uuid
                WHERE engram_fts MATCH ? AND e.deleted=0
                ORDER BY f.rank
                LIMIT ?
            """, (safe_query, limit))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def filter_candidates(self, filters: Dict[str, Any], candidate_uuids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """A basic metadata filter implementation."""
        query = "SELECT * FROM engram_metadata WHERE deleted=0"
        params = []
        if candidate_uuids is not None:
            if not candidate_uuids:
                return []
            placeholders = ','.join('?' * len(candidate_uuids))
            query += f" AND engram_uuid IN ({placeholders})"
            params.extend(candidate_uuids)
            
        with closing(self._get_connection()) as conn, conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                d = self._row_to_dict(row)
                match = True
                for k, v in filters.items():
                    if k == 'source_uri':
                        if d['source_uri'] != v:
                            match = False
                            break
                    else:
                        if d['metadata_json'].get(k) != v:
                            match = False
                            break
                if match:
                    results.append(d)
            return results

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d['metadata_json'] = json.loads(d['metadata_json']) if d['metadata_json'] else {}
        d['governance_json'] = json.loads(d['governance_json']) if d['governance_json'] else {}
        return d
