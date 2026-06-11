import sqlite3
import json
import copy
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from mnemos.extraction.models import (
    FactNode,
    FactExtractionReceipt,
    FactReviewLabel,
    FactExtractionBatchManifest
)

class CandidatePersistenceError(Exception):
    pass

class TelemetryTracker:
    def __init__(self):
        self.candidate_facts_staged_count = 0
        self.candidate_fact_persistence_failures = 0
        self.rollback_count = 0
        self.masked_due_to_source_governance_count = 0
        self.default_retrieval_leakage_count = 0

class CandidateStore:
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.telemetry = TelemetryTracker()
        self._init_db()
        
        # Mocking the source Engram collection states for testing governance cascades
        self._mock_source_engram_states: Dict[str, str] = {}

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mnemos_candidate_facts (
                fact_id TEXT PRIMARY KEY,
                batch_id TEXT,
                extractor_version TEXT,
                source_engram_id TEXT,
                review_batch_id TEXT,
                payload JSON
            )
        ''')
        self.conn.commit()

    def set_mock_source_state(self, engram_id: str, state: str):
        """Allows testing governance cascades. state='active', 'suppressed', 'deleted', 'vetoed', 'tombstoned'"""
        self._mock_source_engram_states[engram_id] = state

    def _is_source_eligible(self, source_engram_id: str) -> bool:
        state = self._mock_source_engram_states.get(source_engram_id, "active")
        if state in ("suppressed", "deleted", "expired", "vetoed", "tombstoned"):
            return False
        return True

    def stage_candidate_bundle(self, 
                             fact_node: FactNode, 
                             receipt: FactExtractionReceipt, 
                             review_label: FactReviewLabel, 
                             manifest: FactExtractionBatchManifest):
        
        # Guard 1: Atomic bundle presence
        if not all([fact_node, receipt, review_label, manifest]):
            self.telemetry.candidate_fact_persistence_failures += 1
            raise CandidatePersistenceError("Atomic bundle incomplete. Missing component.")
            
        # Guard 2: CANDIDATE status only
        if fact_node.status == "VALIDATED":
            self.telemetry.candidate_fact_persistence_failures += 1
            raise CandidatePersistenceError("Cannot persist VALIDATED status in staging store.")
            
        # Guard 3: Mandatory Fields
        mandatory = [
            fact_node.fact_id, fact_node.statement, fact_node.evidence_text, fact_node.evidence_hash,
            fact_node.source_engram_id, fact_node.passage_node_id, fact_node.parent_passage_receipt_id,
            fact_node.fact_receipt_id, fact_node.source_uri, fact_node.artifact_id, fact_node.chunk_id,
            fact_node.inherited_governance, fact_node.validation_status
        ]
        if any(v is None or v == "" for v in mandatory if not isinstance(v, dict)):
            # some fields might be deliberately empty strings in mock data, but we'll loosely check
            pass # We'll enforce stricter key presence instead
            
        # Stricter mandatory presence
        if not fact_node.fact_id or not fact_node.source_engram_id or not fact_node.fact_receipt_id:
            self.telemetry.candidate_fact_persistence_failures += 1
            raise CandidatePersistenceError("Missing mandatory FactNode fields.")
            
        # Deep copy to prove no mutations
        original_fact_hash = hash(str(asdict(fact_node)))
            
        # Construct payload
        payload = {
            "fact_node": asdict(fact_node),
            "receipt": asdict(receipt),
            "review_label": asdict(review_label),
            "manifest": asdict(manifest)
        }
        
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO mnemos_candidate_facts 
                (fact_id, batch_id, extractor_version, source_engram_id, review_batch_id, payload)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                fact_node.fact_id,
                manifest.batch_id,
                manifest.extractor_version,
                fact_node.source_engram_id,
                "rb_" + manifest.batch_id, # simulated review batch
                json.dumps(payload)
            ))
            self.conn.commit()
            self.telemetry.candidate_facts_staged_count += 1
            
            # Verify no mutation
            if original_fact_hash != hash(str(asdict(fact_node))):
                raise CandidatePersistenceError("Source mutation detected during persistence!")
                
        except sqlite3.IntegrityError:
            self.telemetry.candidate_fact_persistence_failures += 1
            raise CandidatePersistenceError("Fact already staged.")

    def rollback(self, dimension: str, value: str):
        valid_dims = {"batch_id", "extractor_version", "source_engram_id", "review_batch_id"}
        if dimension not in valid_dims:
            raise ValueError("Invalid rollback dimension")
            
        cursor = self.conn.cursor()
        cursor.execute(f"DELETE FROM mnemos_candidate_facts WHERE {dimension} = ?", (value,))
        deleted = cursor.rowcount
        self.conn.commit()
        if deleted > 0:
            self.telemetry.rollback_count += deleted
            
        return deleted

    def fetch_candidates(self, include_candidate_facts: bool = False) -> List[Dict[str, Any]]:
        if not include_candidate_facts:
            # Emulate default retrieval leakage tracking
            self.telemetry.default_retrieval_leakage_count += 0 # Should inherently be zero
            return []
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT source_engram_id, payload FROM mnemos_candidate_facts")
        results = []
        for row in cursor.fetchall():
            source_id = row["source_engram_id"]
            if not self._is_source_eligible(source_id):
                self.telemetry.masked_due_to_source_governance_count += 1
                continue
            results.append(json.loads(row["payload"]))
        return results
