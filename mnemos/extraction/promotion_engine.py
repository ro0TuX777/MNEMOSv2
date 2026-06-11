import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from mnemos.extraction.models import (
    FactNode,
    FactExtractionReceipt,
    FactReviewLabel,
    FactExtractionBatchManifest,
    FactPromotionReceipt,
    FactLifecycleEvent
)
from mnemos.extraction.candidate_store import CandidateStore

class PromotionError(Exception):
    pass

class PromotionEngine:
    def __init__(self, candidate_store: CandidateStore, db_path: str = ":memory:"):
        self.store = candidate_store
        # We share the same sqlite connection for simplicity in testing, or create a new one
        # To make it truly disjoint we can create tables in the same db file but keep them isolated
        # For this prototype we will assume db_path matches candidate_store's db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mnemos_fact_promotion_receipts (
                receipt_id TEXT PRIMARY KEY,
                fact_id TEXT,
                payload JSON
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mnemos_fact_lifecycle_events (
                event_id TEXT PRIMARY KEY,
                fact_id TEXT,
                event_type TEXT,
                payload JSON
            )
        ''')
        self.conn.commit()

    def _log_lifecycle_event(self, fact_id: str, event_type: str, operator_id: str, reason: str, metadata: dict = None):
        event = FactLifecycleEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            fact_id=fact_id,
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            operator_id=operator_id,
            reason=reason,
            metadata=metadata or {}
        )
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO mnemos_fact_lifecycle_events (event_id, fact_id, event_type, payload)
            VALUES (?, ?, ?, ?)
        ''', (event.event_id, event.fact_id, event.event_type, json.dumps(event.to_dict())))
        self.conn.commit()
        return event

    def fetch_promotion_receipts(self, fact_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT payload FROM mnemos_fact_promotion_receipts WHERE fact_id = ?", (fact_id,))
        return [json.loads(r["payload"]) for r in cursor.fetchall()]

    def preflight_conflict_sweep(self, fact: FactNode) -> bool:
        # Stub for conflict candidate sweep
        # Real system would check Qdrant/engrams for logical contradictions
        # Here we mock a collision if statement has "CONFLICT_TEST" in it
        if "CONFLICT_TEST" in fact.statement:
            return False
        return True

    def promote_candidate(self, fact_id: str, operator_id: str) -> FactPromotionReceipt:
        # 1. Fetch Candidate Quad-Tuple from isolated store
        # This requires `include_candidate_facts=True` otherwise we can't see it
        # This proves the RAG boundary remains isolated.
        candidates = self.store.fetch_candidates(include_candidate_facts=True)
        bundle = next((c for c in candidates if c["fact_node"]["fact_id"] == fact_id), None)
        
        if not bundle:
            raise PromotionError("Candidate not found or ineligible.")
            
        fact_data = bundle["fact_node"]
        receipt_data = bundle["receipt"]
        label_data = bundle["review_label"]
        manifest_data = bundle["manifest"]
        
        # 2. Quad-Tuple validation
        if not fact_data or not receipt_data or not label_data or not manifest_data:
            raise PromotionError("Incomplete Quad-Tuple. Promotion halted.")
            
        # 3. Human Review validation
        if label_data.get("reviewer_type") != "human":
            raise PromotionError("Automated LLM promotion strictly forbidden. Human reviewer required.")
        if label_data.get("recommended_action") != "PROMOTE_TO_VALIDATED":
            raise PromotionError("Review label action does not explicitly authorize PROMOTE_TO_VALIDATED.")
        if not label_data.get("atomicity_verified") or not label_data.get("faithfulness_verified"):
            raise PromotionError("Atomicity and faithfulness must be explicitly verified by operator.")
            
        # 4. Governance verification
        src_engram_id = fact_data["source_engram_id"]
        # We check the CURRENT state using the store's eligibility checker
        if not self.store._is_source_eligible(src_engram_id):
            raise PromotionError("Source engram is currently suppressed, deleted, or vetoed. Promotion blocked.")
            
        # Reconstruct FactNode to pass to sweep stub
        fact_node = FactNode(**fact_data)
        
        # 5. ConflictCandidate Sweep
        if not self.preflight_conflict_sweep(fact_node):
            raise PromotionError("ConflictCandidate sweep failed. Logical contradiction detected.")
            
        # 6. Generate Disjoint Receipt
        receipt = FactPromotionReceipt(
            receipt_id=f"prom_{uuid.uuid4().hex[:8]}",
            promoted_fact_id=fact_id,
            human_review_label_id=label_data.get("fact_id"), # usually same as fact_id or specific label ID
            operator_id=operator_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            source_governance_snapshot=fact_data.get("inherited_governance", {}),
            conflict_sweep_hash="mock_hash_clear",
            promotion_status="VALIDATED"
        )
        
        # 7. Persist disjoint receipt
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO mnemos_fact_promotion_receipts (receipt_id, fact_id, payload)
            VALUES (?, ?, ?)
        ''', (receipt.receipt_id, fact_id, json.dumps(receipt.to_dict())))
        self.conn.commit()
        
        # 8. Append Lifecycle event
        self._log_lifecycle_event(fact_id, "PROMOTION_APPROVED", operator_id, "Passed all preflight gates")
        
        return receipt

    def get_terminal_lifecycle_state(self, fact_id: str) -> Tuple[Optional[str], Optional[str]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT event_type, event_id FROM mnemos_fact_lifecycle_events WHERE fact_id=? ORDER BY ROWID DESC LIMIT 1", (fact_id,))
        row = cursor.fetchone()
        if not row:
            return None, None
        return row["event_type"], row["event_id"]

    def fetch_validated_facts(self) -> List[Dict[str, Any]]:
        # This operates entirely isolated from RAG default retrieval
        candidates = self.store.fetch_candidates(include_candidate_facts=True)
        validated = []
        for bundle in candidates:
            fact_id = bundle["fact_node"]["fact_id"]
            
            # Check for a promotion receipt
            receipts = self.fetch_promotion_receipts(fact_id)
            if not receipts:
                continue
                
            # Check terminal lifecycle state
            terminal_state, event_id = self.get_terminal_lifecycle_state(fact_id)
            if terminal_state in ("DOWNGRADED", "REJECTED", "SUPERSEDED"):
                continue
                
            # Live governance check is implicitly covered because store.fetch_candidates masks ineligible ones!
            # We just need to add the conflict tracking metadata.
            
            # Retrieve all events for chain export
            cursor = self.conn.cursor()
            cursor.execute("SELECT payload FROM mnemos_fact_lifecycle_events WHERE fact_id=? ORDER BY ROWID ASC", (fact_id,))
            events = [json.loads(r["payload"]) for r in cursor.fetchall()]
            
            receipt = receipts[-1] # latest receipt
            
            export_chain = {
                "source_engram_id": bundle["fact_node"]["source_engram_id"],
                "passage_node": {"passage_node_id": bundle["fact_node"]["passage_node_id"]}, # Stubbed parent
                "candidate_fact": bundle["fact_node"],
                "human_review_label": bundle["review_label"],
                "promotion_receipt": receipt,
                "lifecycle_events": events,
                "conflict_metadata": {
                    "conflict_check_performed": True,
                    "conflict_check_timestamp": receipt["timestamp"],
                    "conflict_sweep_hash": receipt["conflict_sweep_hash"],
                    "conflict_candidate_ids": [],
                    "conflict_resolution_status": "CLEARED",
                    "conflict_resolution_operator_id": receipt["operator_id"],
                    "supersedes_fact_id": None,
                    "superseded_by_fact_id": None,
                    "terminal_lifecycle_state": terminal_state,
                    "terminal_lifecycle_event_id": event_id
                }
            }
            validated.append(export_chain)
            
        return validated
