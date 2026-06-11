import json
import os
import uuid
import hashlib
import copy
from datetime import datetime
from typing import List, Dict, Any, Tuple

from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.extraction.models import (
    ExtractionReceipt,
    PassageNode,
    ExtractionBatchManifest
)

OUTPUT_DIR = os.path.join("data", "smc_1_output")

def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def dummy_extract_passage(engram: Engram) -> Tuple[str, Tuple[int, int]]:
    """
    Simulates an LLM extraction by finding the first sentence or substring.
    Returns the extracted text and the (start, end) character offsets.
    """
    # Simple extraction logic: grab the first sentence up to a period, or the whole thing if short
    content = engram.content
    period_idx = content.find('.')
    if period_idx != -1 and period_idx > 10:
        span = (0, period_idx + 1)
    else:
        span = (0, min(len(content), 100))
    
    extracted_text = content[span[0]:span[1]]
    return extracted_text, span

def generate_mock_engrams() -> List[Engram]:
    """Generates synthetic Engrams to avoid Qdrant/Database queries."""
    e1 = Engram(
        content="The Artemis II mission is scheduled to launch four astronauts around the Moon in 2025. This will be the first crewed mission to the Moon since Apollo 17.",
        source="doc:artemis_overview",
        metadata={
            "artifact_id": "art_111",
            "chunk_id": "chk_111",
            "source_uri": "s3://nasa/artemis.txt",
        },
        governance=GovernanceMeta(
            memory_type="episodic",
            source_type="document",
            policy_flags=["verified"]
        )
    )
    
    e2 = Engram(
        content="Quantum computing could break RSA encryption. Shor's algorithm running on a sufficiently large quantum computer can factor large primes in polynomial time.",
        source="doc:quantum_threat",
        metadata={
            "artifact_id": "art_222",
            "chunk_id": "chk_222",
            "source_uri": "https://arxiv.org/abs/quant",
        },
        governance=GovernanceMeta(
            memory_type="semantic",
            source_type="academic",
            policy_flags=["experimental"]
        )
    )
    
    return [e1, e2]

def run_extraction_pipeline():
    print("--- Starting SMC-1 Offline Passage Extraction Proof ---")
    
    # 1. Setup Data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    engrams = generate_mock_engrams()
    
    # Keep deep copies to verify 0 mutations
    engrams_pristine = copy.deepcopy(engrams)
    
    batch_id = f"batch_smc1_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    passages: List[PassageNode] = []
    receipts: List[ExtractionReceipt] = []
    failures: List[str] = []
    
    # 2. Extract and Process
    for engram in engrams:
        try:
            # Simulated LLM extraction
            extracted_text, span = dummy_extract_passage(engram)
            
            # Hashes
            source_hash = compute_sha256(engram.content)
            
            # Build Receipt
            receipt_id = f"rcpt_{uuid.uuid4().hex}"
            lineage = engram.lineage()
            
            gov_snapshot = engram.governance.to_dict() if engram.governance else {}
            
            receipt = ExtractionReceipt(
                receipt_id=receipt_id,
                batch_id=batch_id,
                source_engram_id=engram.id,
                source_uri=lineage.get("source_uri", ""),
                artifact_id=lineage.get("artifact_id", ""),
                chunk_id=lineage.get("chunk_id", ""),
                provenance_span=span,
                source_hash=source_hash,
                extractor_version="smc1_dummy_v1",
                prompt_hash="dummy_prompt_hash_0000",
                model_name_version="dummy_llm_v0",
                timestamp=timestamp,
                extraction_mode="offline_shadow",
                governance_snapshot=gov_snapshot,
                output_hash="" # Computed after node is formed
            )
            
            # Build Node
            passage_id = f"psg_{uuid.uuid4().hex}"
            passage = PassageNode(
                passage_id=passage_id,
                text=extracted_text,
                source_engram_id=engram.id,
                provenance_span=span,
                extraction_receipt_id=receipt_id,
                inherited_governance=gov_snapshot
            )
            
            # Output Hash
            passage_json_str = json.dumps(passage.to_dict(), sort_keys=True)
            receipt.output_hash = compute_sha256(passage_json_str)
            
            passages.append(passage)
            receipts.append(receipt)
            
            # Serialize to disk
            with open(os.path.join(OUTPUT_DIR, f"{passage_id}.json"), "w") as f:
                json.dump(passage.to_dict(), f, indent=2)
                
            with open(os.path.join(OUTPUT_DIR, f"{receipt_id}.json"), "w") as f:
                json.dump(receipt.to_dict(), f, indent=2)
                
        except Exception as e:
            failures.append(f"Failed to process engram {engram.id}: {str(e)}")

    # 3. Manifest
    manifest = ExtractionBatchManifest(
        batch_id=batch_id,
        timestamp=timestamp,
        processed_count=len(engrams),
        success_count=len(passages),
        error_count=len(failures),
        failures=failures
    )
    
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
        
    # 4. Strict Validation Gates
    print("\n--- Validation Gates ---")
    
    # Gate 1: 100% have receipts
    receipt_ids = {r.receipt_id for r in receipts}
    all_have_receipts = all(p.extraction_receipt_id in receipt_ids for p in passages)
    print(f"Gate 1: 100% PassageNodes have receipts? {'PASS' if all_have_receipts else 'FAIL'}")
    
    # Gate 2: 100% exact span match
    span_matches = True
    for p, e in zip(passages, engrams):
        start, end = p.provenance_span
        if e.content[start:end] != p.text:
            span_matches = False
            print(f"Span mismatch on {p.passage_id}")
    print(f"Gate 2: 100% span exact match? {'PASS' if span_matches else 'FAIL'}")
    
    # Gate 3: 100% source hash match
    hash_matches = True
    for r, e in zip(receipts, engrams):
        if r.source_hash != compute_sha256(e.content):
            hash_matches = False
            print(f"Hash mismatch on {r.receipt_id}")
    print(f"Gate 3: 100% source_hash validation? {'PASS' if hash_matches else 'FAIL'}")
    
    # Gate 4: 0 Engram Mutations
    mutations = 0
    for e_run, e_orig in zip(engrams, engrams_pristine):
        if e_run.to_dict() != e_orig.to_dict():
            mutations += 1
    print(f"Gate 4: 0 Source Engram Mutations? {'PASS' if mutations == 0 else 'FAIL'} ({mutations} mutations detected)")
    
    # Output report
    print("\n--- SMC-1 Run Summary ---")
    print(f"Batch ID: {batch_id}")
    print(f"Processed: {len(engrams)}")
    print(f"Successfully generated: {len(passages)} PassageNodes, {len(receipts)} Receipts")
    print(f"Data output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Rollback command: rm -rf {os.path.abspath(OUTPUT_DIR)}")
    
    success = all_have_receipts and span_matches and hash_matches and (mutations == 0)
    if success:
        print("\n[PASS] SMC-1: All strict validation gates cleared.")
    else:
        print("\n[FAIL] SMC-1: Validation gates failed. Review required.")

if __name__ == "__main__":
    run_extraction_pipeline()
