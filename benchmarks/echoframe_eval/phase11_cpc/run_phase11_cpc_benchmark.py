import os
import json
from sentence_segmenter import SentenceSegmenter
from cpc_style_sentence_ranker import CPCStyleSentenceRanker
from protected_sentence_policy import ProtectedSentencePolicy

def run_phase11_benchmark():
    print("Running Phase 11 CPC Benchmark...")
    
    # Mock Document
    doc = (
        "Source: [DODM 5240.01-A]\n"
        "Governance: HIGH_RISK, approval_required\n"
        "1.1 The operator shall monitor signals.\n"
        "1.2 The operator may not disclose SIGINT unless authorized.\n"
        "1.3 The sky is blue and the grass is green.\n"
        "1.4 Operations begin on 2024-05-15 with 500 units.\n"
        "1.5 Coffee is served at 0800 in the breakroom."
    )
    question = "When do operations begin?"
    
    segmenter = SentenceSegmenter()
    ranker = CPCStyleSentenceRanker()
    policy = ProtectedSentencePolicy()
    
    sentences = segmenter.segment(doc)
    
    # Analyze protection
    protected = []
    unprotected = []
    for s in sentences:
        res = policy.evaluate(s)
        if res["is_protected"]:
            protected.append(s)
        else:
            unprotected.append(s)
            
    print(f"Total Sentences: {len(sentences)}")
    print(f"Protected: {len(protected)}")
    print(f"Unprotected: {len(unprotected)}")
    
    # CPC Selection on Unprotected
    cpc_selected = ranker.rank(question, unprotected, target_ratio=0.5)
    
    # Reassemble Hybrid
    final_evidence = "\n".join(protected + cpc_selected)
    
    print("\n--- FINAL HYBRID EVIDENCE ---")
    print(final_evidence)
    
if __name__ == "__main__":
    run_phase11_benchmark()
