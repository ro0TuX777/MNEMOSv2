import os
import sys

from mnemos.retrieval.lancedb_tier import LanceDBTier
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.derived_fact_scoring import DerivedFactScorer

def run_test():
    db_dir = "data/pit11a/lance"
    table_name = "mnemos_engrams"
    
    lance_tier = LanceDBTier(db_dir=db_dir, table_name=table_name)
    fusion = TierFusion([lance_tier])
    lance_tier._initialize()

    test_id = "46142bb57f6b253b"
    print(f"Testing with known ID: {test_id}")
    
    # Test get_engrams
    fetched = fusion.get_engrams([test_id])
    if not fetched:
        print("FAIL: get_engrams returned empty list")
        return
        
    print(f"PASS: get_engrams returned {len(fetched)} results")
    source_texts = [f.get("content", "") for f in fetched]
    
    support_evidence_preview = source_texts[0][:200].replace("\n", " ") + "..."
    print(f"PASS: support_evidence_preview: {support_evidence_preview}")
    
    # Test scorer
    scorer = DerivedFactScorer.get_instance()
    
    # Exact match fact
    fact_content = "This is a derived fact: " + source_texts[0][:150]
    scores = scorer.score_candidate("Test query", fact_content, source_texts)
    
    print(f"PASS: Exact match support score: {scores['derived_fact_source_support_score']}")
    print(f"PASS: Exact match alignment score: {scores['derived_fact_answer_alignment_score']}")

if __name__ == "__main__":
    run_test()
