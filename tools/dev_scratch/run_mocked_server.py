import os
import sys
from unittest.mock import patch
from mnemos.config import get_config

def main():
    # Patch Qdrant initialization
    patcher = patch('mnemos.retrieval.qdrant_tier.QdrantTier._initialize')
    patcher.start()

    # Patch search_derived to return a dummy payload for live trial
    dummy_fact = {
        "fact_id": "fact_1",
        "authority_type": "MNEMOS_DERIVED_FACT",
        "display_label": "[MNEMOS-DERIVED]",
        "content": "This is a derived fact.",
        "traceability": {
            "source_engram_ids": ["eng_1"],
            "passage_node_ids": ["pass_1"],
            "fact_id": "fact_1",
            "fact_receipt_id": "rec_1",
            "promotion_receipt_id": "prom_1",
            "lifecycle_event_id": "life_1",
            "source_uri": "uri_1",
            "artifact_id": "art_1",
            "chunk_id": "chunk_1",
            "provenance_span": "span_1",
            "verifier_receipt_id": "ver_1"
        },
        "governance_metadata": {"status": "CERTIFIED_FOR_GOVERNED_EVALUATION_OPERATION"},
        "lifecycle_metadata": {"terminal_state": "ACTIVE"},
        "conflict_metadata": {"conflict_status": "NO_CONFLICT_FOUND"}
    }

    patcher2 = patch('mnemos.retrieval.retrieval_router.RetrievalRouter.search_derived', return_value={"derived_results": [dummy_fact]})
    patcher2.start()

    from service.app import app, get_config
    config = get_config()
    
    print(f"Starting mocked MNEMOS server on port {config.port}...")
    app.run(host="127.0.0.1", port=config.port, debug=False)

if __name__ == "__main__":
    main()
