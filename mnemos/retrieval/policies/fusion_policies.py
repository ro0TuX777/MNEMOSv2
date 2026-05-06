"""
Hybrid fusion policy presets.
"""

from typing import Dict


FUSION_POLICIES: Dict[str, Dict[str, float]] = {
    "semantic_dominant": {"lexical": 0.25, "semantic": 0.75},
    "balanced": {"lexical": 0.50, "semantic": 0.50},
    "lexical_dominant": {"lexical": 0.75, "semantic": 0.25},
    # Server-side RRF via Qdrant prefetch — weights are not used
    # (Qdrant's built-in RRF handles fusion).  The key presence
    # enables the router to select the QdrantHybridFusion engine.
    "qdrant_rrf": {"lexical": 0.50, "semantic": 0.50},
}


DEFAULT_FUSION_POLICY = "balanced"

