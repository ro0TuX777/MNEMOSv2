"""
Hybrid fusion policy presets.
"""

from typing import Dict


FUSION_POLICIES: Dict[str, Dict[str, float]] = {
    "semantic_dominant": {"lexical": 0.25, "semantic": 0.75},
    "balanced": {"lexical": 0.50, "semantic": 0.50},
    "lexical_dominant": {"lexical": 0.75, "semantic": 0.25},
}


DEFAULT_FUSION_POLICY = "balanced"

