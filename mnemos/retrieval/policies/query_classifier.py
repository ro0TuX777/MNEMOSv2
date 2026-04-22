"""
Query Classifier Interface
==========================

Provides pluggable query-family classification backends.
"""

from typing import Tuple, Dict, Any

from mnemos.retrieval.policies.heuristic_classifier import classify_query_heuristic

class QueryFamilyClassifier:
    def classify(self, query: str) -> Tuple[str, float]:
        raise NotImplementedError

class HeuristicFamilyClassifier(QueryFamilyClassifier):
    def classify(self, query: str) -> Tuple[str, float]:
        return classify_query_heuristic(query)

class ZeroShotFamilyClassifier(QueryFamilyClassifier):
    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold

    def classify(self, query: str) -> Tuple[str, float]:
        # Implement real NLP model wrapper here.
        # Fallback to unknown for the dummy mock implementation.
        return "unknown", 0.0

class HybridFamilyClassifier(QueryFamilyClassifier):
    def __init__(self, confidence_threshold: float = 0.65):
        self.heuristic = HeuristicFamilyClassifier()
        self.zeroshot = ZeroShotFamilyClassifier(confidence_threshold)

    def classify(self, query: str) -> Tuple[str, float]:
        family, conf = self.heuristic.classify(query)
        if family == "unknown" or conf < 0.65:
            zs_family, zs_conf = self.zeroshot.classify(query)
            if zs_family != "unknown":
                return zs_family, zs_conf
        return family, conf

def get_classifier(config: Dict[str, Any] = None) -> QueryFamilyClassifier:
    """Factory to retrieve configured classifier."""
    if not config:
        config = {}
    
    mode = config.get("mode", "heuristic")
    threshold = config.get("confidence_threshold", 0.65)
    
    if mode == "zero_shot":
        return ZeroShotFamilyClassifier(threshold)
    elif mode == "hybrid":
        return HybridFamilyClassifier(threshold)
    else:
        return HeuristicFamilyClassifier()
