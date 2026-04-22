import pytest
from mnemos.retrieval.policies.query_classifier import get_classifier

def test_heuristic_classifier_outputs():
    # Setup
    classifier = get_classifier({"mode": "heuristic"})
    
    # Code queries
    family, conf = classifier.classify("def python_api_thing(x): `something`")
    assert family == "code_behavior"
    assert conf == 0.85
    
    # Why/How
    family, conf = classifier.classify("Why does the system fail?")
    assert family == "why_how"
    assert conf == 0.80
    
    # Factoid
    family, conf = classifier.classify("What is the core memory appliance?")
    assert family == "factoid"
    assert conf == 0.90
    
    # Unknown
    family, conf = classifier.classify("I think it is a sunny day in San Francisco.")
    assert family == "unknown"
    assert conf == 0.20

def test_hybrid_fallback():
    classifier = get_classifier({"mode": "hybrid"})
    
    # Heuristic matches first
    family, conf = classifier.classify("how do I fix a bug?")
    assert family == "why_how"
    
    # Fallback to zero shot mock on unknown
    family, conf = classifier.classify("completely generic unstructured string")
    assert family == "unknown"
    assert conf == 0.20 # The heuristic returns this before dropping to ZeroShot which matches unknown.

def test_zeroshot_lane():
    classifier = get_classifier({"mode": "zero_shot", "confidence_threshold": 0.50})
    family, conf = classifier.classify("why is the sky blue?")
    assert family == "unknown" # Current dummy implementation
    assert conf == 0.0
