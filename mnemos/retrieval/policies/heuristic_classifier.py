"""
Query Family Classifier
=======================

Provides heuristic/rule-based query-family classification for retrieval queries 
to determine Cross-Encoder reranking eligibility.
"""

import re
from typing import Tuple

def classify_query_heuristic(query: str) -> Tuple[str, float]:
    """
    Classifies a query into one of the known families:
    factoid, constraint_heavy, multi_clause, why_how, hard_negative, code_behavior, or unknown.
    
    Returns:
        tuple[str, float]: The classified family and a confidence score between 0.0 and 1.0.
    """
    tl = query.lower()
    
    # 1. factoid: 0.90
    if tl.startswith("what is ") or tl.startswith("who ") or tl.startswith("when "):
        if len(query.split()) < 8:
            return "factoid", 0.90
            
    # 2. code_behavior: 0.85
    code_patterns = [r'\bpython\b', r'\bapi\b', r'\bscript\b', r'\bmodule\b', r'\bfunction\b', r'\.py\b', r'`.*`']
    for p in code_patterns:
        if re.search(p, tl):
            return "code_behavior", 0.85
            
    # 3. why_how: 0.80
    if tl.startswith("why ") or tl.startswith("how ") or "explain" in tl:
        return "why_how", 0.80
        
    # 4. constraint_heavy: 0.75
    if "best when" in tl or "only if" in tl or "unless" in tl or "greater than" in tl:
         return "constraint_heavy", 0.75

    # 5. hard_negative: 0.70
    if "instead of" in tl or "versus" in tl or "vs" in tl or "differ" in tl or "loser" in tl:
         return "hard_negative", 0.70

    # 6. multi_clause: 0.65
    if " and " in tl and ("," in tl or "during" in tl):
        return "multi_clause", 0.65

    # unknown: 0.20
    return "unknown", 0.20
