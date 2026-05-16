import re
from protected_span_policy import PROTECTED_PATTERNS

class SpanValidator:
    @staticmethod
    def validate_preservation(original: str, compressed: str) -> dict:
        results = {}
        for key, pattern in PROTECTED_PATTERNS.items():
            orig_matches = re.findall(pattern, original)
            if not orig_matches:
                results[key] = True # Nothing to preserve
                continue
            
            comp_matches = re.findall(pattern, compressed)
            orig_set = set(m.lower() for m in orig_matches)
            comp_set = set(m.lower() for m in comp_matches)
            
            results[key] = orig_set.issubset(comp_set)
            
        return results
