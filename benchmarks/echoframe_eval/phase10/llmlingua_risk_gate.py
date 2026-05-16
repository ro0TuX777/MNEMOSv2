import re
from deep_protected_span_extractor import DeepProtectedSpanExtractor

HIGH_RISK_TAGS = [
    "HIGH_RISK", "SIGINT", "INTEL", "LEGAL", "POLICY_CRITICAL", 
    "MILITARY_GRADE", "APPROVAL_REQUIRED", "CLASSIFICATION_MARKED",
    "CUI", "FOUO", "SCI", "SAP", "SECRET", "TOP SECRET"
]

class LLMLinguaRiskGate:
    def __init__(self, density_threshold=0.15):
        self.extractor = DeepProtectedSpanExtractor()
        self.density_threshold = density_threshold
        
    def check_context_tags(self, text: str) -> bool:
        # Returns True if High Risk
        text_upper = text.upper()
        for tag in HIGH_RISK_TAGS:
            if tag in text_upper:
                return True
        return False
        
    def check_density(self, text: str) -> float:
        words = text.split()
        if not words: return 0.0
        
        facts = self.extractor.extract_all(text)
        total_spans = sum(len(items) for items in facts.values())
        return total_spans / len(words)
        
    def evaluate_admission(self, text: str, bypass_density_if_extracted=False) -> dict:
        is_high_risk = self.check_context_tags(text)
        density = self.check_density(text)
        
        reasons = []
        if is_high_risk:
            reasons.append("HIGH_RISK_CONTEXT_DETECTED")
            
        if density > self.density_threshold and not bypass_density_if_extracted:
            reasons.append(f"PROTECTED_SPAN_DENSITY_TOO_HIGH ({density:.2f} > {self.density_threshold})")
            
        return {
            "admit": len(reasons) == 0,
            "reasons": reasons,
            "density": density
        }
