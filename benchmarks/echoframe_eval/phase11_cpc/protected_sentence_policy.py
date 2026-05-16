import re

PROTECTED_PATTERNS = {
    "source_pointer": r"(?i)source:\s*\[.*?\]|source\s+\d+|Source:",
    "page_ref": r"(?i)page\s+\d+|p\.\s*\d+",
    "classification": r"\b(?:CUI|FOUO|SCI|SAP|SECRET|TOP SECRET|UNCLASSIFIED|SIGINT)\b",
    "dod_ref": r"\b(?:DOD|DoDM|DoDI|NSA|CIA|MCO|MCRP|NAVMC|CNGBM)\b",
    "section_num": r"(?i)\bsection\s+\d+|paragraph\s+\d+",
    "date": r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2} [A-Z][a-z]{2,8} \d{4})\b",
    "number": r"\b\d+(?:\.\d+)?(?:%|M|K|B)?\b",
    "acronym": r"\b[A-Z]{2,5}\b",
    "obligation": r"(?i)\b(?:must|shall|required)\b",
    "negation": r"(?i)\b(?:not|never|cannot|may not|prohibited)\b",
    "exception": r"(?i)\b(?:except|unless)\b",
    "approval": r"(?i)\b(?:approval|authorized|authorization)\b",
    "markers": r"(?i)\[EVIDENCE_GAP\]|\[CONTRADICTION\]|unknown"
}

class ProtectedSentencePolicy:
    def evaluate(self, sentence: str) -> dict:
        matched_flags = []
        for category, pattern in PROTECTED_PATTERNS.items():
            if re.search(pattern, sentence):
                matched_flags.append(category)
        return {
            "is_protected": len(matched_flags) > 0,
            "flags": matched_flags
        }
