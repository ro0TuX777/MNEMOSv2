import re

DEEP_PATTERNS = {
    "date": r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2} [A-Z][a-z]{2,8} \d{4})\b",
    "numeric_code": r"\b\d{4}\.\d{2}-[A-Z](?:\(\d\))?\b", # e.g. 5240.01-A(1)
    "number": r"\b\d+(?:\.\d+)?(?:%|M|K|B)?\b",
    "dod_identifier": r"\b(?:DOD|DoDM|DoDI|NSA|CIA|MCO|MCRP|NAVMC|CNGBM)\b",
    "section_ref": r"(?i)\bsection\s+\d+[a-z]?\b",
    "paragraph_ref": r"(?i)\bparagraph\s+\d+\b",
    "acronym": r"\b[A-Z]{2,}\b", # Simple heuristic for acronyms
    "classification": r"\b(?:CUI|FOUO|SCI|SAP|SECRET|TOP SECRET|UNCLASSIFIED)\b",
    "negation": r"(?i)\b(?:not|never|no|cannot|prohibited)\b",
    "exception": r"(?i)\b(?:except|unless|however|but)\b",
    "obligation": r"(?i)\b(?:shall|must|will|required)\b"
}

class DeepProtectedSpanExtractor:
    def extract_all(self, text: str) -> dict:
        facts = {}
        for category, pattern in DEEP_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                # Store unique matches
                facts[category] = list(set(matches))
        return facts
        
    def format_as_facts(self, facts: dict) -> str:
        lines = []
        for category, items in facts.items():
            for item in items:
                # Truncate if too long
                display = item if len(item) < 50 else item[:47] + "..."
                lines.append(f"- {category}: {display}")
        return "\n".join(lines)
