import re

PROTECTED_PATTERNS = {
    "source_pointer": r"(?i)source:\s*\[.*?\]|source\s+\d+",
    "governance_flag": r"(?i)governance:|approval_required|risk_label",
    "evidence_gap": r"(?i)\[EVIDENCE_GAP\]|unknown|insufficient context",
    "contradiction": r"(?i)\[CONTRADICTION\]|conflict|disagree",
    "date": r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
    "number": r"\b\d+(?:\.\d+)?(?:%|M|K|B)?\b",
    "config_key": r"(?i)config_[a-z_]+",
    "exception_clause": r"(?i)except|unless|however|but",
    "negation": r"(?i)\bnot\b|\bnever\b|\bno\b|\bcannot\b",
}
