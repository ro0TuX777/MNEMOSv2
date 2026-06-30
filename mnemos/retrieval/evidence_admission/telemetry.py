"""Telemetry redaction helpers for Evidence Admission R0.

R0's shadow block is safe to return directly in an API response (it never
contains raw query text or candidate content — see ``models.py``), but any
persisted/logged comparison record must not retain raw query text either.
This module provides the one redaction helper required by the "telemetry
redaction" required test.
"""

from __future__ import annotations

from typing import Any, Dict

from .feature_extraction import normalize_query

#: Keys that may carry raw query text and must never reach a persisted
#: telemetry/comparison record unredacted.
_RAW_QUERY_KEYS = ("query", "raw_query", "normalized_query")


def redact_for_telemetry(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``record`` with raw query text replaced by length
    and a normalized-query hash placeholder; all other fields pass through
    unchanged."""
    redacted = dict(record)
    for key in _RAW_QUERY_KEYS:
        if key in redacted and isinstance(redacted[key], str):
            raw = redacted[key]
            redacted[key] = {
                "redacted": True,
                "length": len(raw),
                "normalized_length": len(normalize_query(raw)),
            }
    return redacted
