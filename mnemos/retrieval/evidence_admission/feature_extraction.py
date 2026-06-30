"""Deterministic feature extraction for Evidence Admission R0.

Pure functions only: ``query`` text plus an ``AdmissionRequestContext`` in,
a plain, JSON-serializable feature dict out. No I/O, no model calls, no
randomness — the same ``(query, context)`` pair always produces the same
feature dict, which is what the "deterministic recommendation for same
input snapshot" required test checks.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict

from .models import AdmissionRequestContext

#: Below this token count a query is treated as weak/underspecified absent
#: any cue/tag/artifact-id match. An R0 development default, not tuned
#: against any frozen or fresh verification pack.
MIN_TOKEN_COUNT_FOR_CONFIDENT_SCOPE = 2

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_query(query: str) -> str:
    return " ".join(str(query).split()).strip().lower()


def _token_count(query: str) -> int:
    return len(_TOKEN_RE.findall(normalize_query(query)))


def extract_features(query: str, context: AdmissionRequestContext) -> Dict[str, Any]:
    """Build a deterministic, JSON-serializable feature snapshot."""
    normalized = normalize_query(query)
    return {
        "normalized_query": normalized,
        "query_token_count": _token_count(query),
        "top_k": int(context.top_k),
        "latency_budget_ms": context.latency_budget_ms,
        "known_collection_scope": context.known_collection_scope,
        "cache_fresh_and_scope_matched": bool(
            context.cache_lookup and context.cache_lookup.fresh_and_scope_matched
        ),
        "cache_name": context.cache_lookup.cache_name if context.cache_lookup else None,
        "complexity_label": context.complexity_label,
        "complexity_confidence": context.complexity_confidence,
        "cue_terms_matched": sorted(context.cue_terms_matched),
        "tag_terms_matched": sorted(context.tag_terms_matched),
        "explicit_artifact_ids": sorted(context.explicit_artifact_ids),
        "associative_expansion_globally_enabled": bool(
            context.associative_expansion_globally_enabled
        ),
        "prior_verified_context_supplied": bool(context.prior_verified_context_supplied),
    }


def input_snapshot_hash(features: Dict[str, Any]) -> str:
    """Stable hash of a feature snapshot, used as the response's
    ``input_snapshot`` field so identical inputs are verifiably identical
    across requests/runs."""
    canonical = json.dumps(features, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
