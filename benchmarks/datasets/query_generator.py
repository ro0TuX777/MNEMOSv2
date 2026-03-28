"""
MNEMOS Benchmark - Query Generator
=====================================

Generates query sets with gold relevance labels for retrieval benchmarks.
Three query regimes: pure semantic, lightly filtered, heavily filtered.
"""

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnemos.engram.model import Engram


@dataclass
class BenchmarkQuery:
    """A benchmark query with optional filters and gold labels."""
    id: str
    text: str
    regime: str                          # "semantic", "light_filter", "heavy_filter"
    # Policy-required filters used for truth labels and compliance checks.
    filters: Dict[str, Any] = field(default_factory=dict)
    # Filters actually applied at retrieval time (can be relaxed for governance testing).
    retrieval_filters: Dict[str, Any] = field(default_factory=dict)
    gold_ids: List[str] = field(default_factory=list)  # known-relevant doc IDs
    domain: str = ""


# ─────────── Query templates per domain ───────────

QUERY_TEMPLATES = {
    "finance": [
        "What were the quarterly revenue results?",
        "Show me risk assessment findings",
        "Capital allocation policy requirements",
        "Audit findings and discrepancies",
        "Budget forecast growth projections",
        "Interest rate sensitivity analysis",
        "Tax provision adjustments needed",
        "Internal controls effectiveness",
        "Portfolio rebalancing strategy targets",
        "Compliance review material findings",
    ],
    "legal": [
        "Contract indemnification obligations",
        "Regulatory filing disclosure requirements",
        "Litigation reserve estimates",
        "Intellectual property protection status",
        "Data privacy compliance gaps",
        "Merger due diligence document review",
        "Board resolution authorisations",
        "Employment agreement provisions",
        "Subpoena discovery response",
        "Compliance training requirements",
    ],
    "medical": [
        "Clinical trial outcomes and efficacy",
        "Patient cohort analysis results",
        "Adverse event reports for drugs",
        "Diagnostic accuracy measurements",
        "Treatment protocol updates",
        "Genomic variant analysis findings",
        "Hospital readmission rate changes",
        "Drug interaction warnings",
        "Imaging review findings",
        "Post-market surveillance incidents",
    ],
    "technical": [
        "Service degradation incidents",
        "Deployment pipeline strategy",
        "Memory leak profiling results",
        "API endpoint latency measurements",
        "Database migration changes",
        "Caching layer hit rates",
        "Container image size optimization",
        "Load test throughput results",
        "Security vulnerability scan findings",
        "Monitoring stack retention config",
    ],
}


def _text_snippet(text: str, max_words: int = 14) -> str:
    words = text.strip().split()
    return " ".join(words[:max_words]) if words else text


def _compute_gold_labels(
    query: BenchmarkQuery,
    corpus: List[Engram],
    embeddings: Optional[np.ndarray] = None,
    query_embedding: Optional[np.ndarray] = None,
    top_k: int = 10,
) -> List[str]:
    """
    Compute gold relevance labels for a query.

    Strategy:
    - If embeddings provided: use exact cosine similarity on float32 embeddings
    - Else: use keyword overlap heuristic (domain + content matching)

    Filters are applied BEFORE ranking (true filtered retrieval).
    """
    # Apply filters to get candidate set
    candidates = list(range(len(corpus)))

    if query.filters:
        filtered = []
        for idx in candidates:
            e = corpus[idx]
            match = True
            for fk, fv in query.filters.items():
                if fk.startswith("metadata."):
                    meta_key = fk.split(".", 1)[1]
                    if meta_key == "timestamp_epoch_min":
                        e_ts = e.metadata.get("timestamp_epoch")
                        if e_ts is None or int(e_ts) < int(fv):
                            match = False
                            break
                    elif meta_key == "timestamp_epoch_max":
                        e_ts = e.metadata.get("timestamp_epoch")
                        if e_ts is None or int(e_ts) > int(fv):
                            match = False
                            break
                    elif e.metadata.get(meta_key) != fv:
                        match = False
                        break
                elif fk == "source":
                    if e.source != fv:
                        match = False
                        break
                elif fk == "confidence_min":
                    if e.confidence < float(fv):
                        match = False
                        break
            if match:
                filtered.append(idx)
        candidates = filtered

    if not candidates:
        return []

    # Rank by embedding similarity if available
    if embeddings is not None and query_embedding is not None:
        candidate_embeddings = embeddings[candidates]
        scores = candidate_embeddings @ query_embedding
        ranked_indices = np.argsort(-scores)[:top_k]
        return [corpus[candidates[i]].id for i in ranked_indices]

    # Fallback: domain + keyword heuristic
    query_words = set(query.text.lower().split())
    scored = []
    for idx in candidates:
        e = corpus[idx]
        content_words = set(e.content.lower().split())
        overlap = len(query_words & content_words)
        domain_match = 1.0 if e.metadata.get("domain") == query.domain else 0.0
        scored.append((idx, overlap + domain_match * 5))

    scored.sort(key=lambda x: -x[1])
    return [corpus[idx].id for idx, _ in scored[:top_k]]


def generate_queries(
    corpus: List[Engram],
    n_per_regime: int = 100,
    seed: int = 99,
    embeddings: Optional[np.ndarray] = None,
    query_embeddings: Optional[np.ndarray] = None,
    filter_mode: str = "strict",
) -> List[BenchmarkQuery]:
    """
    Generate benchmark queries across three regimes.

    Regime A (semantic):       No filters
    Regime B (light_filter):   1-2 metadata filters
    Regime C (heavy_filter):   3-4 metadata filters

    filter_mode:
      - strict: retrieval filters == policy-required filters
      - relaxed: retrieval filters are intentionally weaker than required filters
                 for filtered regimes, enabling governance-correctness evaluation.
    """
    env_mode = os.getenv("MNEMOS_BENCH_FILTER_MODE", "").strip().lower()
    if env_mode:
        filter_mode = env_mode

    if filter_mode not in {"strict", "relaxed"}:
        raise ValueError("filter_mode must be 'strict' or 'relaxed'")

    rng = random.Random(seed)
    domains = list(QUERY_TEMPLATES.keys())
    queries = []

    # Collect unique metadata values from corpus
    departments = list(set(e.metadata.get("department", "") for e in corpus))
    tenants = list(set(e.metadata.get("tenant", "") for e in corpus))
    clearances = list(set(e.metadata.get("clearance", "") for e in corpus))
    sources = list(set(e.source for e in corpus))
    ts_epochs = [
        int(e.metadata.get("timestamp_epoch"))
        for e in corpus
        if e.metadata.get("timestamp_epoch") is not None
    ]

    for regime_idx, (regime, n_filters) in enumerate([
        ("semantic", 0),
        ("light_filter", 2),
        ("heavy_filter", 4),
    ]):
        for i in range(n_per_regime):
            domain = rng.choice(domains)
            template = rng.choice(QUERY_TEMPLATES[domain])

            qid = hashlib.sha256(f"{seed}:{regime}:{i}:{template}".encode()).hexdigest()[:12]

            required_filters = {}
            if n_filters >= 1:
                required_filters["source"] = rng.choice(sources)
            if n_filters >= 2:
                required_filters["metadata.department"] = rng.choice(departments)
            if n_filters >= 3 and regime != "heavy_filter":
                required_filters["metadata.tenant"] = rng.choice(tenants)
            if n_filters >= 4 and regime != "heavy_filter":
                required_filters["metadata.clearance"] = rng.choice(clearances)
            if regime in {"light_filter", "heavy_filter"} and ts_epochs:
                anchor = rng.choice(ts_epochs)
                # Tune windows to keep governance stress measurable but non-saturated.
                # light_filter: medium pressure, heavy_filter: broader window so
                # compliance does not collapse to 0 across all runs.
                if regime == "heavy_filter":
                    window_days = 730
                else:
                    window_days = 365
                window = window_days * 24 * 3600
                required_filters["metadata.timestamp_epoch_min"] = anchor - window
                required_filters["metadata.timestamp_epoch_max"] = anchor + window

            retrieval_filters = dict(required_filters)
            if filter_mode == "relaxed":
                if regime == "light_filter":
                    # Relax by dropping one policy dimension.
                    retrieval_filters = {
                        k: v for k, v in required_filters.items()
                        if k not in {"metadata.department", "metadata.timestamp_epoch_min", "metadata.timestamp_epoch_max"}
                    }
                elif regime == "heavy_filter":
                    # Relax to source-only for stronger governance stress tests.
                    retrieval_filters = {
                        k: v for k, v in required_filters.items()
                        if k == "source"
                    }

            # Adversarial queries: keep policy filters, but use text from a conflicting
            # chunk (same source, different tenant/clearance/time window) to induce
            # semantic pull toward policy-invalid hits.
            if regime in {"light_filter", "heavy_filter"} and rng.random() < 0.10:
                req_source = required_filters.get("source")
                req_tenant = required_filters.get("metadata.tenant")
                req_clearance = required_filters.get("metadata.clearance")
                ts_min = required_filters.get("metadata.timestamp_epoch_min")
                ts_max = required_filters.get("metadata.timestamp_epoch_max")

                candidates = []
                for e in corpus:
                    if req_source and e.source != req_source:
                        continue
                    mismatches = 0
                    if req_tenant and e.metadata.get("tenant") != req_tenant:
                        mismatches += 1
                    if req_clearance and e.metadata.get("clearance") != req_clearance:
                        mismatches += 1
                    if ts_min is not None and ts_max is not None:
                        e_ts = e.metadata.get("timestamp_epoch")
                        if e_ts is not None and not (int(ts_min) <= int(e_ts) <= int(ts_max)):
                            mismatches += 1
                    if mismatches > 0:
                        candidates.append(e)

                if candidates:
                    template = _text_snippet(rng.choice(candidates).content)

            query = BenchmarkQuery(
                id=qid,
                text=template,
                regime=regime,
                filters=required_filters,
                retrieval_filters=retrieval_filters,
                domain=domain,
            )

            # Compute gold labels
            q_emb = None
            if query_embeddings is not None:
                q_idx = regime_idx * n_per_regime + i
                if q_idx < len(query_embeddings):
                    q_emb = query_embeddings[q_idx]

            query.gold_ids = _compute_gold_labels(
                query, corpus, embeddings, q_emb, top_k=10
            )
            queries.append(query)

    return queries


def save_queries(queries: List[BenchmarkQuery], path: Path):
    """Save queries to JSON."""
    data = [
        {
            "id": q.id,
            "text": q.text,
            "regime": q.regime,
            "filters": q.filters,
            "retrieval_filters": q.retrieval_filters,
            "gold_ids": q.gold_ids,
            "domain": q.domain,
        }
        for q in queries
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_queries(path: Path) -> List[BenchmarkQuery]:
    """Load queries from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [
        BenchmarkQuery(
            id=d["id"],
            text=d["text"],
            regime=d["regime"],
            filters=d.get("filters", {}),
            retrieval_filters=d.get("retrieval_filters", d.get("filters", {})),
            gold_ids=d.get("gold_ids", []),
            domain=d.get("domain", ""),
        )
        for d in data
    ]
