"""
MNEMOS Benchmark - Query Generator
=====================================

Generates query sets with gold relevance labels for retrieval benchmarks.
Three query regimes: pure semantic, lightly filtered, heavily filtered.
"""

import hashlib
import json
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
    filters: Dict[str, Any] = field(default_factory=dict)
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
                    if e.metadata.get(meta_key) != fv:
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
) -> List[BenchmarkQuery]:
    """
    Generate benchmark queries across three regimes.

    Regime A (semantic):       No filters
    Regime B (light_filter):   1-2 metadata filters
    Regime C (heavy_filter):   3-4 metadata filters
    """
    rng = random.Random(seed)
    domains = list(QUERY_TEMPLATES.keys())
    queries = []

    # Collect unique metadata values from corpus
    departments = list(set(e.metadata.get("department", "") for e in corpus))
    tenants = list(set(e.metadata.get("tenant", "") for e in corpus))
    clearances = list(set(e.metadata.get("clearance", "") for e in corpus))
    sources = list(set(e.source for e in corpus))

    for regime_idx, (regime, n_filters) in enumerate([
        ("semantic", 0),
        ("light_filter", 2),
        ("heavy_filter", 4),
    ]):
        for i in range(n_per_regime):
            domain = rng.choice(domains)
            template = rng.choice(QUERY_TEMPLATES[domain])

            qid = hashlib.sha256(f"{seed}:{regime}:{i}:{template}".encode()).hexdigest()[:12]

            filters = {}
            if n_filters >= 1:
                filters["source"] = rng.choice(sources)
            if n_filters >= 2:
                filters["metadata.department"] = rng.choice(departments)
            if n_filters >= 3:
                filters["metadata.tenant"] = rng.choice(tenants)
            if n_filters >= 4:
                filters["metadata.clearance"] = rng.choice(clearances)

            query = BenchmarkQuery(
                id=qid,
                text=template,
                regime=regime,
                filters=filters,
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
            gold_ids=d.get("gold_ids", []),
            domain=d.get("domain", ""),
        )
        for d in data
    ]
