"""Associative Routing E2 — opt-in, bounded candidate-expansion lane.

Public surface: ``CandidateExpansionEngine`` and the process-wide
``default_engine`` singleton, plus the flag/kill-switch/bounds constants in
``config``. See ``engine.py`` for the invariants this module must preserve.
"""

from .config import (
    CANDIDATE_EXPANSION_ENABLE_ENV,
    CANDIDATE_EXPANSION_FLAG,
    E2_FIXTURES_DIR,
    MAX_ADDED_CANDIDATES,
    MAX_EXPANSION_LATENCY_MS,
    MAX_PATHS,
    MAX_TRAVERSAL_DEPTH,
)
from .engine import CandidateExpansionEngine, default_engine

__all__ = [
    "CANDIDATE_EXPANSION_ENABLE_ENV",
    "CANDIDATE_EXPANSION_FLAG",
    "CandidateExpansionEngine",
    "E2_FIXTURES_DIR",
    "MAX_ADDED_CANDIDATES",
    "MAX_EXPANSION_LATENCY_MS",
    "MAX_PATHS",
    "MAX_TRAVERSAL_DEPTH",
    "default_engine",
]
