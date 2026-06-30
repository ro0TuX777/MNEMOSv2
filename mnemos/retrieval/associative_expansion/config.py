"""Configuration constants for the Associative Routing E2 candidate-expansion engine.

E2 is opt-in, bounded, and double-gated: a request must set the flag AND the
global kill switch must be enabled, mirroring the existing
ValidatedFactShadowRetriever double opt-in pattern
(mnemos/retrieval/shadow_retriever.py). Nothing here changes default
retrieval behavior by itself.
"""

from __future__ import annotations

from pathlib import Path

#: Request/body field name callers must set to ``true`` to request expansion.
CANDIDATE_EXPANSION_FLAG = "associative_candidate_expansion"

#: Global enable switch. Defaults to disabled: even if a caller sets the
#: request flag, expansion only runs when this is explicitly "true".
CANDIDATE_EXPANSION_ENABLE_ENV = "MNEMOS_ASSOCIATIVE_CANDIDATE_EXPANSION_ENABLED"

#: Expanded E2 fixture corpus: E1's corpus plus a third, unrelated
#: documentation family (PIT-8/PIT-9B/PIT-10) for evaluation fairness. A
#: separate directory from associative_shadow/fixtures so E1 stays frozen.
E2_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: Conservative bounds (see E2 authorization spec). E0's router is
#: structurally depth-1 cue->tag->content, so MAX_TRAVERSAL_DEPTH is
#: satisfied by construction and recorded rather than enforced.
MAX_PATHS = 3
MAX_ADDED_CANDIDATES = 3
MAX_TRAVERSAL_DEPTH = 2
MAX_EXPANSION_LATENCY_MS = 10.0
