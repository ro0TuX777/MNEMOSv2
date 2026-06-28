"""Offline-only prototype for an EpiCache-inspired session-context assembler.

Shadow research lane, Phase 1. See
docs/adr/0007-session-context-assembler-shadow-only.md and
docs/session_context_assembler_spec.md.

This package must never:
  - add a production route or connect to any authorized-consumer runtime path
  - write Engrams, summaries, Resolution Engrams, or evidence bundles
  - alter retrieval ranking or retrieval configuration
  - alter authority, trust, contradiction, governance, or promotion state
  - use corpus `episode_hint` labels as a clustering/selection input
    (see diagnostics.py for the one permitted offline-scoring use)

It depends only on the Python standard library and the frozen corpus under
benchmarks/truthsets/session_context_assembler_r0.json. It imports nothing
from mnemos/, service/, or mnemos_sdk/.
"""

PROTOTYPE_VERSION = "0.1.0-prototype"
