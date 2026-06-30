"""Configuration constants for the Associative Routing E1 shadow adapter.

E1 is a shadow-only, opt-in lane (see ``docs/associative_routing_e1_design_note.md``
once written, and the authorization header carried in the E1 task spec). Nothing
in this module changes default retrieval behavior by itself.
"""

from __future__ import annotations

from pathlib import Path

#: Request/body field name callers must set to ``true`` to receive a shadow block.
#: Absent or false ⇒ no shadow evaluation, no change to the delivered response.
ASSOCIATIVE_SHADOW_FLAG = "associative_routing_shadow"

#: Kill switch: if set to "true" (case-insensitive), the adapter always reports
#: status "unavailable" regardless of the request flag. Mirrors the
#: VFR_DISABLE_SHADOW_MODE pattern in mnemos/retrieval/shadow_retriever.py.
ASSOCIATIVE_SHADOW_DISABLE_ENV = "MNEMOS_DISABLE_ASSOCIATIVE_SHADOW"

#: Expanded E1 fixture corpus (GateMem G4/G5 + R0 retrieval-hygiene + AI-developer
#: memory trial + ADR sources). Deliberately a separate directory from
#: prototype/associative_routing_e0/fixtures so the frozen E0 prototype and its
#: 24 passing tests are never mutated by E1 corpus growth.
E1_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
