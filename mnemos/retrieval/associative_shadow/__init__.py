"""Associative Routing E1 — opt-in, read-only shadow retrieval lane.

Public surface: ``AssociativeShadowAdapter`` and the process-wide
``default_adapter`` singleton, plus the flag/kill-switch/fixtures constants in
``config``. See ``adapter.py`` for the isolation invariants this module must
preserve.
"""

from .adapter import AssociativeShadowAdapter, default_adapter
from .config import (
    ASSOCIATIVE_SHADOW_DISABLE_ENV,
    ASSOCIATIVE_SHADOW_FLAG,
    E1_FIXTURES_DIR,
)

__all__ = [
    "ASSOCIATIVE_SHADOW_DISABLE_ENV",
    "ASSOCIATIVE_SHADOW_FLAG",
    "AssociativeShadowAdapter",
    "E1_FIXTURES_DIR",
    "default_adapter",
]
