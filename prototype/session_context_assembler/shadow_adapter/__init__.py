"""Isolated consumer-neutral read-only shadow adapter (ADR 0008).

No network listener, external consumer connection, live routing, SDK surface,
durable-memory write, retrieval change, or governance mutation exists here.
"""

from .adapter import LocalShadowAdapter
from .kill_switch import KillSwitch
from .models import LocalAssemblyInputs, LocalTransportContext, PolicySnapshot

__all__ = [
    "KillSwitch",
    "LocalAssemblyInputs",
    "LocalShadowAdapter",
    "LocalTransportContext",
    "PolicySnapshot",
]
