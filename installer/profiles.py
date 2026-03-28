"""
MNEMOS Installer — Profile Definitions
========================================

Defines deployment profiles and their requirements.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProfileSpec:
    """Specification for a MNEMOS deployment profile."""
    name: str
    display_name: str
    description: str
    services: List[str]
    containers: int
    tiers: List[str]
    min_ram_gb: float = 1.0
    gpu_required: bool = True
    requires_docker: bool = True
    env_vars: dict = field(default_factory=dict)
    best_for: str = ""


PROFILES = {
    "core_memory_appliance": ProfileSpec(
        name="core_memory_appliance",
        display_name="Core Memory Appliance",
        description="Qdrant + PostgreSQL + MNEMOS (3 containers)",
        services=["qdrant", "postgres", "mnemos"],
        containers=3,
        tiers=["qdrant"],
        min_ram_gb=2.0,
        gpu_required=True,
        requires_docker=True,
        env_vars={
            "MNEMOS_PROFILE": "core_memory_appliance",
            "MNEMOS_TIERS": "qdrant",
        },
        best_for="Semantic memory, agent systems, general-purpose RAG",
    ),
    "governance_native": ProfileSpec(
        name="governance_native",
        display_name="Governance Native",
        description="PostgreSQL/pgvector + MNEMOS (2 containers)",
        services=["postgres", "mnemos"],
        containers=2,
        tiers=["pgvector"],
        min_ram_gb=1.5,
        gpu_required=True,
        requires_docker=True,
        env_vars={
            "MNEMOS_PROFILE": "governance_native",
            "MNEMOS_TIERS": "pgvector",
        },
        best_for="Provenance-heavy, metadata-filtered, compliance-aware retrieval",
    ),
    "custom_manual": ProfileSpec(
        name="custom_manual",
        display_name="Custom Manual",
        description="Operator-defined configuration",
        services=[],
        containers=0,
        tiers=[],
        min_ram_gb=1.0,
        gpu_required=False,
        requires_docker=False,
        env_vars={
            "MNEMOS_PROFILE": "custom_manual",
        },
        best_for="Advanced operators, multi-tier setups, experimentation",
    ),
}


def get_profile(name: str) -> Optional[ProfileSpec]:
    """Get a profile by name."""
    return PROFILES.get(name)


def list_profiles() -> List[ProfileSpec]:
    """List all available profiles."""
    return list(PROFILES.values())
