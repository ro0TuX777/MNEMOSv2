"""
MNEMOS Installer — Profile Recommendation Engine
===================================================

Decision tree (not scoring engine) that recommends a profile
based on user answers and probe results.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from installer.profiles import ProfileSpec, PROFILES
from installer.questions import UserAnswers
from installer.probes import ProbeResults


@dataclass
class Recommendation:
    """Profile recommendation with reasoning."""
    profile: ProfileSpec
    confidence: str = "high"       # high, medium, low
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)


def recommend(answers: UserAnswers, probes: ProbeResults) -> Recommendation:
    """
    Decision tree for profile recommendation.

    Logic:
      1. manual preference → custom_manual
      2. strict_filters OR metadata_governance priority → governance_native
      3. everything else → core_memory_appliance
      4. probes eliminate impossible profiles
    """

    # Step 1: manual override
    if answers.prefer_manual:
        return Recommendation(
            profile=PROFILES["custom_manual"],
            confidence="high",
            reasons=["User chose manual control"],
        )

    # Step 2: governance decision
    governance = (
        answers.strict_filters
        or answers.priority == "metadata_governance"
        or answers.use_case == "compliance_governed"
    )

    if governance:
        selected = PROFILES["governance_native"]
        reasons = []
        if answers.strict_filters:
            reasons.append("Strict metadata/provenance filtering required")
        if answers.priority == "metadata_governance":
            reasons.append("Metadata governance is top priority")
        if answers.use_case == "compliance_governed":
            reasons.append("Compliance-governed use case selected")
        reasons.append("pgvector enables SQL WHERE + ANN in one query")
        reasons.append("Fewer containers (2 vs 3) — simpler operations")
    else:
        selected = PROFILES["core_memory_appliance"]
        reasons = ["Semantic retrieval is the primary need"]
        if answers.use_case == "agent_memory":
            reasons.append("Agent memory maps naturally to Qdrant HNSW")
        if answers.scale in ("100k_to_1m", "over_1m"):
            reasons.append(f"Scale ({answers.scale}) benefits from dedicated Qdrant service")
        reasons.append("Qdrant provides payload filtering, replication, and sharding")

    # Step 3: probe-based warnings
    warnings = []
    alternatives = []

    if not probes.gpu_available:
        warnings.append(
            "⚠️  No GPU detected. MNEMOS requires CUDA for embedding inference. "
            "CPU fallback exists but is significantly slower."
        )

    if not probes.docker_available:
        warnings.append(
            "⚠️  Docker not detected. MNEMOS requires Docker for deployment. "
            "Install Docker and NVIDIA Container Toolkit first."
        )

    if probes.docker_available and not probes.nvidia_runtime:
        warnings.append(
            "⚠️  NVIDIA Container Toolkit not detected. "
            "GPU acceleration will not work in Docker containers."
        )

    if probes.ram_gb > 0 and probes.ram_gb < selected.min_ram_gb:
        warnings.append(
            f"⚠️  Available RAM ({probes.ram_gb} GB) is below the "
            f"recommended minimum ({selected.min_ram_gb} GB) for {selected.display_name}."
        )

    if probes.disk_free_gb > 0 and probes.disk_free_gb < 5:
        warnings.append(
            f"⚠️  Low disk space ({probes.disk_free_gb} GB free). "
            "MNEMOS needs space for model downloads and vector indexes."
        )

    # Suggest alternatives
    if selected.name == "core_memory_appliance":
        alternatives.append("governance_native")
    elif selected.name == "governance_native":
        alternatives.append("core_memory_appliance")

    confidence = "high"
    if warnings:
        confidence = "medium" if len(warnings) <= 2 else "low"

    return Recommendation(
        profile=selected,
        confidence=confidence,
        reasons=reasons,
        warnings=warnings,
        alternatives=alternatives,
    )
