"""Compute-mode resolution for generated MNEMOS deployments."""

from __future__ import annotations

from dataclasses import dataclass

from installer.probes import ProbeResults


@dataclass(frozen=True)
class ComputeMode:
    """Resolved compute settings for generated installer artifacts."""

    mode: str
    gpu_device: str
    reason: str


def resolve_compute_mode(requested: str, probes: ProbeResults) -> ComputeMode:
    """Resolve requested compute mode against host platform capabilities."""

    requested_normalized = (requested or "auto").lower()
    if requested_normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError("compute mode must be one of: auto, cuda, cpu")

    os_name = (probes.os_name or "").lower()
    is_macos = os_name in {"darwin", "macos", "mac os", "mac os x"}

    if requested_normalized == "cpu":
        return ComputeMode("cpu", "cpu", "CPU mode explicitly selected")

    if requested_normalized == "cuda":
        if is_macos:
            return ComputeMode(
                "cpu",
                "cpu",
                "macOS detected - NVIDIA runtime is not supported; using CPU mode",
            )
        if not probes.nvidia_runtime:
            return ComputeMode(
                "cpu",
                "cpu",
                "NVIDIA container runtime not detected; using CPU mode",
            )
        return ComputeMode("cuda", "cuda", "CUDA mode explicitly selected")

    if is_macos:
        return ComputeMode(
            "cpu",
            "cpu",
            "macOS detected - NVIDIA runtime is not supported; using CPU mode",
        )

    if probes.gpu_available and probes.nvidia_runtime:
        return ComputeMode("cuda", "cuda", "NVIDIA GPU and container runtime detected")

    return ComputeMode("cpu", "cpu", "NVIDIA GPU/runtime not detected; using CPU mode")

