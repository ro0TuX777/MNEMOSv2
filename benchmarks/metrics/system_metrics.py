"""
MNEMOS Benchmark - System Metrics
====================================

Captures host environment for benchmark reproducibility.
"""

import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvironmentSnapshot:
    """Captured environment at benchmark time."""
    timestamp: str = ""
    os_name: str = ""
    os_version: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpu_name: str = ""
    vram_mb: int = 0
    python_version: str = ""
    numpy_version: str = ""
    docker_version: str = ""
    qdrant_version: str = ""
    postgres_version: str = ""


def _run(cmd: list, timeout: int = 5) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def capture_environment() -> EnvironmentSnapshot:
    """Capture current system environment."""
    import numpy as np

    snap = EnvironmentSnapshot(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        os_name=platform.system(),
        os_version=platform.version(),
        cpu_model=platform.processor() or "unknown",
        cpu_cores=os.cpu_count() or 0,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
    )

    # RAM
    try:
        import psutil
        snap.ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass

    # GPU
    gpu_out = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader,nounits"])
    if gpu_out:
        parts = gpu_out.split(",")
        snap.gpu_name = parts[0].strip() if parts else ""
        snap.vram_mb = int(float(parts[1].strip())) if len(parts) > 1 else 0

    # Docker
    snap.docker_version = _run(["docker", "version", "--format", "{{.Server.Version}}"])

    return snap


def snapshot_to_dict(snap: EnvironmentSnapshot) -> dict:
    """Convert snapshot to dict for JSON serialization."""
    return {
        "timestamp": snap.timestamp,
        "os": f"{snap.os_name} {snap.os_version}",
        "cpu": snap.cpu_model,
        "cpu_cores": snap.cpu_cores,
        "ram_gb": snap.ram_gb,
        "gpu": snap.gpu_name,
        "vram_mb": snap.vram_mb,
        "python": snap.python_version,
        "numpy": snap.numpy_version,
        "docker": snap.docker_version,
    }
