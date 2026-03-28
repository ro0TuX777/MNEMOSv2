"""
MNEMOS Installer — System Probes
==================================

Detects host capabilities: GPU, RAM, Docker, NVIDIA runtime,
existing Postgres, disk space.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class ProbeResults:
    """Detected host capabilities."""
    gpu_available: bool = False
    gpu_name: str = ""
    vram_mb: int = 0
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    docker_available: bool = False
    nvidia_runtime: bool = False
    existing_postgres: bool = False
    os_name: str = ""
    cpu_cores: int = 0


def _run_cmd(cmd: list, timeout: int = 10) -> str:
    """Run a command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def probe_gpu() -> tuple:
    """Detect NVIDIA GPU via nvidia-smi."""
    output = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total",
                       "--format=csv,noheader,nounits"])
    if output:
        parts = output.split(",")
        name = parts[0].strip() if len(parts) > 0 else ""
        vram = int(float(parts[1].strip())) if len(parts) > 1 else 0
        return True, name, vram
    return False, "", 0


def probe_ram() -> float:
    """Detect total RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        # Fallback: read from OS
        try:
            if os.name == "nt":
                output = _run_cmd(["wmic", "os", "get", "TotalVisibleMemorySize"])
                lines = [l.strip() for l in output.split("\n") if l.strip().isdigit()]
                if lines:
                    return round(int(lines[0]) / (1024 * 1024), 1)
            else:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            return round(kb / (1024 * 1024), 1)
        except Exception:
            pass
    return 0.0


def probe_disk() -> float:
    """Detect free disk space in GB."""
    try:
        usage = shutil.disk_usage(os.getcwd())
        return round(usage.free / (1024 ** 3), 1)
    except Exception:
        return 0.0


def probe_docker() -> bool:
    """Check if Docker is available."""
    return bool(_run_cmd(["docker", "version", "--format", "{{.Server.Version}}"]))


def probe_nvidia_runtime() -> bool:
    """Check if NVIDIA Container Toolkit / runtime is available."""
    output = _run_cmd(["docker", "info", "--format", "{{.Runtimes}}"])
    return "nvidia" in output.lower()


def probe_existing_postgres() -> bool:
    """Check if a Postgres container is already running."""
    output = _run_cmd(["docker", "ps", "--filter", "ancestor=postgres",
                       "--format", "{{.Names}}"])
    return bool(output)


def probe_cpu_cores() -> int:
    """Detect number of CPU cores."""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def run_probes() -> ProbeResults:
    """Run all system probes and return results."""
    gpu_ok, gpu_name, vram = probe_gpu()

    return ProbeResults(
        gpu_available=gpu_ok,
        gpu_name=gpu_name,
        vram_mb=vram,
        ram_gb=probe_ram(),
        disk_free_gb=probe_disk(),
        docker_available=probe_docker(),
        nvidia_runtime=probe_nvidia_runtime(),
        existing_postgres=probe_existing_postgres(),
        os_name=os.name,
        cpu_cores=probe_cpu_cores(),
    )
