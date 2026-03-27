"""
MNEMOS SDK — Boundary Configuration.

Mirrors the MSF BoundaryConfig pattern with MNEMOS-specific defaults.
All settings are driven by ``MNEMOS_*`` environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class MnemosConfig:
    """Connection and resilience settings for the MNEMOS boundary."""

    # Connection
    base_url: str = "http://localhost:8700"
    token: str = ""

    # Resilience
    timeout_s: float = 5.0
    ready_wait_s: float = 10.0
    retries: int = 2
    retry_delay_s: float = 0.5

    # Feature toggle
    enabled: bool = True

    # Auto-start (for local dev / compose scenarios)
    autostart_on_demand: bool = False
    autostart_cmd: str = ""
    autostart_timeout_s: float = 25.0

    @classmethod
    def from_env(cls, prefix: str = "MNEMOS") -> "MnemosConfig":
        """Build configuration from environment variables.

        Default prefix is ``MNEMOS``, producing env vars like:
        - ``MNEMOS_BASE_URL``
        - ``MNEMOS_TOKEN``
        - ``MNEMOS_TIMEOUT_S``
        - ``MNEMOS_READY_WAIT_S``
        - ``MNEMOS_RETRIES``
        - ``MNEMOS_RETRY_DELAY_S``
        - ``MNEMOS_ENABLED``
        - ``MNEMOS_AUTOSTART_ON_DEMAND``
        - ``MNEMOS_AUTOSTART_CMD``
        """
        key = prefix.strip().upper()

        def _env(name: str, default: str = "") -> str:
            return os.getenv(f"{key}_{name}", default).strip()

        def _bool(raw: str, default: bool) -> bool:
            text = raw.strip().lower()
            if not text:
                return default
            return text in {"1", "true", "yes", "on"}

        def _float(raw: str, default: float) -> float:
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        def _int(raw: str, default: int) -> int:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                return default

        return cls(
            base_url=_env("BASE_URL", "http://localhost:8700").rstrip("/"),
            token=_env("TOKEN", ""),
            timeout_s=_float(_env("TIMEOUT_S", "5"), 5.0),
            ready_wait_s=_float(_env("READY_WAIT_S", "10"), 10.0),
            retries=_int(_env("RETRIES", "2"), 2),
            retry_delay_s=_float(_env("RETRY_DELAY_S", "0.5"), 0.5),
            enabled=_bool(_env("ENABLED", "true"), True),
            autostart_on_demand=_bool(_env("AUTOSTART_ON_DEMAND", "false"), False),
            autostart_cmd=_env("AUTOSTART_CMD", ""),
            autostart_timeout_s=_float(_env("AUTOSTART_TIMEOUT_S", "25"), 25.0),
        )
