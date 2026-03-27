"""
MNEMOS SDK — Boundary adapter client for MNEMOS memory service.

Usage:
    from mnemos_sdk import MnemosClient, MnemosConfig

    config = MnemosConfig.from_env()
    client = MnemosClient(config)
    client.wait_until_ready()

    result = client.search("quantum entanglement", top_k=5)
"""

from mnemos_sdk.config import MnemosConfig
from mnemos_sdk.client import MnemosClient

__all__ = ["MnemosClient", "MnemosConfig"]
