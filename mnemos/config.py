"""
MNEMOS Configuration
=====================

Environment-variable driven configuration for all MNEMOS components.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MnemosConfig:
    """Central configuration for MNEMOS service."""

    # Retrieval tiers
    tiers: List[str] = field(default_factory=lambda: ["qdrant"])
    embedding_model: str = "all-MiniLM-L6-v2"
    colbert_model: str = "colbertv2.0"

    # TurboQuant compression
    quant_bits: int = 4   # 0 = disabled, 1-4 = bit-width

    # Forensic audit
    audit_enabled: bool = True
    audit_db_path: str = "data/audit.db"       # SQLite fallback
    audit_retention_days: int = 90
    postgres_dsn: str = ""                      # PostgreSQL DSN (overrides SQLite)

    # API
    port: int = 8700
    token: str = ""
    log_level: str = "INFO"

    # GPU
    gpu_device: str = "cuda"                    # "cuda", "cpu", or "cuda:N"

    # Data directories
    data_dir: str = "data"
    lance_dir: str = "data/lance"
    colbert_index_dir: str = "data/colbert"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "mnemos_engrams"

    @classmethod
    def from_env(cls) -> "MnemosConfig":
        """Build configuration from environment variables."""
        tiers_raw = os.getenv("MNEMOS_TIERS", "qdrant")
        tiers = [t.strip() for t in tiers_raw.split(",") if t.strip()]

        quant_bits = int(os.getenv("MNEMOS_QUANT_BITS", "4"))
        if quant_bits < 0 or quant_bits > 4:
            logger.warning(f"Invalid MNEMOS_QUANT_BITS={quant_bits}, defaulting to 4")
            quant_bits = 4

        config = cls(
            tiers=tiers,
            embedding_model=os.getenv("MNEMOS_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            colbert_model=os.getenv("MNEMOS_COLBERT_MODEL", "colbertv2.0"),
            quant_bits=quant_bits,
            audit_enabled=os.getenv("MNEMOS_AUDIT_ENABLED", "true").lower() in ("true", "1", "yes"),
            audit_db_path=os.getenv("MNEMOS_AUDIT_DB", "data/audit.db"),
            audit_retention_days=int(os.getenv("MNEMOS_AUDIT_RETENTION_DAYS", "90")),
            postgres_dsn=os.getenv("MNEMOS_POSTGRES_DSN", ""),
            port=int(os.getenv("MNEMOS_PORT", "8700")),
            token=os.getenv("MNEMOS_TOKEN", ""),
            log_level=os.getenv("MNEMOS_LOG_LEVEL", "INFO"),
            gpu_device=os.getenv("MNEMOS_GPU_DEVICE", "cuda"),
            data_dir=os.getenv("MNEMOS_DATA_DIR", "data"),
            lance_dir=os.getenv("MNEMOS_LANCE_DIR", "data/lance"),
            colbert_index_dir=os.getenv("MNEMOS_COLBERT_DIR", "data/colbert"),
            qdrant_url=os.getenv("MNEMOS_QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("MNEMOS_QDRANT_COLLECTION", "mnemos_engrams"),
        )

        logger.info(
            f"⚙️ MNEMOS config: tiers={config.tiers}, quant={config.quant_bits}-bit, "
            f"audit={config.audit_enabled}, gpu={config.gpu_device}"
        )
        return config

    @property
    def has_qdrant(self) -> bool:
        return "qdrant" in self.tiers

    @property
    def has_lancedb(self) -> bool:
        return "lancedb" in self.tiers

    @property
    def has_colbert(self) -> bool:
        return "colbert" in self.tiers

    @property
    def has_compression(self) -> bool:
        return self.quant_bits > 0

    @property
    def has_postgres(self) -> bool:
        return bool(self.postgres_dsn)


# Global config singleton
_config: Optional[MnemosConfig] = None


def get_config() -> MnemosConfig:
    """Get the global MNEMOS configuration."""
    global _config
    if _config is None:
        _config = MnemosConfig.from_env()
    return _config
