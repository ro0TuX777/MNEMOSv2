"""
MNEMOS Configuration
=====================

Environment-variable driven configuration for all MNEMOS components.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from mnemos.retrieval.policies.fusion_policies import FUSION_POLICIES

logger = logging.getLogger(__name__)


@dataclass
class MnemosConfig:
    """Central configuration for MNEMOS service."""

    # Deployment profile
    profile: str = "core_memory_appliance"

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
    postgres_dsn: str = ""                      # PostgreSQL DSN (audit + pgvector)

    # API
    port: int = 8700
    token: str = ""
    log_level: str = "INFO"

    # GPU
    gpu_device: str = "cuda"                    # "cuda", "cpu", or "cuda:N"

    # Data directories
    data_dir: str = "data"
    colbert_index_dir: str = "data/colbert"

    # Qdrant (Core Memory Appliance)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "mnemos_engrams"

    # pgvector (Governance Native)
    pgvector_table: str = "mnemos_vectors"
    lexical_table: str = "mnemos_lexical"

    # Retrieval mode + hybrid fusion
    retrieval_mode: str = "semantic"
    fusion_policy: str = "balanced"
    lexical_top_k: int = 25
    semantic_top_k: int = 25
    explain_default: bool = False

    @staticmethod
    def _parse_bool(name: str, default: str) -> bool:
        raw = os.getenv(name, default).strip().lower()
        if raw in ("true", "1", "yes"):
            return True
        if raw in ("false", "0", "no"):
            return False
        raise ValueError(f"{name} must be one of: true,false,1,0,yes,no (got '{raw}')")

    @staticmethod
    def _parse_int(name: str, default: str, *, min_value: int = 0) -> int:
        raw = os.getenv(name, default).strip()
        try:
            value = int(raw)
        except ValueError as e:
            raise ValueError(f"{name} must be an integer (got '{raw}')") from e
        if value < min_value:
            raise ValueError(f"{name} must be >= {min_value} (got {value})")
        return value

    @staticmethod
    def _parse_retrieval_mode(name: str = "MNEMOS_RETRIEVAL_MODE", default: str = "semantic") -> str:
        raw = os.getenv(name, default).strip().lower()
        if raw not in {"semantic", "hybrid"}:
            raise ValueError(f"{name} must be one of: semantic,hybrid (got '{raw}')")
        return raw

    @staticmethod
    def _parse_fusion_policy(name: str = "MNEMOS_FUSION_POLICY", default: str = "balanced") -> str:
        raw = os.getenv(name, default).strip().lower()
        if raw not in FUSION_POLICIES:
            allowed = ",".join(sorted(FUSION_POLICIES.keys()))
            raise ValueError(f"{name} must be one of: {allowed} (got '{raw}')")
        return raw

    @classmethod
    def from_env(cls) -> "MnemosConfig":
        """Build configuration from environment variables."""
        tiers_raw = os.getenv("MNEMOS_TIERS", "qdrant")
        tiers = [t.strip() for t in tiers_raw.split(",") if t.strip()]

        quant_bits = cls._parse_int("MNEMOS_QUANT_BITS", "4", min_value=0)
        if quant_bits > 4:
            raise ValueError(f"MNEMOS_QUANT_BITS must be <= 4 (got {quant_bits})")

        retrieval_mode = cls._parse_retrieval_mode()
        fusion_policy = cls._parse_fusion_policy()
        lexical_top_k = cls._parse_int("MNEMOS_LEXICAL_TOP_K", "25", min_value=1)
        semantic_top_k = cls._parse_int("MNEMOS_SEMANTIC_TOP_K", "25", min_value=1)
        explain_default = cls._parse_bool("MNEMOS_EXPLAIN_DEFAULT", "false")

        config = cls(
            profile=os.getenv("MNEMOS_PROFILE", "core_memory_appliance"),
            tiers=tiers,
            embedding_model=os.getenv("MNEMOS_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            colbert_model=os.getenv("MNEMOS_COLBERT_MODEL", "colbertv2.0"),
            quant_bits=quant_bits,
            audit_enabled=cls._parse_bool("MNEMOS_AUDIT_ENABLED", "true"),
            audit_db_path=os.getenv("MNEMOS_AUDIT_DB", "data/audit.db"),
            audit_retention_days=cls._parse_int("MNEMOS_AUDIT_RETENTION_DAYS", "90", min_value=1),
            postgres_dsn=os.getenv("MNEMOS_POSTGRES_DSN", ""),
            port=cls._parse_int("MNEMOS_PORT", "8700", min_value=1),
            token=os.getenv("MNEMOS_TOKEN", ""),
            log_level=os.getenv("MNEMOS_LOG_LEVEL", "INFO"),
            gpu_device=os.getenv("MNEMOS_GPU_DEVICE", "cuda"),
            data_dir=os.getenv("MNEMOS_DATA_DIR", "data"),
            colbert_index_dir=os.getenv("MNEMOS_COLBERT_DIR", "data/colbert"),
            qdrant_url=os.getenv("MNEMOS_QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("MNEMOS_QDRANT_COLLECTION", "mnemos_engrams"),
            pgvector_table=os.getenv("MNEMOS_PGVECTOR_TABLE", "mnemos_vectors"),
            lexical_table=os.getenv("MNEMOS_LEXICAL_TABLE", "mnemos_lexical"),
            retrieval_mode=retrieval_mode,
            fusion_policy=fusion_policy,
            lexical_top_k=lexical_top_k,
            semantic_top_k=semantic_top_k,
            explain_default=explain_default,
        )

        logger.info(
            f"⚙️ MNEMOS config: profile={config.profile}, tiers={config.tiers}, "
            f"quant={config.quant_bits}-bit, audit={config.audit_enabled}, "
            f"gpu={config.gpu_device}"
        )
        return config

    @property
    def has_qdrant(self) -> bool:
        return "qdrant" in self.tiers

    @property
    def has_lancedb(self) -> bool:
        return "lancedb" in self.tiers

    @property
    def has_pgvector(self) -> bool:
        return "pgvector" in self.tiers

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
