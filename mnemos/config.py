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
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    # Optional explicit vector dimension. None (default) => derived from the
    # embedding model itself by the vector tier; set MNEMOS_EMBEDDING_DIM only
    # to pin/validate a specific dimension.
    embedding_dim: Optional[int] = None
    long_context_model: str = "nomic-ai/nomic-embed-text-v1"

    # Reranking
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"

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
    adaptive_routing: bool = True

    # Governance layer
    governance_mode: str = "off"            # off | advisory | enforced
    governance_min_score: float = 0.0       # veto threshold (0.0 = disabled)
    governance_freshness_half_life: float = 180.0   # days
    governance_volatility_bias: bool = True

    # Memory Over Maps (phase-gated rollout)
    memory_over_maps_phase1: bool = False
    memory_over_maps_phase2: bool = False
    memory_over_maps_phase3: bool = False
    memory_over_maps_phase4: bool = False
    memory_over_maps_phase5: bool = False

    # PIT-1 Governed Derived Fact Lane Scaffold
    derived_enabled: bool = False
    derived_whitelist: List[str] = field(default_factory=lambda: ["eval_dashboard", "governance_auditor"])

    # PIT-3 Derived Fact Shadow Packet Limits
    pit3_max_derived_facts_per_shadow_packet: int = 5
    pit3_max_derived_fact_tokens: int = 500

    # PIT-1 leakage guard behavior when a derived fact survives the
    # server-side filters: "strip" removes it, "canary" returns it and logs.
    pit_leakage_mode: str = "strip"

    # MNEMOS-Thinking Pulse forecasting
    timesfm_enabled: bool = True
    pulse_actions: str = "advisory"          # off | advisory | autonomous
    pulse_horizon_minutes: int = 15
    timesfm_sidecar_url: str = "http://mnemos-timesfm:8711"
    timesfm_timeout_s: float = 0.08
    pulse_p95_budget_ms: float = 250.0
    pulse_warmup_cooldown_s: int = 900

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
    def _parse_governance_mode(name: str = "MNEMOS_GOVERNANCE_MODE", default: str = "off") -> str:
        raw = os.getenv(name, default).strip().lower()
        if raw not in {"off", "advisory", "enforced"}:
            raise ValueError(f"{name} must be one of: off,advisory,enforced (got '{raw}')")
        return raw

    @staticmethod
    def _parse_float(name: str, default: str, *, min_value: float = 0.0) -> float:
        raw = os.getenv(name, default).strip()
        try:
            value = float(raw)
        except ValueError as e:
            raise ValueError(f"{name} must be a float (got '{raw}')") from e
        if value < min_value:
            raise ValueError(f"{name} must be >= {min_value} (got {value})")
        return value

    @staticmethod
    def _parse_fusion_policy(name: str = "MNEMOS_FUSION_POLICY", default: str = "balanced") -> str:
        raw = os.getenv(name, default).strip().lower()
        if raw not in FUSION_POLICIES:
            allowed = ",".join(sorted(FUSION_POLICIES.keys()))
            raise ValueError(f"{name} must be one of: {allowed} (got '{raw}')")
        return raw

    @staticmethod
    def _parse_pulse_actions(name: str = "MNEMOS_PULSE_ACTIONS", default: str = "advisory") -> str:
        raw = os.getenv(name, default).strip().lower()
        if raw not in {"off", "advisory", "autonomous"}:
            raise ValueError(f"{name} must be one of: off,advisory,autonomous (got '{raw}')")
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

        governance_mode = cls._parse_governance_mode()
        governance_min_score = cls._parse_float("MNEMOS_GOVERNANCE_MIN_SCORE", "0.0")
        governance_freshness_half_life = cls._parse_float(
            "MNEMOS_GOVERNANCE_FRESHNESS_HALF_LIFE", "180.0", min_value=1.0
        )
        governance_volatility_bias = cls._parse_bool("MNEMOS_GOVERNANCE_VOLATILITY_BIAS", "true")
        memory_over_maps_phase1 = cls._parse_bool("MNEMOS_MEMORY_OVER_MAPS_PHASE1", "false")
        memory_over_maps_phase2 = cls._parse_bool("MNEMOS_MEMORY_OVER_MAPS_PHASE2", "false")
        memory_over_maps_phase3 = cls._parse_bool("MNEMOS_MEMORY_OVER_MAPS_PHASE3", "false")
        memory_over_maps_phase4 = cls._parse_bool("MNEMOS_MEMORY_OVER_MAPS_PHASE4", "false")
        memory_over_maps_phase5 = cls._parse_bool("MNEMOS_MEMORY_OVER_MAPS_PHASE5", "false")
        timesfm_enabled = cls._parse_bool("MNEMOS_TIMESFM_ENABLED", "true")
        pulse_actions = cls._parse_pulse_actions()
        pulse_horizon_minutes = cls._parse_int("MNEMOS_PULSE_HORIZON_MINUTES", "15", min_value=1)
        timesfm_timeout_s = cls._parse_float("MNEMOS_TIMESFM_TIMEOUT_S", "0.08", min_value=0.01)
        pulse_p95_budget_ms = cls._parse_float("MNEMOS_PULSE_P95_BUDGET_MS", "250.0", min_value=1.0)
        pulse_warmup_cooldown_s = cls._parse_int("MNEMOS_PULSE_WARMUP_COOLDOWN_S", "900", min_value=1)

        derived_enabled = cls._parse_bool("MNEMOS_DERIVED_ENABLED", "false")
        derived_whitelist_raw = os.getenv("MNEMOS_DERIVED_WHITELIST", "eval_dashboard,governance_auditor")
        derived_whitelist = [w.strip() for w in derived_whitelist_raw.split(",") if w.strip()]

        pit3_max_derived_facts_per_shadow_packet = cls._parse_int("MNEMOS_PIT3_MAX_DERIVED_FACTS_PER_SHADOW_PACKET", "5", min_value=1)
        pit3_max_derived_fact_tokens = cls._parse_int("MNEMOS_PIT3_MAX_DERIVED_FACT_TOKENS", "500", min_value=1)

        pit_leakage_mode = os.getenv("MNEMOS_PIT_LEAKAGE_MODE", "strip").strip().lower()
        if pit_leakage_mode not in ("strip", "canary"):
            logger.warning(
                f"Invalid MNEMOS_PIT_LEAKAGE_MODE={pit_leakage_mode!r}; falling back to 'strip'"
            )
            pit_leakage_mode = "strip"

        config = cls(
            profile=os.getenv("MNEMOS_PROFILE", "core_memory_appliance"),
            tiers=tiers,
            embedding_model=os.getenv("MNEMOS_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
            embedding_dim=(
                cls._parse_int("MNEMOS_EMBEDDING_DIM", "1", min_value=1)
                if os.getenv("MNEMOS_EMBEDDING_DIM", "").strip() else None
            ),
            long_context_model=os.getenv("MNEMOS_LONG_CONTEXT_MODEL", "nomic-ai/nomic-embed-text-v1"),
            use_reranker=cls._parse_bool("MNEMOS_USE_RERANKER", "true"),
            reranker_model=os.getenv("MNEMOS_RERANKER_MODEL", "BAAI/bge-reranker-base"),
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
            qdrant_url=os.getenv("MNEMOS_QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("MNEMOS_QDRANT_COLLECTION", "mnemos_engrams"),
            pgvector_table=os.getenv("MNEMOS_PGVECTOR_TABLE", "mnemos_vectors"),
            lexical_table=os.getenv("MNEMOS_LEXICAL_TABLE", "mnemos_lexical"),
            retrieval_mode=retrieval_mode,
            fusion_policy=fusion_policy,
            lexical_top_k=lexical_top_k,
            semantic_top_k=semantic_top_k,
            explain_default=explain_default,
            adaptive_routing=cls._parse_bool("MNEMOS_ADAPTIVE_ROUTING", "true"),
            governance_mode=governance_mode,
            governance_min_score=governance_min_score,
            governance_freshness_half_life=governance_freshness_half_life,
            governance_volatility_bias=governance_volatility_bias,
            memory_over_maps_phase1=memory_over_maps_phase1,
            memory_over_maps_phase2=memory_over_maps_phase2,
            memory_over_maps_phase3=memory_over_maps_phase3,
            memory_over_maps_phase4=memory_over_maps_phase4,
            memory_over_maps_phase5=memory_over_maps_phase5,
            derived_enabled=derived_enabled,
            derived_whitelist=derived_whitelist,
            pit3_max_derived_facts_per_shadow_packet=pit3_max_derived_facts_per_shadow_packet,
            pit3_max_derived_fact_tokens=pit3_max_derived_fact_tokens,
            pit_leakage_mode=pit_leakage_mode,
            timesfm_enabled=timesfm_enabled,
            pulse_actions=pulse_actions,
            pulse_horizon_minutes=pulse_horizon_minutes,
            timesfm_sidecar_url=os.getenv("MNEMOS_TIMESFM_SIDECAR_URL", "http://mnemos-timesfm:8711"),
            timesfm_timeout_s=timesfm_timeout_s,
            pulse_p95_budget_ms=pulse_p95_budget_ms,
            pulse_warmup_cooldown_s=pulse_warmup_cooldown_s,
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
