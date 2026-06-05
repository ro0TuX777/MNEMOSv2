import os
from dataclasses import dataclass
from typing import Dict, Any, List

class ConfigError(Exception):
    pass

@dataclass
class ProfileConfig:
    profile_name: str
    retrieval_backend: str
    turbovec_enabled: bool
    experimental: bool

def resolve_profile() -> ProfileConfig:
    """
    Resolve the active MNEMOS profile based on environment variables.
    Fails closed if the combination is unsafe or invalid.
    """
    profile = os.environ.get("MNEMOS_PROFILE", "core_memory_appliance")
    backend = os.environ.get("MNEMOS_RETRIEVAL_BACKEND", "qdrant")
    tb_enabled = os.environ.get("MNEMOS_TURBOVEC_ENABLED", "false").lower() == "true"
    
    if profile == "core_memory_appliance":
        if backend != "qdrant":
            raise ConfigError(f"Profile 'core_memory_appliance' requires backend 'qdrant', got '{backend}'")
        return ProfileConfig(
            profile_name=profile,
            retrieval_backend=backend,
            turbovec_enabled=False,
            experimental=False
        )
        
    elif profile == "portable_memory_appliance":
        if backend != "turbovec":
            raise ConfigError(f"Profile 'portable_memory_appliance' requires backend 'turbovec', got '{backend}'")
        if not tb_enabled:
            raise ConfigError("Profile 'portable_memory_appliance' requires MNEMOS_TURBOVEC_ENABLED=true")
        return ProfileConfig(
            profile_name=profile,
            retrieval_backend=backend,
            turbovec_enabled=True,
            experimental=True
        )
        
    elif profile == "governance_native":
        if backend != "pgvector":
            raise ConfigError(f"Profile 'governance_native' requires backend 'pgvector', got '{backend}'")
        return ProfileConfig(
            profile_name=profile,
            retrieval_backend=backend,
            turbovec_enabled=False,
            experimental=False
        )
        
    else:
        raise ConfigError(f"Unsupported profile: '{profile}'")

def get_capabilities() -> Dict[str, Any]:
    """Return health and capability reporting for the resolved profile."""
    try:
        config = resolve_profile()
    except ConfigError as e:
        return {
            "status": "degraded",
            "error": str(e),
            "degraded_components": ["profile_resolution"]
        }
        
    if config.profile_name == "portable_memory_appliance":
        try:
            import turbovec
            return {
                "status": "healthy",
                "profile": config.profile_name,
                "retrieval_backend": config.retrieval_backend,
                "tiers": ["turbovec", "sqlite_fts"],
                "compression": {
                    "enabled": True,
                    "engine": "turbovec",
                    "bits": 4
                },
                "degraded_components": [],
                "experimental": config.experimental,
                "cross_platform_status": "warn",
                "known_constraints": [
                    "Linux/WSL requires nightly Rust or compatible prebuilt turbovec wheel",
                    "Qdrant remains recommended default for frictionless standard deployments"
                ]
            }
        except ImportError:
            return {
                "status": "degraded",
                "profile": config.profile_name,
                "retrieval_backend": config.retrieval_backend,
                "degraded_components": ["turbovec_import"],
                "error": "turbovec package unavailable"
            }
            
    # Default capabilities block for core appliance
    return {
        "status": "healthy",
        "profile": config.profile_name,
        "retrieval_backend": config.retrieval_backend,
        "tiers": [config.retrieval_backend],
        "degraded_components": [],
        "experimental": config.experimental
    }
