import os
import pytest
from mnemos.runtime.profile_selector import resolve_profile, ConfigError

def test_default_profile_remains_core_memory_appliance(monkeypatch):
    monkeypatch.delenv("MNEMOS_PROFILE", raising=False)
    monkeypatch.delenv("MNEMOS_RETRIEVAL_BACKEND", raising=False)
    monkeypatch.delenv("MNEMOS_TURBOVEC_ENABLED", raising=False)
    
    config = resolve_profile()
    assert config.profile_name == "core_memory_appliance"
    assert config.retrieval_backend == "qdrant"
    assert config.turbovec_enabled is False
    assert config.experimental is False

def test_portable_profile_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("MNEMOS_PROFILE", "portable_memory_appliance")
    monkeypatch.setenv("MNEMOS_RETRIEVAL_BACKEND", "turbovec")
    monkeypatch.setenv("MNEMOS_TURBOVEC_ENABLED", "true")
    
    config = resolve_profile()
    assert config.profile_name == "portable_memory_appliance"
    assert config.retrieval_backend == "turbovec"
    assert config.turbovec_enabled is True
    assert config.experimental is True

def test_portable_profile_fails_without_enabled_flag(monkeypatch):
    monkeypatch.setenv("MNEMOS_PROFILE", "portable_memory_appliance")
    monkeypatch.setenv("MNEMOS_RETRIEVAL_BACKEND", "turbovec")
    monkeypatch.setenv("MNEMOS_TURBOVEC_ENABLED", "false")
    
    with pytest.raises(ConfigError, match="requires MNEMOS_TURBOVEC_ENABLED=true"):
        resolve_profile()

def test_invalid_profile_backend_combination(monkeypatch):
    monkeypatch.setenv("MNEMOS_PROFILE", "core_memory_appliance")
    monkeypatch.setenv("MNEMOS_RETRIEVAL_BACKEND", "turbovec")
    
    with pytest.raises(ConfigError, match="requires backend 'qdrant'"):
        resolve_profile()

def test_unsupported_profile_fails(monkeypatch):
    monkeypatch.setenv("MNEMOS_PROFILE", "unknown_profile")
    with pytest.raises(ConfigError, match="Unsupported profile"):
        resolve_profile()
