import pytest
import sys
from mnemos.runtime.profile_selector import get_capabilities

def test_capabilities_default_qdrant(monkeypatch):
    monkeypatch.delenv("MNEMOS_PROFILE", raising=False)
    monkeypatch.delenv("MNEMOS_RETRIEVAL_BACKEND", raising=False)
    
    caps = get_capabilities()
    assert caps["status"] == "healthy"
    assert caps["profile"] == "core_memory_appliance"
    assert caps["retrieval_backend"] == "qdrant"
    assert "qdrant" in caps["tiers"]

def test_capabilities_portable_profile_healthy(monkeypatch):
    monkeypatch.setenv("MNEMOS_PROFILE", "portable_memory_appliance")
    monkeypatch.setenv("MNEMOS_RETRIEVAL_BACKEND", "turbovec")
    monkeypatch.setenv("MNEMOS_TURBOVEC_ENABLED", "true")
    
    caps = get_capabilities()
    assert caps["status"] == "healthy"
    assert caps["profile"] == "portable_memory_appliance"
    assert caps["retrieval_backend"] == "turbovec"
    assert "turbovec" in caps["tiers"]
    assert caps["experimental"] is True

def test_capabilities_portable_profile_degraded(monkeypatch):
    monkeypatch.setenv("MNEMOS_PROFILE", "portable_memory_appliance")
    monkeypatch.setenv("MNEMOS_RETRIEVAL_BACKEND", "turbovec")
    monkeypatch.setenv("MNEMOS_TURBOVEC_ENABLED", "true")
    
    # Simulate turbovec missing
    monkeypatch.setitem(sys.modules, "turbovec", None)
    
    caps = get_capabilities()
    assert caps["status"] == "degraded"
    assert caps["profile"] == "portable_memory_appliance"
    assert "turbovec_import" in caps["degraded_components"]
    assert caps.get("experimental") is None # or check error message
