import pytest
from mnemos.retrieval.factory import get_retrieval_tier

def test_factory_returns_qdrant_by_default(monkeypatch):
    monkeypatch.delenv("MNEMOS_PROFILE", raising=False)
    monkeypatch.delenv("MNEMOS_RETRIEVAL_BACKEND", raising=False)
    
    tier = get_retrieval_tier()
    assert tier.name == "qdrant", "Factory did not return Qdrant stub for default profile"

def test_factory_returns_turbovec_fusion(monkeypatch):
    monkeypatch.setenv("MNEMOS_PROFILE", "portable_memory_appliance")
    monkeypatch.setenv("MNEMOS_RETRIEVAL_BACKEND", "turbovec")
    monkeypatch.setenv("MNEMOS_TURBOVEC_ENABLED", "true")
    
    tier = get_retrieval_tier()
    assert type(tier).__name__ == "TurbovecFusion", "Factory did not return TurbovecFusion for portable profile"
