"""Tests for hybrid-related config controls."""

import pytest

from mnemos.config import MnemosConfig


def _clear_mnemos_env(monkeypatch):
    for key in [
        "MNEMOS_RETRIEVAL_MODE",
        "MNEMOS_FUSION_POLICY",
        "MNEMOS_LEXICAL_TOP_K",
        "MNEMOS_SEMANTIC_TOP_K",
        "MNEMOS_EXPLAIN_DEFAULT",
        "MNEMOS_QUANT_BITS",
        "MNEMOS_AUDIT_ENABLED",
        "MNEMOS_AUDIT_RETENTION_DAYS",
        "MNEMOS_PORT",
        "MNEMOS_MEMORY_OVER_MAPS_PHASE2",
        "MNEMOS_MEMORY_OVER_MAPS_PHASE3",
        "MNEMOS_MEMORY_OVER_MAPS_PHASE4",
        "MNEMOS_MEMORY_OVER_MAPS_PHASE5",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_config_hybrid_defaults_preserve_semantic(monkeypatch):
    _clear_mnemos_env(monkeypatch)
    cfg = MnemosConfig.from_env()
    assert cfg.retrieval_mode == "semantic"
    assert cfg.fusion_policy == "balanced"
    assert cfg.lexical_top_k == 25
    assert cfg.semantic_top_k == 25
    assert cfg.explain_default is False


def test_config_hybrid_valid_values(monkeypatch):
    _clear_mnemos_env(monkeypatch)
    monkeypatch.setenv("MNEMOS_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("MNEMOS_FUSION_POLICY", "lexical_dominant")
    monkeypatch.setenv("MNEMOS_LEXICAL_TOP_K", "40")
    monkeypatch.setenv("MNEMOS_SEMANTIC_TOP_K", "30")
    monkeypatch.setenv("MNEMOS_EXPLAIN_DEFAULT", "true")
    monkeypatch.setenv("MNEMOS_MEMORY_OVER_MAPS_PHASE2", "true")
    monkeypatch.setenv("MNEMOS_MEMORY_OVER_MAPS_PHASE3", "true")
    monkeypatch.setenv("MNEMOS_MEMORY_OVER_MAPS_PHASE4", "true")
    monkeypatch.setenv("MNEMOS_MEMORY_OVER_MAPS_PHASE5", "true")

    cfg = MnemosConfig.from_env()
    assert cfg.retrieval_mode == "hybrid"
    assert cfg.fusion_policy == "lexical_dominant"
    assert cfg.lexical_top_k == 40
    assert cfg.semantic_top_k == 30
    assert cfg.explain_default is True
    assert cfg.memory_over_maps_phase2 is True
    assert cfg.memory_over_maps_phase3 is True
    assert cfg.memory_over_maps_phase4 is True
    assert cfg.memory_over_maps_phase5 is True


@pytest.mark.parametrize(
    "name,value,contains",
    [
        ("MNEMOS_RETRIEVAL_MODE", "hybrid_plus", "MNEMOS_RETRIEVAL_MODE"),
        ("MNEMOS_FUSION_POLICY", "aggressive", "MNEMOS_FUSION_POLICY"),
        ("MNEMOS_LEXICAL_TOP_K", "0", "MNEMOS_LEXICAL_TOP_K"),
        ("MNEMOS_SEMANTIC_TOP_K", "-1", "MNEMOS_SEMANTIC_TOP_K"),
        ("MNEMOS_EXPLAIN_DEFAULT", "sometimes", "MNEMOS_EXPLAIN_DEFAULT"),
    ],
)
def test_config_invalid_hybrid_values_fail_clearly(monkeypatch, name, value, contains):
    _clear_mnemos_env(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError) as exc:
        MnemosConfig.from_env()
    assert contains in str(exc.value)
