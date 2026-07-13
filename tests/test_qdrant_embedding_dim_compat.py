"""Qdrant collection dimension compatibility — generic embedding-profile correctness.

Retrieval-plumbing remediation tests: the tier must derive, validate, or
explicitly configure collection vector dimensions from the active embedding
profile/model rather than assuming a 384-dimensional fallback. Covers:

* BGE profile provisions/validates 768-dimensional vectors
* Nomic profile retains its expected named-vector configuration
* dimension mismatch fails clearly before misleading retrieval results
* collection configuration is reproducible from repository configuration
* R1 behavior unchanged when the embedding tier is compatible
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _MockCollections:
    def __init__(self, collections):
        self.collections = collections


class _NamedCollection:
    def __init__(self, name):
        self.name = name


def _mock_client(existing=(), collection_info=None):
    client = MagicMock()
    client.get_collections.return_value = _MockCollections(
        [_NamedCollection(n) for n in existing]
    )
    if collection_info is not None:
        client.get_collection.return_value = collection_info
    return client


def _fake_model(dim):
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim
    return model


def _make_tier(embedding_model, *, model_dim=None, embedding_dim=None,
               existing=(), collection_info=None):
    """Construct a real QdrantTier through __init__ with mocked client/model."""
    from mnemos.retrieval import qdrant_tier as qt

    client = _mock_client(existing=existing, collection_info=collection_info)
    with patch.object(qt.QdrantTier, "_get_embedder",
                      return_value=_fake_model(model_dim)) as get_embedder, \
         patch("qdrant_client.QdrantClient", return_value=client):
        tier = qt.QdrantTier(
            url="http://localhost:6333",
            collection_name="dim_compat_test",
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            gpu_device="cpu",
        )
    return tier, client, get_embedder


class TestBgeProfile:
    def test_bge_derives_768_and_creates_768_collection(self):
        tier, client, get_embedder = _make_tier(
            "BAAI/bge-base-en-v1.5", model_dim=768,
        )
        assert tier._embedding_dim == 768
        assert get_embedder.called  # dimension came from the model, not a guess
        vectors_config = client.create_collection.call_args.kwargs["vectors_config"]
        assert vectors_config.size == 768

    def test_no_384_fallback_assumption(self):
        """A non-nomic model must never silently get a 384-dim collection."""
        tier, client, _ = _make_tier("BAAI/bge-base-en-v1.5", model_dim=768)
        vectors_config = client.create_collection.call_args.kwargs["vectors_config"]
        assert vectors_config.size != 384

    def test_derivation_probe_fallback_when_getter_unavailable(self):
        """If get_sentence_embedding_dimension is absent/None, probe an encode."""
        import numpy as np
        from mnemos.retrieval import qdrant_tier as qt

        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = None
        model.encode.return_value = np.zeros((1, 768), dtype=np.float32)
        client = _mock_client()
        with patch.object(qt.QdrantTier, "_get_embedder", return_value=model), \
             patch("qdrant_client.QdrantClient", return_value=client):
            tier = qt.QdrantTier(
                collection_name="dim_compat_test",
                embedding_model="BAAI/bge-base-en-v1.5",
                gpu_device="cpu",
            )
        assert tier._embedding_dim == 768


class TestNomicProfile:
    def test_nomic_retains_named_vector_configuration(self):
        from mnemos.retrieval.qdrant_tier import NOMIC_FULL_DIM, NOMIC_MRL_DIM

        tier, client, get_embedder = _make_tier(
            "nomic-ai/nomic-embed-text-v1.5", model_dim=768,
        )
        assert tier._embedding_dim == NOMIC_FULL_DIM
        # Nomic layout is fixed; no derivation model-load is required for it.
        assert not get_embedder.called
        vectors_config = client.create_collection.call_args.kwargs["vectors_config"]
        assert set(vectors_config) == {"dense_64", "dense_768"}
        assert vectors_config["dense_64"].size == NOMIC_MRL_DIM
        assert vectors_config["dense_768"].size == NOMIC_FULL_DIM


class TestDimensionMismatchFailsClosed:
    def _existing_info(self, size=None, named=None):
        info = MagicMock()
        if named is not None:
            info.config.params.vectors = named
        else:
            vec = MagicMock()
            vec.size = size
            info.config.params.vectors = vec
        return info

    def test_existing_384_collection_with_768_model_raises(self):
        with pytest.raises(ValueError, match="incompatible"):
            _make_tier(
                "BAAI/bge-base-en-v1.5", model_dim=768,
                existing=("dim_compat_test",),
                collection_info=self._existing_info(size=384),
            )

    def test_existing_matching_collection_initializes(self):
        tier, client, _ = _make_tier(
            "BAAI/bge-base-en-v1.5", model_dim=768,
            existing=("dim_compat_test",),
            collection_info=self._existing_info(size=768),
        )
        assert tier._embedding_dim == 768
        client.create_collection.assert_not_called()

    def test_named_vector_collection_with_single_vector_profile_raises(self):
        with pytest.raises(ValueError, match="incompatible"):
            _make_tier(
                "BAAI/bge-base-en-v1.5", model_dim=768,
                existing=("dim_compat_test",),
                collection_info=self._existing_info(
                    named={"dense_64": MagicMock(), "dense_768": MagicMock()},
                ),
            )

    def test_loaded_model_contradicting_explicit_dim_raises_before_embed(self):
        """Explicit override that contradicts the actual model fails clearly."""
        from mnemos.retrieval import qdrant_tier as qt

        tier, _, _ = _make_tier(
            "BAAI/bge-base-en-v1.5", embedding_dim=768,
        )
        # Simulate first real model load returning a 384-dim model.
        with pytest.raises(ValueError, match="dimensionally invalid|produces"):
            tier._validate_model_dim(_fake_model(384))


class TestReproducibleFromRepositoryConfiguration:
    def test_explicit_embedding_dim_is_honored_without_model_load(self):
        tier, client, get_embedder = _make_tier(
            "BAAI/bge-base-en-v1.5", embedding_dim=768,
        )
        assert tier._embedding_dim == 768
        assert not get_embedder.called  # explicit config, no derivation needed
        vectors_config = client.create_collection.call_args.kwargs["vectors_config"]
        assert vectors_config.size == 768

    def test_config_env_passthrough(self, monkeypatch):
        from mnemos.config import MnemosConfig

        monkeypatch.setenv("MNEMOS_EMBEDDING_DIM", "768")
        assert MnemosConfig.from_env().embedding_dim == 768
        monkeypatch.delenv("MNEMOS_EMBEDDING_DIM")
        assert MnemosConfig.from_env().embedding_dim is None

    def test_config_default_is_derive(self):
        from mnemos.config import MnemosConfig

        assert MnemosConfig().embedding_dim is None


class TestR1UnchangedWhenTierCompatible:
    """The remediation is retrieval plumbing only: R1 decision logic, allowed
    and forbidden route labels, and bounded-override behavior are untouched."""

    def test_r1_route_labels_unchanged(self):
        from mnemos.retrieval.evidence_admission import (
            ALLOWED_ENFORCED_ROUTE_LABELS,
            FORBIDDEN_ENFORCED_ROUTE_LABELS,
        )

        assert ALLOWED_ENFORCED_ROUTE_LABELS == (
            "CUE_ONLY_LOOKUP",
            "CACHE_ONLY",
            "BOUNDED_SEMANTIC_RETRIEVAL",
            "ABSTAIN_OR_REQUEST_SCOPE",
            "NORMAL_RETRIEVAL_FALLBACK",
        )
        assert FORBIDDEN_ENFORCED_ROUTE_LABELS == (
            "HYBRID_RETRIEVAL",
            "ASSOCIATIVE_EXPANSION_ELIGIBLE",
            "graph_hybrid_experimental",
            "derived_facts",
            "summary_inclusion",
            "governance_override",
        )

    def test_r1_bounded_overrides_do_not_depend_on_embedding_dim(self):
        from mnemos.retrieval.evidence_admission import (
            AdmissionRecommendation,
            bounded_retrieval_overrides,
            decide_enforcement,
        )

        rec = AdmissionRecommendation(
            status="recommended",
            recommended_route="SEMANTIC_RETRIEVAL",
            candidate_budget=8,
            context_token_budget=1200,
            expansion_budget=0,
            latency_budget_ms=None,
            stop_condition="minimum_evidence_satisfied",
            reason_codes=[],
            input_snapshot=None,
            latency_ms=0.1,
        )
        decision = decide_enforcement(rec)
        assert decision.enforced_route == "BOUNDED_SEMANTIC_RETRIEVAL"
        overrides = bounded_retrieval_overrides(
            decision, requested_top_k=10, configured_semantic_top_k=25,
        )
        assert overrides == {
            "top_k": 8,
            "semantic_top_k": 8,
            "retrieval_mode": "semantic",
            "adaptive_routing": False,
        }
