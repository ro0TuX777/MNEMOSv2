import json
from pathlib import Path
from types import SimpleNamespace

from mnemos.engram.model import Engram
from mnemos.memory_over_maps.view_cache import (
    DerivedViewCache,
    build_retrieval_cache_context,
)
from mnemos.retrieval.base import SearchResult
from mnemos.retrieval.retrieval_router import RetrievalRouter
from service.app import MnemosRuntime
from tools.mnemos_seed_manifest import load_seed_manifest, update_manifest_section
from tools.run_retrieval_hygiene_benchmark import _compute_summary, _score_query
from tools.run_retrieval_hygiene_benchmark import RUN_MATRIX
from tools.run_retrieval_fresh_verification import (
    _compute_summary as _compute_fresh_summary,
    _score_query as _score_fresh_query,
)
from tools.seed_mnemos_repo_context import _document_for_path
from tools.seed_mnemos_repo_summaries import DEFAULT_SUMMARIES, build_default_summaries
from tools.snapshot_retrieval_reproducibility import compare_configuration


def _router_stub() -> RetrievalRouter:
    router = RetrievalRouter.__new__(RetrievalRouter)
    router._stats = {"retrieval_duplicate_group_count": 0}
    return router


def test_summary_seed_ids_are_deterministic_and_stable():
    first = build_default_summaries()
    second = build_default_summaries()

    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert len({item["id"] for item in first}) == len(first)
    assert all(item["metadata"]["retrieval_only"] is True for item in first)
    assert all(item["metadata"]["source_linked"] is True for item in first)


def test_context_seed_document_uses_deterministic_identity():
    first = _document_for_path(
        "docs/benchmarks/gatemem_program_status.md",
        ["repo_context", "gatemem", "markdown"],
        "gatemem_reference",
    )
    second = _document_for_path(
        "docs/benchmarks/gatemem_program_status.md",
        ["repo_context", "gatemem", "markdown"],
        "gatemem_reference",
    )
    assert first["id"] == second["id"]
    assert first["metadata"]["canonical_source_uri"] == "docs/benchmarks/gatemem_program_status.md"


def test_runtime_index_documents_honors_supplied_ids():
    class StubFusion:
        def __init__(self):
            self.last_engrams = []

        def index(self, engrams, tiers=None):
            self.last_engrams = list(engrams)
            return {"qdrant": len(engrams)}

    class StubLexical:
        def index(self, engrams):
            return len(engrams)

    runtime = MnemosRuntime.__new__(MnemosRuntime)
    runtime._volatility_engine = None
    runtime._semantic_fusion = StubFusion()
    runtime._lexical_tier = StubLexical()
    runtime._audit = lambda *args, **kwargs: None
    runtime._base_payload = lambda: {"status": "healthy"}

    payload = runtime.index_documents(
        [{"id": "seeded-id-1", "content": "hello", "source": "summary://test", "metadata": {}}],
        {},
    )

    assert payload["result"]["engram_ids"] == ["seeded-id-1"]
    assert runtime._semantic_fusion.last_engrams[0].id == "seeded-id-1"


def test_retrieval_duplicate_suppression_groups_seeded_summary_duplicates():
    router = _router_stub()
    base = DEFAULT_SUMMARIES[0]
    r1 = SearchResult(
        engram=Engram(
            id=base["id"],
            content=base["content"],
            source=base["source"],
            metadata=dict(base["metadata"]),
        ),
        score=0.9,
        tier="qdrant",
    )
    r2 = SearchResult(
        engram=Engram(
            id="duplicate-two",
            content=base["content"],
            source=base["source"],
            metadata=dict(base["metadata"]),
        ),
        score=0.8,
        tier="qdrant",
    )

    deduped, meta = router._deduplicate_retrieval_results([r1, r2])

    assert len(deduped) == 1
    assert meta["suppressed_count"] == 1
    assert deduped[0].metadata["duplicate_suppression"]["applied"] is True


def test_duplicate_suppression_preserves_distinct_chunks_from_same_source():
    router = _router_stub()
    source = "docs/example.md"
    r1 = SearchResult(
        engram=Engram(
            id="a",
            content="first chunk",
            source=source,
            metadata={
                "canonical_source_uri": source,
                "normalized_content_hash": "hash-a",
            },
        ),
        score=0.9,
        tier="qdrant",
    )
    r2 = SearchResult(
        engram=Engram(
            id="b",
            content="second chunk",
            source=source,
            metadata={
                "canonical_source_uri": source,
                "normalized_content_hash": "hash-b",
            },
        ),
        score=0.8,
        tier="qdrant",
    )

    deduped, meta = router._deduplicate_retrieval_results([r1, r2])

    assert len(deduped) == 2
    assert meta["suppressed_count"] == 0


def test_pre_cognitive_cache_rejects_context_mismatch():
    cache = DerivedViewCache(ttl_seconds=3600)
    context = build_retrieval_cache_context(
        query="GateMem G4 frozen regression baseline",
        authorized_scope="default",
        collection_snapshot="mnemos_claude_repo_seed:10",
        retrieval_profile="hybrid|balanced|lexical_top_k=25|semantic_top_k=25",
        embedding_model_name="nomic-ai/nomic-embed-text-v1.5",
        seed_snapshot="seed-a",
    )
    cache.set_pre_cognitive(
        key="k1",
        query="GateMem G4 frozen regression baseline",
        cluster_id=1,
        view={"results": [{"engram": {"source": "summary://gatemem"}}]},
        dependency_refs={"session_id": "default"},
        cache_context=context,
    )

    hit = cache.fuzzy_pre_cognitive_get(
        query="GateMem G4 frozen regression baseline",
        cluster_id=1,
        cache_context=context,
    )
    warm_hit = cache.fuzzy_pre_cognitive_get(
        query="GateMem G4 frozen regression baseline",
        cluster_id=1,
        cache_context=context,
    )
    miss = cache.fuzzy_pre_cognitive_get(
        query="GateMem G4 frozen regression baseline",
        cluster_id=1,
        cache_context={**context, "collection_snapshot": "other:10"},
    )

    assert hit is not None
    assert warm_hit == hit
    assert miss is None


def test_configuration_parity_reports_intentional_retrieval_mode_difference():
    result = compare_configuration(
        {
            "base_url": "http://localhost:8700",
            "retrieval_mode_default": "semantic",
        },
        {
            "base_url": "http://localhost:8700",
            "tool_defaults": {"search_memory": {"retrieval_mode": "hybrid"}},
        },
    )

    assert result["pass"] is True
    assert result["intentional_differences"]


def test_seed_manifest_composite_snapshot_is_deterministic(tmp_path: Path):
    manifest_path = tmp_path / "repo_seed_manifest.json"
    first = update_manifest_section(
        section_name="repo_summaries",
        section_payload={
            "seed_schema_version": "summary_seed_v1",
            "seed_snapshot_id": "abc123",
            "seed_identities": ["a", "b"],
        },
        path=manifest_path,
    )
    second = update_manifest_section(
        section_name="repo_context",
        section_payload={
            "seed_schema_version": "repo_context_seed_v1",
            "seed_snapshot_id": "def456",
            "seed_identities": ["c"],
        },
        path=manifest_path,
    )
    loaded = load_seed_manifest(manifest_path)

    assert first["seed_snapshot_id"] != "unknown"
    assert second["seed_snapshot_id"] == loaded["seed_snapshot_id"]


def test_seed_manifest_snapshot_changes_when_section_changes(tmp_path: Path):
    manifest_path = tmp_path / "repo_seed_manifest.json"
    original = update_manifest_section(
        section_name="repo_summaries",
        section_payload={
            "seed_schema_version": "summary_seed_v1",
            "seed_snapshot_id": "abc123",
            "seed_identities": ["a"],
        },
        path=manifest_path,
    )
    changed = update_manifest_section(
        section_name="repo_summaries",
        section_payload={
            "seed_schema_version": "summary_seed_v1",
            "seed_snapshot_id": "abc124",
            "seed_identities": ["a"],
        },
        path=manifest_path,
    )
    assert original["seed_snapshot_id"] != changed["seed_snapshot_id"]


def test_benchmark_scaffold_template_tracks_frozen_query_count():
    frozen = json.loads(
        Path("docs/experiments/retrieval_hygiene_r0_frozen_alias_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    template = json.loads(
        Path("benchmarks/results/retrieval_hygiene_r0_benchmark_result_template.json").read_text(
            encoding="utf-8"
        )
    )

    assert frozen["benchmark_id"] == template["benchmark_id"]
    assert len(frozen["queries"]) == 15


def test_run_matrix_has_expected_eight_runs():
    assert len(RUN_MATRIX) == 8
    assert RUN_MATRIX[0] == ("direct_service", "cold", 1)
    assert RUN_MATRIX[-1] == ("mcp_path", "warm", 2)


def test_retrieval_fingerprint_reports_executed_route_not_only_config_defaults():
    runtime = MnemosRuntime.__new__(MnemosRuntime)
    runtime._config = SimpleNamespace(
        qdrant_collection="mnemos_claude_repo_seed",
        retrieval_mode="semantic",
        fusion_policy="balanced",
        lexical_top_k=25,
        semantic_top_k=25,
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
    )
    runtime._seed_snapshot_id = lambda: "2437be792647c500"

    fingerprint = runtime._retrieval_fingerprint(
        {"retrieval_mode": "hybrid", "fusion_policy": "lexical_dominant"}
    )

    assert fingerprint["retrieval_profile"] == "hybrid|lexical_dominant|lexical_top_k=25|semantic_top_k=25"
    assert fingerprint["configured_retrieval_profile"] == "semantic|balanced|lexical_top_k=25|semantic_top_k=25"


def test_low_relevance_abstention_guard_identifies_q15_style_noise():
    runtime = MnemosRuntime.__new__(MnemosRuntime)
    results = [
        SearchResult(engram=Engram(id="a", content="x", source="doc-a"), score=0.0015, tier="qdrant"),
        SearchResult(engram=Engram(id="b", content="y", source="doc-b"), score=0.0010, tier="qdrant"),
        SearchResult(engram=Engram(id="c", content="z", source="doc-c"), score=0.0009, tier="qdrant"),
    ]

    meta = runtime._low_relevance_abstention_meta(
        results,
        mode_meta={"retrieval_mode": "semantic"},
    )

    assert meta is not None
    assert meta["applied"] is True
    assert meta["reason_code"] == "low_relevance_abstention"
    assert meta["top_scores"] == [0.0015, 0.001, 0.0009]


def test_low_relevance_abstention_guard_does_not_fire_for_meaningful_match():
    runtime = MnemosRuntime.__new__(MnemosRuntime)
    results = [
        SearchResult(engram=Engram(id="a", content="x", source="doc-a"), score=0.6953, tier="qdrant"),
        SearchResult(engram=Engram(id="b", content="y", source="doc-b"), score=0.6124, tier="qdrant"),
    ]

    meta = runtime._low_relevance_abstention_meta(
        results,
        mode_meta={"retrieval_mode": "semantic"},
    )

    assert meta is None


def test_low_relevance_abstention_guard_does_not_fire_for_relevant_low_but_above_floor_score():
    runtime = MnemosRuntime.__new__(MnemosRuntime)
    results = [
        SearchResult(engram=Engram(id="a", content="x", source="doc-a"), score=0.0101, tier="qdrant"),
        SearchResult(engram=Engram(id="b", content="y", source="doc-b"), score=0.0090, tier="qdrant"),
    ]

    meta = runtime._low_relevance_abstention_meta(
        results,
        mode_meta={"retrieval_mode": "semantic"},
    )

    assert meta is None


def test_ambiguous_benchmark_query_accepts_any_declared_responsive_neighborhood():
    benchmark = {
        "neighborhoods": {
            "gatemem_frozen_baseline": {
                "accepted_sources": ["summary://gatemem/g4_alias_frozen_regression_baseline"]
            },
            "gatemem_g5_handoff": {
                "accepted_sources": ["summary://gatemem/g5_blocked_handoff"]
            },
        }
    }
    query_entry = {
        "expected_neighborhood": "gatemem_frozen_baseline",
        "evaluation_mode": "ambiguous_neighborhood",
        "accepted_neighborhoods": ["gatemem_frozen_baseline", "gatemem_g5_handoff"],
    }
    response = {
        "results": [
            {
                "engram": {
                    "source": "summary://gatemem/g5_blocked_handoff",
                    "metadata": {},
                }
            }
        ],
        "meta": {},
    }

    score = _score_query(query_entry, response, benchmark)

    assert score["top1_neighborhood_correct"] is True
    assert score["evaluation_mode"] == "ambiguous_neighborhood"


def test_direct_mcp_agreement_treats_mutual_abstention_as_agreement():
    payload = {
        "paths_compared": ["direct_service", "mcp_path"],
        "per_query_results": [
            {
                "query_id": "q15",
                "runs": [
                    {
                        "path": "direct_service",
                        "cache_state_requested": "cold",
                        "top_results": [],
                        "top1_neighborhood_correct": True,
                        "top3_neighborhood_present": True,
                        "duplicate_groups": 0,
                    },
                    {
                        "path": "mcp_path",
                        "cache_state_requested": "cold",
                        "top_results": [],
                        "top1_neighborhood_correct": True,
                        "top3_neighborhood_present": True,
                        "duplicate_groups": 0,
                    },
                ],
            }
        ],
        "summary": {"notes": []},
    }

    _compute_summary(payload)

    assert payload["summary"]["direct_mcp_top1_agreement"] == 1.0


def test_fresh_verification_negative_control_scores_abstention_as_correct():
    pack = {
        "neighborhoods": {"abstain_non_gatemem": {"accepted_sources": []}},
    }
    query_entry = {
        "expected_behavior": "abstain_non_gatemem",
        "category": "unrelated_negative_control",
    }
    response = {"results": [], "meta": {"retrieval_fingerprint": {"retrieval_profile": "semantic|none", "configured_retrieval_profile": "semantic|balanced"}}}

    score = _score_fresh_query(query_entry, response, pack)

    assert score["top1_behavior_correct"] is True
    assert score["abstention_correct"] is True


def test_fresh_verification_relevant_low_score_query_must_not_abstain():
    pack = {
        "neighborhoods": {
            "gatemem_claim_boundary": {
                "accepted_sources": ["docs/benchmarks/gatemem_g4_offline_reference_implementation.md"]
            },
            "gatemem_frozen_baseline": {
                "accepted_sources": ["benchmarks/results/gatemem_g4_frozen_reference_manifest.md"]
            },
        },
    }
    query_entry = {
        "expected_behavior": "retrieve_gatemem_claim_boundary_or_frozen_baseline",
        "category": "relevant_low_score_query_must_not_abstain",
    }
    abstaining_response = {
        "results": [],
        "meta": {"retrieval_fingerprint": {"retrieval_profile": "semantic|none", "configured_retrieval_profile": "semantic|balanced"}},
    }
    non_abstaining_response = {
        "results": [
            {
                "engram": {
                    "source": "docs/benchmarks/gatemem_g4_offline_reference_implementation.md",
                    "metadata": {"source_uri": "docs/benchmarks/gatemem_g4_offline_reference_implementation.md"},
                }
            }
        ],
        "meta": {"retrieval_fingerprint": {"retrieval_profile": "semantic|none", "configured_retrieval_profile": "semantic|balanced"}},
    }

    abstaining = _score_fresh_query(query_entry, abstaining_response, pack)
    non_abstaining = _score_fresh_query(query_entry, non_abstaining_response, pack)

    assert abstaining["false_abstention"] is True
    assert non_abstaining["false_abstention"] is False
    assert non_abstaining["top1_behavior_correct"] is True


def test_fresh_verification_summary_treats_mutual_abstention_as_agreement():
    payload = {
        "paths_compared": ["direct_service", "mcp_path"],
        "per_query_results": [
            {
                "query_id": "v01",
                "runs": [
                    {
                        "path": "direct_service",
                        "cache_state_requested": "cold",
                        "top_results": [],
                        "top1_behavior_correct": True,
                        "top3_behavior_present": True,
                        "abstention_correct": True,
                        "false_abstention": False,
                        "duplicate_groups": 0,
                        "abstained": True,
                    },
                    {
                        "path": "mcp_path",
                        "cache_state_requested": "cold",
                        "top_results": [],
                        "top1_behavior_correct": True,
                        "top3_behavior_present": True,
                        "abstention_correct": True,
                        "false_abstention": False,
                        "duplicate_groups": 0,
                        "abstained": True,
                    },
                ],
            }
        ],
        "summary": {"notes": []},
    }

    _compute_fresh_summary(payload)

    assert payload["summary"]["direct_mcp_top1_agreement"] == 1.0
