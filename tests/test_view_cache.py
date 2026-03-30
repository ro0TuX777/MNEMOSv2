"""Tests for Phase 4 derived-view cache and invalidation."""

from mnemos.memory_over_maps.view_cache import (
    DerivedViewCache,
    build_cache_key,
    query_fingerprint,
)


def test_cache_key_is_deterministic():
    key1 = build_cache_key(
        view_type="evidence_bundle",
        query_fingerprint_value=query_fingerprint("hello world"),
        artifact_ids=["a1", "a2"],
        chunk_ids=["c1", "c2"],
        governance_state_hash_value="gov-1",
        synthesis_policy_version="default",
    )
    key2 = build_cache_key(
        view_type="evidence_bundle",
        query_fingerprint_value=query_fingerprint("hello world"),
        artifact_ids=["a2", "a1"],
        chunk_ids=["c2", "c1"],
        governance_state_hash_value="gov-1",
        synthesis_policy_version="default",
    )
    assert key1 == key2


def test_cache_hit_miss_and_invalidation_dry_run():
    cache = DerivedViewCache(ttl_seconds=3600)
    key = "k1"
    cache.set(
        key=key,
        view={"view_id": "v1", "view_type": "evidence_bundle"},
        dependency_refs={
            "artifact_ids": ["art-1"],
            "chunk_ids": ["chunk-1"],
            "governance_state_hash": "gov-a",
            "synthesis_policy_version": "default",
        },
    )
    assert cache.get(key) is not None

    dry = cache.invalidate(
        event_type="source_artifact_updated",
        refs={"artifact_id": "art-1"},
        dry_run=True,
    )
    assert key in dry["impacted_keys"]
    assert cache.get(key) is not None  # dry-run should not invalidate

    live = cache.invalidate(
        event_type="source_artifact_updated",
        refs={"artifact_id": "art-1"},
        dry_run=False,
    )
    assert key in live["impacted_keys"]
    assert cache.get(key) is None

    stats = cache.stats()
    assert stats["hit_count"] >= 1
    assert stats["miss_count"] >= 1
    assert stats["invalidation_event_count"] >= 2
    assert stats["invalidated_key_total"] >= 1
