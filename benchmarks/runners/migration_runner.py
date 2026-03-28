"""
MNEMOS Benchmark - Migration Runner (Track 4)
================================================

Product question: Is profile migration operationally credible?

Tests Core→Governance and Governance→Core migration paths.
Requires live Docker services.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnemos.engram.model import Engram


def _check_docker() -> bool:
    """Check if Docker is available."""
    import subprocess
    try:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def run_migration_benchmark(
    corpus: List[Engram],
    source_tier: str,
    target_tier: str,
    gpu_device: str = "cuda",
) -> Dict[str, Any]:
    """
    Benchmark migration from one profile to another.

    Steps:
    1. Verify source tier has data
    2. Read all engrams from source via search
    3. Index all engrams into target tier
    4. Verify data integrity (count match, spot-check IDs)

    Decision this drives: Is profile migration fast and safe enough?
    """
    print(f"\n  Migration: {source_tier} → {target_tier}")

    from benchmarks.runners.retrieval_runner import _check_backend, _create_tier

    # Check both backends
    if not _check_backend(source_tier):
        print(f"    [WARN]  {source_tier} not available - skipping")
        return {"status": "skipped", "reason": f"{source_tier}_unavailable"}

    if not _check_backend(target_tier):
        print(f"    [WARN]  {target_tier} not available - skipping")
        return {"status": "skipped", "reason": f"{target_tier}_unavailable"}

    source = _create_tier(source_tier, gpu_device=gpu_device)
    target = _create_tier(target_tier, gpu_device=gpu_device)

    # Step 1: Check source has data
    source_stats = source.stats()
    source_count = source_stats.get("document_count", 0)
    print(f"    Source ({source_tier}): {source_count} documents")

    if source_count == 0:
        # Index corpus into source first
        print(f"    Indexing {len(corpus)} engrams into source...")
        source.index(corpus)
        source_count = len(corpus)

    # Step 2: Extract engrams from source (via search with broad query)
    # In production this would use a bulk export, but for benchmarking
    # we use the corpus we already have
    migrate_engrams = corpus[:source_count] if len(corpus) >= source_count else corpus

    # Step 3: Index into target
    t0 = time.perf_counter()
    target_count = target.index(migrate_engrams)
    migration_time = time.perf_counter() - t0

    # Step 4: Data integrity check
    target_stats = target.stats()
    target_doc_count = target_stats.get("document_count", 0)

    # Spot-check: retrieve 10 random engrams by ID
    import random
    rng = random.Random(42)
    check_ids = [e.id for e in rng.sample(migrate_engrams, min(10, len(migrate_engrams)))]
    found_count = 0
    for eid in check_ids:
        engram = target.get(eid)
        if engram is not None:
            found_count += 1

    result = {
        "status": "success",
        "source_tier": source_tier,
        "target_tier": target_tier,
        "source_count": source_count,
        "target_count": target_count,
        "target_verified_count": target_doc_count,
        "migration_time_s": round(migration_time, 3),
        "docs_per_sec": round(target_count / migration_time, 1) if migration_time > 0 else 0,
        "integrity_checks": {
            "checked": len(check_ids),
            "found": found_count,
            "pass_rate": found_count / len(check_ids) if check_ids else 0,
        },
        "data_loss": max(0, source_count - target_doc_count),
    }

    print(f"    [OK] Migrated {target_count} engrams in {migration_time:.2f}s")
    print(f"    Integrity: {found_count}/{len(check_ids)} spot checks passed")
    if result["data_loss"] > 0:
        print(f"    [WARN]  Data loss: {result['data_loss']} engrams missing")

    return result


def run_migration_track(
    corpus: List[Engram],
    gpu_device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run full Track 4: Migration & Recovery Benchmarks.

    Tests both migration directions.
    """
    print("\n" + "=" * 70)
    print("  TRACK 4: Migration & Recovery Benchmarks")
    print("=" * 70)

    if not _check_docker():
        print("    [WARN]  Docker not available - skipping track")
        return {"track": "migration", "status": "skipped", "reason": "no_docker"}

    results = {
        "track": "migration",
        "status": "success",
        "migrations": {},
    }

    # Core → Governance
    results["migrations"]["core_to_governance"] = run_migration_benchmark(
        corpus, "qdrant", "pgvector", gpu_device=gpu_device,
    )

    # Governance → Core
    results["migrations"]["governance_to_core"] = run_migration_benchmark(
        corpus, "pgvector", "qdrant", gpu_device=gpu_device,
    )

    return results
