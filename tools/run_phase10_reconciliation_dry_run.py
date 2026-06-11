"""Run the Phase 10 knowledge reconciliation pre-check or action mode."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.engram.model import Engram
from mnemos.governance.hygiene.reconciliation_runner import ReconciliationRunner
from mnemos.governance.models.memory_state import GovernanceMeta

DEFAULT_TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "knowledge_reconciliation_v1.json"
RAW_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "summaries"


def _make_engram(case: Dict[str, Any], item: Dict[str, Any]) -> Engram:
    return Engram(
        id=item["id"],
        content=item["content"],
        source=item.get("source", ""),
        metadata={"truthset_case_id": case["id"]},
        governance=GovernanceMeta(
            entity_key=case["entity_key"],
            attribute_key=case["attribute_key"],
            normalized_value=item["normalized_value"],
            trust_score=float(item.get("trust_score", 0.8)),
            utility_score=float(item.get("utility_score", 0.8)),
            source_authority=float(item.get("source_authority", 0.5)),
        ),
    )


def _load_engrams(truthset: Path) -> tuple[Dict[str, Any], List[Engram]]:
    payload = json.loads(truthset.read_text(encoding="utf-8"))
    engrams: List[Engram] = []
    for case in payload["collisions"]:
        for item in case["engrams"]:
            engrams.append(_make_engram(case, item))
    return payload, engrams


def _qdrant_count(qdrant_url: str, collection: str) -> int:
    from qdrant_client import QdrantClient

    return QdrantClient(url=qdrant_url, timeout=30).count(
        collection, exact=True
    ).count


def evaluate(
    truthset: Path,
    *,
    mode: str,
    qdrant_url: str,
    collection: str,
    embedding_model: str,
    gpu_device: str,
    seed_parents: bool,
) -> Dict[str, Any]:
    payload, engrams = _load_engrams(truthset)
    dry_run = mode == "dry-run"
    before_count: Optional[int] = None
    after_count: Optional[int] = None
    indexer = None
    if not dry_run:
        from mnemos.retrieval.qdrant_tier import QdrantTier

        before_count = _qdrant_count(qdrant_url, collection)
        indexer = QdrantTier(
            url=qdrant_url,
            collection_name=collection,
            embedding_model=embedding_model,
            embedding_dim=768,
            gpu_device=gpu_device,
        )
        if seed_parents:
            indexer.index(engrams)

    report = ReconciliationRunner().run(
        engrams,
        dry_run=dry_run,
        indexer=indexer,
    )
    if not dry_run:
        after_count = _qdrant_count(qdrant_url, collection)

    expected = len(payload["collisions"])
    detected = report.contradictions_found
    synthesized = len(report.records)
    writes = report.resolution_engram_writes
    schema_pass = all(
        record.resolution_engram.metadata.get("is_resolution_engram") is True
        and record.resolution_engram.source.startswith("derived://reconciliation/")
        and len(record.resolution_engram.edges) >= 2
        for record in report.records
    )
    detection_pass = detected == expected and synthesized == expected
    write_pass = dry_run or writes == expected
    count_delta = (
        None
        if before_count is None or after_count is None
        else after_count - before_count
    )

    return {
        "truthset": str(truthset.resolve().relative_to(PROJECT_ROOT)),
        "mode": mode,
        "collection": collection,
        "expected_collisions": expected,
        "detected_collisions": detected,
        "synthesized_resolution_engrams": synthesized,
        "resolution_engram_writes": writes,
        "parent_fixture_writes": len(engrams) if (not dry_run and seed_parents) else 0,
        "qdrant_count_before": before_count,
        "qdrant_count_after": after_count,
        "qdrant_count_delta": count_delta,
        "report": report.to_dict(),
        "gates": {
            "collision_detection": detection_pass,
            "resolution_schema": schema_pass,
            "resolution_writes": write_pass,
        },
        "overall_pass": detection_pass and schema_pass and write_pass,
    }


def write_artifacts(result: Dict[str, Any]) -> tuple[Path, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    mode = str(result["mode"]).replace("-", "_")
    raw_path = RAW_DIR / f"phase10_reconciliation_{mode}_{timestamp}_raw.json"
    summary_path = SUMMARY_DIR / f"phase10_reconciliation_{mode}_{timestamp}_summary.md"
    raw_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_path.write_text(
        "# Phase 10 Reconciliation Dry-Run Summary\n\n"
        f"- Mode: `{result['mode']}`\n"
        f"- Collection: `{result['collection']}`\n"
        f"- Truthset: `{result['truthset']}`\n"
        f"- Expected collisions: `{result['expected_collisions']}`\n"
        f"- Detected collisions: `{result['detected_collisions']}`\n"
        f"- Resolution engrams synthesized: `{result['synthesized_resolution_engrams']}`\n"
        f"- Resolution engram writes: `{result['resolution_engram_writes']}`\n"
        f"- Parent fixture writes: `{result['parent_fixture_writes']}`\n"
        f"- Qdrant count delta: `{result['qdrant_count_delta']}`\n"
        f"- Collision detection: `{'PASS' if result['gates']['collision_detection'] else 'FAIL'}`\n"
        f"- Resolution schema: `{'PASS' if result['gates']['resolution_schema'] else 'FAIL'}`\n"
        f"- Resolution writes: `{'PASS' if result['gates']['resolution_writes'] else 'FAIL'}`\n"
        f"- Gate: **{'PASS' if result['overall_pass'] else 'FAIL'}**\n\n"
        f"- Raw: `{raw_path.name}`\n",
        encoding="utf-8",
    )
    return raw_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truthset", type=Path, default=DEFAULT_TRUTHSET)
    parser.add_argument(
        "--mode",
        choices=["dry-run", "apply"],
        default="dry-run",
        help="dry-run previews; apply indexes synthesized resolution engrams",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Compatibility alias for --mode apply",
    )
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="mnemos_engrams_nomic_mrl")
    parser.add_argument("--embedding-model", default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--gpu-device", default="cuda")
    parser.add_argument(
        "--fail-on-conflict",
        action="store_true",
        help="Return non-zero when the reconciliation gate does not pass",
    )
    parser.add_argument(
        "--seed-parents",
        action="store_true",
        help="In apply mode, index the truthset parent collision fixtures before resolutions",
    )
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args()

    mode = "apply" if args.apply else args.mode
    result = evaluate(
        args.truthset,
        mode=mode,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        embedding_model=args.embedding_model,
        gpu_device=args.gpu_device,
        seed_parents=args.seed_parents,
    )
    print(f"mode: {result['mode']}")
    print(f"expected collisions: {result['expected_collisions']}")
    print(f"detected collisions: {result['detected_collisions']}")
    print(f"resolution engrams: {result['synthesized_resolution_engrams']}")
    print(f"resolution writes: {result['resolution_engram_writes']}")
    print(f"parent fixture writes: {result['parent_fixture_writes']}")
    if result["qdrant_count_delta"] is not None:
        print(
            "qdrant count: "
            f"{result['qdrant_count_before']} -> {result['qdrant_count_after']} "
            f"(delta {result['qdrant_count_delta']})"
        )
    print(f"gate: {'PASS' if result['overall_pass'] else 'FAIL'}")
    if not args.no_artifacts:
        raw, summary = write_artifacts(result)
        print(f"raw: {raw}")
        print(f"summary: {summary}")
    if args.fail_on_conflict and not result["overall_pass"]:
        return 1
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
