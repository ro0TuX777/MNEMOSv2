"""Seed concise, high-signal MNEMOS repo summary cards into a running service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.mnemos_seed_manifest import DEFAULT_MANIFEST_PATH, update_manifest_section
from tools.mnemos_seed_utils import (
    build_seed_engram_id,
    build_seed_identity,
    build_seed_snapshot_id,
    normalized_content_hash,
)


DEFAULT_BASE_URL = "http://localhost:8700"
SEED_SCHEMA_VERSION = "summary_seed_v1"
DEFAULT_PARAPHRASE_QUERIES = [
    "GateMem G4 frozen regression baseline",
    "GateMem G4 frozen reference baseline",
    "no further internal prototyping authorized",
    "Further GateMem policy and implementation work is paused",
]

_SUMMARY_SPECS = [
    {
        "content": (
            "GateMem G4 is frozen for regression testing only. "
            "The frozen G4 implementation/corpus composite is "
            "ed3b5c7672e591b039183eaa2d8c7c7630a655575bfc7866558d53f2eb874c52. "
            "Changes require a new development iteration rather than rewriting the frozen result."
        ),
        "source": "summary://gatemem/g4_frozen_baseline",
        "neuro_tags": ["repo_summary", "gatemem", "g4", "frozen_baseline"],
        "confidence": 0.99,
        "metadata": {
            "source_uri": "benchmarks/results/gatemem_g4_frozen_reference_manifest.md",
            "filename": "gatemem_g4_frozen_reference_manifest.md",
            "ingestion_source": "seed_mnemos_repo_summaries",
            "topic": "gatemem_reference",
            "summary_id": "gatemem_g4_frozen_baseline",
        },
    },
    {
        "content": (
            "GateMem G4 frozen regression baseline. GateMem G4 frozen reference baseline. "
            "This refers to the frozen G4 implementation/corpus composite and the regression-only "
            "baseline that must not be rewritten in place."
        ),
        "source": "summary://gatemem/g4_alias_frozen_regression_baseline",
        "neuro_tags": ["repo_summary", "gatemem", "g4", "alias", "frozen_baseline"],
        "confidence": 0.99,
        "metadata": {
            "source_uri": "benchmarks/results/gatemem_g4_frozen_reference_manifest.md",
            "filename": "gatemem_g4_frozen_reference_manifest.md",
            "ingestion_source": "seed_mnemos_repo_summaries",
            "topic": "gatemem_reference",
            "summary_id": "gatemem_g4_alias_frozen_regression_baseline",
        },
    },
    {
        "content": (
            "Further GateMem policy and implementation work is paused. "
            "The next meaningful milestone requires an independent sealed-evaluation custodian, "
            "a newly sealed or independent corpus, completed preregistration, a frozen candidate "
            "policy artifact, and one-shot evaluation under evaluator-only label access."
        ),
        "source": "summary://gatemem/pause_and_blocker",
        "neuro_tags": ["repo_summary", "gatemem", "pause", "blocker"],
        "confidence": 0.99,
        "metadata": {
            "source_uri": "docs/benchmarks/gatemem_program_status.md",
            "filename": "gatemem_program_status.md",
            "ingestion_source": "seed_mnemos_repo_summaries",
            "topic": "gatemem_reference",
            "summary_id": "gatemem_pause_and_blocker",
        },
    },
    {
        "content": (
            "No further internal prototyping authorized. GateMem work is paused and internal policy "
            "or implementation work must not continue as though G4 were still an open prototype lane."
        ),
        "source": "summary://gatemem/alias_no_further_internal_prototyping",
        "neuro_tags": ["repo_summary", "gatemem", "alias", "pause", "blocker"],
        "confidence": 0.99,
        "metadata": {
            "source_uri": "docs/benchmarks/gatemem_program_status.md",
            "filename": "gatemem_program_status.md",
            "ingestion_source": "seed_mnemos_repo_summaries",
            "topic": "gatemem_reference",
            "summary_id": "gatemem_alias_no_further_internal_prototyping",
        },
    },
    {
        "content": (
            "GateMem G4 demonstrated reference-contract conformance on inspectable synthetic "
            "development cases only. It is not authorization security, production readiness, "
            "held-out evaluation, legal compliance, or deletion capability."
        ),
        "source": "summary://gatemem/g4_claim_boundary",
        "neuro_tags": ["repo_summary", "gatemem", "claim_boundary", "g4"],
        "confidence": 0.99,
        "metadata": {
            "source_uri": "docs/benchmarks/gatemem_g4_offline_reference_implementation.md",
            "filename": "gatemem_g4_offline_reference_implementation.md",
            "ingestion_source": "seed_mnemos_repo_summaries",
            "topic": "gatemem_reference",
            "summary_id": "gatemem_g4_claim_boundary",
        },
    },
    {
        "content": (
            "The GateMem continuation packet is ready for external handoff, but sealed evaluation "
            "is still blocked because external inputs are required. No evaluation run is authorized."
        ),
        "source": "summary://gatemem/g5_blocked_handoff",
        "neuro_tags": ["repo_summary", "gatemem", "g5", "handoff"],
        "confidence": 0.99,
        "metadata": {
            "source_uri": "docs/benchmarks/gatemem_g5/README.md",
            "filename": "README.md",
            "ingestion_source": "seed_mnemos_repo_summaries",
            "topic": "gatemem_reference",
            "summary_id": "gatemem_g5_blocked_handoff",
        },
    },
]


def _seed_snapshot_id() -> str:
    return build_seed_snapshot_id(
        [
            SEED_SCHEMA_VERSION,
            *(
                f"{item['source']}|{item['metadata']['source_uri']}|{item['metadata']['summary_id']}|{normalized_content_hash(item['content'])}"
                for item in _SUMMARY_SPECS
            ),
        ]
    )


def _with_seed_identity(item: dict[str, Any], *, seed_snapshot_id: str) -> dict[str, Any]:
    metadata = dict(item.get("metadata", {}))
    canonical_source_uri = str(metadata.get("source_uri") or item.get("source") or "")
    summary_card_kind = str(metadata.get("summary_id") or item.get("source") or "summary_card")
    seed_identity = build_seed_identity(
        canonical_source_uri=canonical_source_uri,
        seed_kind=summary_card_kind,
        schema_version=SEED_SCHEMA_VERSION,
    )
    content_hash = normalized_content_hash(item["content"])
    metadata.update(
        {
            "canonical_source_uri": canonical_source_uri,
            "summary_card_kind": summary_card_kind,
            "schema_version": SEED_SCHEMA_VERSION,
            "seed_identity": seed_identity,
            "normalized_content_hash": content_hash,
            "seed_snapshot_id": seed_snapshot_id,
            "retrieval_only": True,
            "source_linked": True,
            "is_seeded_summary": True,
            "is_superseded": False,
        }
    )
    return {
        "id": build_seed_engram_id(seed_identity),
        "content": item["content"],
        "source": item["source"],
        "neuro_tags": list(item.get("neuro_tags", [])),
        "confidence": item.get("confidence", 1.0),
        "metadata": metadata,
    }


def build_default_summaries() -> list[dict[str, Any]]:
    snapshot_id = _seed_snapshot_id()
    return [_with_seed_identity(item, seed_snapshot_id=snapshot_id) for item in _SUMMARY_SPECS]


# Short cards are easier for semantic retrieval than full markdown documents.
DEFAULT_SUMMARIES = build_default_summaries()


def seed_summaries(*, base_url: str, timeout_s: float) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/mnemos/index",
        json={"documents": DEFAULT_SUMMARIES},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("index response must be a JSON object")
    return payload


def search_summaries(*, base_url: str, query: str, timeout_s: float, top_k: int) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/mnemos/search",
        json={
            "query": query,
            "top_k": top_k,
            "retrieval_mode": "semantic",
            "explain": True,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("search response must be a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--query", default="GateMem G4 frozen regression baseline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-paraphrase-check", action="store_true")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args()

    seed_result = seed_summaries(base_url=args.base_url, timeout_s=args.timeout_s)
    manifest = update_manifest_section(
        section_name="repo_summaries",
        section_payload={
            "seed_schema_version": SEED_SCHEMA_VERSION,
            "seed_snapshot_id": _seed_snapshot_id(),
            "document_count": len(DEFAULT_SUMMARIES),
            "seed_identities": [
                item["metadata"]["seed_identity"] for item in DEFAULT_SUMMARIES
            ],
            "normalized_content_hashes": [
                item["metadata"]["normalized_content_hash"] for item in DEFAULT_SUMMARIES
            ],
        },
        path=Path(args.manifest_path),
    )

    print(f"Indexed summaries: {seed_result.get('result', {}).get('indexed')}")
    print(f"Engram IDs: {len(seed_result.get('result', {}).get('engram_ids', []))}")
    print(f"Seed manifest: {args.manifest_path}")
    print(f"Composite seed snapshot: {manifest.get('seed_snapshot_id')}")
    queries = (
        list(DEFAULT_PARAPHRASE_QUERIES)
        if args.run_paraphrase_check
        else [args.query]
    )
    for query in queries:
        search_result = search_summaries(
            base_url=args.base_url,
            query=query,
            timeout_s=args.timeout_s,
            top_k=args.top_k,
        )
        print(f"Query: {query}")
        print(f"Search status: {search_result.get('status')}")
        print(f"Search result count: {len(search_result.get('results', []))}")
        if search_result.get("results"):
            for idx, item in enumerate(search_result["results"][:3], start=1):
                engram = item.get("engram", {})
                print(f"{idx}. source={engram.get('source')} score={item.get('score')}")
                print(f"   {str(engram.get('content', ''))[:180]}")
        print()


if __name__ == "__main__":
    main()
