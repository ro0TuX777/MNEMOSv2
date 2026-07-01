"""Freeze the Evidence Admission and Budgeting R1 non-empty corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "evidence_admission_and_budgeting_r1_corpus_manifest.json"
R0_BASELINE_COMMIT = "bef472112a751436c7af35cf472e13ccfa3a2329"

CHUNKING_CONFIG = {
    "strategy": "word_window",
    "max_words": 120,
    "overlap_words": 20,
    "normalization": "collapse_whitespace",
}

EMBEDDING_PROFILE = {
    "retrieval_mode": "semantic",
    "fusion_policy": "balanced",
    "embedding_model_name": "BAAI/bge-base-en-v1.5",
    "lexical_top_k": 25,
    "semantic_top_k": 25,
    "candidate_envelope": "not_applied_during_corpus_freeze",
}

COLLECTION = {
    "name": "evidence_admission_r1_frozen_corpus",
    "intended_runtime": "local_http_service_or_direct_runtime_after_explicit_seed",
}

DEFAULT_SOURCE_SPECS: List[Dict[str, str]] = [
    {"path": "docs/benchmarks/gatemem_program_status.md", "family": "gatemem_governance_status", "role": "current_state_record"},
    {"path": "docs/benchmarks/gatemem_g4_offline_reference_implementation.md", "family": "gatemem_governance_status", "role": "current_state_record"},
    {"path": "docs/benchmarks/gatemem_g4_offline_reference_implementation_proposal.md", "family": "gatemem_governance_status", "role": "superseded_record"},
    {"path": "docs/adr/0013-gatemem-g4-offline-reference-implementation-proposal.md", "family": "gatemem_governance_status", "role": "dependency_blocker_record"},
    {"path": "docs/benchmarks/gatemem_g5/README.md", "family": "gatemem_governance_status", "role": "dependency_blocker_record"},
    {"path": "docs/benchmarks/gatemem_g5/handoff_checklist.md", "family": "gatemem_governance_status", "role": "dependency_blocker_record"},
    {"path": "docs/benchmarks/gatemem_g5/preregistration.md", "family": "gatemem_governance_status", "role": "dependency_blocker_record"},
    {"path": "docs/benchmarks/gatemem_g5/evaluator_protocol.md", "family": "gatemem_governance_status", "role": "negative_control_material"},
    {"path": "docs/benchmarks/gatemem_g5/custodian_charter.md", "family": "gatemem_governance_status", "role": "negative_control_material"},
    {"path": "docs/benchmarks/gatemem_g3_authorization_disclosure_semantics.md", "family": "gatemem_governance_status", "role": "duplicate_or_near_duplicate_condition"},
    {"path": "docs/benchmarks/gatemem_g2_offline_adapter.md", "family": "gatemem_governance_status", "role": "superseded_record"},
    {"path": "benchmarks/results/gatemem_g4_gate.md", "family": "gatemem_governance_status", "role": "current_state_record"},
    {"path": "benchmarks/results/gatemem_g4_frozen_reference_manifest.md", "family": "gatemem_governance_status", "role": "duplicate_or_near_duplicate_condition"},
    {"path": "benchmarks/results/gatemem_g2_baseline_manifest.md", "family": "gatemem_governance_status", "role": "superseded_record"},
    {"path": "benchmarks/results/gatemem_g2a_gate.md", "family": "gatemem_governance_status", "role": "superseded_record"},
    {"path": "docs/evidence_admission_and_budgeting_r0_closeout.md", "family": "retrieval_hygiene_associative_evidence_admission", "role": "current_state_record"},
    {"path": "docs/evidence_admission_and_budgeting_r0_design_note.md", "family": "retrieval_hygiene_associative_evidence_admission", "role": "current_state_record"},
    {"path": "benchmarks/results/evidence_admission_r0_formal_http_service_run_001.json", "family": "retrieval_hygiene_associative_evidence_admission", "role": "current_state_record"},
    {"path": "benchmarks/results/evidence_admission_r0_fresh_http_service_run_001.json", "family": "retrieval_hygiene_associative_evidence_admission", "role": "current_state_record"},
    {"path": "benchmarks/results/evidence_admission_r0_development_direct_runtime_run_001.json", "family": "retrieval_hygiene_associative_evidence_admission", "role": "negative_control_material"},
    {"path": "benchmarks/results/retrieval_hygiene_r0_closeout.md", "family": "retrieval_hygiene_associative_evidence_admission", "role": "current_state_record"},
    {"path": "benchmarks/results/retrieval_hygiene_r0_run_003.json", "family": "retrieval_hygiene_associative_evidence_admission", "role": "current_state_record"},
    {"path": "benchmarks/results/retrieval_hygiene_r0_run_001.json", "family": "retrieval_hygiene_associative_evidence_admission", "role": "superseded_record"},
    {"path": "docs/experiments/retrieval_hygiene_r0_frozen_alias_benchmark.json", "family": "retrieval_hygiene_associative_evidence_admission", "role": "duplicate_or_near_duplicate_condition"},
    {"path": "docs/experiments/retrieval_hygiene_r0_fresh_verification_pack.json", "family": "retrieval_hygiene_associative_evidence_admission", "role": "current_state_record"},
    {"path": "docs/associative_routing_e2_closeout.md", "family": "retrieval_hygiene_associative_evidence_admission", "role": "current_state_record"},
    {"path": "docs/associative_routing_e2_design_note.md", "family": "retrieval_hygiene_associative_evidence_admission", "role": "current_state_record"},
    {"path": "docs/associative_routing_e0_design_note.md", "family": "retrieval_hygiene_associative_evidence_admission", "role": "superseded_record"},
    {"path": "benchmarks/results/associative_routing_e2_live_comparison_run_001.json", "family": "retrieval_hygiene_associative_evidence_admission", "role": "current_state_record"},
    {"path": "benchmarks/results/associative_routing_e0_benchmark.md", "family": "retrieval_hygiene_associative_evidence_admission", "role": "superseded_record"},
    {"path": "README.md", "family": "unrelated_mnemos_documentation", "role": "current_state_record"},
    {"path": "docs/README.md", "family": "unrelated_mnemos_documentation", "role": "current_state_record"},
    {"path": "docs/architecture.md", "family": "unrelated_mnemos_documentation", "role": "current_state_record"},
    {"path": "docs/mnemos_operator_playbook.md", "family": "unrelated_mnemos_documentation", "role": "negative_control_material"},
    {"path": "mcp_servers/mnemos/README.md", "family": "unrelated_mnemos_documentation", "role": "negative_control_material"},
    {"path": "docs/adr/0001-deployment-profiles.md", "family": "unrelated_mnemos_documentation", "role": "negative_control_material"},
    {"path": "docs/adr/0002-summary-isolation.md", "family": "unrelated_mnemos_documentation", "role": "negative_control_material"},
    {"path": "docs/adr/0006-chat-evidence-contract.md", "family": "unrelated_mnemos_documentation", "role": "negative_control_material"},
    {"path": "docs/adr/0008-consumer-neutral-read-only-shadow-adapter-implementation.md", "family": "unrelated_mnemos_documentation", "role": "duplicate_or_near_duplicate_condition"},
    {"path": "docs/reports/post_gatemem_authorized_backlog_closeout.md", "family": "unrelated_mnemos_documentation", "role": "current_state_record"},
    {"path": "docs/whitepaper.md", "family": "unrelated_mnemos_documentation", "role": "duplicate_or_near_duplicate_condition"},
]


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _chunk_words(text: str, *, max_words: int, overlap_words: int) -> List[str]:
    words = _normalize_text(text).split()
    if not words:
        return []
    chunks: List[str] = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def build_manifest(*, freeze_date: str | None = None) -> Dict[str, Any]:
    sources: List[Dict[str, Any]] = []
    retrieval_units: List[Dict[str, Any]] = []
    source_hash_components: List[str] = []
    freeze_day = freeze_date or date.today().isoformat()

    for spec in DEFAULT_SOURCE_SPECS:
        path = ROOT / spec["path"]
        if not path.exists():
            raise FileNotFoundError(spec["path"])
        text = path.read_text(encoding="utf-8", errors="replace")
        source_hash = _sha256_bytes(path)
        chunks = _chunk_words(
            text,
            max_words=int(CHUNKING_CONFIG["max_words"]),
            overlap_words=int(CHUNKING_CONFIG["overlap_words"]),
        )
        source_hash_components.append(f"{spec['path']}:{source_hash}:{len(chunks)}")
        sources.append(
            {
                **spec,
                "sha256": source_hash,
                "byte_count": path.stat().st_size,
                "retrieval_unit_count": len(chunks),
            }
        )
        for idx, chunk in enumerate(chunks):
            retrieval_units.append(
                {
                    "unit_id": f"r1u-{len(retrieval_units) + 1:04d}",
                    "source_path": spec["path"],
                    "chunk_index": idx,
                    "text_sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                    "word_count": len(chunk.split()),
                    "family": spec["family"],
                    "role": spec["role"],
                }
            )

    manifest_digest = hashlib.sha256(
        "||".join(
            [
                R0_BASELINE_COMMIT,
                freeze_day,
                json.dumps(CHUNKING_CONFIG, sort_keys=True),
                json.dumps(EMBEDDING_PROFILE, sort_keys=True),
                *source_hash_components,
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]

    collection = {
        **COLLECTION,
        "snapshot": f"{COLLECTION['name']}:{len(retrieval_units)}",
    }
    return {
        "manifest_id": f"evidence_admission_and_budgeting_r1_corpus_{manifest_digest}",
        "status": "FROZEN_BEFORE_R1_ENFORCEMENT",
        "r0_baseline_commit": R0_BASELINE_COMMIT,
        "service_revision": _git_head(),
        "corpus_curator": "Codex implementation team",
        "freeze_date": freeze_day,
        "rules": [
            "Do not modify corpus content after the formal evaluation pack is frozen.",
            "Do not tune R1 enforcement policy, thresholds, or implementation against this corpus after preregistration without refreezing and invalidating prior claim evidence.",
            "The actual formal evaluation pack must be authored by an independent_non_implementation_author.",
        ],
        "chunking": CHUNKING_CONFIG,
        "embedding_profile": EMBEDDING_PROFILE,
        "collection": collection,
        "sources": sources,
        "retrieval_units": retrieval_units,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--freeze-date")
    args = parser.parse_args()

    output = Path(args.output)
    manifest = build_manifest(freeze_date=args.freeze_date)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
