"""Local-only G4 harness keeping identity keys and expectations isolated."""

from __future__ import annotations

import secrets
import json
from pathlib import Path
from typing import Any

from .audit import ContentFreeAuditSink
from .canonical import digest, file_sha256, load_jsonl, write_json, write_jsonl
from .engine import OfflineAuthorizationEngine
from .generator import generate_development_corpus
from .identity import FixtureIdentityAuthority, _HARNESS_CAPABILITY


def _all_canaries(cases: list[dict[str, Any]]) -> tuple[str, ...]:
    values: list[str] = []
    for case in cases:
        artifact = case.get("artifact", {})
        values.extend(str(value) for value in artifact.get("forbidden_after_redaction", []))
        values.extend(str(value) for value in artifact.get("denied_output_canaries", []))
    return tuple(values)


def run_reference_harness(
    corpus_root: str | Path,
    output_root: str | Path,
    *,
    fixture_key: bytes | None = None,
) -> dict[str, Any]:
    corpus = Path(corpus_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    cases = load_jsonl(corpus / "cases.jsonl")
    canaries = _all_canaries(cases)
    secret = fixture_key if fixture_key is not None else secrets.token_bytes(32)
    authority = FixtureIdentityAuthority(key=secret, _capability=_HARNESS_CAPABILITY)
    engine = OfflineAuthorizationEngine(authority)
    audit = ContentFreeAuditSink(prohibited_canaries=canaries)
    results = [engine.evaluate(case, audit) for case in cases]
    rows = [result.content_free_dict() for result in results]
    results_path = output / "case_results.jsonl"
    audit_path = output / "audit.jsonl"
    write_jsonl(results_path, rows)
    audit.write(audit_path)

    serialized_packages = json.dumps(
        [result.package for result in results if result.package is not None],
        ensure_ascii=False,
        sort_keys=True,
    )
    serialized_outputs = (
        results_path.read_text(encoding="utf-8")
        + audit_path.read_text(encoding="utf-8")
        + serialized_packages
    )
    leaked_canaries = sorted({value for value in canaries if value in serialized_outputs})
    counts: dict[str, int] = {}
    for result in results:
        counts[result.outcome] = counts.get(result.outcome, 0) + 1
    summary = {
        "schema_version": "gatemem-g4-reference-run-v1",
        "status": "GATEMEM_G4_REFERENCE_RUN_COMPLETE",
        "classification": "REFERENCE_CONTRACT_CONFORMANCE_ON_INSPECTABLE_SYNTHETIC_DEVELOPMENT_CASES",
        "case_count": len(results),
        "outcome_counts": counts,
        "provenance_integrity_count": sum(result.provenance_integrity for result in results),
        "audit_event_count": len(audit.events),
        "leaked_canaries": leaked_canaries,
        "corpus_manifest_sha256": file_sha256(corpus / "manifest.json"),
        "case_results_sha256": file_sha256(results_path),
        "audit_sha256": file_sha256(audit_path),
        "offline_only": True,
        "network_used": False,
        "runtime_route_used": False,
        "gatemem_data_used": False,
        "held_out_evaluation": False,
        "deletion_capability_claim": False,
        "production_authority_claim": False,
    }
    write_json(output / "run_summary.json", summary)
    return {"summary": summary, "results": results, "canaries": canaries}


def generate_and_run(
    corpus_root: str | Path,
    output_root: str | Path,
    *,
    fixture_key: bytes | None = None,
) -> dict[str, Any]:
    manifest = generate_development_corpus(corpus_root)
    run = run_reference_harness(corpus_root, output_root, fixture_key=fixture_key)
    run["manifest"] = manifest
    return run


def evaluate_case_in_memory(
    case: dict[str, Any], *, fixture_key: bytes | None = None
):
    """Test helper that still creates/loads key material only inside the harness."""

    canaries = tuple(case.get("artifact", {}).get("forbidden_after_redaction", []))
    audit = ContentFreeAuditSink(prohibited_canaries=canaries)
    secret = fixture_key if fixture_key is not None else secrets.token_bytes(32)
    authority = FixtureIdentityAuthority(key=secret, _capability=_HARNESS_CAPABILITY)
    return OfflineAuthorizationEngine(authority).evaluate(case, audit)


def artifact_contains_secret(roots: list[str | Path], secret: bytes) -> bool:
    """Scan persisted corpus/output artifacts for raw, hex, or base64 key forms."""

    import base64

    needles = {
        secret,
        secret.hex().encode("ascii"),
        base64.b64encode(secret),
        base64.urlsafe_b64encode(secret),
    }
    for root in map(Path, roots):
        for path in root.rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                if any(needle in data for needle in needles):
                    return True
    return False


def cleanup_generated_artifacts(
    artifact_root: str | Path, *, allowed_parent: str | Path
) -> None:
    """Remove only known G4 outputs inside an explicitly allowed parent."""

    root = Path(artifact_root).resolve()
    parent = Path(allowed_parent).resolve()
    if root == parent or parent not in root.parents:
        raise ValueError("artifact root is outside the allowed rollback parent")
    allowed = {
        "cases.jsonl",
        "expectations.jsonl",
        "manifest.json",
        "case_results.jsonl",
        "audit.jsonl",
        "run_summary.json",
    }
    if not root.exists():
        return
    children = list(root.iterdir())
    if any(child.is_dir() or child.name not in allowed for child in children):
        raise ValueError("artifact root contains an unknown rollback target")
    for child in children:
        child.unlink()
    root.rmdir()
