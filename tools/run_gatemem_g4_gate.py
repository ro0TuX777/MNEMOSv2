"""Evaluate G4 reference-contract conformance and isolation gates."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.gatemem_g4 import (  # noqa: E402
    artifact_contains_secret,
    cleanup_generated_artifacts,
    generate_and_run,
)
from prototype.gatemem_g4.audit import AUDIT_FIELDS  # noqa: E402
from prototype.gatemem_g4.canonical import file_sha256, load_jsonl, write_json  # noqa: E402
from tools.compile_gatemem_g2a_cross_domain import verify_frozen_baseline  # noqa: E402

DEFAULT_CORPUS = Path(r"G:\MNEMOS-research\gatemem_g4_development_corpus")
DEFAULT_OUTPUT = Path(r"G:\MNEMOS-research\gatemem_g4_reference_run")
DEFAULT_JSON = ROOT / "benchmarks" / "results" / "gatemem_g4_gate.json"
DEFAULT_MD = ROOT / "benchmarks" / "results" / "gatemem_g4_gate.md"
FROZEN_HASH = "4211bc91e7dbe53f588a8ecb00a04e6e44d2fb775ca1e888d058680acb5ad209"
IMPLEMENTATION_FILES = (
    "prototype/gatemem_g4/__init__.py",
    "prototype/gatemem_g4/audit.py",
    "prototype/gatemem_g4/canonical.py",
    "prototype/gatemem_g4/engine.py",
    "prototype/gatemem_g4/generator.py",
    "prototype/gatemem_g4/harness.py",
    "prototype/gatemem_g4/identity.py",
    "tools/run_gatemem_g4_offline.py",
    "tools/run_gatemem_g4_gate.py",
)


def _import_isolation() -> tuple[bool, bool, bool]:
    package = ROOT / "prototype" / "gatemem_g4"
    runtime = {"mnemos", "mnemos_sdk", "service", "bench"}
    network = {"requests", "httpx", "urllib", "socket"}
    hosted = {"openai", "anthropic", "transformers", "torch"}
    seen_runtime: set[str] = set()
    seen_network: set[str] = set()
    seen_hosted: set[str] = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            roots: set[str] = set()
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.split(".", 1)[0]}
            seen_runtime.update(roots & runtime)
            seen_network.update(roots & network)
            seen_hosted.update(roots & hosted)
    return not seen_runtime, not seen_network, not seen_hosted


def _key_authority_is_harness_owned() -> bool:
    package = ROOT / "prototype" / "gatemem_g4"
    constructor_sites: set[str] = set()
    generation_sites: set[str] = set()
    for path in package.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "FixtureIdentityAuthority(" in text and path.name != "identity.py":
            constructor_sites.add(path.name)
        if "secrets.token_bytes(" in text:
            generation_sites.add(path.name)
    return constructor_sites == {"harness.py"} and generation_sites == {"harness.py"}


def _no_runtime_reverse_imports() -> bool:
    allowed = {
        ROOT / "tools" / "run_gatemem_g4_offline.py",
        ROOT / "tools" / "run_gatemem_g4_gate.py",
        ROOT / "tests" / "test_gatemem_g4.py",
    }
    package = ROOT / "prototype" / "gatemem_g4"
    for path in ROOT.rglob("*.py"):
        if package in path.parents or path in allowed:
            continue
        if "prototype.gatemem_g4" in path.read_text(encoding="utf-8"):
            return False
    return True


def _expectations_isolated_from_decision_path() -> bool:
    package = ROOT / "prototype" / "gatemem_g4"
    decision_files = ("identity.py", "engine.py", "audit.py")
    core_clean = all(
        "expectations.jsonl" not in (package / name).read_text(encoding="utf-8")
        for name in decision_files
    )
    harness = (package / "harness.py").read_text(encoding="utf-8")
    return core_clean and 'load_jsonl(corpus / "expectations.jsonl")' not in harness


def _implementation_fingerprint(corpus_composite: str) -> dict[str, Any]:
    files = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in IMPLEMENTATION_FILES
    }
    canonical = json.dumps(
        {"files": files, "corpus_composite_sha256": corpus_composite},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "source_sha256": files,
        "corpus_composite_sha256": corpus_composite,
        "composite_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _expectation_matches(
    expectations: list[dict[str, Any]], results: list[dict[str, Any]]
) -> tuple[int, list[str]]:
    actual = {row["case_id"]: row for row in results}
    mismatches: list[str] = []
    for expected in expectations:
        row = actual.get(expected["case_id"])
        if row is None or row["outcome"] != expected["outcome"] or row["reason_code"] != expected["reason_code"]:
            mismatches.append(expected["case_id"])
    return len(expectations) - len(mismatches), mismatches


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# GateMem G4 Offline Reference Implementation Gate",
        "",
        "```text",
        report["status"],
        report["classification"],
        "SYNTHETIC_DEVELOPMENT_ONLY",
        "NO_PRODUCTION_AUTHORITY_CLAIM",
        "NO_GATEMEM_SCORE_OR_HELD_OUT_CLAIM",
        "NO_DELETION_ENGINEERING",
        "```",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["gates"].items()
    )
    lines.extend(
        [
            "",
            f"**Overall: {'PASS' if report['all_passed'] else 'FAIL'}**",
            "",
            f"Cases: `{report['counts']['cases']}`; exact expectation matches: `{report['counts']['exact_matches']}`.",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_gate(corpus_root: Path, output_root: Path) -> dict[str, Any]:
    manifest = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    cases = load_jsonl(corpus_root / "cases.jsonl")
    expectations = load_jsonl(corpus_root / "expectations.jsonl")
    results = load_jsonl(output_root / "case_results.jsonl")
    audit = load_jsonl(output_root / "audit.jsonl")
    exact, mismatches = _expectation_matches(expectations, results)
    by_id = {row["case_id"]: row for row in results}
    no_runtime, no_network, no_hosted = _import_isolation()
    forbidden_case_keys = {"hmac_key", "fixture_key", "secret_key", "key_material"}
    serialized_cases = json.dumps(cases, ensure_ascii=False, sort_keys=True)

    sentinel = b"G4-HMAC-KEY-MUST-NEVER-PERSIST!!"
    with tempfile.TemporaryDirectory(prefix="gatemem-g4-key-isolation-") as temp:
        root = Path(temp)
        mutation_corpus = root / "corpus"
        mutation_output = root / "output"
        generate_and_run(mutation_corpus, mutation_output, fixture_key=sentinel)
        key_isolated = not artifact_contains_secret(
            [mutation_corpus, mutation_output], sentinel
        )
        repeat_one = generate_and_run(root / "repeat-one-corpus", root / "repeat-one-output")
        repeat_two = generate_and_run(root / "repeat-two-corpus", root / "repeat-two-output")
        deterministic_rerun = (
            repeat_one["manifest"]["composite_sha256"]
            == repeat_two["manifest"]["composite_sha256"]
            and repeat_one["summary"]["case_results_sha256"]
            == repeat_two["summary"]["case_results_sha256"]
            and repeat_one["summary"]["audit_sha256"]
            == repeat_two["summary"]["audit_sha256"]
        )
        rollback_corpus = root / "rollback-corpus"
        rollback_output = root / "rollback-output"
        generate_and_run(rollback_corpus, rollback_output)
        cleanup_generated_artifacts(rollback_output, allowed_parent=root)
        cleanup_generated_artifacts(rollback_corpus, allowed_parent=root)
        rollback_rehearsed = not rollback_output.exists() and not rollback_corpus.exists()

    frozen = verify_frozen_baseline(
        ROOT / "benchmarks" / "results" / "gatemem_g2_baseline_manifest.json"
    )
    replay_ids = [case["case_id"] for case in cases if case["case_id"].startswith("replay_")]
    gates = {
        "authorized_scope_exact": manifest["case_count"] == len(cases) == len(expectations) == len(results),
        "corpus_files_match_manifest": all(
            file_sha256(corpus_root / relative) == expected
            for relative, expected in manifest["file_sha256"].items()
        ),
        "exact_development_expectations": exact == len(expectations) and not mismatches,
        "expectations_isolated_from_decision_path": _expectations_isolated_from_decision_path(),
        "synthetic_mnemos_owned_only": manifest["contains_gatemem_data"] is False and manifest["contains_production_data"] is False,
        "corpus_permanently_non_held_out": manifest["held_out_eligible"] is False and summary["held_out_evaluation"] is False,
        "hmac_key_absent_from_case_schema": not any(f'"{key}"' in serialized_cases for key in forbidden_case_keys),
        "hmac_key_created_or_loaded_only_by_harness": _key_authority_is_harness_owned(),
        "hmac_key_isolation_mutation": key_isolated,
        "forged_and_tampered_envelopes_denied": all(by_id[name]["outcome"] == "DENIED" for name in ("forged_signature", "unknown_issuer", "tampered_claims")),
        "caller_widening_denied": all(by_id[name]["outcome"] == "DENIED" for name in ("caller_tenant_widening", "caller_session_widening")),
        "query_wording_grants_no_authority": by_id["query_claims_authority"]["reason_code"] == "ENTITLEMENT_MISSING",
        "scoped_delegation_bounded": by_id["permit_delegated"]["outcome"] == "DISCLOSED" and by_id["delegation_operation_widening"]["outcome"] == "DENIED" and by_id["delegation_expired"]["outcome"] == "DENIED",
        "role_alone_cannot_permit": by_id["entitlement_missing"]["outcome"] == "DENIED",
        "explicit_denial_precedence": by_id["explicit_deny"]["reason_code"] == "OVERRIDING_DENIAL",
        "classification_and_lineage_fail_closed": all(by_id[name]["outcome"] == "DENIED" for name in ("artifact_class_denied", "source_class_denied", "classification_exceeded", "lineage_incomplete")),
        "redaction_success_and_failure_bounded": by_id["permit_redacted"]["outcome"] == "DISCLOSED_WITH_REDACTION" and by_id["redaction_residue"]["outcome"] == "DENIED" and by_id["unknown_redaction"]["outcome"] == "DENIED",
        "package_budgets_fail_closed": by_id["package_character_budget"]["outcome"] == "DENIED" and by_id["package_record_budget"]["outcome"] == "DENIED",
        "all_replay_drift_denied": bool(replay_ids) and all(by_id[name]["reason_code"] == "REPLAY_CONFLICT" for name in replay_ids),
        "evaluator_field_injection_rejected": by_id["evaluator_field_injection"]["reason_code"] == "CASE_SCHEMA_INVALID",
        "no_denied_canary_leakage": summary["leaked_canaries"] == [],
        "content_free_audit_complete": len(audit) == len(cases) == summary["audit_event_count"],
        "strict_audit_allowlist": all(set(event) == AUDIT_FIELDS for event in audit),
        "audit_retention_metadata_bounded": all(event["retention_days"] == 30 for event in audit),
        "provenance_integrity_complete": summary["provenance_integrity_count"] == len(cases),
        "package_imports_isolated_from_runtime": no_runtime,
        "runtime_has_no_reverse_import": _no_runtime_reverse_imports(),
        "no_network_or_hosted_provider_imports": no_network and no_hosted and summary["network_used"] is False,
        "no_runtime_route_or_production_authority_claim": summary["runtime_route_used"] is False and summary["production_authority_claim"] is False,
        "no_deletion_capability_claim": summary["deletion_capability_claim"] is False,
        "frozen_g2_core_unchanged": frozen["composite_sha256"] == FROZEN_HASH,
        "deterministic_rerun_equivalence": deterministic_rerun,
        "bounded_rollback_rehearsed": rollback_rehearsed,
    }
    return {
        "schema_version": "gatemem-g4-gate-v1",
        "authorization": "GATEMEM_G4_OFFLINE_REFERENCE_IMPLEMENTATION_AUTHORIZED",
        "status": "GATEMEM_G4_OFFLINE_REFERENCE_IMPLEMENTATION_COMPLETE",
        "classification": "REFERENCE_CONTRACT_CONFORMANCE_ON_INSPECTABLE_SYNTHETIC_DEVELOPMENT_CASES",
        "gates": gates,
        "all_passed": all(gates.values()),
        "counts": {"cases": len(cases), "exact_matches": exact, "mismatches": mismatches},
        "implementation_fingerprint": _implementation_fingerprint(
            manifest["composite_sha256"]
        ),
        "frozen_g2_composite_sha256": frozen["composite_sha256"],
        "claim_boundary": (
            "This result demonstrates reference-contract conformance on inspectable "
            "synthetic development cases. It is not authorization security, production "
            "readiness, held-out evaluation, or benchmark performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    if args.regenerate:
        generate_and_run(args.corpus_root, args.output_root)
    report = evaluate_gate(args.corpus_root, args.output_root)
    write_json(DEFAULT_JSON, report)
    DEFAULT_MD.write_text(_render(report), encoding="utf-8", newline="\n")
    print(f"All G4 gates passed: {report['all_passed']}")
    print(f"Wrote {DEFAULT_JSON}")
    print(f"Wrote {DEFAULT_MD}")
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
