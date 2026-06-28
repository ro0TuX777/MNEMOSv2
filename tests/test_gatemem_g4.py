"""Focused tests for the G4 local offline reference implementation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from prototype.gatemem_g4 import (
    ContentFreeAuditSink,
    artifact_contains_secret,
    build_development_cases,
    cleanup_generated_artifacts,
    evaluate_case_in_memory,
    generate_and_run,
    generate_development_corpus,
)
from prototype.gatemem_g4.audit import AUDIT_FIELDS
from prototype.gatemem_g4.canonical import load_jsonl
from tools.run_gatemem_g4_gate import evaluate_gate
from tools.verify_gatemem_g4_frozen import verify_frozen_reference


def _case(case_id: str):
    cases, _ = build_development_cases()
    return deepcopy(next(case for case in cases if case["case_id"] == case_id))


def _evaluate(case):
    return evaluate_case_in_memory(case)


def test_generator_is_deterministic_and_marks_corpus_non_held_out(tmp_path):
    first = generate_development_corpus(tmp_path / "one")
    second = generate_development_corpus(tmp_path / "two")
    assert first["composite_sha256"] == second["composite_sha256"]
    assert first["held_out_eligible"] is False
    assert first["contains_gatemem_data"] is False
    assert first["contains_production_data"] is False


def test_expectations_are_separate_from_decision_cases(tmp_path):
    generate_development_corpus(tmp_path)
    cases = load_jsonl(tmp_path / "cases.jsonl")
    assert all("outcome" not in case and "reason_code" not in case for case in cases)
    assert (tmp_path / "expectations.jsonl").is_file()


def test_permit_and_redaction_paths_preserve_provenance():
    permit = _evaluate(_case("permit_owned"))
    redacted = _evaluate(_case("permit_redacted"))
    assert permit.outcome == "DISCLOSED"
    assert redacted.outcome == "DISCLOSED_WITH_REDACTION"
    assert "SYNTH-SPAN-SECRET" not in json.dumps(redacted.package)
    assert "SYNTH-FIELD-SECRET" not in json.dumps(redacted.package)
    assert permit.provenance_integrity and redacted.provenance_integrity


@pytest.mark.parametrize("case_id", ["forged_signature", "unknown_issuer", "tampered_claims"])
def test_forged_identity_envelopes_fail_closed(case_id):
    assert _evaluate(_case(case_id)).outcome == "DENIED"


def test_query_wording_and_caller_scope_never_grant_authority():
    assert _evaluate(_case("query_claims_authority")).reason_code == "ENTITLEMENT_MISSING"
    assert _evaluate(_case("caller_tenant_widening")).outcome == "DENIED"
    assert _evaluate(_case("caller_session_widening")).outcome == "DENIED"


def test_role_without_entitlement_and_explicit_denial_fail_closed():
    assert _evaluate(_case("entitlement_missing")).outcome == "DENIED"
    assert _evaluate(_case("explicit_deny")).reason_code == "OVERRIDING_DENIAL"


def test_scoped_delegation_permits_only_within_operation_and_time():
    assert _evaluate(_case("permit_delegated")).outcome == "DISCLOSED"
    assert _evaluate(_case("delegation_operation_widening")).outcome == "DENIED"
    assert _evaluate(_case("delegation_expired")).outcome == "DENIED"


def test_redaction_residue_and_unknown_transform_become_denial():
    assert _evaluate(_case("redaction_residue")).reason_code == "REDACTION_VERIFICATION_FAILED"
    assert _evaluate(_case("unknown_redaction")).reason_code == "REDACTION_OBLIGATION_UNKNOWN"


def test_package_budgets_fail_closed():
    assert _evaluate(_case("package_character_budget")).reason_code == "PACKAGE_BUDGET_EXCEEDED"
    assert _evaluate(_case("package_record_budget")).reason_code == "PACKAGE_BUDGET_EXCEEDED"


@pytest.mark.parametrize("dimension", ["identity", "entitlement", "policy", "descriptor", "redaction"])
def test_replay_rejected_on_every_registered_drift_dimension(dimension):
    result = _evaluate(_case(f"replay_{dimension}_drift"))
    assert result.reason_code == "REPLAY_CONFLICT"


def test_evaluator_field_injection_is_rejected_before_policy():
    result = _evaluate(_case("evaluator_field_injection"))
    assert result.reason_code == "CASE_SCHEMA_INVALID"
    assert result.package is None


def test_audit_rejects_unknown_fields_and_protected_canaries():
    event = {field: None for field in AUDIT_FIELDS}
    event.update({"consumer_id": "consumer", "schema_version": "v1"})
    sink = ContentFreeAuditSink(prohibited_canaries=["PROTECTED-CANARY"])
    with pytest.raises(ValueError, match="schema violation"):
        sink.emit({**event, "raw_query": "hello"})
    with pytest.raises(ValueError, match="content canary"):
        sink.emit({**event, "consumer_id": "PROTECTED-CANARY"})


def test_fixture_hmac_key_never_persists_in_case_corpus_or_audit(tmp_path):
    sentinel = b"G4-HMAC-KEY-MUST-NEVER-PERSIST!!"
    corpus = tmp_path / "corpus"
    output = tmp_path / "output"
    generate_and_run(corpus, output, fixture_key=sentinel)
    assert not artifact_contains_secret([corpus, output], sentinel)
    cases = (corpus / "cases.jsonl").read_text(encoding="utf-8")
    assert "hmac_key" not in cases
    assert "fixture_key" not in cases
    rows = load_jsonl(corpus / "cases.jsonl")
    assert all("signature" not in row for row in rows)


def test_harness_matches_all_inspectable_development_expectations(tmp_path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "output"
    run = generate_and_run(corpus, output)
    expected = {row["case_id"]: row for row in load_jsonl(corpus / "expectations.jsonl")}
    actual = {row["case_id"]: row for row in load_jsonl(output / "case_results.jsonl")}
    assert len(actual) == run["manifest"]["case_count"]
    assert all(actual[key]["outcome"] == value["outcome"] for key, value in expected.items())
    assert all(actual[key]["reason_code"] == value["reason_code"] for key, value in expected.items())
    assert run["summary"]["leaked_canaries"] == []


def test_bounded_rollback_removes_only_known_generated_artifacts(tmp_path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "output"
    generate_and_run(corpus, output)
    cleanup_generated_artifacts(output, allowed_parent=tmp_path)
    cleanup_generated_artifacts(corpus, allowed_parent=tmp_path)
    assert not output.exists() and not corpus.exists()
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "unknown.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown rollback target"):
        cleanup_generated_artifacts(unsafe, allowed_parent=tmp_path)
    assert (unsafe / "unknown.txt").is_file()


def test_gate_passes_and_retains_reference_only_claim(tmp_path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "output"
    generate_and_run(corpus, output)
    gate = evaluate_gate(corpus, output)
    assert gate["all_passed"] is True
    assert gate["classification"] == "REFERENCE_CONTRACT_CONFORMANCE_ON_INSPECTABLE_SYNTHETIC_DEVELOPMENT_CASES"
    assert "not authorization security" in gate["claim_boundary"]


def test_published_g4_reference_baseline_remains_frozen():
    result = verify_frozen_reference()
    assert result["verified"] is True
    assert all(result["checks"].values())
    assert result["manifest"]["claims"]["regression_testing_only"] is True
    assert result["manifest"]["claims"]["generalization_claim"] is False
