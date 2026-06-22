"""Phase 5A held-out technical verification and mutation sensitivity tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from prototype.session_context_assembler.corpus import load_validated_corpus
from prototype.session_context_assembler.replay import run_condition_c, run_condition_c1
from tools.run_session_context_assembler_r2_verification import (
    EXPECTED_R1_SHA256,
    OWNER_MANIFEST,
    OWNER_PACK,
    R1_MANIFEST_PATH,
    R1_PATH,
    R2_MANIFEST_PATH,
    R2_PATH,
    SCORING_ONLY_FIELD,
    SELECTOR_PATH,
    _label_coverage,
    _lineage_loss,
    _score_case,
    run_verification,
    selector_boundary_violations,
)

R2_SHA256 = "ba990a35d507811105f879003d9c4b9ea8acea3884a8a694046e38f6bcb51ef5"
ADAPTER_DESIGN = Path(
    "docs/session_context_assembler_consumer_neutral_shadow_adapter_design.md"
)
ADR_0008 = Path(
    "docs/adr/0008-consumer-neutral-read-only-shadow-adapter-implementation.md"
)
EXPECTED_CLASSES = {
    "old_decisive_decision_vs_recent_lexical_distractor",
    "multiple_eligible_unresolved_contradictions",
    "mixed_resolved_and_unresolved_related_decisions",
    "source_linked_evidence_split_across_episodes",
    "high_salience_irrelevant_incident_interruption",
    "near_budget_mandatory_artifact_overflow",
    "missing_or_ineligible_source_link",
    "turn_reorder_within_an_episode",
    "irrelevant_turn_injection",
    "paraphrased_current_task",
}


@pytest.fixture(scope="module")
def corpus():
    return load_validated_corpus(R2_PATH, R2_MANIFEST_PATH)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(R2_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def verification():
    return run_verification()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r1_remains_unchanged_and_hash_valid():
    manifest = json.loads(R1_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert _sha256(R1_PATH) == EXPECTED_R1_SHA256 == manifest["file_sha256"]
    load_validated_corpus(R1_PATH, R1_MANIFEST_PATH)


def test_r2_is_frozen_hash_valid_and_held_out(corpus, manifest):
    assert _sha256(R2_PATH) == R2_SHA256 == manifest["file_sha256"]
    assert manifest["frozen"] is True
    assert len(corpus["cases"]) == manifest["case_count"] == 10
    assert {case["verification_class"] for case in corpus["cases"]} == EXPECTED_CLASSES
    assert all(case["id"].startswith("sca_r2_") for case in corpus["cases"])


def test_scoring_only_expectations_never_enter_selector_source():
    source = SELECTOR_PATH.read_text(encoding="utf-8")
    assert SCORING_ONLY_FIELD not in source
    assert selector_boundary_violations(source) == []


@pytest.mark.parametrize("seed", [0, 7, 31])
def test_s1_is_deterministic_for_fixed_seeds(corpus, manifest, seed):
    for case in corpus["cases"]:
        budget = case["expected_context_budget"]
        first = run_condition_c1(case, manifest["file_sha256"], seed, budget)
        second = run_condition_c1(case, manifest["file_sha256"], seed, budget)
        assert first == second


def test_all_phase5a_gates_pass(verification):
    result, _, _ = verification
    assert result["all_phase5a_advancement_gates_passed"] is True
    assert result["r1_unchanged_and_hash_valid"] is True
    assert all(gate["passed"] for gate in result["gates"].values())
    assert all(
        check["passed"] for check in result["mutation_sensitivity_self_checks"].values()
    )


def test_feasible_cases_retain_all_required_artifacts(verification):
    result, _, _ = verification
    rows = [
        row for row in result["records"]
        if row["condition"] == "C1_selector_s1_mandatory_preservation"
        and row["budget_feasible"]
    ]
    assert len(rows) == 9
    assert all(not row["missing_required_artifact_ids"] for row in rows)
    assert all(row["context_budget_insufficient"] is False for row in rows)


def test_infeasible_set_abstains_without_silent_omission(verification):
    result, _, _ = verification
    row = next(
        row for row in result["records"]
        if row["condition"] == "C1_selector_s1_mandatory_preservation"
        and not row["budget_feasible"]
    )
    assert row["missing_required_artifact_ids"]
    assert row["context_budget_insufficient"] is True
    assert row["selection_abstention_reason"]
    assert {"prior_decision_artifact", "source_linked_evidence"} <= set(
        row["omitted_required_artifact_types"]
    )
    assert row["silent_required_artifact_omission"] is False


def test_budget_lineage_labels_and_ineligible_sources_are_safe(verification):
    result, _, _ = verification
    rows = [
        row for row in result["records"]
        if row["condition"] == "C1_selector_s1_mandatory_preservation"
    ]
    assert all(row["token_estimate"] <= row["context_budget"] for row in rows)
    assert all(row["provenance_loss_count"] == 0 for row in rows)
    assert all(row["synthetic_context_label_coverage"] == 1.0 for row in rows)
    assert all(not row["absent_source_violations"] for row in rows)


def test_mutation_bypassing_mandatory_ordering_is_detected(corpus, manifest):
    failures = []
    for case in corpus["cases"]:
        if not case[SCORING_ONLY_FIELD]["budget_feasible"]:
            continue
        legacy = run_condition_c(
            case, manifest["file_sha256"], 7, case["expected_context_budget"]
        )
        scored = _score_case(case, legacy)
        if scored["missing_required_artifact_ids"]:
            failures.append(case["id"])
    assert failures, "ordering mutation would be vacuously undetected"


def test_mutation_removing_parent_source_id_is_detected(corpus, manifest):
    case = corpus["cases"][0]
    record = run_condition_c1(case, manifest["file_sha256"], 7, case["expected_context_budget"])
    target = record["selected_source_ids"][0]
    mutated = copy.deepcopy(record)
    for label in mutated["labels"]:
        if target in label["parent_source_ids"]:
            label["parent_source_ids"].remove(target)
            break
    _, source_loss = _lineage_loss(mutated)
    assert source_loss == 1


def test_mutation_removing_synthetic_label_is_detected(corpus, manifest):
    case = corpus["cases"][0]
    record = run_condition_c1(case, manifest["file_sha256"], 7, case["expected_context_budget"])
    mutated = copy.deepcopy(record)
    mutated["labels"].pop()
    assert _label_coverage(mutated) < 1.0


def test_mutation_suppressing_abstention_is_detected(corpus, manifest):
    case = next(
        item for item in corpus["cases"]
        if not item[SCORING_ONLY_FIELD]["budget_feasible"]
    )
    record = run_condition_c1(case, manifest["file_sha256"], 7, case["expected_context_budget"])
    record["context_budget_insufficient"] = False
    scored = _score_case(case, record)
    assert scored["silent_required_artifact_omission"] is True


def test_mutation_reading_scoring_field_is_detected():
    source = SELECTOR_PATH.read_text(encoding="utf-8")
    mutated = source + "\n_PROBE = case['verification_expectations']\n"
    assert "scoring_only_field_access" in selector_boundary_violations(mutated)


def test_verification_core_has_no_filesystem_write(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("verification core attempted filesystem write")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    result, _, _ = run_verification()
    assert result["all_phase5a_advancement_gates_passed"] is True


def test_owner_pack_is_blinded_stratified_and_not_run(verification):
    _, pack, owner_manifest = verification
    assert len(pack["tasks"]) == 10
    assert pack["review_not_run"] is True
    assert pack["labels"] == [
        "PRODUCT_OWNER_REVIEW", "NOT_INDEPENDENT_HUMAN_STUDY", "NOT_GENERALIZABLE"
    ]
    serialized = json.dumps(pack)
    assert SCORING_ONLY_FIELD not in serialized
    assert "A_full_history" not in serialized
    assert "B_sliding_window" not in serialized
    assert "C1_selector_s1_mandatory_preservation" not in serialized
    assert owner_manifest["restricted"] is True
    expected_hash = hashlib.sha256(
        (json.dumps(pack, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    assert owner_manifest["pack_sha256"] == expected_hash


def test_written_owner_pack_matches_restricted_manifest_hash():
    owner_manifest = json.loads(OWNER_MANIFEST.read_text(encoding="utf-8"))
    assert _sha256(OWNER_PACK) == owner_manifest["pack_sha256"]


def test_optional_model_assisted_lane_was_not_run(verification):
    result, _, _ = verification
    assert result["model_assisted_surrogate_evaluation"] == "NOT_RUN_OPTIONAL_WORKSTREAM"
    assert result["product_owner_review"] == "PACK_PREPARED_NOT_RUN"


def test_consumer_neutral_adapter_contract_is_design_only():
    text = ADAPTER_DESIGN.read_text(encoding="utf-8")
    for required in (
        "current_task", "consumer_session_reference", "eligible_context_scope",
        "requested_budget", "consumer_identity", "authorization_context",
        "context_package", "synthetic_context_labels", "parent_engram_ids",
        "parent_source_ids", "provenance_metadata", "abstention_state",
        "token_estimate", "policy_identifiers", "artifact_id", "artifact_type",
        "lineage_complete", "package_digest", "issued_at", "expires_at",
        "consumer_id", "adapter_contract_version", "disclosure_policy_id",
        "redaction_policy_id", "REQUEST_REPLAY_CONFLICT", "AUTHORIZATION_DENIED",
        "CONTRACT_VERSION_UNSUPPORTED", "Rollback and data retention",
    ):
        assert required in text
    assert "consumer-neutral technical shadow adapter" in text
    assert "isolated local prototype" in text
    assert "cannot guarantee deletion" in text
    assert "Provenance is artifact-local" in text
    assert not Path(
        "prototype/session_context_assembler/consumer_adapter.py"
    ).exists()
    assert Path("prototype/session_context_assembler/shadow_adapter").is_dir()


def test_phase5a_advancement_wording_is_consumer_neutral():
    protocol = Path("docs/session_context_assembler_phase_5a_protocol.md").read_text(
        encoding="utf-8"
    )
    notes = Path("docs/session_context_assembler_phase_5a_notes.md").read_text(
        encoding="utf-8"
    )
    report = Path(
        "benchmarks/results/session_context_assembler_r2_verification.md"
    ).read_text(encoding="utf-8")
    for text in (protocol, notes, report):
        assert "SAM-facing" not in text
        assert "consumer-neutral" in text


def test_adr_0008_is_accepted_only_for_isolated_implementation():
    text = ADR_0008.read_text(encoding="utf-8")
    for required in (
        "Status: Accepted — isolated shadow implementation only",
        "Replay Policy Pinning",
        "Transport Authenticity and Delivery Binding",
        "REPLAY_POLICY_MISMATCH",
        "content-free technical telemetry",
        "Kill Switch and Rollback",
        "live consumer routing",
        "SDK release",
        "retrieval-ranking change",
        "It does not authorize integration or connection",
        "authorized consumer-neutral shadow-evaluation proposal",
    ):
        assert required in text
    assert "Status: Proposed" not in text
