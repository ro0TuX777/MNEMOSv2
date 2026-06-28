"""Adversarial instrumentation stress tests for the Phase 5 packet/compiler tooling.

DEBUG-ONLY. These tests probe leakage resistance, condition-key separation, and
malformed-pseudonym rejection in the packet builder and response compiler. They
are not a human-review study, do not generate reviewer responses, and never
write to the real distributed packet directory
(benchmarks/review_packets/session_context_assembler_phase_5/). Every packet
build below targets pytest's tmp_path. Nothing produced here is compiled,
cited, or stored as human-value evidence; see
docs/session_context_assembler_phase_5_human_review_protocol.md:109-110.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_session_context_assembler_review_packets import (
    CONDITIONS,
    DEFAULT_SEED,
    _materialize,
)
from tools.compile_session_context_assembler_review_responses import (
    compile_responses,
    validate_response,
)

REAL_DISTRIBUTION_DIR = Path(
    "benchmarks/review_packets/session_context_assembler_phase_5"
)

# Superset of tests/test_session_context_assembler_human_review_protocol.py's
# FORBIDDEN_REVIEWER_KEYS, plus near-miss spellings/casings/separators an
# implementer might accidentally introduce.
FORBIDDEN_SUBSTRINGS = {
    "case_id",
    "caseid",
    "case-id",
    "condition_key",
    "conditionkey",
    "required_prior_decision_ids",
    "required_source_ids",
    "expected_contradiction_status",
    "episode_hint",
    "episode_hints",
    "known_irrelevant_history_turn_ids",
    "decision_artifact_retention",
    "required_source_recall",
    "contradiction_awareness_matches_expected",
    "selection_rationale",
    "coordinator_only_condition_key",
    "internal_case_key",
}


@pytest.fixture()
def stress_packets(tmp_path):
    output = tmp_path / "stress_review_output"
    manifest = _materialize(output, DEFAULT_SEED, overwrite=False)
    return output, manifest


def test_stress_output_is_never_the_real_distribution_directory(stress_packets):
    output, _ = stress_packets
    assert output.resolve() != REAL_DISTRIBUTION_DIR.resolve()
    assert REAL_DISTRIBUTION_DIR.resolve() not in output.resolve().parents


def test_packet_bytes_resist_forbidden_substring_leakage(stress_packets):
    output, manifest = stress_packets
    for entry in manifest["packets"]:
        text = (output / entry["path"]).read_text(encoding="utf-8").lower()
        hits = {s for s in FORBIDDEN_SUBSTRINGS if s.replace("-", "_") in text or s in text}
        assert not hits, f"{entry['path']} leaked: {hits}"


def test_packet_bytes_resist_condition_name_leakage_case_insensitive(stress_packets):
    output, manifest = stress_packets
    lowered_conditions = [c.lower() for c in CONDITIONS]
    for entry in manifest["packets"]:
        text = (output / entry["path"]).read_text(encoding="utf-8").lower()
        for condition in lowered_conditions:
            assert condition not in text, f"{entry['path']} leaked condition {condition}"


def test_condition_key_exists_only_in_coordinator_manifest(stress_packets):
    output, manifest = stress_packets
    manifest_text = (output / "coordinator_manifest.json").read_text(encoding="utf-8")
    for condition in CONDITIONS:
        assert condition in manifest_text
    packet_files = sorted((output / "packets").glob("*.json"))
    assert len(packet_files) == manifest["packet_count"]
    for path in packet_files:
        assert "condition_key" not in path.read_text(encoding="utf-8")


def test_condition_key_removal_does_not_affect_packet_readability(stress_packets):
    output, manifest = stress_packets
    (output / "coordinator_manifest.json").unlink()
    for entry in manifest["packets"]:
        packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
        assert {p["condition_code"] for p in packet["packages"]} == {
            "PACKAGE-1", "PACKAGE-2", "PACKAGE-3"
        }


ADVERSARIAL_PSEUDONYMS = [
    "",
    "rev-lower-01",
    "REV_UNDERSCORE",
    "REV",
    "REV-",
    "REV-A",
    "X-NOT-REV",
    "REV-" + "A" * 40,
    "REV-DROP TABLE",
    "REV-<script>alert(1)</script>",
    "REV-../../etc/passwd",
    "REV-РЕВ-01",  # Cyrillic homoglyphs of REV
    "REV-NAME:bob@example.com",
    " REV-PADDED-01",
    "REV-PADDED-01 ",
    "REV-PADDED\n01",
    "rev-bob",
    "Rev-Bob-01",
]

WELL_FORMED_PSEUDONYMS = ["REV-A12", "REV-BOB-01", "REV-Z9-Z9-Z9"]


def _base_response(packet: dict, pseudonym: str) -> dict:
    reviews = []
    for package in packet["packages"]:
        has_warning = "warning" in package
        reviews.append(
            {
                "condition_code": package["condition_code"],
                "prior_decision_preserved": "uncertain",
                "source_references_understandable": 3,
                "source_references_sufficient": 3,
                "contradiction_representation": "not_applicable",
                "necessary_material_omitted": "uncertain",
                "synthetic_context_distinguishable": 3,
                "abstention_clear_and_cautious": 3 if has_warning else "not_applicable",
                "task_effect": "no_different",
                "short_rationale": "DEBUG_STRESS_TEST_NOT_HUMAN_DATA fixture only.",
            }
        )
    return {
        "schema": "sca_phase5_review_response_v1",
        "study_id": packet["study_id"],
        "reviewer_pseudonym": pseudonym,
        "task_code": packet["task_code"],
        "package_reviews": reviews,
    }


@pytest.mark.parametrize("pseudonym", ADVERSARIAL_PSEUDONYMS)
def test_compiler_rejects_malformed_pseudonyms(stress_packets, pseudonym):
    output, manifest = stress_packets
    entry = manifest["packets"][0]
    packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
    response = _base_response(packet, pseudonym)
    with pytest.raises(ValueError, match="reviewer_pseudonym"):
        validate_response(response, packet)


@pytest.mark.parametrize("pseudonym", WELL_FORMED_PSEUDONYMS)
def test_compiler_accepts_well_formed_pseudonyms(stress_packets, pseudonym):
    output, manifest = stress_packets
    entry = manifest["packets"][0]
    packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
    response = _base_response(packet, pseudonym)
    validate_response(response, packet)  # must not raise


def test_pseudonym_regex_permits_trailing_hyphen_runs(stress_packets):
    """Documented finding, not a defect: PSEUDONYM allows a trailing run of
    hyphens (e.g. "REV-ID--") because the character class after the first
    suffix character includes "-" with no constraint on position. Flagging
    here so a future protocol revision can decide whether to tighten it;
    this stress suite does not change gating behavior on its own."""
    output, manifest = stress_packets
    entry = manifest["packets"][0]
    packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
    response = _base_response(packet, "REV-ID--")
    validate_response(response, packet)  # currently accepted; see docstring


def test_compile_responses_never_invoked_without_real_human_files(stress_packets, tmp_path):
    """Guards against silently treating this fixture data as study evidence."""
    output, _ = stress_packets
    empty_dir = tmp_path / "empty_responses"
    empty_dir.mkdir()
    compiled = compile_responses(output / "coordinator_manifest.json", [])
    assert compiled["human_response_file_count"] == 0
    assert compiled["unique_reviewer_task_submissions"] == 0
    assert "DEBUG_STRESS_TEST" not in json.dumps(compiled)


def test_duplicate_reviewer_task_submission_is_rejected(stress_packets, tmp_path):
    output, manifest = stress_packets
    entry = manifest["packets"][0]
    packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
    response = _base_response(packet, "REV-DUP-01")
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(response), encoding="utf-8")
    path_b.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate reviewer/task submission"):
        compile_responses(output / "coordinator_manifest.json", [path_a, path_b])


def test_mismatched_package_codes_are_rejected(stress_packets):
    output, manifest = stress_packets
    entry = manifest["packets"][0]
    packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
    response = _base_response(packet, "REV-CODE-01")
    response["package_reviews"][0]["condition_code"] = "PACKAGE-99"
    with pytest.raises(ValueError, match="do not match the masked packet"):
        validate_response(response, packet)
