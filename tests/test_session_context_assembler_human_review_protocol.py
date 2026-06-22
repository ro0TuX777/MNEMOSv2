"""Acceptance tests for the Phase 5 human-review design (not study data)."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.build_session_context_assembler_review_packets import (
    DEFAULT_SEED,
    _materialize,
    _verify,
)
from tools.compile_session_context_assembler_review_responses import (
    compile_responses,
)

PROTOCOL = Path("docs/session_context_assembler_phase_5_human_review_protocol.md")
FORM = Path("docs/session_context_assembler_phase_5_review_form.md")
BUILDER = Path("tools/build_session_context_assembler_review_packets.py")
COMPILER = Path("tools/compile_session_context_assembler_review_responses.py")
R1 = Path("benchmarks/truthsets/session_context_assembler_r1.json")
R1_MANIFEST = Path("benchmarks/truthsets/session_context_assembler_r1.manifest.json")
R1_SHA256 = "9dc5682ec08ffad24a9c329ef8b581d3d68c3f83c92e078502f3d37c837e53dc"

FORBIDDEN_REVIEWER_KEYS = {
    "case_id",
    "condition",
    "required_prior_decision_ids",
    "required_source_ids",
    "expected_contradiction_status",
    "episode_hint",
    "known_irrelevant_history_turn_ids",
    "decision_artifact_retention",
    "required_source_recall",
    "contradiction_awareness_matches_expected",
    "selection_rationale",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


@pytest.fixture()
def frozen_packets(tmp_path):
    output = tmp_path / "review"
    manifest = _materialize(output, DEFAULT_SEED, overwrite=False)
    return output, manifest


def test_required_design_deliverables_exist_and_state_review_not_run():
    for path in (PROTOCOL, FORM, BUILDER, COMPILER):
        assert path.is_file(), path
    protocol = PROTOCOL.read_text(encoding="utf-8").lower()
    form = FORM.read_text(encoding="utf-8").lower()
    assert "study not run" in protocol
    assert "does not authorize" in protocol
    assert "ai-generated" in protocol
    assert "no review responses have been collected" in form


def test_r1_and_manifest_are_unchanged():
    manifest = json.loads(R1_MANIFEST.read_text(encoding="utf-8"))
    assert _sha256(R1) == R1_SHA256 == manifest["file_sha256"]


def test_generation_is_byte_reproducible(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _materialize(first, DEFAULT_SEED, overwrite=False)
    _materialize(second, DEFAULT_SEED, overwrite=False)
    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files


def test_manifest_integrity_and_verify_mode(frozen_packets):
    output, manifest = frozen_packets
    assert manifest["frozen"] is True
    assert manifest["design_only_review_not_run"] is True
    assert manifest["packet_count"] == 29
    assert manifest["packages_per_packet"] == 3
    for entry in manifest["packets"]:
        assert _sha256(output / entry["path"]) == entry["sha256"]
    _verify(output, DEFAULT_SEED)


def test_tampered_packet_fails_manifest_verification(frozen_packets):
    output, manifest = frozen_packets
    packet = output / manifest["packets"][0]["path"]
    packet.write_bytes(packet.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify(output, DEFAULT_SEED)


def test_reviewer_packets_are_condition_masked_and_gold_free(frozen_packets):
    output, manifest = frozen_packets
    internal_case_keys = {
        task["internal_case_key"]
        for task in manifest["coordinator_only_condition_key"]
    }
    actual_conditions = {
        condition
        for task in manifest["coordinator_only_condition_key"]
        for condition in task["condition_key"].values()
    }
    for entry in manifest["packets"]:
        packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
        assert not (FORBIDDEN_REVIEWER_KEYS & set(_walk_keys(packet)))
        serialized = json.dumps(packet, sort_keys=True)
        assert not any(case_key in serialized for case_key in internal_case_keys)
        assert not any(condition in serialized for condition in actual_conditions)
        assert {p["condition_code"] for p in packet["packages"]} == {
            "PACKAGE-1", "PACKAGE-2", "PACKAGE-3"
        }


def test_condition_positions_are_balanced(frozen_packets):
    _, manifest = frozen_packets
    positions = {}
    for task in manifest["coordinator_only_condition_key"]:
        for code, condition in task["condition_key"].items():
            positions.setdefault(condition, {}).setdefault(code, 0)
            positions[condition][code] += 1
    for counts in positions.values():
        assert max(counts.values()) - min(counts.values()) <= 1


def test_c1_provenance_and_abstention_are_reviewer_visible(frozen_packets):
    output, manifest = frozen_packets
    task_keys = {
        task["task_code"]: task["condition_key"]
        for task in manifest["coordinator_only_condition_key"]
    }
    warning_count = 0
    for entry in manifest["packets"]:
        packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
        c1_code = next(
            code for code, condition in task_keys[packet["task_code"]].items()
            if condition == "C1_selector_s1_mandatory_preservation"
        )
        c1 = next(p for p in packet["packages"] if p["condition_code"] == c1_code)
        assert c1["artifacts"]
        for artifact in c1["artifacts"]:
            assert artifact["artifact_type"] == "synthetic_context"
            assert artifact["label"] == "synthetic_context"
            assert artifact["synthetic_context"] is True
            assert artifact["non_authoritative"] is True
            assert artifact["non_promotable"] is True
            assert "parent_engram_ids" in artifact
            assert "parent_source_ids" in artifact
        if "warning" in c1:
            warning_count += 1
            assert set(c1["warning"]) == {
                "context_budget_insufficient",
                "omitted_required_artifact_types",
                "selection_abstention_reason",
            }
            assert c1["warning"]["context_budget_insufficient"] is True
            assert "ep0" not in c1["warning"]["selection_abstention_reason"]
            assert "ep1" not in c1["warning"]["selection_abstention_reason"]
    assert warning_count == 5


def test_coordinator_key_is_not_under_reviewer_packet_directory(frozen_packets):
    output, _ = frozen_packets
    assert (output / "coordinator_manifest.json").is_file()
    assert not list((output / "packets").glob("*manifest*"))
    notice = (output / "REVIEWER_DISTRIBUTION_NOTICE.txt").read_text(encoding="utf-8")
    assert "Do not distribute" in notice


def test_builder_and_compiler_are_separate_and_have_no_runtime_imports():
    for path, other_name in (
        (BUILDER, "compile_session_context_assembler_review_responses"),
        (COMPILER, "build_session_context_assembler_review_packets"),
    ):
        text = path.read_text(encoding="utf-8")
        assert other_name not in text
        tree = ast.parse(text)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(
            name.startswith(("mnemos", "service", "mnemos_sdk"))
            for name in imports
        )


def _example_human_response(packet: dict, reviewer="REV-TEST-01") -> dict:
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
                "short_rationale": "Test-only validation fixture, not human-study evidence.",
            }
        )
    return {
        "schema": "sca_phase5_review_response_v1",
        "study_id": packet["study_id"],
        "reviewer_pseudonym": reviewer,
        "task_code": packet["task_code"],
        "package_reviews": reviews,
    }


def test_compiler_validates_pseudonymous_structured_responses(frozen_packets, tmp_path):
    output, manifest = frozen_packets
    entry = manifest["packets"][0]
    packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps(_example_human_response(packet)), encoding="utf-8"
    )
    compiled = compile_responses(output / "coordinator_manifest.json", [response_path])
    assert compiled["human_response_file_count"] == 1
    assert compiled["unique_reviewer_task_submissions"] == 1
    assert sum(
        item["review_count"]
        for item in compiled["aggregates_by_unblinded_condition"].values()
    ) == 3


def test_compiler_rejects_direct_identity_fields(frozen_packets, tmp_path):
    output, manifest = frozen_packets
    entry = manifest["packets"][0]
    packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
    response = _example_human_response(packet)
    response["email"] = "forbidden@example.test"
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="identity fields"):
        compile_responses(output / "coordinator_manifest.json", [response_path])


def test_compiler_rejects_abstention_rating_mismatch(frozen_packets, tmp_path):
    output, manifest = frozen_packets
    warning_entry = None
    warning_packet = None
    for entry in manifest["packets"]:
        packet = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
        if any("warning" in package for package in packet["packages"]):
            warning_entry, warning_packet = entry, packet
            break
    assert warning_packet is not None
    response = _example_human_response(warning_packet)
    warning_code = next(
        package["condition_code"] for package in warning_packet["packages"]
        if "warning" in package
    )
    next(
        review for review in response["package_reviews"]
        if review["condition_code"] == warning_code
    )["abstention_clear_and_cautious"] = "not_applicable"
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="abstention rating is required"):
        compile_responses(output / "coordinator_manifest.json", [response_path])
