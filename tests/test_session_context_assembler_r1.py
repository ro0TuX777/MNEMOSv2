"""Phase 2R corpus and measurement-revision checks."""

from __future__ import annotations

import json
from pathlib import Path

from prototype.session_context_assembler.corpus import load_validated_corpus
from prototype.session_context_assembler.extractor import extract_ids_from_turn
from prototype.session_context_assembler.models import turn_from_dict
from prototype.session_context_assembler.replay import run_replay
from prototype.session_context_assembler.segmenter import segment_turns

R0 = Path("benchmarks/truthsets/session_context_assembler_r0.json")
R1 = Path("benchmarks/truthsets/session_context_assembler_r1.json")
R1_MANIFEST = Path("benchmarks/truthsets/session_context_assembler_r1.manifest.json")


def _load_r1():
    return load_validated_corpus(R1, R1_MANIFEST)


def test_r1_manifest_validates_and_r0_file_remains_separate():
    r1 = _load_r1()
    assert len(r1["cases"]) == 29
    assert R0.read_bytes() != R1.read_bytes()


def test_inherited_r0_semantics_are_preserved_additively():
    r0 = json.loads(R0.read_text(encoding="utf-8"))
    r1 = _load_r1()
    r1_by_id = {case["id"]: case for case in r1["cases"]}
    stable_fields = (
        "case_family", "session_id", "task_id", "current_task",
        "required_prior_decision_ids", "required_source_ids",
        "known_irrelevant_history_turn_ids", "notes",
    )
    for old in r0["cases"]:
        revised = r1_by_id[old["id"]]
        for field in stable_fields:
            assert revised[field] == old[field]
        assert [t["content"] for t in revised["conversation_history"]] == [
            t["content"] for t in old["conversation_history"]
        ]


def test_every_r1_budget_is_formula_derived_and_binding():
    for case in _load_r1()["cases"]:
        full_tokens = sum(len(t["content"].split()) for t in case["conversation_history"])
        turns = [turn_from_dict(t) for t in case["conversation_history"]]
        turns_by_id = {turn.turn_id: turn for turn in turns}
        largest_episode = max(
            sum(len(turns_by_id[tid].content.split()) for tid in episode["turn_ids"])
            for episode in segment_turns(turns)
        )
        assert case["expected_context_budget"] == max(
            round(0.5 * full_tokens), 15, largest_episode
        )
        assert case["expected_context_budget"] < full_tokens


def test_b_and_c_never_exceed_their_shared_r1_budget():
    r1 = _load_r1()
    manifest = json.loads(R1_MANIFEST.read_text(encoding="utf-8"))
    records = run_replay(r1, manifest["file_sha256"], seed=7)
    budgets = {case["id"]: case["expected_context_budget"] for case in r1["cases"]}
    for row in records:
        if row["condition"] != "A_full_history":
            assert row["token_estimate"] <= budgets[row["case_id"]]


def test_structured_source_links_are_extracted_and_eligibility_gated():
    linked = {
        "turn_id": "t1", "speaker": "agent", "content": "No inline identifier.",
        "linked_source_ids": ["SRC-SCA-structured-only"],
    }
    assert extract_ids_from_turn(turn_from_dict(linked))[1] == {"SRC-SCA-structured-only"}
    linked["eligible"] = False
    assert extract_ids_from_turn(turn_from_dict(linked)) == (frozenset(), frozenset())


def test_full_history_has_no_r1_source_measurement_ceiling():
    r1 = _load_r1()
    manifest = json.loads(R1_MANIFEST.read_text(encoding="utf-8"))
    records = run_replay(r1, manifest["file_sha256"], seed=7)
    scored = [
        row for row in records
        if row["condition"] == "A_full_history" and row["required_source_recall"] is not None
    ]
    assert scored
    assert all(row["required_source_recall"] == 1.0 for row in scored)


def test_explicit_contradiction_status_is_measurable_with_full_history():
    r1 = _load_r1()
    manifest = json.loads(R1_MANIFEST.read_text(encoding="utf-8"))
    records = run_replay(r1, manifest["file_sha256"], seed=7)
    expected = {
        case["id"]: case["expected_contradiction_status"]
        for case in r1["cases"] if "expected_contradiction_status" in case
    }
    assert set(expected) == {
        "sca_r0_caf_001", "sca_r0_caf_002", "sca_r0_caf_003",
        "sca_r0_urd_001", "sca_r0_urd_002", "sca_r0_urd_003",
        "sca_r1_caf_004", "sca_r1_urd_004",
    }
    for row in records:
        if row["condition"] == "A_full_history" and row["case_id"] in expected:
            assert row["contradiction_awareness_result"] == expected[row["case_id"]]
            assert row["contradiction_awareness_matches_expected"] is True


def test_decision_artifact_retention_is_explicit_and_new_stress_classes_exist():
    r1 = _load_r1()
    manifest = json.loads(R1_MANIFEST.read_text(encoding="utf-8"))
    records = run_replay(r1, manifest["file_sha256"], seed=7)
    for row in records:
        assert row["decision_artifact_retention"] == row["required_prior_decision_recall"]
    difficulties = {
        case.get("decision_retention_difficulty") for case in r1["cases"]
    }
    assert {
        "old_decisive_low_lexical_similarity",
        "recent_irrelevant_distractor",
        "semantically_similar_distractor",
    } <= difficulties
