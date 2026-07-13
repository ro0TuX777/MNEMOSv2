"""Phase 4R tests for the isolated governed selector S1."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from prototype.session_context_assembler.corpus import load_validated_corpus
from prototype.session_context_assembler.models import turn_from_dict
from prototype.session_context_assembler.replay import (
    compute_condition_comparison,
    compute_s1_advancement_gates,
    run_replay,
)
from prototype.session_context_assembler.segmenter import segment_turns
from prototype.session_context_assembler.selector_s1 import (
    assemble_context_package_s1,
    select_episodes_s1,
)

R1 = Path("benchmarks/truthsets/session_context_assembler_r1.json")
MANIFEST = Path("benchmarks/truthsets/session_context_assembler_r1.manifest.json")
SELECTOR = Path("prototype/session_context_assembler/selector_s1.py")
R1_FILE_SHA256 = "9dc5682ec08ffad24a9c329ef8b581d3d68c3f83c92e078502f3d37c837e53dc"


@pytest.fixture(scope="module")
def corpus():
    return load_validated_corpus(R1, MANIFEST)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records(corpus, manifest):
    return run_replay(corpus, manifest["file_sha256"], seed=7, include_s1=True)


def test_r1_remains_frozen_and_hash_validated(manifest):
    assert manifest["file_sha256"] == R1_FILE_SHA256
    assert load_validated_corpus(R1, MANIFEST)["version"] == "session-context-assembler-r1"


def test_selector_source_contains_no_scoring_only_field_access():
    tree = ast.parse(SELECTOR.read_text(encoding="utf-8"))
    forbidden = {
        "required_prior_decision_ids",
        "required_source_ids",
        "expected_contradiction_status",
        "episode_hint",
        "known_irrelevant_history_turn_ids",
    }
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert forbidden.isdisjoint(literals)


def test_selector_has_no_runtime_governance_import_or_io_calls():
    tree = ast.parse(SELECTOR.read_text(encoding="utf-8"))
    imports = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
    assert not any(name.startswith(("mnemos", "service", "mnemos_sdk")) for name in imports)
    assert not ({"open", "write_text", "write_bytes", "unlink", "remove", "rename"} & set(calls))


def test_higher_artifact_tier_beats_lexically_closer_context():
    case = {
        "current_task": "Which queue backend handles retry scheduling?",
        "conversation_history": [
            {
                "turn_id": "t1", "speaker": "agent",
                "content": "Decision recorded (DEC-SCA-900): dedicated scheduler service.",
            },
            {
                "turn_id": "t2", "speaker": "user",
                "content": "Which queue backend handles retry scheduling right now?",
            },
        ],
    }
    turns = [turn_from_dict(item) for item in case["conversation_history"]]
    by_id = {turn.turn_id: turn for turn in turns}
    episodes = [
        {"episode_id": "ep0", "turn_ids": ("t1",)},
        {"episode_id": "ep1", "turn_ids": ("t2",)},
    ]
    selected, _, rationale, status = select_episodes_s1(
        episodes, case["current_task"], by_id, token_budget=9, seed=3
    )
    assert selected[0]["turn_ids"] == ("t1",)
    assert status["context_budget_insufficient"] is False
    assert "tier=1" in rationale[0]


def test_ineligible_artifact_is_not_mandatory():
    turns = [
        turn_from_dict({
            "turn_id": "t1", "speaker": "agent", "eligible": False,
            "content": "Blocked decision DEC-SCA-901 and SRC-SCA-blocked.",
        }),
        turn_from_dict({
            "turn_id": "t2", "speaker": "user", "content": "Current safe context."
        }),
    ]
    by_id = {turn.turn_id: turn for turn in turns}
    selected, _, _, status = select_episodes_s1(
        segment_turns(turns), "Current safe context", by_id, token_budget=3, seed=0
    )
    assert selected[0]["turn_ids"] == ("t2",)
    assert status["context_budget_insufficient"] is False


def test_mandatory_budget_shortfall_emits_abstention_contract():
    turns = [
        turn_from_dict({
            "turn_id": "t1", "speaker": "agent",
            "content": "First eligible decision DEC-SCA-910 has several supporting words.",
        }),
        turn_from_dict({
            "turn_id": "t2", "speaker": "agent",
            "content": "Second eligible decision DEC-SCA-911 has several supporting words.",
        }),
    ]
    by_id = {turn.turn_id: turn for turn in turns}
    _, used, _, status = select_episodes_s1(
        segment_turns(turns), "eligible decision", by_id, token_budget=9, seed=0
    )
    assert used <= 9
    assert status["context_budget_insufficient"] is True
    assert "prior_decision_artifact" in status["omitted_required_artifact_types"]
    assert status["selection_abstention_reason"]


def test_semantic_fill_stops_after_mandatory_budget_abstention():
    turns = [
        turn_from_dict({
            "turn_id": "t1", "speaker": "agent",
            "content": "Mandatory decision DEC-SCA-920 has too many supporting words to fit.",
        }),
        turn_from_dict({
            "turn_id": "t2", "speaker": "user", "content": "Relevant short fill."
        }),
    ]
    by_id = {turn.turn_id: turn for turn in turns}
    episodes = [
        {"episode_id": "ep0", "turn_ids": ("t1",)},
        {"episode_id": "ep1", "turn_ids": ("t2",)},
    ]
    selected, _, rationale, status = select_episodes_s1(
        episodes, "Relevant short fill", by_id, token_budget=3, seed=0
    )
    assert selected == []
    assert status["context_budget_insufficient"] is True
    assert any("semantic fill blocked" in item for item in rationale)


def test_s1_replay_is_deterministic_and_has_four_conditions(corpus, manifest, records):
    repeated = run_replay(corpus, manifest["file_sha256"], seed=7, include_s1=True)
    assert records == repeated
    by_case = {}
    for row in records:
        by_case.setdefault(row["case_id"], set()).add(row["condition"])
    expected = {
        "A_full_history", "B_sliding_window", "C_governed_episode_selected",
        "C1_selector_s1_mandatory_preservation",
    }
    assert all(conditions == expected for conditions in by_case.values())


def test_s1_meets_advancement_requirements(records):
    result = compute_s1_advancement_gates(records)
    assert result["all_advancement_requirements_passed"] is True
    assert all(gate["passed"] for gate in result["gates"].values())
    assert result["safety_gates"]["all_required_gates_passed"] is True


def test_s1_stays_within_every_binding_budget(corpus, records):
    budgets = {case["id"]: case["expected_context_budget"] for case in corpus["cases"]}
    c1 = [row for row in records if row["condition"].startswith("C1_")]
    assert len(c1) == 29
    assert all(row["token_estimate"] <= budgets[row["case_id"]] for row in c1)


def test_b_c0_and_c1_receive_identical_binding_budgets(corpus, records):
    budgets = {case["id"]: case["expected_context_budget"] for case in corpus["cases"]}
    bounded = {"B_sliding_window", "C_governed_episode_selected", "C1_selector_s1_mandatory_preservation"}
    for row in records:
        if row["condition"] in bounded:
            assert row["context_budget"] == budgets[row["case_id"]]


def test_adversarial_decisions_and_all_contradictions_are_not_silently_omitted(records):
    adversarial = {"sca_r1_pad_004", "sca_r1_pad_005", "sca_r1_pad_006"}
    c1 = [row for row in records if row["condition"].startswith("C1_")]
    for row in c1:
        if row["case_id"] in adversarial:
            assert row["decision_artifact_retention"] == 1.0
        if row["contradiction_awareness_matches_expected"] is not None:
            assert row["contradiction_awareness_matches_expected"] is True
        if row["context_budget_insufficient"]:
            assert row["omitted_required_artifact_types"]
            assert row["selection_abstention_reason"]


def test_c0_metrics_are_unchanged_by_optional_s1_condition(records):
    comparison = compute_condition_comparison(records)
    c0 = comparison["C_governed_episode_selected"]
    assert c0["mean_decision_artifact_retention"] == pytest.approx(0.1176470588)
    assert c0["mean_required_source_recall"] == pytest.approx(0.1379310345)
    assert c0["mean_prompt_token_reduction"] == pytest.approx(0.6125812812)


def test_s1_package_preserves_lineage_labels(corpus, manifest):
    case = next(item for item in corpus["cases"] if item["id"] == "sca_r1_pad_004")
    package = assemble_context_package_s1(
        case, manifest["file_sha256"], seed=7,
        token_budget=case["expected_context_budget"],
    )
    assert "DEC-SCA-201" in package["selected_parent_engram_ids"]
    assert "SRC-SCA-storage-tier-decision" in package["selected_source_ids"]
    label_decisions = {
        item for label in package["synthetic_context_labels"]
        for item in label["parent_engram_ids"]
    }
    label_sources = {
        item for label in package["synthetic_context_labels"]
        for item in label["parent_source_ids"]
    }
    assert set(package["selected_parent_engram_ids"]) <= label_decisions
    assert set(package["selected_source_ids"]) <= label_sources
