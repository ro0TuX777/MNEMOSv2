"""Tests for the session-context-assembler Phase 3 offline A/B/C replay
harness (prototype/session_context_assembler/replay.py and
tools/run_session_context_assembler_replay.py).

Mirrors the Phase 1 safety-invariant test style: structural guarantees
(no governance/runtime imports, no filesystem writes/deletes) are checked
via AST and patched-call techniques, not by trusting prose comments.

No test here asserts a quality, non-inferiority, or production-readiness
claim. See docs/session_context_assembler_phase_3_notes.md.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

from prototype.session_context_assembler.corpus import load_validated_corpus
from prototype.session_context_assembler.replay import (
    CONTRADICTION_GROUND_TRUTH,
    _contradiction_awareness_result,
    compute_aggregate_gates,
    compute_condition_comparison,
    run_condition_a,
    run_condition_b,
    run_replay,
)

CORPUS_PATH = Path("benchmarks/truthsets/session_context_assembler_r0.json")
MANIFEST_PATH = Path("benchmarks/truthsets/session_context_assembler_r0.manifest.json")

PROTOTYPE_PACKAGE_DIR = Path("prototype/session_context_assembler")
TOOLS_SCRIPT_PATH = Path("tools/run_session_context_assembler_replay.py")

_FORBIDDEN_MODULE_PREFIXES = ("mnemos", "service", "mnemos_sdk")


@pytest.fixture(scope="module")
def corpus_data():
    return load_validated_corpus(CORPUS_PATH, MANIFEST_PATH)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _case_by_id(corpus_data, case_id):
    return next(c for c in corpus_data["cases"] if c["id"] == case_id)


# --- required harness properties --------------------------------------------


def test_same_corpus_manifest_hash_used_for_all_conditions(corpus_data, manifest):
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    hashes = {r["corpus_manifest_hash"] for r in records}
    assert hashes == {manifest["file_sha256"]}


def test_same_case_hash_across_conditions_for_a_given_case(corpus_data, manifest):
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    by_case: dict = {}
    for r in records:
        by_case.setdefault(r["case_id"], set()).add(r["case_hash"])
    for case_id, hashes in by_case.items():
        assert len(hashes) == 1, f"{case_id} has inconsistent case_hash across conditions"


def test_all_three_conditions_present_for_every_case(corpus_data, manifest):
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    by_case: dict = {}
    for r in records:
        by_case.setdefault(r["case_id"], set()).add(r["condition"])
    expected = {"A_full_history", "B_sliding_window", "C_governed_episode_selected"}
    for case_id, conditions in by_case.items():
        assert conditions == expected, f"{case_id} missing a condition: {conditions}"


def test_required_per_case_output_fields_present(corpus_data, manifest):
    required_fields = {
        "session_id",
        "task_id",
        "condition",
        "prototype_version",
        "seed",
        "corpus_manifest_hash",
        "case_hash",
        "selected_turn_ids",
        "selected_episode_ids",
        "selected_parent_engram_ids",
        "selected_source_ids",
        "token_estimate",
        "required_prior_decision_recall",
        "required_source_recall",
        "contradiction_awareness_result",
        "known_irrelevant_history_selected",
        "provenance_loss_count",
        "synthetic_context_label_coverage",
    }
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    for r in records:
        missing = required_fields - r.keys()
        assert not missing, f"record for {r['case_id']}/{r['condition']} missing fields: {missing}"


def test_condition_a_includes_every_turn_unbounded(corpus_data):
    case = _case_by_id(corpus_data, "sca_r0_lrid_001")
    a = run_condition_a(case)
    assert set(a["selected_turn_ids"]) == {t["turn_id"] for t in case["conversation_history"]}


# --- deterministic seeds where selection is stochastic ----------------------


def test_deterministic_replay_for_fixed_seed(corpus_data, manifest):
    records_a = run_replay(corpus_data, manifest["file_sha256"], seed=5)
    records_b = run_replay(corpus_data, manifest["file_sha256"], seed=5)
    assert records_a == records_b


def test_condition_a_and_b_are_seed_independent(corpus_data):
    """A and B have no stochastic component; only C's tie-break uses the
    seed. Confirms A/B output is identical regardless of seed value."""
    case = _case_by_id(corpus_data, "sca_r0_mspd_001")
    a1, a2 = run_condition_a(case), run_condition_a(case)
    assert a1 == a2
    b1 = run_condition_b(case, token_budget=case["expected_context_budget"])
    b2 = run_condition_b(case, token_budget=case["expected_context_budget"])
    assert b1 == b2


# --- manifest and per-case hash validation before execution -----------------


def test_replay_uses_hash_validated_corpus_loader(corpus_data, manifest):
    # load_validated_corpus (used by the fixture and the CLI script) raises
    # on any tamper; reaching this point means validation already occurred.
    assert corpus_data["cases"], "expected a non-empty validated corpus"
    assert manifest["case_count"] == len(corpus_data["cases"])


# --- metric definitions -------------------------------------------------------


def test_required_prior_decision_recall_is_none_when_nothing_required(corpus_data, manifest):
    case = _case_by_id(corpus_data, "sca_r0_caf_001")  # required_prior_decision_ids == []
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    case_records = [r for r in records if r["case_id"] == case["id"]]
    for r in case_records:
        assert r["required_prior_decision_recall"] is None


def test_required_prior_decision_recall_full_when_decision_inline_and_unbudgeted():
    case = {
        "id": "inline_recall_test",
        "session_id": "S1",
        "task_id": "T1",
        "case_family": "synthetic",
        "current_task": "what was decided?",
        "conversation_history": [
            {"turn_id": "t1", "speaker": "user", "episode_hint": "e", "content": "Decision made (DEC-SCA-900)."},
        ],
        "required_prior_decision_ids": ["DEC-SCA-900"],
        "required_source_ids": ["SRC-SCA-x"],
        "known_irrelevant_history_turn_ids": [],
        "expected_context_budget": 9999,
    }
    a = run_condition_a(case)
    decisions = set(a["selected_parent_engram_ids"])
    assert decisions == {"DEC-SCA-900"}


def test_known_irrelevant_history_selected_counts_overlap():
    case = {
        "id": "irrelevant_count_test",
        "session_id": "S2",
        "task_id": "T2",
        "case_family": "synthetic",
        "current_task": "anything",
        "conversation_history": [
            {"turn_id": "t1", "speaker": "user", "episode_hint": "e", "content": "irrelevant filler one"},
            {"turn_id": "t2", "speaker": "user", "episode_hint": "e", "content": "irrelevant filler two"},
            {"turn_id": "t3", "speaker": "user", "episode_hint": "e", "content": "relevant content here"},
        ],
        "required_prior_decision_ids": [],
        "required_source_ids": ["SRC-SCA-x"],
        "known_irrelevant_history_turn_ids": ["t1", "t2"],
        "expected_context_budget": 9999,
    }
    a = run_condition_a(case)
    selected = set(a["selected_turn_ids"])
    known_irrelevant = set(case["known_irrelevant_history_turn_ids"])
    assert len(selected & known_irrelevant) == 2  # full history includes everything


def test_prompt_token_reduction_is_zero_for_condition_a(corpus_data, manifest):
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    for r in records:
        if r["condition"] == "A_full_history":
            assert r["prompt_token_reduction"] == 0.0


def test_sliding_window_never_exceeds_full_history_token_estimate(corpus_data, manifest):
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    full_by_case = {r["case_id"]: r["token_estimate"] for r in records if r["condition"] == "A_full_history"}
    for r in records:
        if r["condition"] == "B_sliding_window":
            assert r["token_estimate"] <= full_by_case[r["case_id"]]


# --- contradiction awareness classification ----------------------------------


def test_contradiction_awareness_not_applicable_outside_ground_truth_table(corpus_data, manifest):
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    for r in records:
        if r["case_id"] not in CONTRADICTION_GROUND_TRUTH:
            assert r["contradiction_awareness_result"] == "not_applicable"


def test_contradiction_awareness_full_history_recovers_caf_categories(corpus_data, manifest):
    """Condition A includes every turn, and every contradiction_aware_followup
    case has its required source IDs inline in turn text, so full history
    must recover the case's authored category (resolved/unresolved)."""
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    expected = {
        "sca_r0_caf_001": "resolved",
        "sca_r0_caf_002": "resolved",
        "sca_r0_caf_003": "unresolved",
    }
    for r in records:
        if r["condition"] == "A_full_history" and r["case_id"] in expected:
            assert r["contradiction_awareness_result"] == expected[r["case_id"]]
            assert r["contradiction_awareness_matches_expected"] is True


def test_contradiction_awareness_omitted_when_required_ids_not_selected():
    result, matches = _contradiction_awareness_result("sca_r0_caf_001", set(), set())
    assert result == "omitted"
    assert matches is False


def test_contradiction_awareness_partial_recovery_is_still_omitted():
    """Recovering only one of two required poles of a contradiction must
    not be reported as the full ground-truth category - a partial view
    cannot faithfully represent that a contradiction existed at all."""
    result, matches = _contradiction_awareness_result(
        "sca_r0_caf_001", set(), {"SRC-SCA-policy-b17"}
    )
    assert result == "omitted"
    assert matches is False


# --- gates: source_id_preservation_rate / parent_engram_lineage_preservation_rate /
#     provenance_loss_count / synthetic_context_label_coverage --------------


def test_required_phase_3_gates_pass_on_frozen_r0(corpus_data, manifest):
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    gate_summary = compute_aggregate_gates(records)
    assert gate_summary["all_required_gates_passed"] is True
    gates = gate_summary["gates"]
    assert gates["source_id_preservation_rate"]["value"] == 1.0
    assert gates["parent_engram_lineage_preservation_rate"]["value"] == 1.0
    assert gates["provenance_loss_count"]["value"] == 0
    assert gates["synthetic_context_label_coverage"]["value"] == 1.0
    assert gates["unauthorized_memory_write_count"]["value"] == 0
    assert gates["unauthorized_governance_mutation_count"]["value"] == 0


def test_synthetic_context_label_coverage_is_zero_for_naive_baselines(corpus_data, manifest):
    """A and B never emit synthetic_context labels - only the governed
    assembler does. This is the structural point of building it."""
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    for r in records:
        if r["condition"] in ("A_full_history", "B_sliding_window"):
            assert r["synthetic_context_label_coverage"] == 0.0
            assert r["synthetic_context_labels"] == []


def test_gates_fail_loudly_if_a_label_were_dropped(corpus_data, manifest):
    """Sanity-check the gate math itself: corrupt one C record's labels and
    confirm the gate then reports a failure, proving the gate isn't
    vacuously true."""
    records = run_replay(corpus_data, manifest["file_sha256"], seed=0)
    for r in records:
        if r["condition"] == "C_governed_episode_selected" and r["selected_source_ids"]:
            r["source_lineage_loss_count"] = len(r["selected_source_ids"])
            r["provenance_loss_count"] += r["source_lineage_loss_count"]
            break
    else:
        pytest.fail("expected at least one condition C record with a selected source id")

    gate_summary = compute_aggregate_gates(records)
    assert gate_summary["gates"]["source_id_preservation_rate"]["passed"] is False
    assert gate_summary["all_required_gates_passed"] is False


# --- no write / no governance mutation (extends Phase 1 invariants to replay) -


def test_no_write_behavior_during_full_replay(monkeypatch, corpus_data, manifest):
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("replay harness attempted a filesystem write/delete call")

    monkeypatch.setattr(Path, "write_bytes", _forbidden)
    monkeypatch.setattr(Path, "write_text", _forbidden)
    monkeypatch.setattr(os, "remove", _forbidden)
    monkeypatch.setattr(os, "unlink", _forbidden)
    monkeypatch.setattr(os, "rename", _forbidden)

    run_replay(corpus_data, manifest["file_sha256"], seed=0)
    run_replay(corpus_data, manifest["file_sha256"], seed=0, token_budget_override=20)
    compute_aggregate_gates(run_replay(corpus_data, manifest["file_sha256"], seed=0))
    compute_condition_comparison(run_replay(corpus_data, manifest["file_sha256"], seed=0))


def _forbidden_import_names(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_MODULE_PREFIXES:
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _FORBIDDEN_MODULE_PREFIXES:
                yield node.module


def test_replay_module_has_no_governance_or_retrieval_imports():
    tree = ast.parse((PROTOTYPE_PACKAGE_DIR / "replay.py").read_text(encoding="utf-8"))
    forbidden = list(_forbidden_import_names(tree))
    assert not forbidden, f"replay.py imports forbidden module(s): {forbidden}"


def test_tools_script_has_no_governance_or_retrieval_imports():
    tree = ast.parse(TOOLS_SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden = list(_forbidden_import_names(tree))
    assert not forbidden, f"{TOOLS_SCRIPT_PATH} imports forbidden module(s): {forbidden}"


def test_tools_script_only_writes_inside_benchmarks_results(tmp_path, monkeypatch):
    """The CLI script must only ever write the two declared report paths,
    never anything else. Patch Path.write_text globally and assert every
    call's `self` path resolves to one of the two redirected report paths."""
    calls = []
    real_write_text = Path.write_text

    def _tracking_write_text(self, *args, **kwargs):
        calls.append(self)
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _tracking_write_text)

    import tools.run_session_context_assembler_replay as runner

    monkeypatch.setattr(runner, "RESULTS_JSON_PATH", tmp_path / "result.json")
    monkeypatch.setattr(runner, "RESULTS_MD_PATH", tmp_path / "result.md")
    monkeypatch.setattr(sys, "argv", ["prog", "--seed", "0"])

    runner.main()

    assert calls, "expected at least one write_text call"
    for path in calls:
        assert path in (tmp_path / "result.json", tmp_path / "result.md")
