"""Tests for the session-context-assembler Phase 1 offline prototype.

Covers the required test list from the Phase 1 authorization: manifest hash
validation, determinism, episode_hint non-use, synthetic-context label
coverage, parent-Engram/source-ID preservation, no-write behavior, no
governance/retrieval-ranking mutation, and blocked-artifact exclusion.

This prototype makes no Phase 3 baseline claims. These tests verify
plumbing and safety invariants only.
"""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path

import pytest

from prototype.session_context_assembler.assembler import assemble_context_package
from prototype.session_context_assembler.corpus import (
    ManifestMismatchError,
    compute_case_hash,
    load_corpus,
    load_manifest,
    load_validated_corpus,
    validate_manifest,
)
from prototype.session_context_assembler.diagnostics import pairwise_boundary_agreement
from prototype.session_context_assembler.models import turn_from_dict
from prototype.session_context_assembler.segmenter import segment_turns

CORPUS_PATH = Path("benchmarks/truthsets/session_context_assembler_r0.json")
MANIFEST_PATH = Path("benchmarks/truthsets/session_context_assembler_r0.manifest.json")

PROTOTYPE_PACKAGE_DIR = Path("prototype/session_context_assembler")


@pytest.fixture(scope="module")
def corpus_data():
    return load_validated_corpus(CORPUS_PATH, MANIFEST_PATH)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST_PATH)


def _case_by_id(corpus_data, case_id):
    return next(c for c in corpus_data["cases"] if c["id"] == case_id)


# --- manifest_hash_validation -----------------------------------------------


def test_manifest_hash_validation_passes_for_frozen_corpus(corpus_data, manifest):
    # Fixture already calls load_validated_corpus; re-validate explicitly too.
    data, raw = load_corpus(CORPUS_PATH)
    validate_manifest(data, raw, manifest)  # must not raise


def test_manifest_hash_validation_rejects_tampered_corpus(tmp_path, manifest):
    data, raw = load_corpus(CORPUS_PATH)
    tampered = json.loads(raw.decode("utf-8"))
    tampered["cases"][0]["current_task"] = "tampered task text"

    tampered_path = tmp_path / "tampered_corpus.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")

    tampered_data, tampered_raw = load_corpus(tampered_path)
    with pytest.raises(ManifestMismatchError):
        validate_manifest(tampered_data, tampered_raw, manifest)


def test_manifest_hash_validation_rejects_tampered_case_only(manifest):
    data, _raw = load_corpus(CORPUS_PATH)
    case = copy.deepcopy(data["cases"][0])
    case["current_task"] = "mutated"
    assert compute_case_hash(case) != manifest["case_hashes"][case["id"]]


# --- deterministic_output_for_fixed_seed ------------------------------------


def test_deterministic_output_for_fixed_seed(corpus_data, manifest):
    case = _case_by_id(corpus_data, "sca_r0_lrid_001")
    pkg_a = assemble_context_package(case, manifest["file_sha256"], seed=7)
    pkg_b = assemble_context_package(case, manifest["file_sha256"], seed=7)
    assert pkg_a == pkg_b


def test_deterministic_output_across_all_cases(corpus_data, manifest):
    for case in corpus_data["cases"]:
        pkg_a = assemble_context_package(case, manifest["file_sha256"], seed=3)
        pkg_b = assemble_context_package(case, manifest["file_sha256"], seed=3)
        assert pkg_a == pkg_b, f"non-deterministic output for {case['id']}"


# --- episode_hint_not_used_as_input -----------------------------------------


def test_episode_hint_not_used_as_input(corpus_data):
    case = _case_by_id(corpus_data, "sca_r0_tsr_001")
    turns_with_hints = [turn_from_dict(t) for t in case["conversation_history"]]

    stripped = [dict(t, episode_hint=None) for t in case["conversation_history"]]
    turns_without_hints = [turn_from_dict(t) for t in stripped]

    assert segment_turns(turns_with_hints) == segment_turns(turns_without_hints)


def test_episode_hint_not_used_as_input_all_cases(corpus_data):
    for case in corpus_data["cases"]:
        turns_with_hints = [turn_from_dict(t) for t in case["conversation_history"]]
        stripped = [dict(t, episode_hint=None) for t in case["conversation_history"]]
        turns_without_hints = [turn_from_dict(t) for t in stripped]
        assert segment_turns(turns_with_hints) == segment_turns(turns_without_hints), (
            f"segmentation changed when episode_hint was removed for {case['id']}"
        )


def _episode_hint_access_sites(tree: ast.AST):
    """Yield AST nodes that actually read an `episode_hint` field: attribute
    access (`turn.episode_hint`) or a string constant used as a dict key /
    `.get(...)` argument. Docstrings/comments are not code and are never
    visited by ast, so they can't produce false positives here."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "episode_hint":
            yield node
        elif isinstance(node, ast.Constant) and node.value == "episode_hint":
            yield node


def test_diagnostics_module_is_the_only_episode_hint_reader():
    """episode_hint may only be read by diagnostics.py (offline scoring) and
    models.py (where the field itself is defined/parsed from input dicts)."""
    for path in PROTOTYPE_PACKAGE_DIR.glob("*.py"):
        if path.name in {"diagnostics.py", "models.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        sites = list(_episode_hint_access_sites(tree))
        assert not sites, f"{path} must not read episode_hint in code (found {len(sites)} site(s))"


def test_pairwise_boundary_agreement_is_diagnostic_only(corpus_data):
    case = _case_by_id(corpus_data, "sca_r0_tsr_001")
    turns = [turn_from_dict(t) for t in case["conversation_history"]]
    predicted = segment_turns(turns)
    score = pairwise_boundary_agreement(predicted, turns)
    assert 0.0 <= score <= 1.0


# --- synthetic_context_label_coverage ---------------------------------------


def test_synthetic_context_label_coverage(corpus_data, manifest):
    for case in corpus_data["cases"]:
        pkg = assemble_context_package(case, manifest["file_sha256"], seed=1)
        labeled_episode_ids = {label["episode_id"] for label in pkg["synthetic_context_labels"]}
        assert labeled_episode_ids == set(pkg["selected_episode_ids"]), (
            f"label coverage mismatch for {case['id']}"
        )
        for label in pkg["synthetic_context_labels"]:
            assert label["label"] == "synthetic_context"
            assert label["non_authoritative"] is True
            assert label["non_promotable"] is True


# --- parent_engram_id_preservation / source_id_preservation -----------------


def test_parent_engram_id_preservation():
    """Decision IDs embedded in selected turns must appear in output, with
    no fabrication beyond what extraction actually found."""
    case = {
        "id": "inline_test_pad",
        "session_id": "S1",
        "task_id": "T1",
        "current_task": "What did we decide?",
        "conversation_history": [
            {"turn_id": "t1", "speaker": "user", "episode_hint": "e", "content": "Decision recorded (DEC-SCA-101)."},
            {"turn_id": "t2", "speaker": "agent", "episode_hint": "e", "content": "Confirmed (DEC-SCA-101)."},
        ],
    }
    pkg = assemble_context_package(case, "manifest-hash", seed=0, token_budget=10_000)
    assert "DEC-SCA-101" in pkg["selected_parent_engram_ids"]
    all_label_decisions = {
        d for label in pkg["synthetic_context_labels"] for d in label["parent_engram_ids"]
    }
    assert all_label_decisions == set(pkg["selected_parent_engram_ids"])


def test_source_id_preservation():
    case = {
        "id": "inline_test_ssf",
        "session_id": "S2",
        "task_id": "T2",
        "current_task": "What did the audit say?",
        "conversation_history": [
            {"turn_id": "t1", "speaker": "agent", "episode_hint": "e", "content": "Audit (SRC-SCA-audit-2026-04) recommends 15 minutes."},
        ],
    }
    pkg = assemble_context_package(case, "manifest-hash", seed=0, token_budget=10_000)
    assert "SRC-SCA-audit-2026-04" in pkg["selected_source_ids"]
    all_label_sources = {
        s for label in pkg["synthetic_context_labels"] for s in label["parent_source_ids"]
    }
    assert all_label_sources == set(pkg["selected_source_ids"])


def test_no_fabricated_ids_outside_selected_episodes():
    """A decision ID present in an unselected (budget-excluded) episode must
    not appear in output."""
    case = {
        "id": "inline_test_budget",
        "session_id": "S3",
        "task_id": "T3",
        "current_task": "Tiny budget case.",
        "conversation_history": [
            {"turn_id": "t1", "speaker": "user", "episode_hint": "e1", "content": "Completely unrelated filler content about weather and lunch and parking."},
            {"turn_id": "t2", "speaker": "agent", "episode_hint": "e2", "content": "Decision recorded (DEC-SCA-999) about something else entirely with lots of extra padding words to inflate token count well beyond budget."},
        ],
    }
    pkg = assemble_context_package(case, "manifest-hash", seed=0, token_budget=1)
    if "ep1" not in pkg["selected_episode_ids"]:
        assert "DEC-SCA-999" not in pkg["selected_parent_engram_ids"]


# --- no_write_behavior -------------------------------------------------------


def test_no_write_behavior(monkeypatch, corpus_data, manifest):
    def _forbidden(*args, **kwargs):
        raise AssertionError("prototype attempted a filesystem write/delete call")

    monkeypatch.setattr(Path, "write_bytes", _forbidden)
    monkeypatch.setattr(Path, "write_text", _forbidden)
    monkeypatch.setattr(os, "remove", _forbidden)
    monkeypatch.setattr(os, "unlink", _forbidden)
    monkeypatch.setattr(os, "rename", _forbidden)

    # Re-run the full read path plus assembly for every case; must not raise.
    data = load_validated_corpus(CORPUS_PATH, MANIFEST_PATH)
    for case in data["cases"]:
        assemble_context_package(case, manifest["file_sha256"], seed=0)


# --- no_governance_mutation / no_retrieval_ranking_mutation ------------------


_FORBIDDEN_MODULE_PREFIXES = ("mnemos", "service", "mnemos_sdk")


def _forbidden_import_names(tree: ast.AST):
    """Yield actual imported module names that touch mnemos/, service/, or
    mnemos_sdk/. Only real Import/ImportFrom nodes are inspected, so a
    docstring explaining the rule (e.g. "must not import mnemos_sdk") cannot
    trigger a false positive."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_MODULE_PREFIXES:
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _FORBIDDEN_MODULE_PREFIXES:
                yield node.module


def test_no_governance_or_retrieval_imports():
    for path in PROTOTYPE_PACKAGE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden = list(_forbidden_import_names(tree))
        assert not forbidden, f"{path} imports forbidden module(s): {forbidden}"


def test_assembler_output_has_no_governance_or_ranking_fields():
    """The output contract is closed: no key implies trust/authority/ranking
    mutation capability."""
    forbidden_keys = {
        "trust_score",
        "authority_class",
        "promotion_status",
        "ranking_override",
        "governance_state",
        "contradiction_state",
    }
    case = {
        "id": "inline_test_contract",
        "session_id": "S4",
        "task_id": "T4",
        "current_task": "check output keys",
        "conversation_history": [
            {"turn_id": "t1", "speaker": "user", "episode_hint": "e", "content": "hello world"},
        ],
    }
    pkg = assemble_context_package(case, "manifest-hash", seed=0)
    assert forbidden_keys.isdisjoint(pkg.keys())


# --- blocked_artifact_exclusion ----------------------------------------------


def test_blocked_artifact_exclusion():
    case = {
        "id": "inline_test_blocked",
        "session_id": "S5",
        "task_id": "T5",
        "current_task": "What was decided?",
        "conversation_history": [
            {
                "turn_id": "t1",
                "speaker": "user",
                "episode_hint": "e",
                "content": "Blocked claim, ineligible for selection (DEC-SCA-555).",
                "eligible": False,
            },
            {
                "turn_id": "t2",
                "speaker": "agent",
                "episode_hint": "e",
                "content": "Eligible follow-up (DEC-SCA-556).",
                "eligible": True,
            },
        ],
    }
    pkg = assemble_context_package(case, "manifest-hash", seed=0, token_budget=10_000)
    assert "DEC-SCA-555" not in pkg["selected_parent_engram_ids"]
    assert "DEC-SCA-556" in pkg["selected_parent_engram_ids"]
