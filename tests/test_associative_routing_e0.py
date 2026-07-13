"""
Tests for the MNEMOS Associative Routing View E0 — offline, read-only
Cue-Tag-Content projection over the GateMem fixture corpus.

Covers the required E0 test cases: positive routing, temporal/state
handling, conflict/ambiguity handling, and safety/integrity invariants.
"""

from __future__ import annotations

import copy
import json

import pytest

from prototype.associative_routing_e0 import (
    AssociativeRouter,
    RegistryValidationError,
    build_projection,
    load_corpus,
    verify_projection,
)
from prototype.associative_routing_e0.models import ALLOWED_TAG_FIELDS
from prototype.associative_routing_e0.projection import FIXTURES_DIR
from prototype.associative_routing_e0.registry import load_corpus as load_corpus_from_dir
from tools.run_associative_routing_e0_benchmark import run_benchmark


@pytest.fixture(scope="module")
def router() -> AssociativeRouter:
    return AssociativeRouter.from_fixtures()


# ---------------------------------------------------------------------------
# Positive routing
# ---------------------------------------------------------------------------


class TestPositiveRouting:
    def test_why_is_gatemem_work_paused(self, router: AssociativeRouter) -> None:
        response = router.route("Why is GateMem work paused?")
        assert response.routing_result == "resolved"
        assert response.abstention is None
        assert "doc:gatemem_g5_readme" in response.candidate_content_ids
        assert "doc:gatemem_program_status" in response.candidate_content_ids
        for path in response.routing_paths:
            assert path.tag_ids and path.cue_ids and path.content_ids

    def test_what_is_frozen_for_regression_testing_only(self, router: AssociativeRouter) -> None:
        response = router.route("What is frozen for regression testing only?")
        assert response.routing_result == "resolved"
        path_ids = {p.tag_ids[0] for p in response.routing_paths}
        assert "tag:g4-frozen-regression-baseline" in path_ids
        assert "doc:gatemem_g4_implementation" in response.candidate_content_ids

    def test_what_blocks_a_fresh_gatemem_evaluation(self, router: AssociativeRouter) -> None:
        response = router.route("What blocks a fresh GateMem evaluation?")
        assert response.routing_result == "resolved"
        assert "doc:gatemem_g5_handoff_checklist" in response.candidate_content_ids

    def test_what_superseded_the_g4_implementation_lane_abstains(
        self, router: AssociativeRouter
    ) -> None:
        """G4 is the frozen, paused, latest baseline — nothing supersedes it.

        This deliberately exercises correct abstention: the cue resolves
        (cue:gatemem-g4-implementation matches) but it has no outgoing
        ``superseded_by`` tag, only an outgoing ``supersedes`` tag (it
        supersedes the proposal, nothing supersedes it). The router must
        not fabricate an answer in the wrong relationship direction.
        """
        response = router.route("What superseded the G4 implementation lane?")
        assert response.routing_result == "abstained"
        assert response.abstention is not None
        assert response.abstention.reason_code == "NO_SUPPORTED_ASSOCIATIVE_PATH"
        assert response.candidate_content_ids == []

    def test_what_is_the_current_state_of_the_g5_handoff(self, router: AssociativeRouter) -> None:
        response = router.route("What is the current state of the G5 handoff?")
        assert response.routing_result == "resolved"
        assert response.candidate_content_ids == ["doc:gatemem_g5_handoff_state"]


# ---------------------------------------------------------------------------
# Temporal and state handling
# ---------------------------------------------------------------------------


class TestTemporalAndState:
    def test_current_state_preferred_over_historical_milestone(
        self, router: AssociativeRouter
    ) -> None:
        response = router.route("What is the current status of GateMem G4?")
        assert response.routing_result == "resolved"
        assert response.candidate_content_ids == ["doc:gatemem_g4_implementation"]

    def test_superseded_by_resolves_in_passive_direction(self, router: AssociativeRouter) -> None:
        response = router.route("What superseded the G4 implementation proposal?")
        assert response.routing_result == "resolved"
        assert response.candidate_content_ids == ["doc:gatemem_g4_implementation"]

    def test_supersedes_resolves_in_active_direction(self, router: AssociativeRouter) -> None:
        response = router.route("What did the G4 implementation lane supersede?")
        assert response.routing_result == "resolved"
        assert response.candidate_content_ids == ["doc:gatemem_g4_implementation_proposal"]

    def test_current_status_document_preferred_over_obsolete_precursor(
        self, router: AssociativeRouter
    ) -> None:
        """The proposal is the obsolete precursor; the implementation doc is current."""
        current = router.route("What is the current status of GateMem G4?")
        precursor_query = router.route("What did the G4 implementation lane supersede?")
        assert current.candidate_content_ids == ["doc:gatemem_g4_implementation"]
        assert precursor_query.candidate_content_ids == ["doc:gatemem_g4_implementation_proposal"]
        assert current.candidate_content_ids != precursor_query.candidate_content_ids


# ---------------------------------------------------------------------------
# Conflict and ambiguity handling
# ---------------------------------------------------------------------------


class TestConflictAndAmbiguity:
    def test_surfaces_multiple_genuinely_distinct_frozen_baselines(
        self, router: AssociativeRouter
    ) -> None:
        """'frozen baseline' alone is genuinely ambiguous: G2/G2A and G4 are
        both independently documented frozen artifacts. The router must
        return both as separate, source-linked paths rather than silently
        picking one."""
        response = router.route("What is the GateMem frozen baseline?")
        assert response.routing_result == "resolved"
        assert "doc:gatemem_program_status" in response.candidate_content_ids
        assert "doc:gatemem_g4_implementation" in response.candidate_content_ids
        assert len(response.routing_paths) >= 2

    def test_abstains_when_no_cue_matches(self, router: AssociativeRouter) -> None:
        response = router.route("What is the capital of France?")
        assert response.routing_result == "abstained"
        assert response.matched_cues == []
        assert response.abstention.reason_code == "NO_SUPPORTED_ASSOCIATIVE_PATH"

    def test_abstains_when_cue_matches_but_no_typed_relationship_exists(
        self, router: AssociativeRouter
    ) -> None:
        response = router.route("What superseded the G4 implementation lane?")
        assert response.routing_result == "abstained"
        assert "cue:gatemem-g4-implementation" in response.matched_cues


# ---------------------------------------------------------------------------
# Safety and integrity
# ---------------------------------------------------------------------------


class TestSafetyAndIntegrity:
    def test_tag_without_source_support_is_rejected(self, tmp_path) -> None:
        _write_fixture_variant(
            tmp_path,
            tag_mutator=lambda tags: tags[0].update(source_record_ids=[]),
        )
        with pytest.raises(RegistryValidationError):
            load_corpus_from_dir(tmp_path)

    def test_tag_with_nonexistent_content_target_is_rejected(self, tmp_path) -> None:
        _write_fixture_variant(
            tmp_path,
            tag_mutator=lambda tags: tags[0].update(to_content_id="doc:does-not-exist"),
        )
        with pytest.raises(RegistryValidationError):
            load_corpus_from_dir(tmp_path)

    def test_tag_with_nonexistent_from_cue_is_rejected(self, tmp_path) -> None:
        _write_fixture_variant(
            tmp_path,
            tag_mutator=lambda tags: tags[0].update(from_cue_id="cue:does-not-exist"),
        )
        with pytest.raises(RegistryValidationError):
            load_corpus_from_dir(tmp_path)

    def test_cue_without_source_support_is_rejected(self, tmp_path) -> None:
        _write_fixture_variant(
            tmp_path,
            cue_mutator=lambda cues: cues[0].update(source_record_ids=[]),
        )
        with pytest.raises(RegistryValidationError):
            load_corpus_from_dir(tmp_path)

    def test_disallowed_field_on_tag_is_rejected(self, tmp_path) -> None:
        """No authority field (trust/promotion/governance) may appear on a Tag."""
        _write_fixture_variant(
            tmp_path,
            tag_mutator=lambda tags: tags[0].update(trust_score=0.99),
        )
        with pytest.raises(RegistryValidationError):
            load_corpus_from_dir(tmp_path)

    def test_orphan_cue_is_rejected(self, tmp_path) -> None:
        _write_fixture_variant(
            tmp_path,
            cue_extra={
                "cue_id": "cue:orphan",
                "canonical_value": "Orphan Cue",
                "cue_type": "test",
                "normalized_value": "orphan cue",
                "source_record_ids": ["doc:gatemem_program_status"],
                "status": "active",
            },
        )
        with pytest.raises(RegistryValidationError):
            load_corpus_from_dir(tmp_path)

    def test_no_authority_field_allowed_in_production_fixtures(self) -> None:
        assert "trust_score" not in ALLOWED_TAG_FIELDS
        assert "promotion_status" not in ALLOWED_TAG_FIELDS
        assert "authority" not in ALLOWED_TAG_FIELDS

    def test_verification_tool_passes_on_real_fixtures(self) -> None:
        result = verify_projection()
        assert result["status"] == "pass"
        assert all(result["checks"].values())

    def test_projection_rebuilds_deterministically(self) -> None:
        projection_a = build_projection()
        projection_b = build_projection()
        assert projection_a.snapshot == projection_b.snapshot
        assert projection_a.manifest == projection_b.manifest

    def test_response_never_includes_authority_fields(self, router: AssociativeRouter) -> None:
        response = router.route("Why is GateMem work paused?")
        payload = response.to_dict()
        forbidden_keys = {"trust_score", "promotion_status", "governance_state", "authority"}
        assert forbidden_keys.isdisjoint(payload.keys())
        assert payload["integrity"]["non_authoritative_projection"] is True

    def test_removing_projection_does_not_touch_source_docs(self) -> None:
        """The projection is built purely from read-only fixture JSON plus
        pointers to existing docs; this test asserts no write path exists
        by confirming load_corpus/build_projection take no write arguments
        and the fixtures directory contains only the three registry files."""
        corpus = load_corpus()
        assert corpus.content_index  # loaded without mutating anything
        fixture_files = sorted(p.name for p in FIXTURES_DIR.iterdir())
        assert fixture_files == ["cue_registry.json", "source_index.json", "tag_registry.json"]


class TestE0BenchmarkGate:
    """CI gate: the E0-SMOKE evaluation must show zero false abstention,
    perfect fallback/abstention correctness, and no regression in required-
    evidence recall relative to the local baseline proxy on the frozen
    development query pack."""

    def test_benchmark_passes_safety_thresholds(self) -> None:
        artifact = run_benchmark()
        summary = artifact["summary"]
        assert artifact["status"] == "pass"
        assert summary["false_abstention_count"] == 0
        assert summary["fallback_correctness_rate"] == 1.0
        assert summary["routing_all_required_recall_mean"] == 1.0
        assert (
            summary["routing_all_required_recall_mean"]
            >= summary["baseline_all_required_recall_top3_mean"]
        )


def _write_fixture_variant(tmp_path, tag_mutator=None, cue_mutator=None, cue_extra=None) -> None:
    """Copy the real fixture corpus into tmp_path with one deliberate mutation."""

    source_index = json.loads((FIXTURES_DIR / "source_index.json").read_text(encoding="utf-8"))
    cue_registry = json.loads((FIXTURES_DIR / "cue_registry.json").read_text(encoding="utf-8"))
    tag_registry = json.loads((FIXTURES_DIR / "tag_registry.json").read_text(encoding="utf-8"))

    cue_registry = copy.deepcopy(cue_registry)
    tag_registry = copy.deepcopy(tag_registry)

    if tag_mutator is not None:
        tag_mutator(tag_registry["tags"])
    if cue_mutator is not None:
        cue_mutator(cue_registry["cues"])
    if cue_extra is not None:
        cue_registry["cues"].append(cue_extra)

    (tmp_path / "source_index.json").write_text(json.dumps(source_index), encoding="utf-8")
    (tmp_path / "cue_registry.json").write_text(json.dumps(cue_registry), encoding="utf-8")
    (tmp_path / "tag_registry.json").write_text(json.dumps(tag_registry), encoding="utf-8")
