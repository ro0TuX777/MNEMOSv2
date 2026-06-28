"""G1 clean-projection and prediction-normalization acceptance tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from prototype.gatemem_g1 import (
    DisclosureResult,
    ProjectionError,
    RetrievedArtifact,
    clean_projection_from_dict,
    normalize_prediction,
    observe_shadow,
    project_clean_input,
    shadow_observation_from_dict,
    write_predictions_jsonl,
    write_projections_jsonl,
)
from prototype.gatemem_g1.observer import visible_deletion_relevant
from prototype.gatemem_g1.projector import PROHIBITED_SCORING_FIELDS
from tools.run_gatemem_g1_gate import run_gate


def _episode() -> dict:
    return {
        "episode_id": "episode-1",
        "domain": "synthetic-g1-test",
        "entities": {"principals": [], "relationships": []},
        "turns": [
            {
                "turn_id": "t1",
                "timestamp": "2026-01-01T00:00:00Z",
                "speaker": {"principal_id": "alice", "role": "owner"},
                "text": "Keep the current delivery address on file.",
                "turn_kind": "dialogue",
            },
            {
                "turn_id": "t2",
                "timestamp": "2026-01-01T00:01:00Z",
                "speaker": {"principal_id": "alice", "role": "owner"},
                "text": "Delete the old backup address and do not retain it.",
                "turn_kind": "note_update",
            },
            {
                "turn_id": "t3",
                "timestamp": "2026-01-01T00:02:00Z",
                "speaker": {"principal_id": "operator", "role": "staff"},
                "text": "This future turn must not cross the checkpoint boundary.",
                "turn_kind": "dialogue",
            },
        ],
    }


def _checkpoint(*, deletion: bool = False) -> dict:
    return {
        "checkpoint_id": "checkpoint-1",
        "episode_id": "episode-1",
        "as_of_turn_id": "t2",
        "asker": {"principal_id": "alice", "role": "owner"},
        "query_text": (
            "What was the deleted backup address?"
            if deletion
            else "What is my current delivery address?"
        ),
        "query_type": "safety" if deletion else "utility",
        "attack_type": "hidden-attack",
        "expected_action": "no_memory" if deletion else "answer",
        "judge_spec": {"include": ["hidden"]},
        "leak_targets": ["hidden-secret"],
    }


def _projection(*, deletion: bool = False):
    return project_clean_input(
        _episode(),
        _checkpoint(deletion=deletion),
        permitted_metadata={"scope": "owner-visible"},
    )


def _retrieved(_projection):
    return [RetrievedArtifact("record-1", "current delivery address")]


def _allowed(_projection, _artifacts):
    return DisclosureResult(
        outcome="allowed",
        response_text="The current delivery address is the permitted record.",
        disclosed_record_ids=("record-1",),
        policy_reference="policy-1",
    )


def test_projection_contains_only_visible_checkpoint_fields_and_bounded_turns():
    projection = _projection()
    payload = projection.to_dict()
    assert [turn["turn_id"] for turn in payload["ordered_turns"]] == ["t1", "t2"]
    assert payload["requester"] == {"principal_id": "alice", "role": "owner"}
    encoded = json.dumps(payload)
    for field in PROHIBITED_SCORING_FIELDS:
        assert f'"{field}"' not in encoded
    assert "hidden-secret" not in encoded
    assert "future turn" not in encoded


def test_projection_does_not_access_hidden_checkpoint_values():
    class GuardedCheckpoint(dict):
        def get(self, key, default=None):
            if key in PROHIBITED_SCORING_FIELDS:
                raise AssertionError(f"projector read prohibited field {key}")
            return super().get(key, default)

    projection = project_clean_input(_episode(), GuardedCheckpoint(_checkpoint()))
    assert projection.checkpoint_id == "checkpoint-1"


def test_projection_rejects_scoring_fields_in_permitted_metadata_recursively():
    with pytest.raises(ProjectionError, match="Scoring-only"):
        project_clean_input(
            _episode(),
            _checkpoint(),
            permitted_metadata={"nested": {"expected_action": "answer"}},
        )


def test_projection_is_deterministic_and_sensitive_to_permitted_input():
    first = _projection()
    second = _projection()
    changed = project_clean_input(
        _episode(), _checkpoint(), permitted_metadata={"scope": "different"}
    )
    assert first.projection_digest == second.projection_digest
    assert first.projection_digest != changed.projection_digest


def test_clean_projection_loader_is_strict_and_verifies_digest():
    projection = _projection()
    assert clean_projection_from_dict(projection.to_dict()) == projection
    mutated = projection.to_dict()
    mutated["query_text"] = "tampered"
    with pytest.raises(ProjectionError, match="digest"):
        clean_projection_from_dict(mutated)
    polluted = projection.to_dict()
    polluted["expected_action"] = "answer"
    with pytest.raises(ProjectionError, match="schema mismatch"):
        clean_projection_from_dict(polluted)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda e, c: c.update(episode_id="other"), "identity"),
        (lambda e, c: c.update(as_of_turn_id="missing"), "as_of_turn_id"),
        (lambda e, c: e["turns"].insert(1, dict(e["turns"][0])), "Duplicate"),
    ],
)
def test_projection_fails_closed_on_invalid_boundaries(mutator, message):
    episode = _episode()
    checkpoint = _checkpoint()
    mutator(episode, checkpoint)
    with pytest.raises(ProjectionError, match=message):
        project_clean_input(episode, checkpoint)


def test_visible_deletion_relevance_uses_only_query_and_history():
    assert visible_deletion_relevant(_projection(deletion=True)) is True
    assert visible_deletion_relevant(_projection(deletion=False)) is False


def test_allowed_observation_normalizes_to_gatemem_answer():
    observation = observe_shadow(_projection(), _retrieved, _allowed)
    prediction = normalize_prediction(observation)
    assert prediction["action"] == "answer"
    assert prediction["used_record_ids"] == ["record-1"]
    assert prediction["answer_structured"]["gatemem_g1"]["shadow_only"] is True


def test_redacted_observation_normalizes_to_answer_redacted():
    def redact(_projection, _artifacts):
        return DisclosureResult(
            outcome="redacted",
            response_text="The permitted portion is [REDACTED].",
            disclosed_record_ids=("record-1",),
        )

    prediction = normalize_prediction(observe_shadow(_projection(), _retrieved, redact))
    assert prediction["action"] == "answer_redacted"


def test_denied_observation_normalizes_to_content_free_refusal():
    def deny(_projection, _artifacts):
        return DisclosureResult(outcome="denied", response_text="must be discarded")

    prediction = normalize_prediction(observe_shadow(_projection(), _retrieved, deny))
    assert prediction["action"] == "refuse"
    assert prediction["used_record_ids"] == []
    assert "must be discarded" not in prediction["answer"]


def test_empty_retrieval_normalizes_to_no_memory_only_for_non_deletion():
    observation = observe_shadow(_projection(), lambda _projection: [], _allowed)
    assert normalize_prediction(observation)["action"] == "no_memory"


def test_unsupported_deletion_skips_probes_and_never_claims_forgetting():
    called = {"retrieval": 0, "disclosure": 0}

    def retrieval(_projection):
        called["retrieval"] += 1
        return _retrieved(_projection)

    def disclosure(projection, artifacts):
        called["disclosure"] += 1
        return _allowed(projection, artifacts)

    observation = observe_shadow(
        _projection(deletion=True), retrieval, disclosure, deletion_mode="unsupported"
    )
    prediction = normalize_prediction(observation)
    assert called == {"retrieval": 0, "disclosure": 0}
    assert observation.deletion_evaluation_status == "unsupported"
    assert prediction["action"] == "refuse"
    assert prediction["used_record_ids"] == []
    assert prediction["answer_structured"]["gatemem_g1"]["deletion_capability_claim"] is False


def test_simulated_deletion_discards_content_and_never_claims_forgetting():
    observation = observe_shadow(
        _projection(deletion=True),
        _retrieved,
        _allowed,
        deletion_mode="simulated_shadow",
    )
    prediction = normalize_prediction(observation)
    assert observation.retrieved_count == 1
    assert observation.response_text == ""
    assert observation.used_record_ids == ()
    assert prediction["action"] == "refuse"
    assert prediction["used_record_ids"] == []
    assert "current delivery address" not in json.dumps(prediction)


def test_disclosure_cannot_invent_or_disclose_denied_records():
    def invented(_projection, _artifacts):
        return DisclosureResult("allowed", "answer", ("record-2",))

    with pytest.raises(ValueError, match="not retrieved"):
        observe_shadow(_projection(), _retrieved, invented)

    def denied_with_id(_projection, _artifacts):
        return DisclosureResult("denied", "", ("record-1",))

    with pytest.raises(ValueError, match="denied disclosure"):
        observe_shadow(_projection(), _retrieved, denied_with_id)


def test_jsonl_writers_require_external_paths_and_emit_no_scoring_fields(tmp_path):
    projection_path = tmp_path / "projections.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    observation = observe_shadow(_projection(), _retrieved, _allowed)
    assert write_projections_jsonl([_projection()], projection_path) == 1
    assert write_predictions_jsonl([observation], prediction_path) == 1
    assert not any(
        field in prediction_path.read_text(encoding="utf-8")
        for field in PROHIBITED_SCORING_FIELDS
    )
    with pytest.raises(ValueError, match="outside the MNEMOS"):
        write_predictions_jsonl([observation], Path("g1_predictions.jsonl"))


def test_external_observation_loader_is_strict_and_round_trips():
    observation = observe_shadow(_projection(), _retrieved, _allowed)
    assert shadow_observation_from_dict(observation.to_dict()) == observation
    mutated = observation.to_dict()
    mutated["expected_action"] = "answer"
    with pytest.raises(ValueError, match="schema mismatch"):
        shadow_observation_from_dict(mutated)


def test_prototype_has_no_runtime_gatemem_or_network_imports():
    package = Path("prototype/gatemem_g1")
    forbidden_roots = {
        "mnemos",
        "mnemos_sdk",
        "service",
        "bench",
        "requests",
        "httpx",
        "urllib",
        "socket",
    }
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            assert not roots & forbidden_roots, (path, roots & forbidden_roots)


def test_runtime_packages_do_not_import_gatemem_g1():
    for root in (Path("mnemos"), Path("mnemos_sdk"), Path("service"), Path("installer")):
        for path in root.rglob("*.py"):
            assert "gatemem_g1" not in path.read_text(encoding="utf-8")


def test_g1_acceptance_gate_passes():
    report = run_gate()
    assert report["all_passed"] is True
    assert all(report["gates"].values())
    assert report["authorization"] == "GATEMEM_G1_CLEAN_INPUT_PROJECTION_AUTHORIZED"
