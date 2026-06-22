"""ADR 0008 isolated shadow-adapter acceptance and mutation tests."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import prototype.session_context_assembler.shadow_adapter.adapter as adapter_module

from prototype.session_context_assembler.shadow_adapter import (
    KillSwitch,
    LocalAssemblyInputs,
    LocalShadowAdapter,
    LocalTransportContext,
    PolicySnapshot,
)
from prototype.session_context_assembler.shadow_adapter.canonical import (
    response_digest,
    verify_response_digest,
)
from prototype.session_context_assembler.shadow_adapter.content_free_shadow_sink import (
    ALLOWED_EVENT_FIELDS,
    ContentFreeShadowSink,
)
from prototype.session_context_assembler.shadow_adapter.errors import ShadowAdapterError
from prototype.session_context_assembler.shadow_adapter.policy_and_disclosure_boundary import (
    evaluate_policy,
)
from prototype.session_context_assembler.shadow_adapter.response_builder_and_digest import (
    validate_response_contract,
)
from tools.run_session_context_assembler_shadow_adapter_gate import run_gate

PACKAGE_DIR = Path("prototype/session_context_assembler/shadow_adapter")
R1 = Path("benchmarks/truthsets/session_context_assembler_r1.json")
R2 = Path("benchmarks/truthsets/session_context_assembler_r2.json")
NOW = datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def r2_cases():
    return json.loads(R2.read_text(encoding="utf-8"))["cases"]


def _case(r2_cases, suffix="old_decisive_001"):
    return next(case for case in r2_cases if case["id"].endswith(suffix))


def _request(case, *, request_id="REQ-001", budget=None):
    return {
        "request_id": request_id,
        "current_task": case["current_task"],
        "consumer_session_reference": case["session_id"],
        "eligible_context_scope": {
            "tenant_scope": "TENANT-TEST",
            "session_scope": case["session_id"],
            "allowed_artifact_classes": ["session_turn"],
            "eligibility_policy_id": "ELIG-1",
        },
        "requested_budget": {
            "token_limit": budget or case["expected_context_budget"],
            "budget_policy_id": "BUD-1",
        },
        "consumer_identity": {
            "consumer_id": "CONSUMER-TEST",
            "adapter_id": "ADAPTER-LOCAL",
            "purpose": "technical_shadow",
        },
        "authorization_context": {
            "authorization_reference": "AUTH-1",
            "permitted_operation": "read_context_package",
            "expiry": (NOW + timedelta(hours=1)).isoformat(),
        },
        "adapter_contract_version": "1.0.0",
    }


def _inputs(case):
    return LocalAssemblyInputs(
        session_id=case["session_id"],
        task_id=case["task_id"],
        conversation_history=tuple(case["conversation_history"]),
        snapshot_reference="R2-FROZEN-SNAPSHOT",
    )


def _policy(case):
    return PolicySnapshot(
        consumer_id="CONSUMER-TEST",
        adapter_id="ADAPTER-LOCAL",
        authorization_reference="AUTH-1",
        authorization_grant_fingerprint="GRANT-FP-1",
        permitted_purpose="technical_shadow",
        tenant_scope="TENANT-TEST",
        session_scope=case["session_id"],
        allowed_artifact_classes=frozenset({"session_turn"}),
        allowed_source_ids=frozenset({"*"}),
        allowed_engram_ids=frozenset({"*"}),
        denied_turn_ids=frozenset(),
        redacted_content_by_turn_id={},
        eligibility_policy_id="ELIG-1",
        disclosure_policy_id="DISC-1",
        redaction_policy_id="RED-1",
        budget_policy_id="BUD-1",
        max_token_budget=case["expected_context_budget"],
        assembler_policy_version="S1-1.0",
        adapter_contract_version="1.0.0",
        authorization_expires_at=NOW + timedelta(hours=1),
        package_ttl_seconds=300,
    )


def _transport(**changes):
    values = {
        "peer_consumer_id": "CONSUMER-TEST",
        "channel_id": "LOCAL-AUTH-CHANNEL-1",
        "authenticated": True,
        "confidentiality_protected": True,
        "integrity_protected": True,
    }
    values.update(changes)
    return LocalTransportContext(**values)


def _run(case, adapter=None, request=None, policy=None, transport=None, now=NOW):
    adapter = adapter or LocalShadowAdapter()
    result = adapter.process(
        request or _request(case),
        _inputs(case),
        policy or _policy(case),
        transport or _transport(),
        now=now,
    )
    return adapter, result


def _redigest(response):
    response["package_digest"]["value"] = response_digest(response)


def test_00_kill_switch_blocks_assembly_cache_sink_and_delivery(r2_cases):
    case = _case(r2_cases)
    active_path_sentinel = {"retrieval_calls": 0, "memory_state": "unchanged"}
    adapter = LocalShadowAdapter(kill_switch=KillSwitch(active=True, reason="test"))
    before = copy.deepcopy(active_path_sentinel)
    _, result = _run(case, adapter=adapter)
    assert result["ok"] is False
    assert result["error"]["error_code"] == "KILL_SWITCH_ACTIVE"
    assert adapter.assembler.invocation_count == 0
    assert adapter.replay.write_count == adapter.replay.size == 0
    assert adapter.sink.events == []
    assert adapter.delivery_attempt_count == 0
    assert active_path_sentinel == before


def test_kill_switch_state_cannot_be_cleared_by_direct_assignment():
    switch = KillSwitch(active=True, reason="locked")
    with pytest.raises(AttributeError):
        switch.active = False
    assert switch.active is True


def test_happy_path_is_shadow_only_and_artifact_local(r2_cases):
    case = _case(r2_cases)
    adapter, result = _run(case)
    assert result["ok"] is result["shadow_only"] is True
    response = result["response"]
    assert verify_response_digest(response)
    artifacts = response["context_package"]["selected_session_artifacts"]
    assert artifacts
    for artifact in artifacts:
        assert artifact["artifact_id"]
        assert artifact["synthetic_context"] is True
        assert artifact["non_authoritative"] is True
        assert artifact["non_promotable"] is True
        assert artifact["lineage_complete"] is True
        assert "parent_engram_ids" in artifact
        assert "parent_source_ids" in artifact
    assert adapter.assembler.invocation_count == 1
    assert adapter.replay.write_count == 1
    assert adapter.delivery_attempt_count == 1


def test_digest_is_reproducible_and_tamper_evident(r2_cases):
    case = _case(r2_cases)
    _, first = _run(case)
    _, second = _run(case)
    assert first["response"] == second["response"]
    mutated = copy.deepcopy(first["response"])
    mutated["token_estimate"] += 1
    assert verify_response_digest(mutated) is False
    with pytest.raises(ShadowAdapterError, match="integrity"):
        validate_response_contract(mutated)


def test_exact_replay_is_byte_equivalent_and_does_not_reassemble(r2_cases):
    case = _case(r2_cases)
    adapter, first = _run(case)
    _, replay = _run(case, adapter=adapter, now=NOW + timedelta(seconds=1))
    assert replay["ok"] is True and replay["replayed"] is True
    assert replay["response"] == first["response"]
    assert adapter.assembler.invocation_count == 1
    assert adapter.replay.write_count == 1
    assert adapter.sink.events[-1]["replayed"] is True


def test_request_id_reuse_with_changed_request_fails_closed(r2_cases):
    case = _case(r2_cases)
    adapter, _ = _run(case)
    changed = _request(case)
    changed["current_task"] = "Different canonical task"
    _, result = _run(case, adapter=adapter, request=changed, now=NOW + timedelta(seconds=1))
    assert result["error"]["error_code"] == "REQUEST_REPLAY_CONFLICT"
    assert adapter.assembler.invocation_count == 1


@pytest.mark.parametrize(
    "policy_change",
    [
        {"authorization_grant_fingerprint": "GRANT-FP-2"},
        {"disclosure_policy_id": "DISC-2"},
        {"redaction_policy_id": "RED-2"},
        {"assembler_policy_version": "S1-1.1"},
        {"allowed_source_ids": frozenset({"SRC-SCA-r2-archive-decision"})},
    ],
)
def test_replay_policy_drift_fails_closed(r2_cases, policy_change):
    case = _case(r2_cases)
    adapter, _ = _run(case)
    changed_policy = replace(_policy(case), **policy_change)
    _, result = _run(
        case, adapter=adapter, policy=changed_policy, now=NOW + timedelta(seconds=1)
    )
    assert result["error"]["error_code"] == "REPLAY_POLICY_MISMATCH"
    assert adapter.assembler.invocation_count == 1


def test_contract_minor_version_drift_is_policy_mismatch(r2_cases):
    case = _case(r2_cases)
    adapter, _ = _run(case)
    request = _request(case)
    request["adapter_contract_version"] = "1.1.0"
    policy = replace(_policy(case), adapter_contract_version="1.1.0")
    _, result = _run(
        case, adapter=adapter, request=request, policy=policy,
        now=NOW + timedelta(seconds=1),
    )
    assert result["error"]["error_code"] == "REQUEST_REPLAY_CONFLICT"


def test_eligibility_policy_version_drift_fails_closed(r2_cases):
    case = _case(r2_cases)
    adapter, _ = _run(case)
    changed = replace(_policy(case), eligibility_policy_id="ELIG-2")
    _, result = _run(case, adapter=adapter, policy=changed, now=NOW + timedelta(seconds=1))
    assert result["error"]["error_code"] == "POLICY_VERSION_INCOMPATIBLE"


def test_redaction_payload_drift_changes_replay_fingerprint(r2_cases):
    case = _case(r2_cases)
    base_policy = replace(
        _policy(case), redacted_content_by_turn_id={"t3": "[REDACTED-V1]"}
    )
    adapter, _ = _run(case, policy=base_policy)
    changed = replace(
        base_policy, redacted_content_by_turn_id={"t3": "[REDACTED-V2]"}
    )
    _, result = _run(
        case, adapter=adapter, policy=changed, now=NOW + timedelta(seconds=1)
    )
    assert result["error"]["error_code"] == "REPLAY_POLICY_MISMATCH"


def test_snapshot_or_eligible_content_drift_changes_replay_fingerprint(r2_cases):
    case = _case(r2_cases)
    adapter, _ = _run(case)
    changed_inputs = replace(_inputs(case), snapshot_reference="R2-NEW-SNAPSHOT")
    result = adapter.process(
        _request(case), changed_inputs, _policy(case), _transport(),
        now=NOW + timedelta(seconds=1),
    )
    assert result["error"]["error_code"] == "REPLAY_POLICY_MISMATCH"


def test_expired_cached_package_is_not_reassembled(r2_cases):
    case = _case(r2_cases)
    adapter, _ = _run(case)
    _, result = _run(case, adapter=adapter, now=NOW + timedelta(seconds=301))
    assert result["error"]["error_code"] == "PACKAGE_EXPIRED"
    assert adapter.assembler.invocation_count == 1


@pytest.mark.parametrize(
    "transport",
    [
        _transport(authenticated=False),
        _transport(confidentiality_protected=False),
        _transport(integrity_protected=False),
        _transport(peer_consumer_id="OTHER"),
        _transport(channel_id=""),
    ],
)
def test_transport_authenticity_is_required(r2_cases, transport):
    case = _case(r2_cases)
    adapter, result = _run(case, transport=transport)
    assert result["error"]["error_code"] == "AUTHORIZATION_DENIED"
    assert adapter.assembler.invocation_count == adapter.replay.write_count == 0


def test_transport_delivery_binding_is_content_free(r2_cases):
    case = _case(r2_cases)
    adapter, result = _run(case)
    event = adapter.sink.events[0]
    assert event["channel_id"] == "LOCAL-AUTH-CHANNEL-1"
    assert event["consumer_id"] == "CONSUMER-TEST"
    assert event["request_id"] == result["response"]["request_id"]
    assert event["package_id"] == result["response"]["package_id"]
    assert event["package_digest"] == result["response"]["package_digest"]["value"]
    assert event["event_time"] == NOW.isoformat()


def test_authorization_scope_and_operation_fail_closed(r2_cases):
    case = _case(r2_cases)
    request = _request(case)
    request["authorization_context"]["permitted_operation"] = "write_engram"
    adapter, result = _run(case, request=request)
    assert result["error"]["error_code"] == "AUTHORIZATION_DENIED"
    request = _request(case, request_id="REQ-002")
    request["eligible_context_scope"]["tenant_scope"] = "OTHER-TENANT"
    _, result = _run(case, request=request)
    assert result["error"]["error_code"] == "AUTHORIZATION_DENIED"


def test_expired_authorization_and_session_reference_fail_closed(r2_cases):
    case = _case(r2_cases)
    expired = _request(case)
    expired["authorization_context"]["expiry"] = (NOW - timedelta(seconds=1)).isoformat()
    _, result = _run(case, request=expired)
    assert result["error"]["error_code"] == "AUTHORIZATION_DENIED"
    mismatch = _request(case, request_id="REQ-REF-MISMATCH")
    mismatch["consumer_session_reference"] = "OTHER-SESSION"
    _, result = _run(case, request=mismatch)
    assert result["error"]["error_code"] == "AUTHORIZATION_DENIED"


def test_contract_budget_and_artifact_policy_versions_fail_closed(r2_cases):
    case = _case(r2_cases)
    contract = _request(case)
    contract["adapter_contract_version"] = "2.0.0"
    _, result = _run(case, request=contract)
    assert result["error"]["error_code"] == "CONTRACT_VERSION_UNSUPPORTED"
    budget = _request(case, request_id="REQ-BUDGET-POLICY")
    budget["requested_budget"]["budget_policy_id"] = "OTHER"
    _, result = _run(case, request=budget)
    assert result["error"]["error_code"] == "POLICY_VERSION_INCOMPATIBLE"
    artifact = _request(case, request_id="REQ-ARTIFACT-POLICY")
    artifact["eligible_context_scope"]["allowed_artifact_classes"] = ["secret"]
    _, result = _run(case, request=artifact)
    assert result["error"]["error_code"] == "DISCLOSURE_DENIED"


def test_source_and_engram_disclosure_are_deny_by_default(r2_cases):
    case = _case(r2_cases)
    source_denied = replace(_policy(case), allowed_source_ids=frozenset())
    adapter, result = _run(case, policy=source_denied)
    assert result["error"]["error_code"] == "DISCLOSURE_DENIED"
    assert adapter.assembler.invocation_count == 0
    engram_denied = replace(_policy(case), allowed_engram_ids=frozenset())
    _, result = _run(case, policy=engram_denied)
    assert result["error"]["error_code"] == "DISCLOSURE_DENIED"


def test_denied_and_ineligible_turns_are_removed_before_assembly(r2_cases):
    case = _case(r2_cases, "ineligible_source_007")
    adapter, result = _run(case)
    assert result["ok"] is True
    serialized = json.dumps(result["response"])
    assert "withdrawn-md5" not in serialized
    assert "Withdrawn draft" not in serialized


def test_redaction_is_applied_before_policy_fingerprint_and_assembly(r2_cases):
    case = _case(r2_cases)
    policy = replace(
        _policy(case), redacted_content_by_turn_id={"t2": "[REDACTED POLICY CONTENT]"}
    )
    effective = evaluate_policy(_request(case), _inputs(case), policy, NOW)
    turn = next(item for item in effective.filtered_history if item["turn_id"] == "t2")
    assert turn["content"] == "[REDACTED POLICY CONTENT]"
    assert "Approved immutable object storage" not in json.dumps(effective.filtered_history)


def test_budget_compliance_and_explicit_overflow_abstention(r2_cases):
    case = _case(r2_cases, "overflow_006")
    _, result = _run(case)
    response = result["response"]
    assert response["token_estimate"] <= case["expected_context_budget"]
    assert response["abstention_state"]["context_budget_insufficient"] is True
    assert response["abstention_state"]["omitted_required_artifact_types"]
    assert "MANDATORY_CANDIDATE_OMITTED_BUDGET" in response["context_package"][
        "selection_metadata"
    ]["selection_rationale_codes"]


def test_parent_source_mutation_is_detected_even_after_redigest(r2_cases):
    case = _case(r2_cases)
    _, result = _run(case)
    mutated = copy.deepcopy(result["response"])
    artifact = next(
        item for item in mutated["context_package"]["selected_session_artifacts"]
        if item["parent_source_ids"]
    )
    artifact["parent_source_ids"].pop()
    _redigest(mutated)
    with pytest.raises(ShadowAdapterError, match="Source lineage"):
        validate_response_contract(mutated)


def test_synthetic_label_mutation_is_detected_after_redigest(r2_cases):
    case = _case(r2_cases)
    _, result = _run(case)
    mutated = copy.deepcopy(result["response"])
    mutated["synthetic_context_labels"].pop()
    _redigest(mutated)
    with pytest.raises(ShadowAdapterError, match="Synthetic labels"):
        validate_response_contract(mutated)


def test_abstention_suppression_mutation_is_detected_after_redigest(r2_cases):
    case = _case(r2_cases, "overflow_006")
    _, result = _run(case)
    mutated = copy.deepcopy(result["response"])
    mutated["abstention_state"] = {
        "context_budget_insufficient": False,
        "omitted_required_artifact_types": [],
        "selection_abstention_reason": None,
    }
    _redigest(mutated)
    with pytest.raises(ShadowAdapterError, match="Abstention state"):
        validate_response_contract(mutated)


def test_telemetry_contains_no_task_or_context_content(r2_cases):
    case = _case(r2_cases)
    adapter, _ = _run(case)
    serialized = json.dumps(adapter.sink.events)
    assert set(adapter.sink.events[0]) == ALLOWED_EVENT_FIELDS
    for forbidden in (
        case["current_task"], "DEC-SCA-301", "SRC-SCA-r2-archive-decision",
        "immutable object storage", "authorization_reference",
    ):
        assert forbidden not in serialized


def test_telemetry_allowlist_mutation_is_rejected():
    event = {field: None for field in ALLOWED_EVENT_FIELDS}
    event["raw_context"] = "forbidden"
    with pytest.raises(ShadowAdapterError, match="Telemetry"):
        ContentFreeShadowSink.validate_event(event)


def test_structured_errors_do_not_leak_internal_identifiers(r2_cases):
    case = _case(r2_cases)
    policy = replace(_policy(case), allowed_source_ids=frozenset())
    adapter, result = _run(case, policy=policy)
    error = result["error"]
    assert set(error) == {
        "request_id", "adapter_contract_version", "error_code", "retryable",
        "safe_retry_after",
    }
    serialized = json.dumps(error)
    assert "SRC-SCA" not in serialized and "DEC-SCA" not in serialized
    assert "DISC-1" not in serialized and "AUTH-1" not in serialized
    assert adapter.sink.events[0]["outcome_code"] == "DISCLOSURE_DENIED"


def test_kill_switch_activation_invalidates_cache_and_blocks_reenableless_use(r2_cases):
    case = _case(r2_cases)
    adapter, _ = _run(case)
    assert adapter.replay.size == 1
    counts = (
        adapter.assembler.invocation_count,
        adapter.replay.write_count,
        len(adapter.sink.events),
        adapter.delivery_attempt_count,
    )
    adapter.activate_kill_switch("rollback")
    assert adapter.replay.size == 0
    _, result = _run(case, adapter=adapter, now=NOW + timedelta(seconds=1))
    assert result["error"]["error_code"] == "KILL_SWITCH_ACTIVE"
    assert counts == (
        adapter.assembler.invocation_count,
        adapter.replay.write_count,
        len(adapter.sink.events),
        adapter.delivery_attempt_count,
    )


def test_no_filesystem_writes_during_processing(r2_cases, monkeypatch):
    case = _case(r2_cases)

    def fail(*args, **kwargs):
        raise AssertionError("shadow adapter attempted filesystem write")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    _, result = _run(case)
    assert result["ok"] is True


def test_package_has_no_network_runtime_or_mnemos_authority_imports():
    forbidden_imports = ("socket", "http", "requests", "fastapi", "service", "mnemos")
    forbidden_calls = {
        "index", "upsert", "promote", "write_text", "write_bytes", "delete", "remove"
    }
    for path in PACKAGE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            assert not any(name.startswith(forbidden_imports) for name in names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls


def test_no_listener_sdk_route_or_consumer_connection_artifact_exists():
    assert not Path("service/shadow_adapter.py").exists()
    assert not Path("mnemos/shadow_adapter.py").exists()
    assert not Path("mnemos_sdk/shadow_adapter.py").exists()
    all_text = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE_DIR.glob("*.py"))
    for forbidden in ("listen(", "FastAPI(", "APIRouter(", "requests.", "httpx."):
        assert forbidden not in all_text


def test_core_corpus_hashes_unchanged_by_adapter_run(r2_cases):
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (R1, R2)}
    _run(_case(r2_cases))
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (R1, R2)}
    assert before == after


def test_mutation_bypassing_kill_switch_is_observable(r2_cases, monkeypatch):
    case = _case(r2_cases)
    switch = KillSwitch(active=True, reason="mutation")
    adapter = LocalShadowAdapter(kill_switch=switch)
    monkeypatch.setattr(switch, "begin", lambda: 0)
    monkeypatch.setattr(switch, "require_enabled", lambda *args: None)
    monkeypatch.setattr(switch, "guard", lambda *args: nullcontext())
    _, result = _run(case, adapter=adapter)
    assert result["ok"] is True
    assert adapter.assembler.invocation_count > 0
    assert adapter.replay.write_count > 0
    assert adapter.sink.events


def test_mid_assembly_kill_activation_discards_package_without_side_effects(
    r2_cases, monkeypatch
):
    case = _case(r2_cases)
    adapter = LocalShadowAdapter()
    original = adapter.assembler.assemble

    def activating_assemble(*args, **kwargs):
        package = original(*args, **kwargs)
        adapter.kill_switch.activate("mid-assembly mutation")
        return package

    monkeypatch.setattr(adapter.assembler, "assemble", activating_assemble)
    _, result = _run(case, adapter=adapter)
    assert result["error"]["error_code"] == "KILL_SWITCH_ACTIVE"
    assert adapter.assembler.invocation_count == 1
    assert adapter.replay.write_count == adapter.replay.size == 0
    assert adapter.sink.events == []
    assert adapter.delivery_attempt_count == 0


def test_mutation_bypassing_policy_pin_is_observable(r2_cases, monkeypatch):
    case = _case(r2_cases)
    adapter, first = _run(case)
    monkeypatch.setattr(
        adapter.replay,
        "lookup",
        lambda request, request_digest, policy_fingerprint, now: copy.deepcopy(
            first["response"]
        ),
    )
    changed = replace(_policy(case), disclosure_policy_id="DISC-MUTATED")
    _, result = _run(
        case, adapter=adapter, policy=changed, now=NOW + timedelta(seconds=1)
    )
    assert result["ok"] is True and result["replayed"] is True
    assert result["response"]["provenance_metadata"]["disclosure_policy_id"] == "DISC-1"


def test_mutation_bypassing_transport_authorization_is_observable(r2_cases, monkeypatch):
    case = _case(r2_cases)
    monkeypatch.setattr(adapter_module, "validate_request", lambda *args, **kwargs: None)
    adapter, result = _run(case, transport=_transport(authenticated=False))
    assert result["ok"] is True
    assert adapter.assembler.invocation_count == 1


def test_mutation_bypassing_redaction_is_observable(r2_cases, monkeypatch):
    case = _case(r2_cases)
    policy = replace(
        _policy(case), redacted_content_by_turn_id={"t2": "[REDACTED POLICY CONTENT]"}
    )
    original_evaluate = adapter_module.evaluate_policy

    def bypass_redaction(request, inputs, snapshot, now):
        effective = original_evaluate(request, inputs, snapshot, now)
        return replace(effective, filtered_history=tuple(inputs.conversation_history))

    monkeypatch.setattr(adapter_module, "evaluate_policy", bypass_redaction)
    _, result = _run(case, policy=policy)
    serialized = json.dumps(result["response"])
    assert "Approved immutable object storage" in serialized
    assert "[REDACTED POLICY CONTENT]" not in serialized


def test_isolated_adapter_gate_passes_and_mutations_are_non_vacuous():
    result = run_gate()
    assert result["all_gates_passed"] is True
    assert result["case_count"] == 10
    assert all(gate["passed"] for gate in result["gates"].values())
    assert all(check["passed"] for check in result["mutation_checks"].values())


def test_committed_gate_result_matches_current_implementation():
    committed = json.loads(
        Path(
            "benchmarks/results/session_context_assembler_shadow_adapter_gate.json"
        ).read_text(encoding="utf-8")
    )
    assert committed == run_gate()
