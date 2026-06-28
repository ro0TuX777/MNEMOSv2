"""Run ADR 0008 isolated shadow-adapter acceptance gates.

Writes benchmark evidence only. It creates no listener, consumer connection,
SDK, runtime route, durable-memory write, retrieval change, or governance path.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prototype.session_context_assembler.shadow_adapter import (  # noqa: E402
    KillSwitch,
    LocalAssemblyInputs,
    LocalShadowAdapter,
    LocalTransportContext,
    PolicySnapshot,
)
from prototype.session_context_assembler.shadow_adapter.canonical import (  # noqa: E402
    response_digest,
    verify_response_digest,
)
from prototype.session_context_assembler.shadow_adapter.content_free_shadow_sink import (  # noqa: E402
    ALLOWED_EVENT_FIELDS,
    ContentFreeShadowSink,
)
from prototype.session_context_assembler.shadow_adapter.errors import (  # noqa: E402
    ShadowAdapterError,
)
from prototype.session_context_assembler.shadow_adapter.response_builder_and_digest import (  # noqa: E402
    validate_response_contract,
)

R1 = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r1.json"
R1_MANIFEST = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r1.manifest.json"
R2 = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r2.json"
R2_MANIFEST = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r2.manifest.json"
PACKAGE_DIR = REPO_ROOT / "prototype" / "session_context_assembler" / "shadow_adapter"
RESULT_JSON = REPO_ROOT / "benchmarks" / "results" / "session_context_assembler_shadow_adapter_gate.json"
RESULT_MD = REPO_ROOT / "benchmarks" / "results" / "session_context_assembler_shadow_adapter_gate.md"
NOW = datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(case: dict) -> dict:
    return {
        "request_id": f"REQ-{case['id']}",
        "current_task": case["current_task"],
        "consumer_session_reference": case["session_id"],
        "eligible_context_scope": {
            "tenant_scope": "TENANT-GATE",
            "session_scope": case["session_id"],
            "allowed_artifact_classes": ["session_turn"],
            "eligibility_policy_id": "ELIG-GATE-1",
        },
        "requested_budget": {
            "token_limit": case["expected_context_budget"],
            "budget_policy_id": "BUDGET-GATE-1",
        },
        "consumer_identity": {
            "consumer_id": "CONSUMER-GATE",
            "adapter_id": "ADAPTER-LOCAL-GATE",
            "purpose": "technical_shadow_gate",
        },
        "authorization_context": {
            "authorization_reference": "AUTH-GATE-1",
            "permitted_operation": "read_context_package",
            "expiry": (NOW + timedelta(hours=1)).isoformat(),
        },
        "adapter_contract_version": "1.0.0",
    }


def _inputs(case: dict) -> LocalAssemblyInputs:
    return LocalAssemblyInputs(
        case["session_id"], case["task_id"], tuple(case["conversation_history"]),
        "R2-FROZEN-GATE-SNAPSHOT",
    )


def _policy(case: dict) -> PolicySnapshot:
    return PolicySnapshot(
        "CONSUMER-GATE", "ADAPTER-LOCAL-GATE", "AUTH-GATE-1", "GRANT-GATE-1",
        "technical_shadow_gate", "TENANT-GATE", case["session_id"],
        frozenset({"session_turn"}), frozenset({"*"}), frozenset({"*"}),
        frozenset(), {}, "ELIG-GATE-1", "DISC-GATE-1", "RED-GATE-1",
        "BUDGET-GATE-1", case["expected_context_budget"], "S1-1.0", "1.0.0",
        NOW + timedelta(hours=1), 300, 30,
    )


def _transport() -> LocalTransportContext:
    return LocalTransportContext("CONSUMER-GATE", "LOCAL-GATE-CHANNEL")


def _boundary_clean() -> bool:
    forbidden_imports = ("socket", "http", "requests", "fastapi", "service", "mnemos")
    for path in PACKAGE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            if any(name.startswith(forbidden_imports) for name in names):
                return False
    return True


def _mutation_checks(case: dict, overflow_case: dict, response: dict) -> dict:
    checks = {}
    tampered = copy.deepcopy(response)
    tampered["token_estimate"] += 1
    checks["digest_tamper_detected"] = not verify_response_digest(tampered)

    lineage = copy.deepcopy(response)
    artifact = next(
        item for item in lineage["context_package"]["selected_session_artifacts"]
        if item["parent_source_ids"]
    )
    artifact["parent_source_ids"].pop()
    lineage["package_digest"]["value"] = response_digest(lineage)
    try:
        validate_response_contract(lineage)
        checks["lineage_removal_detected"] = False
    except ShadowAdapterError:
        checks["lineage_removal_detected"] = True

    telemetry = {field: None for field in ALLOWED_EVENT_FIELDS}
    telemetry["raw_context"] = "forbidden"
    try:
        ContentFreeShadowSink.validate_event(telemetry)
        checks["telemetry_escape_detected"] = False
    except ShadowAdapterError:
        checks["telemetry_escape_detected"] = True

    killed = LocalShadowAdapter(kill_switch=KillSwitch(active=True, reason="gate"))
    killed_result = killed.process(
        _request(case), _inputs(case), _policy(case), _transport(), now=NOW
    )
    checks["kill_switch_bypass_detected"] = (
        killed_result["error"]["error_code"] == "KILL_SWITCH_ACTIVE"
        and killed.assembler.invocation_count == 0
        and killed.replay.write_count == 0
        and not killed.sink.events
        and killed.delivery_attempt_count == 0
    )

    replay_adapter = LocalShadowAdapter()
    replay_adapter.process(
        _request(case), _inputs(case), _policy(case), _transport(), now=NOW
    )
    drifted = replace(_policy(case), disclosure_policy_id="DISC-GATE-2")
    drift_result = replay_adapter.process(
        _request(case), _inputs(case), drifted, _transport(),
        now=NOW + timedelta(seconds=1),
    )
    checks["policy_pin_bypass_detected"] = (
        drift_result["error"]["error_code"] == "REPLAY_POLICY_MISMATCH"
    )

    denied_transport = LocalTransportContext(
        "CONSUMER-GATE", "LOCAL-GATE-CHANNEL", authenticated=False
    )
    denied_result = LocalShadowAdapter().process(
        _request(case), _inputs(case), _policy(case), denied_transport, now=NOW
    )
    checks["authorization_bypass_detected"] = (
        denied_result["error"]["error_code"] == "AUTHORIZATION_DENIED"
    )

    redaction_policy = replace(
        _policy(case), redacted_content_by_turn_id={"t2": "[REDACTED]"}
    )
    redacted_result = LocalShadowAdapter().process(
        _request(case), _inputs(case), redaction_policy, _transport(), now=NOW
    )
    checks["redaction_bypass_detected"] = (
        "Approved immutable object storage" not in json.dumps(redacted_result)
    )

    overflow_response = LocalShadowAdapter().process(
        _request(overflow_case), _inputs(overflow_case), _policy(overflow_case),
        _transport(), now=NOW,
    )["response"]
    suppressed = copy.deepcopy(overflow_response)
    suppressed["abstention_state"] = {
        "context_budget_insufficient": False,
        "omitted_required_artifact_types": [],
        "selection_abstention_reason": None,
    }
    suppressed["package_digest"]["value"] = response_digest(suppressed)
    try:
        validate_response_contract(suppressed)
        checks["abstention_suppression_detected"] = False
    except ShadowAdapterError:
        checks["abstention_suppression_detected"] = True
    return checks


def run_gate() -> dict:
    r1_manifest = json.loads(R1_MANIFEST.read_text(encoding="utf-8"))
    r2_manifest = json.loads(R2_MANIFEST.read_text(encoding="utf-8"))
    cases = json.loads(R2.read_text(encoding="utf-8"))["cases"]
    records = []
    deterministic = True
    for case in cases:
        first_adapter = LocalShadowAdapter()
        first = first_adapter.process(
            _request(case), _inputs(case), _policy(case), _transport(), now=NOW
        )
        second = LocalShadowAdapter().process(
            _request(case), _inputs(case), _policy(case), _transport(), now=NOW
        )
        deterministic &= first == second
        response = first.get("response", {})
        artifacts = response.get("context_package", {}).get("selected_session_artifacts", [])
        records.append(
            {
                "case_id": case["id"],
                "ok": first["ok"],
                "digest_valid": bool(response) and verify_response_digest(response),
                "artifact_local_lineage_complete": bool(artifacts) and all(
                    artifact["lineage_complete"]
                    and artifact["synthetic_context"]
                    and artifact["non_authoritative"]
                    and artifact["non_promotable"]
                    for artifact in artifacts
                ),
                "budget_compliant": bool(response)
                and response["token_estimate"] <= case["expected_context_budget"],
                "telemetry_content_free": bool(first_adapter.sink.events)
                and set(first_adapter.sink.events[0]) == ALLOWED_EVENT_FIELDS,
                "shadow_only": first.get("shadow_only") is True,
            }
        )
    sample_case = cases[0]
    sample = LocalShadowAdapter().process(
        _request(sample_case), _inputs(sample_case), _policy(sample_case),
        _transport(), now=NOW,
    )["response"]
    overflow_case = next(case for case in cases if case["id"].endswith("overflow_006"))
    mutation_checks = _mutation_checks(sample_case, overflow_case, sample)
    r1_valid = _sha256(R1) == r1_manifest["file_sha256"]
    r2_valid = _sha256(R2) == r2_manifest["file_sha256"]
    gates = {
        "r1_hash_valid": r1_valid,
        "r2_hash_valid": r2_valid,
        "all_cases_assembled": all(row["ok"] for row in records),
        "digest_verification_rate_1_0": all(row["digest_valid"] for row in records),
        "artifact_local_lineage_rate_1_0": all(
            row["artifact_local_lineage_complete"] for row in records
        ),
        "budget_compliance_rate_1_0": all(row["budget_compliant"] for row in records),
        "content_free_telemetry_rate_1_0": all(
            row["telemetry_content_free"] for row in records
        ),
        "shadow_only_rate_1_0": all(row["shadow_only"] for row in records),
        "fixed_seed_determinism": deterministic,
        "no_runtime_or_network_import_path": _boundary_clean(),
        "all_mutations_detected": all(mutation_checks.values()),
    }
    return {
        "schema": "session_context_assembler_shadow_adapter_gate_v1",
        "labels": [
            "ISOLATED_SHADOW_ONLY", "NO_NETWORK_LISTENER",
            "NO_EXTERNAL_CONSUMER_CONNECTION", "NO_LIVE_ROUTING",
            "NO_MEMORY_OR_GOVERNANCE_MUTATION",
        ],
        "case_count": len(cases),
        "gates": {
            name: {"value": value, "required": True, "passed": value is True}
            for name, value in gates.items()
        },
        "mutation_checks": {
            name: {"passed": passed} for name, passed in mutation_checks.items()
        },
        "all_gates_passed": all(gates.values()),
        "records": records,
    }


def _markdown(result: dict) -> str:
    lines = [
        "# Session Context Assembler — Isolated Shadow Adapter Gate",
        "",
        " ".join(f"`{label}`" for label in result["labels"]),
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, gate in result["gates"].items():
        lines.append(f"| {name} | {'PASS' if gate['passed'] else 'FAIL'} |")
    lines.extend(["", "## Mutation sensitivity", "", "| Mutation | Result |", "|---|---|"])
    for name, check in result["mutation_checks"].items():
        lines.append(f"| {name} | {'PASS' if check['passed'] else 'FAIL'} |")
    lines.extend(
        [
            "", f"**Overall: {'PASS' if result['all_gates_passed'] else 'FAIL'}**", "",
            "A PASS authorizes review of an authorized consumer-neutral "
            "shadow-evaluation proposal only. It does not authorize a listener, "
            "consumer connection, live routing, SDK, deployment, writes, retrieval "
            "changes, or governance mutation.", "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    result = run_gate()
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    RESULT_MD.write_bytes(_markdown(result).encode("utf-8"))
    print(f"Isolated shadow cases: {result['case_count']}")
    print(f"All adapter gates passed: {result['all_gates_passed']}")
    print(f"Wrote {RESULT_JSON}")
    print(f"Wrote {RESULT_MD}")
    return 0 if result["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
