"""Orchestrator for the isolated, local-only ADR 0008 shadow adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from .assembler_invocation_boundary import AssemblerInvocationBoundary
from .canonical import sha256_digest, verify_response_digest
from .content_free_shadow_sink import ContentFreeShadowSink
from .errors import ShadowAdapterError
from .kill_switch import KillSwitch
from .models import LocalAssemblyInputs, LocalTransportContext, PolicySnapshot
from .policy_and_disclosure_boundary import evaluate_policy
from .replay_controller import ReplayController
from .request_validator import validate_request
from .response_builder_and_digest import build_response


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class LocalShadowAdapter:
    """In-process technical shadow only; never a live consumer delivery path."""

    def __init__(
        self,
        kill_switch: KillSwitch | None = None,
        replay: ReplayController | None = None,
        sink: ContentFreeShadowSink | None = None,
        assembler: AssemblerInvocationBoundary | None = None,
    ) -> None:
        self.kill_switch = kill_switch or KillSwitch()
        self.replay = replay or ReplayController()
        self.sink = sink or ContentFreeShadowSink()
        self.assembler = assembler or AssemblerInvocationBoundary()
        self.delivery_attempt_count = 0

    def activate_kill_switch(self, reason: str) -> None:
        self.kill_switch.activate(reason, rollback=self.replay.invalidate_all)

    def process(
        self,
        request: dict,
        inputs: LocalAssemblyInputs,
        policy: PolicySnapshot,
        transport: LocalTransportContext,
        *,
        now: datetime | None = None,
        seed: int = 7,
    ) -> dict:
        now = now or datetime.now(timezone.utc)
        request_id = request.get("request_id") if isinstance(request, dict) else None
        contract_version = (
            request.get("adapter_contract_version", "unknown")
            if isinstance(request, dict) else "unknown"
        )
        consumer_id = None
        adapter_id = None
        try:
            kill_generation = self.kill_switch.begin()
            validate_request(request, transport, now)
            consumer_id = request["consumer_identity"].get("consumer_id")
            adapter_id = request["consumer_identity"].get("adapter_id")
            effective = evaluate_policy(request, inputs, policy, now)
            request_digest = sha256_digest(request)

            self.kill_switch.require_enabled(kill_generation)
            cached = self.replay.lookup(request, request_digest, effective.policy_fingerprint, now)
            if cached is not None:
                self.kill_switch.require_enabled(kill_generation)
                if not verify_response_digest(cached):
                    raise ShadowAdapterError(
                        "LINEAGE_INCOMPLETE", "Cached package integrity failed."
                    )
                with self.kill_switch.guard(kill_generation):
                    self.sink.record_success(
                        cached,
                        policy.adapter_id,
                        transport.channel_id,
                        now.astimezone(timezone.utc).isoformat(),
                        replayed=True,
                    )
                    self.delivery_attempt_count += 1
                return {"ok": True, "shadow_only": True, "replayed": True, "response": cached}

            self.kill_switch.require_enabled(kill_generation)
            package = self.assembler.assemble(request, inputs, effective, seed)
            self.kill_switch.require_enabled(kill_generation)
            response = build_response(request, inputs, policy, effective, package, now)
            with self.kill_switch.guard(kill_generation):
                self.replay.put(
                    request,
                    request_digest,
                    effective.policy_fingerprint,
                    _parse_time(response["expires_at"]),
                    response,
                )
                self.sink.record_success(
                    response,
                    policy.adapter_id,
                    transport.channel_id,
                    now.astimezone(timezone.utc).isoformat(),
                    replayed=False,
                )
                self.delivery_attempt_count += 1
            return {"ok": True, "shadow_only": True, "replayed": False, "response": response}
        except ShadowAdapterError as error:
            error_response = error.response(request_id, contract_version)
            if error.code != "KILL_SWITCH_ACTIVE" and not self.kill_switch.active:
                self.sink.record_error(
                    request_id,
                    consumer_id,
                    adapter_id,
                    contract_version,
                    transport.channel_id,
                    now.astimezone(timezone.utc).isoformat(),
                    error.code,
                )
            return {"ok": False, "shadow_only": True, "error": error_response}
