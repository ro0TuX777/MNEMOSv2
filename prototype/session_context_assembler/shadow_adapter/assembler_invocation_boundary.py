"""Whitelisted invocation of the existing isolated S1 assembler."""

from __future__ import annotations

from dataclasses import dataclass

from prototype.session_context_assembler.selector_s1 import assemble_context_package_s1

from .models import LocalAssemblyInputs
from .policy_and_disclosure_boundary import EffectivePolicy


@dataclass
class AssemblerInvocationBoundary:
    invocation_count: int = 0

    def assemble(
        self,
        request: dict,
        inputs: LocalAssemblyInputs,
        effective: EffectivePolicy,
        seed: int,
    ) -> dict:
        self.invocation_count += 1
        case = {
            "id": f"local-shadow:{request['request_id']}",
            "session_id": inputs.session_id,
            "task_id": inputs.task_id,
            "current_task": request["current_task"],
            "conversation_history": [dict(turn) for turn in effective.filtered_history],
        }
        return assemble_context_package_s1(
            case,
            corpus_manifest_hash=inputs.snapshot_reference,
            seed=seed,
            token_budget=effective.effective_budget,
        )
