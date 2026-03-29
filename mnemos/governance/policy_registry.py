"""
PolicyRegistry — ordered, config-driven governance policy chain.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.policies import BasePolicy

logger = logging.getLogger("mnemos.governance.registry")


class PolicyRegistry:
    """
    Maintains an ordered list of governance policies and runs them as a
    sequential pipeline on each search candidate.

    Policies are evaluated in registration order.  The pipeline
    short-circuits once a candidate is suppressed (veto_modifier == 0),
    unless ``run_all_on_suppressed`` is True.

    Individual policies can be disabled at runtime via ``disable()``.
    """

    def __init__(
        self,
        policies: Optional[List[BasePolicy]] = None,
        disabled_policies: Optional[List[str]] = None,
        run_all_on_suppressed: bool = False,
    ):
        self._policies: List[BasePolicy] = list(policies or [])
        self._disabled: Set[str] = set(disabled_policies or [])
        self._run_all = run_all_on_suppressed

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, policy: BasePolicy) -> None:
        """Append a policy to the end of the chain."""
        self._policies.append(policy)

    def disable(self, policy_name: str) -> None:
        self._disabled.add(policy_name)

    def enable(self, policy_name: str) -> None:
        self._disabled.discard(policy_name)

    @property
    def policy_names(self) -> List[str]:
        return [p.policy_name for p in self._policies]

    @property
    def active_policy_names(self) -> List[str]:
        return [p.policy_name for p in self._policies
                if p.policy_name not in self._disabled]

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        result: SearchResult,
        context: Dict[str, Any],
    ) -> GovernanceDecision:
        """
        Run all active policies on a single SearchResult.

        Computes the final governed_score after all modifiers are applied.

        governed_score =
            retrieval_score
            × trust_modifier
            × utility_modifier
            × freshness_modifier
            × contradiction_modifier
            × veto_modifier
        """
        decision = GovernanceDecision(
            engram_id=result.engram.id,
            retrieval_score=result.score,
            governed_score=result.score,
        )

        for policy in self._policies:
            if policy.policy_name in self._disabled:
                continue
            try:
                decision = policy.evaluate(result, decision, context)
            except Exception:
                logger.exception(
                    "Policy %r raised on engram %s — skipping",
                    policy.policy_name,
                    result.engram.id,
                )
            # Short-circuit once suppressed (saves cost on remaining policies)
            if not self._run_all and decision.veto_modifier == 0.0:
                break

        # Final governed score: product of all modifiers × retrieval score
        decision.governed_score = round(
            decision.retrieval_score
            * decision.trust_modifier
            * decision.utility_modifier
            * decision.freshness_modifier
            * decision.contradiction_modifier
            * decision.veto_modifier,
            6,
        )

        return decision
