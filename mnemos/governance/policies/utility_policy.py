"""
UtilityPolicy — applies trust and utility modifiers to the governed score.

Wave 1: reads trust_score and utility_score from GovernanceMeta when
available, otherwise falls back to engram.confidence.

The reflect path (Wave 2) will write differentiated trust/utility values
over time.  Until then, these modifiers will be close to 1.0 for most
memories, which is correct behaviour for advisory mode.
"""

from __future__ import annotations

from typing import Any, Dict

from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.policies import BasePolicy

# Modifier bounds — kept explicit and inspectable.
_TRUST_MIN = 0.0
_TRUST_MAX = 1.25
_UTILITY_MIN = 0.5
_UTILITY_MAX = 1.25


class UtilityPolicy(BasePolicy):

    @property
    def policy_name(self) -> str:
        return "utility"

    def evaluate(
        self,
        result: SearchResult,
        decision: GovernanceDecision,
        context: Dict[str, Any],
    ) -> GovernanceDecision:
        engram = result.engram
        gov = engram.governance

        # Trust modifier: maps [0, 1] trust_score → [0.75, 1.25]
        # A fully trusted memory is boosted; a zero-trust memory is heavily penalised.
        raw_trust = gov.trust_score if gov is not None else engram.confidence
        decision.trust_modifier = _clamp(
            0.75 + 0.5 * float(raw_trust), _TRUST_MIN, _TRUST_MAX
        )

        # Utility modifier: maps [0, 1] utility_score → [0.5, 1.25]
        # A high-utility memory is boosted; a never-used memory gets 0.5 floor.
        raw_utility = gov.utility_score if gov is not None else 1.0
        decision.utility_modifier = _clamp(
            0.5 + 0.75 * float(raw_utility), _UTILITY_MIN, _UTILITY_MAX
        )

        return decision


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
