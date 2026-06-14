"""
RelevanceVetoPolicy — drop candidates that fail basic quality gates.

Wave 1 heuristics:
  1. Veto if retrieval_score < min_score_threshold  (default 0.0 = disabled)
  2. Veto if deletion_state is soft_deleted or tombstone
  3. Veto if "toxic" in policy_flags
  4. Compute freshness modifier for all surviving candidates
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.policies import BasePolicy
from mnemos.governance.hygiene.volatility import family_key_from_engram


class RelevanceVetoPolicy(BasePolicy):

    def __init__(
        self,
        min_score_threshold: float = 0.0,
        freshness_half_life_days: float = 180.0,
        volatility_bias_enabled: bool = False,
        volatility_bias_provider: Optional[Callable[[str], float]] = None,
    ):
        self._min_score = min_score_threshold
        self._half_life = max(freshness_half_life_days, 1.0)
        self._volatility_bias_enabled = bool(volatility_bias_enabled)
        self._volatility_bias_provider = volatility_bias_provider

    @property
    def policy_name(self) -> str:
        return "relevance_veto"

    def evaluate(
        self,
        result: SearchResult,
        decision: GovernanceDecision,
        context: Dict[str, Any],
    ) -> GovernanceDecision:
        engram = result.engram
        gov = engram.governance

        # ── Gate 1: score floor ───────────────────────────────────────────
        if self._min_score > 0.0 and result.score < self._min_score:
            return self._veto(
                decision,
                f"score {result.score:.4f} below threshold {self._min_score:.4f}",
            )

        # ── Gate 2: deleted memory ────────────────────────────────────────
        if gov is not None and gov.deletion_state in ("soft_deleted", "tombstone"):
            return self._veto(decision, f"deletion_state={gov.deletion_state}")

        # ── Gate 3: toxic flag ────────────────────────────────────────────
        if gov is not None and "toxic" in gov.policy_flags:
            return self._veto(decision, "policy_flag=toxic")

        # ── Freshness modifier ────────────────────────────────────────────
        half_life = self._half_life
        if self._volatility_bias_enabled and self._volatility_bias_provider is not None:
            family_key = family_key_from_engram(engram)
            try:
                bias = max(1.0, float(self._volatility_bias_provider(family_key)))
            except Exception:
                bias = 1.0
            half_life = max(1.0, self._half_life / bias)
            if bias > 1.0:
                decision.policy_trace = getattr(decision, "policy_trace", {}) or {}
                decision.policy_trace["volatility_bias"] = {
                    "family_key": family_key,
                    "bias": round(bias, 4),
                    "effective_half_life_days": round(half_life, 4),
                }

        decision.freshness_modifier = _compute_freshness(engram.created_at, half_life)

        return decision

    @staticmethod
    def _veto(
        decision: GovernanceDecision, reason: str
    ) -> GovernanceDecision:
        decision.veto_pass = False
        decision.veto_reason = reason
        decision.veto_modifier = 0.0
        decision.suppressed = True
        decision.suppressed_reason = reason
        return decision


def _compute_freshness(created_at: str, half_life_days: float) -> float:
    """Exponential freshness decay.

    Returns 1.0 for brand-new memories, 0.5 at half_life_days,
    approaching 0 for very old memories.  Unknown age → 1.0.
    """
    try:
        ts = created_at.rstrip("Z")
        created = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        age_days = (now - created).total_seconds() / 86400.0
        if age_days <= 0:
            return 1.0
        lam = math.log(2) / half_life_days
        return round(math.exp(-lam * age_days), 6)
    except Exception:
        return 1.0
