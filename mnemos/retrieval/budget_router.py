"""
BudgetAwareRouter — latency-budget stage planning for tiered retrieval.

W6/W7 synthesis: given a per-request ``latency_budget_ms``, decide which
retrieval stages run. Pure planning logic — no retrieval calls. The router
composes knobs that already exist (oversampling depth, conditional rerank,
HNSW ef) plus the planned Matryoshka prefetch/rescore split (Phase 7).

Degradation ladder (applied in order until the plan fits the budget):

    0. full        — prefetch + rescore + conditional rerank, default ef
    1. drop rerank
    2. shrink oversample factor (3.0 → 1.5)
    3. reduce hnsw_ef (default → fast)
    4. drop rescore — prefetch-only results, response flagged degraded

The plan never degrades below prefetch: a response is always producible.
When even prefetch exceeds the budget, the plan is returned with
``budget_infeasible=True`` so the caller can answer honestly rather than
fail (design principle #7 — graceful degradation, honestly reported).

Stage costs are estimated by an EWMA cost model seeded with conservative
priors and updated from observed latencies (the retrieval router already
measures dense and rerank stage latency in telemetry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Conservative cold-start priors (ms) — overwritten by observations.
DEFAULT_STAGE_PRIORS_MS: Dict[str, float] = {
    "embed": 10.0,
    "prefetch": 8.0,    # low-dim ANN (64-dim Matryoshka head)
    "rescore": 12.0,    # server-side full-dim rescore of oversampled set
    "rerank": 60.0,     # cross-encoder lane
}

OVERSAMPLE_FULL = 3.0
OVERSAMPLE_REDUCED = 1.5
HNSW_EF_DEFAULT = 128
HNSW_EF_FAST = 64


@dataclass
class StagePlan:
    """Resolved execution plan for one search request."""

    prefetch: bool = True
    rescore: bool = True
    rerank: bool = True
    oversample_factor: float = OVERSAMPLE_FULL
    hnsw_ef: int = HNSW_EF_DEFAULT

    latency_budget_ms: Optional[float] = None
    estimated_total_ms: float = 0.0
    degraded: bool = False
    degradation_steps: List[str] = field(default_factory=list)
    budget_infeasible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prefetch": self.prefetch,
            "rescore": self.rescore,
            "rerank": self.rerank,
            "oversample_factor": self.oversample_factor,
            "hnsw_ef": self.hnsw_ef,
            "latency_budget_ms": self.latency_budget_ms,
            "estimated_total_ms": round(self.estimated_total_ms, 2),
            "degraded": self.degraded,
            "degradation_steps": list(self.degradation_steps),
            "budget_infeasible": self.budget_infeasible,
        }


class StageCostModel:
    """EWMA latency estimates per retrieval stage."""

    def __init__(self, alpha: float = 0.2, priors: Optional[Dict[str, float]] = None):
        self._alpha = min(max(alpha, 0.01), 1.0)
        self._estimates: Dict[str, float] = dict(priors or DEFAULT_STAGE_PRIORS_MS)
        self._observation_counts: Dict[str, int] = {}

    def observe(self, stage: str, latency_ms: float) -> None:
        if latency_ms < 0:
            return
        current = self._estimates.get(stage)
        if current is None or self._observation_counts.get(stage, 0) == 0:
            self._estimates[stage] = float(latency_ms)
        else:
            self._estimates[stage] = (1 - self._alpha) * current + self._alpha * float(latency_ms)
        self._observation_counts[stage] = self._observation_counts.get(stage, 0) + 1

    def estimate(self, stage: str) -> float:
        return self._estimates.get(stage, 0.0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "estimates_ms": {k: round(v, 2) for k, v in sorted(self._estimates.items())},
            "observation_counts": dict(sorted(self._observation_counts.items())),
        }


class BudgetAwareRouter:
    """
    Plans stage execution for a latency budget.

    The router does not execute retrieval — it produces a ``StagePlan`` the
    retrieval router consumes, and receives observed stage latencies back via
    ``observe()`` to keep the cost model current.
    """

    def __init__(self, cost_model: Optional[StageCostModel] = None):
        self._costs = cost_model or StageCostModel()

    # ── Cost model passthrough ────────────────────────────────────────────

    def observe(self, stage: str, latency_ms: float) -> None:
        self._costs.observe(stage, latency_ms)

    def stats(self) -> Dict[str, Any]:
        return self._costs.snapshot()

    # ── Planning ──────────────────────────────────────────────────────────

    def plan(self, latency_budget_ms: Optional[float] = None) -> StagePlan:
        """
        Resolve a stage plan. ``None`` budget → full plan, never degraded.
        """
        full = StagePlan(latency_budget_ms=latency_budget_ms)
        full.estimated_total_ms = self._estimate(full)
        if latency_budget_ms is None or full.estimated_total_ms <= latency_budget_ms:
            return full

        plan = full
        for step, mutate in (
            ("drop_rerank", self._drop_rerank),
            ("reduce_oversample", self._reduce_oversample),
            ("reduce_hnsw_ef", self._reduce_hnsw_ef),
            ("drop_rescore", self._drop_rescore),
        ):
            mutate(plan)
            plan.degraded = True
            plan.degradation_steps.append(step)
            plan.estimated_total_ms = self._estimate(plan)
            if plan.estimated_total_ms <= latency_budget_ms:
                return plan

        plan.budget_infeasible = True
        return plan

    # ── Internals ─────────────────────────────────────────────────────────

    def _estimate(self, plan: StagePlan) -> float:
        total = self._costs.estimate("embed")
        if plan.prefetch:
            ef_scale = plan.hnsw_ef / HNSW_EF_DEFAULT
            oversample_scale = plan.oversample_factor / OVERSAMPLE_FULL
            total += self._costs.estimate("prefetch") * max(ef_scale, 0.25)
            if plan.rescore:
                total += self._costs.estimate("rescore") * max(oversample_scale, 0.25)
        if plan.rerank:
            total += self._costs.estimate("rerank")
        return total

    @staticmethod
    def _drop_rerank(plan: StagePlan) -> None:
        plan.rerank = False

    @staticmethod
    def _reduce_oversample(plan: StagePlan) -> None:
        plan.oversample_factor = OVERSAMPLE_REDUCED

    @staticmethod
    def _reduce_hnsw_ef(plan: StagePlan) -> None:
        plan.hnsw_ef = HNSW_EF_FAST

    @staticmethod
    def _drop_rescore(plan: StagePlan) -> None:
        plan.rescore = False
        plan.oversample_factor = 1.0
