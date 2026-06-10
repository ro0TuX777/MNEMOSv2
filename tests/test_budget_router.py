"""Unit tests for the W6/W7 BudgetAwareRouter stage planner."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.retrieval.budget_router import (
    HNSW_EF_DEFAULT,
    HNSW_EF_FAST,
    OVERSAMPLE_FULL,
    OVERSAMPLE_REDUCED,
    BudgetAwareRouter,
    StageCostModel,
)


def _router(embed=10.0, prefetch=8.0, rescore=12.0, rerank=60.0):
    model = StageCostModel(priors={"embed": embed, "prefetch": prefetch, "rescore": rescore, "rerank": rerank})
    return BudgetAwareRouter(cost_model=model)


def test_no_budget_yields_full_plan():
    plan = _router().plan(None)
    assert plan.rescore and plan.rerank and not plan.degraded
    assert plan.oversample_factor == OVERSAMPLE_FULL
    assert plan.hnsw_ef == HNSW_EF_DEFAULT
    # 10 + 8 + 12 + 60
    assert abs(plan.estimated_total_ms - 90.0) < 1e-6


def test_generous_budget_keeps_full_plan():
    plan = _router().plan(200.0)
    assert not plan.degraded and plan.degradation_steps == []


def test_tight_budget_drops_rerank_first():
    plan = _router().plan(40.0)  # full=90; without rerank=30
    assert plan.degraded
    assert plan.degradation_steps == ["drop_rerank"]
    assert plan.rerank is False
    assert plan.rescore is True


def test_tighter_budget_reduces_oversample_then_ef():
    # without rerank = 10 + 8 + 12 = 30; reduced oversample = 10 + 8 + 6 = 24
    plan = _router().plan(25.0)
    assert plan.degradation_steps == ["drop_rerank", "reduce_oversample"]
    assert plan.oversample_factor == OVERSAMPLE_REDUCED

    # then reduced ef = 10 + 4 + 6 = 20
    plan = _router().plan(21.0)
    assert plan.degradation_steps == ["drop_rerank", "reduce_oversample", "reduce_hnsw_ef"]
    assert plan.hnsw_ef == HNSW_EF_FAST
    assert plan.rescore is True


def test_minimal_budget_drops_rescore_but_keeps_prefetch():
    # prefetch-only = 10 + 4 = 14
    plan = _router().plan(15.0)
    assert plan.degradation_steps[-1] == "drop_rescore"
    assert plan.rescore is False
    assert plan.prefetch is True
    assert plan.degraded and not plan.budget_infeasible


def test_infeasible_budget_is_flagged_not_failed():
    plan = _router().plan(5.0)
    assert plan.budget_infeasible is True
    assert plan.prefetch is True  # still returns a runnable plan


def test_cost_model_ewma_updates_plans():
    router = _router(rerank=60.0)
    # rerank observed much faster — full plan should start fitting 50ms budget
    for _ in range(40):
        router.observe("rerank", 10.0)
    plan = router.plan(50.0)
    assert plan.rerank is True
    assert not plan.degraded


def test_plan_serialization_contract():
    plan = _router().plan(40.0)
    payload = plan.to_dict()
    for key in (
        "prefetch", "rescore", "rerank", "oversample_factor", "hnsw_ef",
        "latency_budget_ms", "estimated_total_ms", "degraded",
        "degradation_steps", "budget_infeasible",
    ):
        assert key in payload


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
