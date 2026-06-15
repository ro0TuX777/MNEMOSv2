"""
CycleEvaluator — R²-Mem-inspired rubric scorer for CognitiveCycleRecord objects.

This module is Phase 16 of the PatternEngramCandidate extraction harness.
It implements a deterministic, LLM-free rubric evaluator that scores each
CognitiveCycleRecord on six dimensions and labels it "good", "bad", or
"neutral".  The labels and scores feed Phase 17 (PatternLearner) for
IF-THEN pattern distillation.

R²-Mem reference: arXiv:2605.13486v1 §3.2 and Appendix D.1.
Rubric structure is adapted from R²-Mem's Planning/Reflection rubrics
(four dimensions × 0–3 each) to six MNEMOS-specific dimensions.
Quality thresholds mirror R²-Mem's stable operating range (Klow=5, Khigh=10)
scaled to the MNEMOS 18-point maximum.

Rubric dimensions
-----------------
routing_precision       Was the query class (CLASS_A/B/C) assigned with
                        high confidence and did the cycle complete?

candidate_efficiency    Was the candidate pool sized appropriately relative
                        to final returned candidates?

governance_appropriate  Was the governance mode matched to the query's
                        risk profile?

forecast_utilisation    Were available forecast signals acted upon?

attention_coverage      Were enough attention dimensions populated to
                        provide a meaningful cycle record?

write_integrity         Do all learning_writes carry a valid write_class?

All scoring is deterministic — no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from mnemos.cognitive.cycle import CognitiveCycleRecord
from mnemos.cognitive.learning_boundary import LEARNING_WRITE_CLASSES


# ── Quality constants (R²-Mem §3.2 stable threshold range) ───────────────────

QUALITY_GOOD = "good"
QUALITY_BAD = "bad"
QUALITY_NEUTRAL = "neutral"

# Scaled to 18-point max (6 dimensions × 3 points each).
# R²-Mem uses Klow=5, Khigh=10 out of 12; our equivalents are 7 / 13 out of 18.
KLOW: int = 7    # scores ≤ KLOW  → "bad"
KHIGH: int = 13  # scores ≥ KHIGH → "good"

EVALUATOR_VERSION = "v1"

# Attention dimension coverage thresholds
_ATTENTION_FULL = 9     # ≥ 9 populated dimensions → score 3
_ATTENTION_PARTIAL = 7  # ≥ 7 populated dimensions → score 2
_ATTENTION_SPARSE = 4   # ≥ 4 populated dimensions → score 1
# < 4 → score 0

# Candidate pool ratio thresholds
# ratio = candidate_count_pre_governance / max(candidate_count_post_governance, 1)
_POOL_TIGHT = 2   # ≤ 2× → score 3
_POOL_OK = 3      # ≤ 3× → score 2
_POOL_WIDE = 5    # ≤ 5× → score 1
# > 5× → score 0


# ── Result schema ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CycleEvaluationRecord:
    """
    Rubric evaluation result for a single CognitiveCycleRecord.

    Produced by CycleEvaluator.evaluate().  Immutable once created.
    """

    cycle_id: str
    """ID of the evaluated CognitiveCycleRecord."""

    trigger_type: str
    """Trigger type copied from the source cycle."""

    rubric_scores: Dict[str, int]
    """
    Per-dimension scores:
      routing_precision, candidate_efficiency, governance_appropriate,
      forecast_utilisation, attention_coverage, write_integrity
    Each value is 0–3.
    """

    aggregate_score: int
    """Sum of all rubric_scores values.  Range 0–18."""

    quality_label: str
    """'good' | 'bad' | 'neutral'.  Derived from aggregate_score vs KLOW/KHIGH."""

    reasons: Dict[str, str]
    """Human-readable reason string per dimension."""

    evaluator_version: str = EVALUATOR_VERSION

    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "trigger_type": self.trigger_type,
            "rubric_scores": dict(self.rubric_scores),
            "aggregate_score": self.aggregate_score,
            "quality_label": self.quality_label,
            "reasons": dict(self.reasons),
            "evaluator_version": self.evaluator_version,
            "evaluated_at": self.evaluated_at,
        }


# ── Rubric scoring helpers ────────────────────────────────────────────────────


def _score_routing_precision(cycle: CognitiveCycleRecord) -> tuple[int, str]:
    """
    Was the query-class assignment confident and did the cycle complete?

    Score 3: class set, confidence ≥ 0.70, status completed
    Score 2: class set, confidence 0.50–0.70, status completed
    Score 1: class set but confidence < 0.50 or status degraded
    Score 0: class None/unknown or status blocked/no_op
    """
    snap = cycle.working_memory_snapshot
    qc = snap.query_class if snap else None
    conf = snap.query_class_confidence if snap else 0.0
    status = cycle.final_status

    if not qc or qc == "unknown":
        return 0, "query_class not set or unknown"
    if status in ("blocked", "no_op"):
        return 0, f"cycle did not complete: final_status={status}"
    if status == "degraded":
        return 1, f"cycle degraded; query_class={qc} conf={conf:.2f}"
    if conf >= 0.70:
        return 3, f"high-confidence routing: {qc} conf={conf:.2f}"
    if conf >= 0.50:
        return 2, f"moderate-confidence routing: {qc} conf={conf:.2f}"
    return 1, f"low-confidence routing: {qc} conf={conf:.2f}"


def _score_candidate_efficiency(cycle: CognitiveCycleRecord) -> tuple[int, str]:
    """
    Was the candidate pool sized appropriately?

    Uses ratio = pre_governance / max(post_governance, 1).

    Score 3: ratio ≤ 2 and suppression_count == 0
    Score 2: ratio ≤ 3 (or pre_cognitive cache hit, no pool needed)
    Score 1: ratio ≤ 5
    Score 0: ratio > 5, or pre_governance > 0 and post_governance == 0
             without a cache hit (over-filtered to zero)
    """
    snap = cycle.working_memory_snapshot
    if snap is None:
        return 1, "no working_memory_snapshot; cannot assess candidate pool"

    pre = snap.candidate_count_pre_governance
    post = snap.candidate_count_post_governance
    supp = snap.suppression_count
    cache_hit = snap.pre_cognitive_cache_hit

    if cache_hit:
        return 3, "pre-cognitive cache hit; no retrieval pool required"

    if pre == 0 and post == 0:
        return 2, "no candidates retrieved (empty corpus or filtered query)"

    if pre > 0 and post == 0:
        return 0, f"all {pre} candidates suppressed; zero returned"

    ratio = pre / max(post, 1)

    if ratio <= _POOL_TIGHT and supp == 0:
        return 3, f"tight pool: pre={pre} post={post} ratio={ratio:.1f} no suppressions"
    if ratio <= _POOL_OK:
        return 2, f"acceptable pool: pre={pre} post={post} ratio={ratio:.1f}"
    if ratio <= _POOL_WIDE:
        return 1, f"wide pool: pre={pre} post={post} ratio={ratio:.1f}"
    return 0, f"oversized pool: pre={pre} post={post} ratio={ratio:.1f}"


def _score_governance_appropriate(cycle: CognitiveCycleRecord) -> tuple[int, str]:
    """
    Was the governance mode matched to the query's risk profile?

    Score 3: enforced + suppressions > 0  (enforcement was warranted and used)
             OR off + no contradictions + no suppressions (off was safe)
             OR advisory + vetoes > 0 (advisory caught issues)
    Score 2: advisory + no vetoes (advisory present but nothing to catch)
    Score 1: off + contradictions detected (off when enforcement was warranted)
    Score 0: enforced + net_candidates == 0 (enforcement eliminated all results)
             OR no governance eval present at all when suppressions > 0
    """
    snap = cycle.working_memory_snapshot
    mode = snap.active_governance_mode if snap else "off"
    contradiction_count = snap.contradiction_count if snap else 0
    suppression_count = snap.suppression_count if snap else 0

    evals = cycle.governance_evaluations
    total_vetoed = sum(e.vetoed for e in evals)
    total_net = sum(e.net_candidates_returned for e in evals)

    if not evals and suppression_count > 0:
        return 0, "suppressions occurred without governance evaluation record"

    if mode == "enforced":
        if total_net == 0 and (total_vetoed > 0 or suppression_count > 0):
            return 0, "enforced governance eliminated all candidates"
        if suppression_count > 0 or total_vetoed > 0:
            return 3, f"enforced governance appropriately suppressed {suppression_count} candidates"
        return 2, "enforced governance applied with no suppressions needed"

    if mode == "advisory":
        if total_vetoed > 0:
            return 3, f"advisory governance flagged {total_vetoed} vetoes; issues surfaced"
        return 2, "advisory governance present; no vetoes triggered"

    # mode == "off"
    if contradiction_count > 0:
        return 1, f"governance off but {contradiction_count} contradictions detected; enforcement may be warranted"
    return 3, "governance off; no contradictions or suppressions detected"


def _score_forecast_utilisation(cycle: CognitiveCycleRecord) -> tuple[int, str]:
    """
    Were available forecast signals acted upon?

    Score 3: forecast advisory present and pre-cognitive cache hit, or
             route is pre_warm/advise with completed forecast actions
    Score 2: forecast advisory present, cycle completed normally
    Score 1: forecast present but route unchanged and cycle degraded
    Score 0: forecast actions present but all failed/skipped (signal wasted)

    If no forecast signals at all: score 2 (neutral — nothing to act on).
    """
    snap = cycle.working_memory_snapshot
    forecast_present = snap.forecast_advisory_present if snap else False
    cache_hit = snap.pre_cognitive_cache_hit if snap else False
    selected_route = snap.selected_route if snap else cycle.selected_route

    forecast_actions = cycle.forecast_actions
    if not forecast_actions and not forecast_present:
        return 2, "no forecast signals present; dimension not applicable"

    completed_forecasts = [a for a in forecast_actions if a.status == "completed"]
    failed_forecasts = [a for a in forecast_actions if a.status == "failed"]

    if forecast_actions and not completed_forecasts:
        return 0, f"all {len(failed_forecasts)} forecast actions failed or skipped"

    if cache_hit:
        return 3, "pre-cognitive cache hit; intent trajectory forecast was acted upon"

    if selected_route in ("pre_warm", "advise") and completed_forecasts:
        return 3, f"forecast routed to '{selected_route}' with {len(completed_forecasts)} completed forecast actions"

    if forecast_present and cycle.final_status == "completed":
        return 2, f"forecast advisory present; cycle completed via route='{selected_route}'"

    if cycle.final_status == "degraded":
        return 1, "forecast present but cycle degraded; signal not fully utilised"

    return 2, "forecast actions completed; route was standard return"


def _score_attention_coverage(cycle: CognitiveCycleRecord) -> tuple[int, str]:
    """
    Were enough attention dimensions populated?

    Score 3: ≥ 9 unique dimensions
    Score 2: ≥ 7 unique dimensions
    Score 1: ≥ 4 unique dimensions
    Score 0: < 4 unique dimensions
    """
    dims = {d.dimension for d in cycle.attention_decisions}
    n = len(dims)
    if n >= _ATTENTION_FULL:
        return 3, f"{n} attention dimensions present (full coverage)"
    if n >= _ATTENTION_PARTIAL:
        return 2, f"{n} attention dimensions present (partial coverage)"
    if n >= _ATTENTION_SPARSE:
        return 1, f"{n} attention dimensions present (sparse coverage)"
    return 0, f"only {n} attention dimensions present (insufficient)"


def _score_write_integrity(cycle: CognitiveCycleRecord) -> tuple[int, str]:
    """
    Do all learning_writes carry a valid write_class?

    Score 3: all writes have a valid write_class, OR no writes present
    Score 2: ≥ 80% of writes have a valid write_class
    Score 1: ≥ 50% of writes have a valid write_class
    Score 0: < 50% valid, or any write with write_class=None when writes exist
    """
    writes = cycle.learning_writes
    if not writes:
        return 3, "no learning_writes present; integrity constraint vacuously satisfied"

    total = len(writes)
    valid = sum(
        1 for w in writes
        if w.write_class is not None and w.write_class in LEARNING_WRITE_CLASSES
    )
    none_count = sum(1 for w in writes if w.write_class is None)

    ratio = valid / total

    if none_count > 0:
        return 0, f"{none_count}/{total} writes have write_class=None"
    if ratio >= 1.0:
        return 3, f"all {total} writes have valid write_class"
    if ratio >= 0.8:
        return 2, f"{valid}/{total} writes have valid write_class"
    if ratio >= 0.5:
        return 1, f"{valid}/{total} writes have valid write_class"
    return 0, f"only {valid}/{total} writes have valid write_class"


# ── Evaluator ─────────────────────────────────────────────────────────────────


class CycleEvaluator:
    """
    Deterministic rubric evaluator for CognitiveCycleRecord objects.

    Produces a CycleEvaluationRecord per cycle with a quality label
    ("good", "bad", "neutral") derived from the aggregate rubric score.

    Usage::

        evaluator = CycleEvaluator()
        record = evaluator.evaluate(cycle)
        print(record.quality_label, record.aggregate_score)

    Thresholds can be overridden at construction time for domain tuning.
    The paper-recommended stable range is klow in [4,6], khigh in [11,14].
    """

    def __init__(
        self,
        klow: int = KLOW,
        khigh: int = KHIGH,
    ) -> None:
        if not (0 <= klow < khigh <= 18):
            raise ValueError(
                f"Invalid thresholds: klow={klow} khigh={khigh}. "
                "Must satisfy 0 <= klow < khigh <= 18."
            )
        self.klow = klow
        self.khigh = khigh

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, cycle: CognitiveCycleRecord) -> CycleEvaluationRecord:
        """Score a single cycle and return a CycleEvaluationRecord."""
        scores: Dict[str, int] = {}
        reasons: Dict[str, str] = {}

        for dim, fn in self._SCORERS.items():
            score, reason = fn(cycle)
            scores[dim] = score
            reasons[dim] = reason

        aggregate = sum(scores.values())
        label = self._label(aggregate)

        return CycleEvaluationRecord(
            cycle_id=cycle.cycle_id,
            trigger_type=cycle.trigger_type,
            rubric_scores=scores,
            aggregate_score=aggregate,
            quality_label=label,
            reasons=reasons,
        )

    def evaluate_batch(
        self,
        cycles: List[CognitiveCycleRecord],
    ) -> List[CycleEvaluationRecord]:
        """Score a list of cycles, preserving order."""
        return [self.evaluate(c) for c in cycles]

    def filter_by_quality(
        self,
        records: List[CycleEvaluationRecord],
        quality: str,
    ) -> List[CycleEvaluationRecord]:
        """
        Return only records matching the given quality label.

        quality must be one of "good", "bad", "neutral".
        """
        if quality not in (QUALITY_GOOD, QUALITY_BAD, QUALITY_NEUTRAL):
            raise ValueError(
                f"Unknown quality label: {quality!r}. "
                f"Must be one of {QUALITY_GOOD!r}, {QUALITY_BAD!r}, {QUALITY_NEUTRAL!r}."
            )
        return [r for r in records if r.quality_label == quality]

    # ── Internal ───────────────────────────────────────────────────────────────

    _SCORERS = {
        "routing_precision":        _score_routing_precision,
        "candidate_efficiency":     _score_candidate_efficiency,
        "governance_appropriate":   _score_governance_appropriate,
        "forecast_utilisation":     _score_forecast_utilisation,
        "attention_coverage":       _score_attention_coverage,
        "write_integrity":          _score_write_integrity,
    }

    def _label(self, aggregate: int) -> str:
        if aggregate >= self.khigh:
            return QUALITY_GOOD
        if aggregate <= self.klow:
            return QUALITY_BAD
        return QUALITY_NEUTRAL
