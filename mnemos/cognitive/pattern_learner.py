"""
PatternLearner — Phase 17 of the PatternEngramCandidate extraction harness.

Translates scored CognitiveCycleRecord objects (from Phase 16 CycleEvaluator)
into advisory PatternEngramCandidate objects via:

  1. SituationAbstractor — deterministic, entity-free situation description
     built from structured cycle fields (no LLM calls).

  2. PatternLearner — assembles an IF-THEN pattern_summary and constructs
     a PatternEngramCandidate.  Only "descriptive" and
     "operational_recommendation" types are ever produced; all *_mutation
     types remain blocked.

  3. PatternConsolidator — A-MEM-inspired deduplication that merges
     near-duplicate situation texts or flags contradiction when two
     candidates share the same situation but differ in quality label.

R²-Mem reference: arXiv:2605.13486v1 §3.3, Appendix E.2.
ExpeL reference: conceptual basis for cross-cycle insight extraction.

Safety invariants
-----------------
- PatternLearner never produces pattern_type in {"policy_mutation",
  "routing_mutation", "template_mutation"}.
- PatternEngramCandidate.authoritative_for_retrieval is always False
  (enforced by the PatternEngramCandidate class itself).
- No LLM calls anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from mnemos.cognitive.cycle import CognitiveCycleRecord
from mnemos.cognitive.cycle_evaluator import (
    CycleEvaluationRecord,
    QUALITY_GOOD,
    QUALITY_BAD,
)
from mnemos.cognitive.learning_boundary import PATTERN_TYPE_TO_WRITE_CLASS
from mnemos.cognitive.pattern_engram import PatternEngramCandidate


# ── Consolidation action types ────────────────────────────────────────────────

ConsolidationAction = Literal["new", "merged", "contradicted"]

# Jaccard similarity threshold for treating two situations as the same
_SITUATION_MERGE_THRESHOLD = 0.85

# Human-readable dimension labels for IF-THEN template assembly
_DIM_LABELS: Dict[str, str] = {
    "routing_precision": "high-confidence query classification",
    "candidate_efficiency": "efficient candidate pool sizing",
    "governance_appropriate": "appropriate governance mode selection",
    "forecast_utilisation": "forecast signal utilisation",
    "attention_coverage": "full attention dimension coverage",
    "write_integrity": "valid learning write classification",
}

# Dimensions whose weakness most directly warrants an "operational_recommendation"
_OPERATIONAL_WEAK_DIMS = frozenset({
    "routing_precision",
    "governance_appropriate",
    "forecast_utilisation",
})


# ── Situation summary ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SituationSummary:
    """
    Entity-free abstract description of a cognitive cycle's situation.

    All fields are derived deterministically from cycle and evaluation data —
    no entity names, query content, or raw text from the query_or_event field
    is included.  This mirrors R²-Mem's σᵢ abstraction stage (§3.3).
    """

    trigger_type: str
    """Type of event that triggered the cycle: search | reflect | pulse | etc."""

    route_class: str
    """Derived query class: CLASS_A | CLASS_B | CLASS_C | unknown."""

    governance_mode: str
    """Active governance mode during the cycle: off | advisory | enforced."""

    forecast_active: bool
    """True if any forecast_actions were present in the cycle."""

    candidate_pressure: str
    """Relative candidate pool pressure: low | medium | high."""

    weak_dimensions: List[str]
    """Rubric dimension names that scored 0 or 1."""

    strong_dimensions: List[str]
    """Rubric dimension names that scored 3."""

    quality_label: str
    """good | bad (neutral cycles are not abstracted)."""

    situation_text: str
    """
    Template-assembled entity-free prose description.
    Example: "A CLASS_B search cycle with advisory governance where
              routing_precision and forecast_utilisation were weak."
    """

    def token_set(self) -> frozenset:
        """Lowercase token set of situation_text for Jaccard similarity."""
        return frozenset(self.situation_text.lower().split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_type": self.trigger_type,
            "route_class": self.route_class,
            "governance_mode": self.governance_mode,
            "forecast_active": self.forecast_active,
            "candidate_pressure": self.candidate_pressure,
            "weak_dimensions": list(self.weak_dimensions),
            "strong_dimensions": list(self.strong_dimensions),
            "quality_label": self.quality_label,
            "situation_text": self.situation_text,
        }


# ── Situation abstractor ──────────────────────────────────────────────────────


class SituationAbstractor:
    """
    Deterministic, LLM-free extractor of entity-free situation summaries.

    Derives route_class from working_memory_snapshot.query_class,
    governance_mode from working_memory_snapshot.active_governance_mode,
    candidate_pressure from pre/post governance counts, and weak/strong
    dimensions from the CycleEvaluationRecord rubric_scores.

    The situation_text is assembled from a small set of templates — it
    contains no entity names, query text, or user-identifiable content.
    """

    def abstract(
        self,
        cycle: CognitiveCycleRecord,
        evaluation: CycleEvaluationRecord,
    ) -> SituationSummary:
        """Produce a SituationSummary from a cycle + its evaluation record."""
        snap = cycle.working_memory_snapshot

        route_class = self._derive_route_class(snap)
        governance_mode = snap.active_governance_mode if snap else "off"
        forecast_active = bool(cycle.forecast_actions)
        candidate_pressure = self._derive_candidate_pressure(snap)

        weak = [
            dim for dim, score in evaluation.rubric_scores.items()
            if score <= 1
        ]
        strong = [
            dim for dim, score in evaluation.rubric_scores.items()
            if score == 3
        ]

        situation_text = self._assemble_text(
            trigger_type=cycle.trigger_type,
            route_class=route_class,
            governance_mode=governance_mode,
            forecast_active=forecast_active,
            candidate_pressure=candidate_pressure,
            weak=weak,
            strong=strong,
            quality_label=evaluation.quality_label,
        )

        return SituationSummary(
            trigger_type=cycle.trigger_type,
            route_class=route_class,
            governance_mode=governance_mode,
            forecast_active=forecast_active,
            candidate_pressure=candidate_pressure,
            weak_dimensions=weak,
            strong_dimensions=strong,
            quality_label=evaluation.quality_label,
            situation_text=situation_text,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _derive_route_class(snap) -> str:
        if snap is None:
            return "unknown"
        qc = snap.query_class or "unknown"
        # Normalise to canonical labels; pass through CLASS_A/B/C directly
        if qc in ("CLASS_A", "CLASS_B", "CLASS_C"):
            return qc
        # Map legacy / service-side labels
        mapping = {
            "factoid": "CLASS_A",
            "multi_hop": "CLASS_B",
            "global_synthesis": "CLASS_C",
        }
        return mapping.get(qc, "unknown")

    @staticmethod
    def _derive_candidate_pressure(snap) -> str:
        if snap is None:
            return "unknown"
        pre = snap.candidate_count_pre_governance
        post = max(snap.candidate_count_post_governance, 1)
        if pre == 0:
            return "low"
        ratio = pre / post
        if ratio <= 2:
            return "low"
        if ratio <= 4:
            return "medium"
        return "high"

    @staticmethod
    def _assemble_text(
        *,
        trigger_type: str,
        route_class: str,
        governance_mode: str,
        forecast_active: bool,
        candidate_pressure: str,
        weak: List[str],
        strong: List[str],
        quality_label: str,
    ) -> str:
        """Assemble a single entity-free sentence describing the situation."""
        parts = [f"A {route_class} {trigger_type} cycle"]
        parts.append(f"with {governance_mode} governance")

        if forecast_active:
            parts.append("and active forecast signals")

        if candidate_pressure != "low" and candidate_pressure != "unknown":
            parts.append(f"under {candidate_pressure} candidate pressure")

        if quality_label == QUALITY_GOOD and strong:
            strong_labels = [_DIM_LABELS.get(d, d) for d in strong]
            parts.append("where " + " and ".join(strong_labels[:3]) + " were strong")
        elif quality_label == QUALITY_BAD and weak:
            weak_labels = [_DIM_LABELS.get(d, d) for d in weak]
            parts.append("where " + " and ".join(weak_labels[:3]) + " were weak")

        return " ".join(parts) + "."


# ── Pattern learner ───────────────────────────────────────────────────────────


class PatternLearner:
    """
    Distils a PatternEngramCandidate from a scored cycle + situation summary.

    Produces IF-THEN pattern_summary strings modelled on R²-Mem Appendix E.2
    experience format: "IF <abstract situation> THEN <strategy>".

    Only "descriptive" and "operational_recommendation" pattern types are
    ever produced.  All *_mutation types are permanently blocked and will
    never be returned by this class.
    """

    # Remediation advice keyed by weak dimension
    _REMEDIATIONS: Dict[str, str] = {
        "routing_precision": (
            "ensure the query classifier has a clear query_class label with confidence ≥ 0.70"
        ),
        "candidate_efficiency": (
            "tighten the candidate pool — reduce oversample factor or raise governance "
            "to enforced mode to prevent zero-return outcomes"
        ),
        "governance_appropriate": (
            "switch to advisory or enforced governance mode when contradictions or "
            "high-risk queries are present"
        ),
        "forecast_utilisation": (
            "act on high-confidence forecast advisories by adjusting routing or "
            "triggering pre-warm before demand peaks"
        ),
        "attention_coverage": (
            "ensure all 9+ attention dimensions are populated before finalising the cycle"
        ),
        "write_integrity": (
            "declare an explicit write_class for every LearningWrite from LEARNING_WRITE_CLASSES"
        ),
    }

    # Strategy advice keyed by strong dimension
    _STRATEGIES: Dict[str, str] = {
        "routing_precision": "apply the same high-confidence query classification approach",
        "candidate_efficiency": "maintain the efficient pool sizing and suppression strategy",
        "governance_appropriate": "continue matching governance mode to query risk profile",
        "forecast_utilisation": "continue acting on forecast advisory signals early",
        "attention_coverage": "maintain full attention dimension population for transparency",
        "write_integrity": "continue declaring explicit write_class on all LearningWrites",
    }

    def extract(
        self,
        cycle: CognitiveCycleRecord,
        evaluation: CycleEvaluationRecord,
        situation: SituationSummary,
    ) -> PatternEngramCandidate:
        """
        Produce a PatternEngramCandidate from a scored cycle + situation.

        Only called for "good" or "bad" quality cycles.  Neutral cycles
        should be filtered out before calling this method.
        """
        if evaluation.quality_label not in (QUALITY_GOOD, QUALITY_BAD):
            raise ValueError(
                f"PatternLearner.extract() requires quality_label 'good' or 'bad', "
                f"got {evaluation.quality_label!r}"
            )

        pattern_type = self._infer_pattern_type(situation)
        proposed_class = PATTERN_TYPE_TO_WRITE_CLASS[pattern_type]
        confidence = self._compute_confidence(evaluation)
        support_score, contradiction_score = self._compute_scores(evaluation, confidence)
        if_then = self._build_if_then(situation, evaluation)
        risk = self._build_risk(situation, pattern_type)

        return PatternEngramCandidate(
            pattern_summary=if_then,
            supporting_cycle_ids=[cycle.cycle_id],
            confidence_score=round(confidence, 4),
            support_score=round(support_score, 4),
            contradiction_score=round(contradiction_score, 4),
            pattern_type=pattern_type,
            recommended_scope="local",
            applies_when=situation.situation_text,
            does_not_apply_when=f"Query classification is not {situation.route_class}",
            risk_if_wrong=risk,
            proposed_learning_class=proposed_class,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _infer_pattern_type(self, situation: SituationSummary) -> str:
        """
        Infer pattern type from the situation.

        Returns "descriptive" for good-quality cycles.
        Returns "operational_recommendation" for bad-quality cycles where at
        least one operational dimension (routing, governance, forecast) was weak.
        Returns "descriptive" for bad cycles where only structural dimensions
        (attention coverage, write integrity) were weak — these describe a
        structural observation, not an operational recommendation.

        INVARIANT: never returns any *_mutation type.
        """
        if situation.quality_label == QUALITY_GOOD:
            return "descriptive"

        # Bad cycle: check for operational weak dimensions
        weak_set = set(situation.weak_dimensions)
        if weak_set & _OPERATIONAL_WEAK_DIMS:
            return "operational_recommendation"
        return "descriptive"

    def _compute_confidence(self, evaluation: CycleEvaluationRecord) -> float:
        """
        Confidence in the pattern.

        For good cycles: confidence = aggregate_score / 18 (higher score → more confident)
        For bad cycles:  confidence = 1.0 - (aggregate_score / 18) (lower score → more confident failure mode)
        """
        ratio = evaluation.aggregate_score / 18.0
        if evaluation.quality_label == QUALITY_GOOD:
            return ratio
        return 1.0 - ratio

    @staticmethod
    def _compute_scores(
        evaluation: CycleEvaluationRecord,
        confidence: float,
    ) -> Tuple[float, float]:
        """Return (support_score, contradiction_score)."""
        if evaluation.quality_label == QUALITY_GOOD:
            return confidence, 0.0
        return 0.0, confidence

    def _build_if_then(
        self,
        situation: SituationSummary,
        evaluation: CycleEvaluationRecord,
    ) -> str:
        """Assemble the IF-THEN pattern_summary string."""
        if evaluation.quality_label == QUALITY_GOOD:
            return self._build_positive_rule(situation)
        return self._build_negative_rule(situation)

    def _build_positive_rule(self, situation: SituationSummary) -> str:
        """IF <good situation> THEN <strategy from strong dims>."""
        strategies = [
            self._STRATEGIES.get(d, d)
            for d in situation.strong_dimensions[:2]
        ]
        then_clause = (
            "; ".join(strategies)
            if strategies
            else "the current configuration produces high-quality results"
        )
        return (
            f"IF {situation.situation_text[:-1].lower()} "
            f"THEN {then_clause}."
        )

    def _build_negative_rule(self, situation: SituationSummary) -> str:
        """IF <bad situation> THEN avoid <weak dims>; apply <remediations>."""
        remediations = [
            self._REMEDIATIONS.get(d, f"address {d}")
            for d in situation.weak_dimensions[:3]
        ]
        then_clause = (
            "; ".join(remediations)
            if remediations
            else "investigate and address the weak rubric dimensions before reuse"
        )
        return (
            f"IF {situation.situation_text[:-1].lower()} "
            f"THEN {then_clause}."
        )

    @staticmethod
    def _build_risk(situation: SituationSummary, pattern_type: str) -> str:
        if pattern_type == "operational_recommendation":
            return (
                f"Incorrect {situation.route_class} routing or governance guidance "
                "may worsen retrieval precision or trigger unnecessary suppression."
            )
        return (
            f"Misapplying this pattern to non-{situation.route_class} queries "
            "may produce misleading quality expectations."
        )


# ── Pattern consolidator ──────────────────────────────────────────────────────


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


class PatternConsolidator:
    """
    A-MEM-inspired deduplication and linking for PatternEngramCandidate objects.

    When a new candidate's situation_text is sufficiently similar (Jaccard ≥
    MERGE_THRESHOLD) to an existing candidate:
    - Same quality label → merge: extend supporting_cycle_ids, update scores.
    - Different quality label → mark contradiction: add to contradicting_engram_ids.

    No LLM calls.  All decisions are deterministic.
    """

    def __init__(self, merge_threshold: float = _SITUATION_MERGE_THRESHOLD) -> None:
        if not 0.0 < merge_threshold <= 1.0:
            raise ValueError(f"merge_threshold must be in (0, 1]; got {merge_threshold}")
        self.merge_threshold = merge_threshold

    def consolidate(
        self,
        new_candidate: PatternEngramCandidate,
        existing_pool: List[PatternEngramCandidate],
        *,
        new_situation: SituationSummary,
        existing_situations: Optional[List[SituationSummary]] = None,
    ) -> Tuple[PatternEngramCandidate, ConsolidationAction]:
        """
        Compare new_candidate against existing_pool by situation similarity.

        Returns a (possibly updated) candidate and a ConsolidationAction:
        - "new"         — no similar existing candidate found
        - "merged"      — merged into an existing same-label candidate
        - "contradicted"— flagged as contradicting an opposite-label candidate

        The returned candidate is always a new object (existing pool entries
        are not mutated by this method).
        """
        new_tokens = new_situation.token_set()

        for i, existing in enumerate(existing_pool):
            existing_sit = (
                existing_situations[i]
                if existing_situations and i < len(existing_situations)
                else None
            )
            if existing_sit is None:
                continue

            sim = _jaccard(new_tokens, existing_sit.token_set())
            if sim < self.merge_threshold:
                continue

            # Similar situation found
            if new_situation.quality_label == existing_sit.quality_label:
                # Same quality → merge
                merged = self._merge(new_candidate, existing)
                return merged, "merged"
            else:
                # Opposite quality → contradiction
                contradicted = self._contradict(new_candidate, existing)
                return contradicted, "contradicted"

        return new_candidate, "new"

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _merge(
        new_candidate: PatternEngramCandidate,
        existing: PatternEngramCandidate,
    ) -> PatternEngramCandidate:
        """Return a new candidate that merges new into existing."""
        merged_cycle_ids = list(
            dict.fromkeys(existing.supporting_cycle_ids + new_candidate.supporting_cycle_ids)
        )
        merged_engram_ids = list(
            dict.fromkeys(existing.supporting_engram_ids + new_candidate.supporting_engram_ids)
        )
        # Blend confidence and support scores upward
        new_confidence = min(
            (existing.confidence_score + new_candidate.confidence_score) / 2.0 * 1.05,
            1.0,
        )
        new_support = min(
            (existing.support_score + new_candidate.support_score) / 2.0 * 1.05,
            1.0,
        )

        return PatternEngramCandidate(
            pattern_summary=existing.pattern_summary,
            supporting_cycle_ids=merged_cycle_ids,
            supporting_engram_ids=merged_engram_ids,
            contradicting_engram_ids=list(existing.contradicting_engram_ids),
            confidence_score=round(new_confidence, 4),
            support_score=round(new_support, 4),
            contradiction_score=existing.contradiction_score,
            pattern_type=existing.pattern_type,
            recommended_scope=existing.recommended_scope,
            applies_when=existing.applies_when,
            does_not_apply_when=existing.does_not_apply_when,
            risk_if_wrong=existing.risk_if_wrong,
            proposed_learning_class=existing.proposed_learning_class,
            candidate_id=existing.candidate_id,
            promotion_status=existing.promotion_status,
            governance_review_required=existing.governance_review_required,
            first_seen_at=existing.first_seen_at,
        )

    @staticmethod
    def _contradict(
        new_candidate: PatternEngramCandidate,
        existing: PatternEngramCandidate,
    ) -> PatternEngramCandidate:
        """Return new_candidate updated to reference the existing as a contradiction."""
        contradicting = list(
            dict.fromkeys(new_candidate.contradicting_engram_ids + [existing.candidate_id])
        )
        new_contradiction_score = min(
            new_candidate.contradiction_score + 0.2, 1.0
        )

        return PatternEngramCandidate(
            pattern_summary=new_candidate.pattern_summary,
            supporting_cycle_ids=new_candidate.supporting_cycle_ids,
            supporting_engram_ids=new_candidate.supporting_engram_ids,
            contradicting_engram_ids=contradicting,
            confidence_score=new_candidate.confidence_score,
            support_score=new_candidate.support_score,
            contradiction_score=round(new_contradiction_score, 4),
            pattern_type=new_candidate.pattern_type,
            recommended_scope=new_candidate.recommended_scope,
            applies_when=new_candidate.applies_when,
            does_not_apply_when=new_candidate.does_not_apply_when,
            risk_if_wrong=new_candidate.risk_if_wrong,
            proposed_learning_class=new_candidate.proposed_learning_class,
            first_seen_at=new_candidate.first_seen_at,
        )
