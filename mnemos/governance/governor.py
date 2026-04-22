"""
Governor — main entry point for the MNEMOS governance layer.

Instantiated once in the service runtime and wired into the search path.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.models.contradiction_record import ContradictionRecord
from mnemos.governance.policies.relevance_veto_policy import RelevanceVetoPolicy
from mnemos.governance.policies.utility_policy import UtilityPolicy
from mnemos.governance.policies.contradiction_policy import ContradictionPolicy
from mnemos.governance.policy_registry import PolicyRegistry
from mnemos.governance.read_path import ReadPath, GOVERNANCE_MODES
from mnemos.governance.reflect_path import ReflectPath, ReflectResult
from mnemos.governance.usage_detector import UsageDetector
from mnemos.governance.reinforcement import Reinforcement
from mnemos.governance.telemetry.reflect_metrics import ReflectMetrics
from mnemos.governance.telemetry.hygiene_metrics import HygieneMetrics
from mnemos.governance.policy_profiles import GovernancePolicyProfile
from mnemos.governance.hygiene import (
    HygienePipeline,
    HygienePipelineReport,
    DecayConfig,
    PruneConfig,
)

logger = logging.getLogger("mnemos.governance")


class Governor:
    """
    Governance layer for MNEMOS retrieval.

    Default policy chain (Wave 2):
      Per-candidate:
        1. RelevanceVetoPolicy  — score floor, deletion state, toxic flag, freshness
        2. UtilityPolicy        — trust and utility modifiers
      Cross-candidate:
        3. ContradictionPolicy  — entity-slot contradiction detection and resolution

    Wave 3 additions:
      reflect()  — post-generation feedback loop; updates trust, utility,
                   stability based on which memories actually contributed.

    Future waves will add:
      4. DecayPolicy hygiene  (Wave 4 — background freshness/prune jobs)

    Configuration
    -------------
    min_score_threshold
        Veto candidates whose retrieval score is below this value.
        Default 0.0 (veto disabled — conservative for advisory mode).

    freshness_half_life_days
        Half-life in days for freshness decay.  A memory created
        ``half_life_days`` ago receives a 0.5 freshness modifier.
        Default 180 days.

    disabled_policies
        List of policy names to skip.
    """

    def __init__(
        self,
        min_score_threshold: float = 0.0,
        freshness_half_life_days: float = 180.0,
        policy_profiles: Optional[Dict[str, GovernancePolicyProfile]] = None,
        disabled_policies: Optional[List[str]] = None,
    ):
        self._base_min_score_threshold = float(min_score_threshold)
        self._base_freshness_half_life_days = float(freshness_half_life_days)
        registry = PolicyRegistry(disabled_policies=disabled_policies)
        registry.register(
            RelevanceVetoPolicy(
                min_score_threshold=self._base_min_score_threshold,
                freshness_half_life_days=self._base_freshness_half_life_days,
            )
        )
        registry.register(UtilityPolicy())

        self._registry = registry
        self._contradiction_policy = ContradictionPolicy()
        self._read_path = ReadPath(registry, contradiction_policy=self._contradiction_policy)
        self._reflect_path = ReflectPath()
        self._hygiene_pipeline = HygienePipeline()
        self._hygiene_metrics = HygieneMetrics()
        self._policy_profiles: Dict[str, GovernancePolicyProfile] = policy_profiles or {
            "default": GovernancePolicyProfile(
                profile_id="default",
                min_score_threshold=self._base_min_score_threshold,
                freshness_half_life_days=self._base_freshness_half_life_days,
            ).validate()
        }

        # ── In-memory aggregate stats (reset on service restart) ───────────
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "total_governed_queries": 0,
            "advisory_queries": 0,
            "enforced_queries": 0,
            "total_candidates_evaluated": 0,
            "total_vetoed": 0,
            "total_suppressed": 0,
            "total_contradictions_detected": 0,
            "total_contradiction_suppressed": 0,
        }
        self._reflect_metrics = ReflectMetrics()

    # ── Public API ────────────────────────────────────────────────────────

    def govern(
        self,
        results: List[SearchResult],
        query: str,
        governance_mode: str = "advisory",
        top_k: int = 10,
        governance_profile: Optional[str] = None,
    ) -> Tuple[List[SearchResult], List[GovernanceDecision], List[ContradictionRecord]]:
        """
        Apply governance to a list of search results.

        Args:
            results:         Raw SearchResult list from the retrieval tier.
            query:           Original query text.
            governance_mode: "off" | "advisory" | "enforced"
            top_k:           Result cap used in enforced mode.

        Returns:
            (governed_results, decisions, contradiction_records)
        """
        if governance_mode not in GOVERNANCE_MODES:
            raise ValueError(
                f"Invalid governance_mode: {governance_mode!r}. "
                f"Must be one of {sorted(GOVERNANCE_MODES)}"
            )

        if governance_mode == "off":
            return results, [], []

        profile = self._resolve_profile(governance_profile)
        read_path = self._read_path if profile.profile_id == "default" else self._build_read_path_for_profile(profile)
        governed, decisions, contradiction_records = read_path.apply(
            results=results,
            query=query,
            governance_mode=governance_mode,
            top_k=top_k,
        )

        self._record_stats(governance_mode, results, decisions, contradiction_records)
        return governed, decisions, contradiction_records

    def reflect(
        self,
        query: str,
        answer: str,
        results: List[SearchResult],
        decisions: List[GovernanceDecision],
        cited_ids: Optional[List[str]] = None,
        governance_mode: str = "advisory",
        governance_profile: Optional[str] = None,
    ) -> ReflectResult:
        """
        Run the post-generation reflect loop for one query/answer pair.

        Detects which memories were used, ignored, contradicted, or vetoed;
        applies deterministic utility/trust/stability reinforcement rules;
        and accumulates reflect telemetry.

        Args:
            query:           Original query text.
            answer:          Generated answer to analyse.
            results:         SearchResult list considered during retrieval.
            decisions:       GovernanceDecision list from govern().
            cited_ids:       Explicit memory IDs the answer is known to cite.
            governance_mode: Mode active during retrieval (for telemetry).

        Returns:
            ReflectResult with usage labels, applied deltas, and counts.
            GovernanceMeta on the supplied results is mutated in place.
        """
        profile = self._resolve_profile(governance_profile)
        reflect_path = self._reflect_path if profile.profile_id == "default" else self._build_reflect_path_for_profile(profile)
        reflect_result = reflect_path.reflect(
            query=query,
            answer=answer,
            results=results,
            decisions=decisions,
            cited_ids=cited_ids,
            governance_mode=governance_mode,
        )
        self._reflect_metrics.record(reflect_result)
        return reflect_result

    def has_policy_profile(self, profile_id: str) -> bool:
        return profile_id in self._policy_profiles

    def policy_profile_ids(self) -> List[str]:
        return sorted(self._policy_profiles.keys())

    def run_hygiene(
        self,
        engrams: list,
        now_iso: Optional[str] = None,
        dry_run: bool = False,
        decay_config: Optional[DecayConfig] = None,
        prune_config: Optional[PruneConfig] = None,
    ) -> HygienePipelineReport:
        """
        Run the Wave 4 hygiene pipeline over a list of Engrams.

        Chains: DecayRunner -> PrunePromoter -> ContradictionSweepRunner.
        All runners mutate GovernanceMeta in place unless dry_run=True.

        Args:
            engrams:       List of Engram objects to sweep.
            now_iso:       ISO UTC timestamp for decay; defaults to current UTC time.
            dry_run:       If True, compute reports without mutating anything.
            decay_config:  Override decay runner configuration.
            prune_config:  Override prune promoter configuration.

        Returns:
            HygienePipelineReport with sub-reports from all three runners.
        """
        if decay_config is not None or prune_config is not None:
            pipeline = HygienePipeline(
                decay_config=decay_config,
                prune_config=prune_config,
            )
        else:
            pipeline = self._hygiene_pipeline

        report = pipeline.run(engrams, now_iso=now_iso, dry_run=dry_run)
        self._hygiene_metrics.record_decay(report.decay)
        self._hygiene_metrics.record_prune(report.prune)
        self._hygiene_metrics.record_sweep(report.sweep)
        return report

    def stats(self) -> Dict[str, Any]:
        """Return aggregate governance + reflect statistics."""
        with self._lock:
            s = dict(self._stats)
        total = s["total_candidates_evaluated"]
        s["veto_rate"] = round(s["total_vetoed"] / total, 4) if total else 0.0
        s["suppression_rate"] = (
            round(s["total_suppressed"] / total, 4) if total else 0.0
        )
        s["active_policies"] = self._registry.active_policy_names
        s.update(self._reflect_metrics.to_dict())
        s.update(self._hygiene_metrics.to_dict())
        return s

    # ── Internals ─────────────────────────────────────────────────────────

    def _record_stats(
        self,
        mode: str,
        raw_results: List[SearchResult],
        decisions: List[GovernanceDecision],
        contradiction_records: List[ContradictionRecord],
    ) -> None:
        vetoed = sum(1 for d in decisions if not d.veto_pass)
        suppressed = sum(1 for d in decisions if d.suppressed)
        contradiction_suppressed = sum(
            1 for d in decisions if d.suppressed_by_contradiction
        )
        with self._lock:
            self._stats["total_governed_queries"] += 1
            key = f"{mode}_queries"
            if key in self._stats:
                self._stats[key] += 1
            self._stats["total_candidates_evaluated"] += len(raw_results)
            self._stats["total_vetoed"] += vetoed
            self._stats["total_suppressed"] += suppressed
            self._stats["total_contradictions_detected"] += len(contradiction_records)
            self._stats["total_contradiction_suppressed"] += contradiction_suppressed

    def _resolve_profile(self, governance_profile: Optional[str]) -> GovernancePolicyProfile:
        pid = (governance_profile or "default").strip()
        profile = self._policy_profiles.get(pid)
        if profile is None:
            raise ValueError(
                f"Unknown governance_profile: {pid!r}. "
                f"Supported profiles: {', '.join(self.policy_profile_ids())}"
            )
        return profile

    def _build_read_path_for_profile(self, profile: GovernancePolicyProfile) -> ReadPath:
        registry = PolicyRegistry(disabled_policies=None)
        registry.register(
            RelevanceVetoPolicy(
                min_score_threshold=profile.min_score_threshold,
                freshness_half_life_days=profile.freshness_half_life_days,
            )
        )
        registry.register(UtilityPolicy())
        return ReadPath(registry, contradiction_policy=self._contradiction_policy)

    def _build_reflect_path_for_profile(self, profile: GovernancePolicyProfile) -> ReflectPath:
        detector = UsageDetector(
            overlap_threshold=profile.overlap_threshold,
            min_memory_tokens_for_overlap=profile.min_memory_tokens_for_overlap,
            min_overlap_tokens=profile.min_overlap_tokens,
        )
        reinforcement = Reinforcement(config=profile.reinforcement_config())
        return ReflectPath(usage_detector=detector, reinforcement=reinforcement)
