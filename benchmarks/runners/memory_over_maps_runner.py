"""
Memory Over Maps benchmark runner.

Phase 1 focuses on M1 lineage integrity and derived-view input completeness.
"""

from __future__ import annotations

import hashlib
import os
import statistics
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from mnemos.audit.forensic_ledger import ForensicLedger
from mnemos.engram.model import Engram
from mnemos.governance.reflect_path import ReflectPath
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.memory_over_maps.models import DerivedView
from mnemos.memory_over_maps.view_builder import build_requested_views
from mnemos.memory_over_maps.view_cache import DerivedViewCache
from mnemos.retrieval.base import SearchResult
from mnemos.retrieval.candidate_envelope import CandidateEnvelopeConfig, apply_candidate_envelope


@dataclass
class Phase1LineageResult:
    lineage_completeness_rate: float
    source_trace_resolution_latency_ms: float
    responses_with_source_artifact_coverage_rate: float
    derived_view_input_completeness_rate: float
    orphan_derived_views: int
    audit_log_derived_view_events: int
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Phase2EnvelopeResult:
    initial_candidate_count: int
    final_candidate_count: int
    compression_ratio: float
    answer_support_retention_rate: float
    duplicate_suppression_rate: float
    source_concentration_ratio: float
    deterministic_replay_match: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Phase3DerivedViewsResult:
    reproducibility_success_rate: float
    regeneration_mismatch_count: int
    input_completeness_rate: float
    generated_view_count: int
    mean_regeneration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Phase4CacheInvalidationResult:
    invalidation_trigger_coverage_rate: float
    stale_cache_survival_rate: float
    dry_run_real_run_parity: bool
    cache_hit_rate: float
    false_invalidation_rate: float
    tested_events: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Phase5ReflectBoundedResult:
    bounded_candidate_adherence_rate: float
    proper_noun_sensitivity_rate: float
    trust_recovery_delta: float
    enforced_mode_drift_rate: float
    concurrent_reflect_success_rate: float
    mean_reflect_latency_ms: float
    scenario_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _build_sample_engrams(sample_size: int = 12) -> List[Engram]:
    out: List[Engram] = []
    for idx in range(sample_size):
        out.append(
            Engram(
                id=f"eng-{idx}",
                content=f"Memory sample {idx}",
                source=f"repo://sample/doc-{idx // 3}.md",
                metadata={
                    "artifact_id": f"artifact-{idx // 3}",
                    "artifact_version": "v1",
                    "chunk_id": f"chunk-{idx}",
                    "source_uri": f"repo://sample/doc-{idx // 3}.md",
                    "provenance_span": {"start": idx * 10, "end": (idx * 10) + 9},
                },
            )
        )
    return out


def _build_phase2_candidates() -> List[SearchResult]:
    rows = [
        ("a1", "SOC2 policy baseline", "art-a", True, 0.99),
        ("a2", "SOC2 policy baseline", "art-a", True, 0.96),
        ("a3", "SOC2 policy update", "art-a", False, 0.93),
        ("b1", "control mapping evidence", "art-b", True, 0.91),
        ("b2", "control mapping evidence expanded", "art-b", False, 0.90),
        ("c1", "vendor checklist unrelated", "art-c", False, 0.89),
        ("d1", "retention window details", "art-d", True, 0.88),
        ("e1", "generic template noise", "art-e", False, 0.87),
    ]
    out: List[SearchResult] = []
    for doc_id, content, artifact_id, supports_answer, score in rows:
        out.append(
            SearchResult(
                engram=Engram(
                    id=doc_id,
                    content=content,
                    metadata={
                        "artifact_id": artifact_id,
                        "chunk_id": f"chunk-{doc_id}",
                        "supports_answer": supports_answer,
                    },
                ),
                score=score,
                tier="hybrid",
            )
        )
    return out


def run_phase1_lineage_track(sample_size: int = 12) -> Phase1LineageResult:
    engrams = _build_sample_engrams(sample_size=sample_size)

    lineage_ok = 0
    source_coverage_ok = 0
    latencies_ms: List[float] = []

    for e in engrams:
        t0 = time.perf_counter()
        lineage = e.lineage()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        required = {"artifact_id", "artifact_version", "chunk_id", "source_uri"}
        if required.issubset(set(lineage.keys())) and all(lineage[k] for k in required):
            lineage_ok += 1
        if lineage.get("artifact_id") and lineage.get("source_uri"):
            source_coverage_ok += 1

    views: List[DerivedView] = []
    for e in engrams[:4]:
        views.append(
            DerivedView(
                view_type="evidence_bundle",
                inputs={"artifact_ids": [e.lineage()["artifact_id"]], "chunk_ids": [e.lineage()["chunk_id"]]},
                query_fingerprint=_fingerprint(e.content),
                governance_state_hash="gov-state-v1",
                cacheable=False,
                reproducible=True,
            )
        )

    fd, ledger_path = tempfile.mkstemp(prefix="mnemos_mom_phase1_", suffix=".db")
    os.close(fd)
    try:
        ledger = ForensicLedger(db_path=str(Path(ledger_path)))
        for view in views:
            ledger.log_derived_view_generation(
                view_type=view.view_type,
                view_id=view.view_id,
                inputs=view.inputs,
                query_fingerprint=view.query_fingerprint,
                governance_state_hash=view.governance_state_hash,
                metadata={"phase": "phase1"},
            )
        derived_events = len(
            [
                tx
                for tx in ledger.get_recent_transactions(limit=100)
                if tx.get("action") == "derived_view_generation"
            ]
        )
    finally:
        try:
            os.remove(ledger_path)
        except OSError:
            # Best-effort cleanup only; benchmark metrics are unaffected.
            pass

    input_complete = 0
    orphan_count = 0
    for view in views:
        inputs = view.inputs or {}
        if inputs.get("artifact_ids") and inputs.get("chunk_ids"):
            input_complete += 1
        else:
            orphan_count += 1

    return Phase1LineageResult(
        lineage_completeness_rate=round(lineage_ok / len(engrams), 4),
        source_trace_resolution_latency_ms=round(statistics.mean(latencies_ms), 4),
        responses_with_source_artifact_coverage_rate=round(source_coverage_ok / len(engrams), 4),
        derived_view_input_completeness_rate=round(input_complete / len(views), 4) if views else 0.0,
        orphan_derived_views=orphan_count,
        audit_log_derived_view_events=derived_events,
        sample_size=len(engrams),
    )


def run_phase2_candidate_envelope_track() -> Phase2EnvelopeResult:
    candidates = _build_phase2_candidates()
    cfg = CandidateEnvelopeConfig(
        enabled=True,
        candidate_pool_limit=6,
        dedupe_similarity_threshold=0.95,
        max_per_source_artifact=2,
        diversity_policy="off",
        bounded_adjudication_enabled=True,
    )
    narrowed_1, meta_1 = apply_candidate_envelope(candidates, cfg)
    narrowed_2, _ = apply_candidate_envelope(candidates, cfg)

    supports_before = {c.engram.id for c in candidates if c.engram.metadata.get("supports_answer")}
    supports_after = {c.engram.id for c in narrowed_1 if c.engram.metadata.get("supports_answer")}
    retained = len(supports_before.intersection(supports_after))
    support_retention = (retained / len(supports_before)) if supports_before else 1.0

    initial = len(candidates)
    final = len(narrowed_1)
    suppressed_dupes = meta_1["suppression_summary"].get("duplicate_similarity", 0)
    duplicate_rate = suppressed_dupes / initial if initial else 0.0

    deterministic = [r.engram.id for r in narrowed_1] == [r.engram.id for r in narrowed_2]

    return Phase2EnvelopeResult(
        initial_candidate_count=initial,
        final_candidate_count=final,
        compression_ratio=round((final / initial), 4) if initial else 0.0,
        answer_support_retention_rate=round(support_retention, 4),
        duplicate_suppression_rate=round(duplicate_rate, 4),
        source_concentration_ratio=float(meta_1.get("source_concentration_ratio", 0.0)),
        deterministic_replay_match=deterministic,
    )


def run_phase3_derived_views_track() -> Phase3DerivedViewsResult:
    results = _build_phase2_candidates()[:3]
    requested = [
        "evidence_bundle",
        "contradiction_bundle",
        "preference_snapshot",
        "timeline_summary",
    ]

    t0 = time.perf_counter()
    run1 = build_requested_views(
        requested=requested,
        query="phase3 reproducibility check",
        results=results,
        decisions=[],
        contradiction_records=[],
        subject_id="bench-subject",
    )
    dt1 = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    run2 = build_requested_views(
        requested=requested,
        query="phase3 reproducibility check",
        results=results,
        decisions=[],
        contradiction_records=[],
        subject_id="bench-subject",
    )
    dt2 = (time.perf_counter() - t1) * 1000.0

    def _canonical(view: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(view)
        out.pop("view_id", None)
        out.pop("created_at", None)
        return out

    mismatches = 0
    for a, b in zip(run1, run2):
        if _canonical(a) != _canonical(b):
            mismatches += 1

    complete = 0
    for v in run1:
        inputs = v.get("inputs", {})
        if inputs.get("artifact_ids") and inputs.get("chunk_ids"):
            complete += 1

    total = len(run1)
    success_rate = 1.0 if mismatches == 0 and total > 0 else 0.0
    completeness = (complete / total) if total else 0.0

    return Phase3DerivedViewsResult(
        reproducibility_success_rate=round(success_rate, 4),
        regeneration_mismatch_count=mismatches,
        input_completeness_rate=round(completeness, 4),
        generated_view_count=total,
        mean_regeneration_ms=round((dt1 + dt2) / 2.0, 4),
    )


def run_phase4_cache_invalidation_track() -> Phase4CacheInvalidationResult:
    cache = DerivedViewCache(ttl_seconds=3600)

    def _insert(tag: str, artifact_id: str, chunk_id: str, governance_hash: str, cluster_id: str = "") -> str:
        key = f"k-{tag}"
        cache.set(
            key=key,
            view={"view_id": f"v-{tag}", "view_type": "evidence_bundle"},
            dependency_refs={
                "artifact_ids": [artifact_id],
                "chunk_ids": [chunk_id],
                "governance_state_hash": governance_hash,
                "synthesis_policy_version": "default",
                "contradiction_cluster_id": cluster_id,
                "lifecycle_states": ["active"],
            },
        )
        return key

    key_a = _insert("a", "art-a", "chunk-a", "gov-a", "cluster-a")
    key_b = _insert("b", "art-b", "chunk-b", "gov-b", "cluster-b")

    # Cache hit/miss sample
    hits = 0
    misses = 0
    if cache.get(key_a) is not None:
        hits += 1
    if cache.get("missing-key") is None:
        misses += 1

    events = [
        ("source_artifact_updated", {"artifact_id": "art-a"}, {key_a}),
        ("chunk_set_changed", {"chunk_id": "chunk-b"}, {key_b}),
        ("contradiction_cluster_changed", {"contradiction_cluster_id": "cluster-a"}, {key_a}),
        ("governance_state_changed", {"governance_state_hash": "gov-x"}, {key_a, key_b}),
        ("synthesis_config_changed", {"synthesis_policy_version": "v2"}, {key_a, key_b}),
    ]

    covered = 0
    parity_ok = True
    stale_survivors = 0
    false_invalidations = 0
    checked = 0

    for event_type, refs, expected in events:
        dry = cache.invalidate(event_type=event_type, refs=refs, dry_run=True)
        live = cache.invalidate(event_type=event_type, refs=refs, dry_run=False)
        dry_keys = set(dry["impacted_keys"])
        live_keys = set(live["impacted_keys"])
        if dry_keys == live_keys:
            pass
        else:
            parity_ok = False
        if expected.issubset(live_keys):
            covered += 1
        # stale survival: expected-invalidated entry should not be returned anymore.
        for key in expected:
            checked += 1
            if cache.get(key) is not None:
                stale_survivors += 1
        # false invalidations: keys outside expected impacted for this event.
        false_invalidations += len(live_keys - expected)

    tested_events = len(events)
    invalidation_coverage = covered / tested_events if tested_events else 0.0
    stale_survival_rate = stale_survivors / checked if checked else 0.0
    cache_hit_rate = hits / (hits + misses) if (hits + misses) else 0.0
    false_invalidation_rate = false_invalidations / (tested_events * 2) if tested_events else 0.0

    return Phase4CacheInvalidationResult(
        invalidation_trigger_coverage_rate=round(invalidation_coverage, 4),
        stale_cache_survival_rate=round(stale_survival_rate, 4),
        dry_run_real_run_parity=parity_ok,
        cache_hit_rate=round(cache_hit_rate, 4),
        false_invalidation_rate=round(false_invalidation_rate, 4),
        tested_events=tested_events,
    )


def _mk_reflect_candidate(
    *,
    eid: str,
    content: str,
    trust: float = 0.5,
    utility: float = 0.5,
    conflict_status: str = "none",
) -> SearchResult:
    e = Engram(
        id=eid,
        content=content,
        metadata={"artifact_id": f"artifact-{eid}", "chunk_id": f"chunk-{eid}"},
    )
    e.governance = GovernanceMeta(
        trust_score=trust,
        utility_score=utility,
        stability_score=0.5,
        conflict_status=conflict_status,
    )
    return SearchResult(engram=e, score=0.8, tier="hybrid")


def _mk_decisions(results: List[SearchResult]) -> List[GovernanceDecision]:
    out: List[GovernanceDecision] = []
    for r in results:
        suppressed_by_contradiction = bool(r.engram.governance and r.engram.governance.conflict_status == "suppressed")
        out.append(
            GovernanceDecision(
                engram_id=r.engram.id,
                retrieval_score=r.score,
                governed_score=r.score,
                veto_pass=True,
                suppressed=suppressed_by_contradiction,
                suppressed_by_contradiction=suppressed_by_contradiction,
                conflict_status=r.engram.governance.conflict_status if r.engram.governance else "none",
            )
        )
    return out


def run_phase5_reflect_bounded_track() -> Phase5ReflectBoundedResult:
    reflect = ReflectPath()

    # 1) Bounded candidate adherence over scenarios.
    envelope_cfg = CandidateEnvelopeConfig(
        enabled=True,
        candidate_pool_limit=4,
        dedupe_similarity_threshold=0.95,
        max_per_source_artifact=2,
        diversity_policy="off",
        bounded_adjudication_enabled=True,
    )
    adherence_checks = 0
    adherence_hits = 0
    for _ in range(3):
        raw = _build_phase2_candidates()
        narrowed, _ = apply_candidate_envelope(raw, envelope_cfg)
        adherence_checks += 1
        if len(narrowed) <= envelope_cfg.candidate_pool_limit:
            adherence_hits += 1

    # 2) Proper-noun/entity sensitivity.
    proper = _mk_reflect_candidate(
        eid="mem-alice",
        content="Alice Johnson leads Project Orion in 2026.",
        trust=0.4,
    )
    noisy = _mk_reflect_candidate(
        eid="mem-noise",
        content="Generic unrelated placeholder content.",
        trust=0.6,
    )
    pn_results = [proper, noisy]
    pn_decisions = _mk_decisions(pn_results)
    t0 = time.perf_counter()
    pn = reflect.reflect(
        query="Who leads Project Orion?",
        answer="Alice Johnson leads Project Orion.",
        results=pn_results,
        decisions=pn_decisions,
        cited_ids=None,
        governance_mode="advisory",
    )
    latency_1 = (time.perf_counter() - t0) * 1000.0
    proper_rate = 1.0 if "mem-alice" in pn.used_ids else 0.0

    # 3) Trust recovery via repeated bounded reflective confirmations.
    recovery = _mk_reflect_candidate(
        eid="mem-recovery",
        content="Critical policy source states retention is 90 days.",
        trust=0.2,
        utility=0.4,
    )
    trust_before = recovery.engram.governance.trust_score if recovery.engram.governance else 0.0
    for _ in range(4):
        reflect.reflect(
            query="What is the retention window?",
            answer="Retention is 90 days.",
            results=[recovery],
            decisions=_mk_decisions([recovery]),
            cited_ids=["mem-recovery"],
            governance_mode="advisory",
        )
    trust_after = recovery.engram.governance.trust_score if recovery.engram.governance else trust_before

    # 4) Enforced-mode drift vs advisory-mode drift.
    drift_a = _mk_reflect_candidate(
        eid="mem-drift",
        content="Vendor contract ends in July 2026.",
        trust=0.5,
    )
    drift_results_a = [drift_a]
    drift_decisions_a = _mk_decisions(drift_results_a)
    out_advisory = reflect.reflect(
        query="When does contract end?",
        answer="Contract ends in July 2026.",
        results=drift_results_a,
        decisions=drift_decisions_a,
        cited_ids=["mem-drift"],
        governance_mode="advisory",
    )
    drift_e = _mk_reflect_candidate(
        eid="mem-drift",
        content="Vendor contract ends in July 2026.",
        trust=0.5,
    )
    drift_results_e = [drift_e]
    drift_decisions_e = _mk_decisions(drift_results_e)
    out_enforced = reflect.reflect(
        query="When does contract end?",
        answer="Contract ends in July 2026.",
        results=drift_results_e,
        decisions=drift_decisions_e,
        cited_ids=["mem-drift"],
        governance_mode="enforced",
    )
    drift_rate = 0.0 if (out_advisory.used_ids == out_enforced.used_ids) else 1.0

    # 5) Concurrent reflect stress.
    lock = threading.Lock()
    successes = 0
    runs = 10
    latencies: List[float] = [latency_1]

    def _worker(idx: int) -> None:
        nonlocal successes
        local = _mk_reflect_candidate(
            eid=f"mem-thread-{idx}",
            content=f"Entity-{idx} appears in bounded candidate set.",
            trust=0.5,
        )
        t = time.perf_counter()
        rr = reflect.reflect(
            query=f"What about Entity-{idx}?",
            answer=f"Entity-{idx} appears in bounded candidate set.",
            results=[local],
            decisions=_mk_decisions([local]),
            cited_ids=[f"mem-thread-{idx}"],
            governance_mode="advisory",
        )
        elapsed = (time.perf_counter() - t) * 1000.0
        with lock:
            latencies.append(elapsed)
            if f"mem-thread-{idx}" in rr.used_ids:
                successes += 1

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(runs)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    return Phase5ReflectBoundedResult(
        bounded_candidate_adherence_rate=round(adherence_hits / adherence_checks, 4) if adherence_checks else 0.0,
        proper_noun_sensitivity_rate=round(proper_rate, 4),
        trust_recovery_delta=round(trust_after - trust_before, 4),
        enforced_mode_drift_rate=round(drift_rate, 4),
        concurrent_reflect_success_rate=round(successes / runs, 4) if runs else 0.0,
        mean_reflect_latency_ms=round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        scenario_count=5,
    )
