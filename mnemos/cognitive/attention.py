"""
AttentionContract builder for MNEMOS cognitive cycles.

Converts the distributed attention-like decisions already made by MNEMOS
(adaptive routing, candidate envelope, summary isolation, graph eligibility,
derived-fact eligibility, forecast advisory, governance gate) into a
structured list of AttentionDecision objects that consuming systems can read
without understanding MNEMOS internal routing code.

This module is pure — it has no side effects and takes all inputs as arguments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mnemos.cognitive.cycle import AttentionDecision


def build_attention_decisions(
    *,
    mode_meta: Dict[str, Any],
    governance_mode: str,
    governance_profile: str,
    phase1_enabled: bool = False,
    phase2_enabled: bool = False,
    phase3_enabled: bool = False,
    derive_views_requested: Optional[List[str]] = None,
    pre_cognitive_hit: bool = False,
    query_family: Optional[str] = None,
    query_family_confidence: float = 0.0,
    rerank_applied: bool = False,
    rerank_skipped_reason: Optional[str] = None,
    shadow_search_present: bool = False,
    intent_forecast_present: bool = False,
    pattern_store_enabled: bool = False,
    pattern_advisory_count: int = 0,
) -> List[AttentionDecision]:
    """
    Build the Attention Contract for a single MNEMOS cognitive cycle.

    Parameters
    ----------
    mode_meta           : Dict returned by RetrievalRouter.search() as the
                          second element of the tuple.
    governance_mode     : "off" | "advisory" | "enforced"
    governance_profile  : Active policy profile ID.
    phase1_enabled      : Memory Over Maps phase 1 (lineage/summary support).
    phase2_enabled      : Memory Over Maps phase 2 (candidate envelope).
    phase3_enabled      : Memory Over Maps phase 3 (derived views).
    derive_views_requested : List of derive_views names from the request.
    pre_cognitive_hit   : Whether query was served from the pre-cognitive cache.
    query_family        : Classified query family string.
    query_family_confidence : Classifier confidence.
    rerank_applied      : Whether cross-encoder reranking was applied.
    rerank_skipped_reason : Reason reranking was skipped, if applicable.
    shadow_search_present : Whether a shadow/intent pre-fetch was triggered.
    intent_forecast_present : Whether an intent trajectory forecast was emitted.

    Returns
    -------
    List[AttentionDecision] — ordered from most general to most specific.
    """
    decisions: List[AttentionDecision] = []

    retrieval_mode = mode_meta.get("retrieval_mode", "semantic")
    fusion_policy = mode_meta.get("fusion_policy")
    envelope_meta = mode_meta.get("candidate_envelope") or {}
    forecast_advisory = mode_meta.get("forecast_advisory") or {}
    lexical_available = bool(mode_meta.get("lexical_available", False))

    # ── 1. Pre-cognitive routing (highest priority — short-circuits everything) ──
    if pre_cognitive_hit:
        decisions.append(AttentionDecision(
            dimension="pre_cognitive_routing",
            decision="cache_hit",
            reason=(
                "Query matched a pre-cognitive intent cache entry from a "
                "prior intent trajectory forecast. Full retrieval was skipped."
            ),
            policy_source="intent_engine.fuzzy_pre_cognitive_get",
        ))
        # When pre-cognitive hits, most downstream gates are irrelevant.
        # Emit only the cache hit decision and return.
        return decisions

    # ── 2. Query classification ───────────────────────────────────────────────
    if query_family:
        conf_note = f" (confidence={round(query_family_confidence, 3)})"
        decisions.append(AttentionDecision(
            dimension="query_classification",
            decision=query_family,
            reason=(
                f"Query classified as '{query_family}'{conf_note} by the "
                "heuristic query family classifier. This classification "
                "influences reranking eligibility and fusion policy selection."
            ),
            policy_source="retrieval.policies.query_classifier",
        ))

    # ── 3. Retrieval mode selection ───────────────────────────────────────────
    if retrieval_mode == "hybrid":
        engine = mode_meta.get("fusion_engine", "python_rrf")
        decisions.append(AttentionDecision(
            dimension="retrieval_mode",
            decision=f"hybrid:{fusion_policy or 'balanced'}",
            reason=(
                f"Hybrid retrieval selected with fusion policy '{fusion_policy}' "
                f"using engine '{engine}'. Both semantic (dense) and lexical "
                "(BM25) lanes were active. Lexical lane available: {lexical_available}."
            ),
            policy_source="retrieval_router.search[retrieval_mode=hybrid]",
        ))
    elif retrieval_mode == "semantic":
        decisions.append(AttentionDecision(
            dimension="retrieval_mode",
            decision="semantic",
            reason=(
                "Semantic-only retrieval was selected. "
                + (
                    "Lexical lane was unavailable (no PostgreSQL FTS configured)."
                    if not lexical_available
                    else "Caller requested semantic-only mode."
                )
            ),
            policy_source="retrieval_router.search[retrieval_mode=semantic]",
        ))

    # ── 4. Candidate envelope ─────────────────────────────────────────────────
    if envelope_meta.get("enabled"):
        initial = envelope_meta.get("initial_candidate_count", 0)
        final = envelope_meta.get("final_candidate_count", 0)
        suppression = envelope_meta.get("suppression_summary", {})
        decisions.append(AttentionDecision(
            dimension="candidate_envelope",
            decision="applied",
            reason=(
                f"Candidate envelope narrowed {initial} raw candidates to "
                f"{final} bounded candidates. "
                f"Deduplication removed {suppression.get('duplicate_similarity', 0)}, "
                f"source-cap removed {suppression.get('source_cap_exceeded', 0)}, "
                f"pool-limit removed {suppression.get('bounded_limit_exceeded', 0)}."
            ),
            policy_source="memory_over_maps_phase2.candidate_envelope",
        ))
    elif phase2_enabled:
        decisions.append(AttentionDecision(
            dimension="candidate_envelope",
            decision="not_requested",
            reason=(
                "Phase 2 (candidate envelope) is enabled but was not requested "
                "in this query (bounded_envelope not present in request body)."
            ),
            policy_source="memory_over_maps_phase2",
        ))
    else:
        decisions.append(AttentionDecision(
            dimension="candidate_envelope",
            decision="disabled",
            reason="Candidate envelope is disabled (memory_over_maps_phase2=false).",
            policy_source="config.memory_over_maps_phase2",
        ))

    # ── 5. Cross-encoder reranking ────────────────────────────────────────────
    if rerank_applied:
        decisions.append(AttentionDecision(
            dimension="cross_encoder_rerank",
            decision="applied",
            reason=(
                "Cross-encoder reranking was applied. Candidate ordering may "
                "differ from raw fusion scores."
            ),
            policy_source="retrieval.policies.rerank_policy",
        ))
    elif rerank_skipped_reason:
        decisions.append(AttentionDecision(
            dimension="cross_encoder_rerank",
            decision="skipped",
            reason=f"Cross-encoder reranking was skipped: {rerank_skipped_reason}.",
            policy_source="retrieval.policies.rerank_policy",
        ))

    # ── 6. Summary / hierarchical node inclusion ──────────────────────────────
    if phase1_enabled:
        decisions.append(AttentionDecision(
            dimension="summary_inclusion",
            decision="eligible",
            reason=(
                "Memory Over Maps phase 1 is enabled. Summary and lineage nodes "
                "are eligible to appear in retrieval results. Summary nodes are "
                "isolated from default factoid paths by source_artifact lineage."
            ),
            policy_source="config.memory_over_maps_phase1",
        ))
    else:
        decisions.append(AttentionDecision(
            dimension="summary_inclusion",
            decision="blocked",
            reason=(
                "Memory Over Maps phase 1 is disabled. Summary and lineage nodes "
                "are not indexed or returned in this deployment."
            ),
            policy_source="config.memory_over_maps_phase1",
        ))

    # ── 7. Graph expansion ────────────────────────────────────────────────────
    # Graph hybrid requires double opt-in (not yet exposed on standard HTTP surface).
    decisions.append(AttentionDecision(
        dimension="graph_expansion",
        decision="blocked",
        reason=(
            "Graph-hybrid expansion is not active on the standard HTTP retrieval "
            "surface. It requires explicit double opt-in and is not enabled by "
            "default. Candidates were drawn from vector and lexical lanes only."
        ),
        policy_source="retrieval_router[graph_hybrid=read_only_non_public]",
    ))

    # ── 8. Derived facts / Memory Over Maps phase 3 ───────────────────────────
    if phase3_enabled and derive_views_requested:
        decisions.append(AttentionDecision(
            dimension="derived_facts",
            decision="eligible",
            reason=(
                f"Derived views were requested ({derive_views_requested}) and "
                "Memory Over Maps phase 3 is enabled. Derived views are "
                "synthesised from governed evidence and returned separately — "
                "they do NOT contaminate the default retrieval lane."
            ),
            policy_source="config.memory_over_maps_phase3",
        ))
    elif phase3_enabled and not derive_views_requested:
        decisions.append(AttentionDecision(
            dimension="derived_facts",
            decision="not_requested",
            reason=(
                "Memory Over Maps phase 3 is enabled, but no derived views were "
                "requested in this query (derive_views not present in request body). "
                "Default retrieval lane is derived-fact-free."
            ),
            policy_source="config.memory_over_maps_phase3",
        ))
    else:
        decisions.append(AttentionDecision(
            dimension="derived_facts",
            decision="blocked",
            reason=(
                "Memory Over Maps phase 3 is disabled. Derived views are "
                "unavailable. Default retrieval remains derived-fact-free."
            ),
            policy_source="config.memory_over_maps_phase3",
        ))

    # ── 9. Governance gate ────────────────────────────────────────────────────
    if governance_mode == "off":
        decisions.append(AttentionDecision(
            dimension="governance_gate",
            decision="off",
            reason=(
                "Governance is off for this query. All retrieved candidates are "
                "returned without scoring, suppression, or contradiction checks."
            ),
            policy_source="config.governance_mode",
        ))
    elif governance_mode == "advisory":
        decisions.append(AttentionDecision(
            dimension="governance_gate",
            decision=f"advisory:{governance_profile}",
            reason=(
                f"Governance is in advisory mode (profile='{governance_profile}'). "
                "All candidates are scored and ranked by governed_score. No "
                "candidates are suppressed in this mode — the caller sees the "
                "full scored set with governance metadata."
            ),
            policy_source=f"config.governance_mode[profile={governance_profile}]",
        ))
    elif governance_mode == "enforced":
        decisions.append(AttentionDecision(
            dimension="governance_gate",
            decision=f"enforced:{governance_profile}",
            reason=(
                f"Governance is in enforced mode (profile='{governance_profile}'). "
                "Vetoed and contradiction-losing candidates are suppressed from "
                "the result set. Survivors are ranked by governed_score."
            ),
            policy_source=f"config.governance_mode[profile={governance_profile}]",
        ))

    # ── 10. Forecast advisory ─────────────────────────────────────────────────
    if forecast_advisory:
        plan = forecast_advisory.get("suggested_plan", "standard")
        pressure = float(forecast_advisory.get("forecast_pressure", 0.0))
        reason_text = forecast_advisory.get("forecast_reason", "")
        confidence = float(forecast_advisory.get("confidence_score", 0.0))
        decisions.append(AttentionDecision(
            dimension="forecast_advisory",
            decision=plan,
            reason=(
                f"Pulse forecast advisory active: suggested_plan='{plan}', "
                f"pressure={round(pressure, 3)}, confidence={round(confidence, 3)}. "
                f"Forecast reason: {reason_text}"
            ),
            policy_source="pulse_engine.advisory",
        ))
    else:
        decisions.append(AttentionDecision(
            dimension="forecast_advisory",
            decision="none",
            reason="No pulse forecast advisory was present for this cycle.",
            policy_source="pulse_engine.advisory",
        ))

    # ── 11. Intent / shadow search ────────────────────────────────────────────
    if shadow_search_present:
        decisions.append(AttentionDecision(
            dimension="shadow_search",
            decision="triggered",
            reason=(
                "Intent trajectory prediction triggered a shadow search. A "
                "speculative pre-fetch was executed for a predicted follow-up "
                "query and stored in the pre-cognitive cache."
            ),
            policy_source="intent_engine.record_and_forecast",
        ))
    if intent_forecast_present:
        decisions.append(AttentionDecision(
            dimension="intent_trajectory",
            decision="forecast_emitted",
            reason=(
                "Intent engine emitted a trajectory forecast. Predicted next "
                "queries from this session's cluster are available for "
                "pre-cognitive cache warming."
            ),
            policy_source="intent_engine.record_and_forecast",
        ))

    # ── 12. Pattern advisory recall ───────────────────────────────────────────
    if pattern_store_enabled:
        if pattern_advisory_count > 0:
            decisions.append(AttentionDecision(
                dimension="pattern_advisory",
                decision="recalled",
                reason=(
                    f"Pattern store advisory recall returned {pattern_advisory_count} "
                    "candidate pattern(s) relevant to this situation. These are "
                    "advisory-only and do not affect retrieval ranking or candidate "
                    "selection. Candidates require explicit governance promotion before "
                    "becoming authoritative."
                ),
                policy_source="pattern_candidate_store.find_relevant",
            ))
        else:
            decisions.append(AttentionDecision(
                dimension="pattern_advisory",
                decision="none",
                reason=(
                    "Pattern store is enabled but no relevant advisory candidates "
                    "were found for this situation (store may be empty or no "
                    "candidates met the similarity threshold)."
                ),
                policy_source="pattern_candidate_store.find_relevant",
            ))
    else:
        decisions.append(AttentionDecision(
            dimension="pattern_advisory",
            decision="disabled",
            reason=(
                "Pattern candidate store is not configured for this deployment "
                "(MNEMOS_PATTERN_STORE_PATH not set). Advisory pattern recall "
                "is unavailable."
            ),
            policy_source="config.pattern_store_path",
        ))

    return decisions
