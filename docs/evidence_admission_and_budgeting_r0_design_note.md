# Evidence Admission and Budgeting R0 — Design Note

```text
EVIDENCE_ADMISSION_AND_BUDGETING_R0_AUTHORIZED
READ_ONLY
SHADOW_ONLY
NO_DEFAULT_RETRIEVAL_CHANGE
```

This note is the required pre-implementation deliverable. It is
repository-grounded: every claim below cites the file and function it is
based on. No R0 code has been written yet.

## 1. Existing code paths R0 will observe

- `RetrievalRouter.search()` — `mnemos/retrieval/retrieval_router.py:1178`.
  Single entry point for semantic/hybrid/graph-hybrid-experimental retrieval.
  R0 must not change its signature or behavior (explicit boundary).
- Adaptive query-complexity classification — `mnemos/retrieval/complexity.py`
  (`EmbeddedComplexityClassifier`, `ZeroShotComplexityClassifier`,
  `route_posture_for_label`) plus the router's own wrapping in
  `_classify_complexity` / `_run_complexity_shadow`
  (`retrieval_router.py:413-489`) and `_adaptive_route_for_complexity`
  (`retrieval_router.py:490-541`). Produces `CLASS_A/B/C` + a
  `routing_posture` dict (`retrieval_mode`, `fusion_policy`,
  `budget_strategy`, `graph`, `rerank`, optionally `summary_layer`).
- `BudgetAwareRouter` / `StagePlan` — `mnemos/retrieval/budget_router.py`.
  This is an **execution-stage** budget (prefetch/rescore/rerank/oversample
  factor/HNSW `ef`), driven by `latency_budget_ms` and `complexity_class`. It
  is a different concept from the R0 spec's `candidate_budget` /
  `context_token_budget`, which describe *how much evidence to return*, not
  *how to execute the search*. R0 does not reuse `StagePlan`; it is a
  separate, non-overlapping budget axis.
- `CandidateEnvelopeConfig` / `apply_candidate_envelope` —
  `mnemos/retrieval/candidate_envelope.py`. Post-retrieval narrowing
  (dedupe, per-source cap, hard limit) gated by request-supplied
  `bounded_envelope`, default `enabled=False`. Produces
  `source_distribution`, `source_concentration_ratio`,
  `average_pairwise_similarity` — directly reusable as R0 sufficiency
  signals for "candidate diversity," but **only present when the caller
  requests `bounded_envelope`**; R0 cannot assume this block always exists.
- Low-relevance abstention — `service/app.py:391-412`
  (`_low_relevance_abstention_meta`, static, `score_floor=0.01` default).
  Applied at `service/app.py:1057-1062`: triggers only when the top result
  **and** all of the top 3 are below the floor; on trigger, `results` is set
  to `[]` and `meta["abstention_guard"]` is populated. This is a real
  post-retrieval gate, not a pre-retrieval heuristic — see §3 for why this
  matters to R0's design.
- Retrieval fingerprints — `service/app.py:372-389`
  (`_retrieval_fingerprint`): `collection_snapshot`, `retrieval_profile`
  (executed), `configured_retrieval_profile`, `embedding_model_name`,
  `cache_schema_version`.
- Cache — `mnemos/memory_over_maps/view_cache.py` (`DerivedViewCache`,
  `query_fingerprint`, `build_cache_key`, `governance_state_hash`,
  `lineage_inputs`, `build_retrieval_cache_context`) and its two call sites
  in `service/app.py`: the pre-cognitive fuzzy cache
  (`_pre_cognitive_cache_context` + `fuzzy_pre_cognitive_get`,
  `app.py:876-924`) and the Phase 3/4 derived-view cache
  (`app.py:1110-1192`). See §3.
- Associative Routing E1/E2 — `mnemos/retrieval/associative_shadow/` and
  `mnemos/retrieval/associative_expansion/`. R0 must recognize "E2 eligible"
  without invoking either engine. The cue/tag registries
  (`associative_expansion/fixtures/{cue,tag,source}_registry.json`) are
  read-only JSON R0 can load directly for cue-term recognition, exactly as
  the spec's "recognized Cue/Tag terms" input describes — without calling
  `AssociativeRouter.route()` or `CandidateExpansionEngine.expand()`.
- Source lineage / governance metadata — `Engram.lineage()` (used at
  `candidate_envelope.py:101`, `retrieval_router.py:150-165`) returns
  `artifact_id`/`chunk_id`; provenance fields checked at
  `retrieval_router.py:138-147` (`canonical_source_uri`, `source_uri`,
  `normalized_content_hash`, `seed_identity`, `schema_version`);
  `GovernanceDecision` fields used in `governance_state_hash`
  (`view_cache.py:52-57`): `engram_id`, `conflict_status`, `suppressed`,
  `governed_score`.

## 2. Exact input fields available without new model calls

Pre-retrieval (no embedding/model call required beyond what the request
already implies):

- `query` (raw text), `len(query)`, simple regex/keyword scans over it.
- `top_k`, `filters`, `tiers`, `latency_budget_ms`, `bounded_envelope`,
  `retrieval_mode`/`fusion_policy` if caller-supplied — all already parsed
  in `service/app.py:1949-1965` before `search_documents()` is called.
- Cue/tag-registry membership (static JSON, see §1).
- Known-collection/source scope: `self._collection_snapshot()` and
  `self._retrieval_profile()` (`service/app.py:330-344`, called by
  `_retrieval_fingerprint`) — both string-valued, already computed without
  a model call.
- Cache state: `DerivedViewCache.get()` / `fuzzy_pre_cognitive_get()` can be
  queried read-only without forcing a miss to become a write; freshness is
  TTL + invalidation-flag based (§3), not a new computation.

Reusable from an *already-running* request (not a *new* call, since
`RetrievalRouter.search()` computes them internally on the request's own
behalf regardless of R0):

- The query embedding vector, if R0 runs after `_get_query_vector` —
  `retrieval_router.py:428-433` — already computed once per request for
  adaptive routing/complexity classification and reused (not
  recomputed) by the dense search call.
- `complexity_meta` (`CLASS_A/B/C` + confidence) once
  `_classify_complexity` has run.
- Post-retrieval only: `mode_meta["retrieval_mode"]`,
  `mode_meta["fusion_policy"]`, result `.score` spread, `candidate_envelope`
  block (if requested), `duplicate_suppression` block,
  `low_relevance_abstention` meta, governance `decisions`.

No field in this list requires draft generation, logits, hidden
chain-of-thought, a new embedding call, or network access — consistent with
the "Do not require" list in the authorization.

## 3. What "cache freshness" means in the current implementation

There is **no generic semantic/hybrid result cache** keyed on raw query
text. Two real caches exist, and R0 must describe freshness against them,
not invent a third:

1. **Pre-cognitive fuzzy cache** (`DerivedViewCache` used as a full-response
   cache via `set_pre_cognitive` / `fuzzy_pre_cognitive_get`,
   `view_cache.py:159-238`). Freshness = entry not `invalidated`, not past
   `ttl_seconds` (`DerivedViewCache.get`, `view_cache.py:131-147`), **and**
   `cache_context` match (`normalized_query`, `authorized_scope`,
   `collection_snapshot`, `retrieval_profile`, `embedding_model_name`,
   `seed_snapshot`, `cache_schema_version`,
   `query_normalization_version` — `view_cache.py:31-49`). Scope match
   beyond TTL is therefore multi-dimensional, not just time-based.
2. **Derived-view cache** (Phase 3/4, `app.py:1110-1192`), keyed by
   `build_cache_key(view_type, query_fingerprint, artifact_ids, chunk_ids,
   governance_state_hash, ...)` (`view_cache.py:82-108`). Freshness here is
   TTL plus dependency-tracked invalidation
   (`_match_invalidation_reason`, `view_cache.py:269-305`): a hit can go
   stale because of `source_artifact_updated/deleted`,
   `chunk_set_changed`, `contradiction_cluster_changed`,
   `governance_state_changed`, `lifecycle_state_changed`, or
   `synthesis_config_changed` — not merely elapsed time.

R0's `CACHE_ONLY` / `CACHE_AVAILABLE` signal is therefore: "does a
not-invalidated, not-TTL-expired entry exist in one of these two caches
whose context (collection snapshot, retrieval profile, embedding model,
seed snapshot, normalized query) matches the incoming request" — read-only
inspection of `DerivedViewCache`'s entry map, never a write, never a
forced-miss promotion.

## 4. Existing route names and route-selection behavior

`retrieval_mode` ∈ `{semantic, hybrid, graph_hybrid_experimental}`;
`fusion_policy` ∈ keys of `FUSION_POLICIES`
(`semantic_dominant`/`balanced`/`lexical_dominant`/`qdrant_rrf`, per
`policies/fusion_policies.py`). Selection happens two ways:

- Caller-supplied `retrieval_mode`/`fusion_policy` (passed straight
  through), or
- Adaptive routing (`adaptive_routing_enabled` / per-request
  `adaptive_routing`): `_adaptive_route_for_complexity` maps
  `CLASS_A → semantic/semantic_dominant/aggressive`,
  `CLASS_B → graph_hybrid_experimental (if available) or requested
  mode/balanced/conservative, force_rerank`,
  `CLASS_C → hybrid (if lexical available)/lexical_dominant/balanced,
  summary_layer=required_with_fallback`.

This vocabulary is orthogonal to the R0 spec's admission-route vocabulary
(`NO_RETRIEVAL`, `CACHE_ONLY`, `CUE_ONLY_LOOKUP`, `SEMANTIC_RETRIEVAL`,
`HYBRID_RETRIEVAL`, `ASSOCIATIVE_EXPANSION_ELIGIBLE`,
`ABSTAIN_OR_REQUEST_SCOPE`). **R0 must keep its own field names
(`recommended_route` etc.) and never write into `retrieval_mode` /
`fusion_policy` / `routing_posture`**, to avoid the two systems being
confused with each other in the response payload or in telemetry.

## 5. Proposed R0 module location and API

Matches the spec exactly; also matches the established E1/E2 package shape
(`config.py` + logic module + `fixtures/`):

```text
mnemos/retrieval/evidence_admission/
  __init__.py        # exports recommend_admission, AdmissionRecommendation, etc.
  models.py           # AdmissionRequestContext, AdmissionRecommendation, dataclasses
  feature_extraction.py  # pure functions: query -> deterministic feature snapshot
  policy.py            # the rule table from the spec; pure function of features
  budget.py            # route -> {candidate_budget, context_token_budget, ...} defaults
  sufficiency.py        # post-retrieval SUFFICIENT/INSUFFICIENT/AMBIGUOUS/OUT_OF_SCOPE
  telemetry.py          # comparison-record assembly + redaction
  fixtures/              # read-only copies/symlink-equivalents are not used;
                          # R0 reads associative_expansion's frozen cue/tag
                          # registries directly (no fixture duplication) unless
                          # a frozen R0-specific corpus is needed for the
                          # evaluation packs (see §8) — kept here if so.
```

Public API:

```python
def recommend_admission(
    query: str,
    request_context: AdmissionRequestContext,
) -> AdmissionRecommendation:
    """Pure, deterministic. No I/O beyond read-only cache/registry inspection
    already passed in via request_context. Never raises into the caller —
    internal errors degrade to status="unavailable"."""
```

`AdmissionRequestContext` carries exactly the §2 fields (query length is
derived, not passed separately) plus optionally-supplied:
`cache_lookup: Optional[CacheLookupResult]` (caller performs the read-only
cache check and hands in the result, so this module never imports
`DerivedViewCache` directly — keeps it import-safe and usable without a
running service, per the spec's "must be usable without a running service"
requirement) and `complexity_meta: Optional[Dict]` (caller passes through
whatever `_classify_complexity` already produced, so R0 never invokes the
classifier itself).

A second, separate entry point for the post-retrieval half:

```python
def assess_sufficiency(
    recommendation: AdmissionRecommendation,
    results: List[SearchResult],
    mode_meta: Dict[str, Any],
) -> SufficiencyAssessment
```

This two-call split (admit-before, assess-after) resolves an ambiguity in
the spec: "R0 Policy Inputs" lists both pre-retrieval fields (query length,
cue terms, cache state) and post-retrieval fields (normal retrieval score
spread, candidate diversity) under one heading, but the architecture diagram
places "shadow route recommendation" and "shadow budget recommendation"
*before* "normal MNEMOS retrieval continues unchanged." Splitting
admission (pre) from sufficiency (post) lets the route/budget
recommendation stay genuinely pre-retrieval (and therefore a meaningful
"would we have needed to retrieve at all" signal) while still using
real post-retrieval signals for the sufficiency/comparison telemetry the
spec also asks for. **Flagging this for explicit confirmation in §7.**

## 6. Proposed shadow response schema

Matches the spec's service-integration block, computed in two pieces and
merged:

```json
{
  "evidence_admission_shadow": {
    "status": "recommended|abstained|unavailable",
    "recommended_route": "cue_only_lookup",
    "candidate_budget": 2,
    "context_token_budget": 600,
    "expansion_budget": 0,
    "latency_budget_ms": 100,
    "stop_condition": "minimum_evidence_satisfied",
    "reason_codes": ["EXPLICIT_ARTIFACT_ID_MATCH"],
    "sufficiency": "sufficient",
    "sufficiency_reason_codes": [],
    "input_snapshot": "sha256:...",
    "latency_ms": 0.0,
    "non_authoritative": true
  }
}
```

`sufficiency` and `sufficiency_reason_codes` are only populated once
`assess_sufficiency` has run (i.e., after normal retrieval); until then
they are `null`. `input_snapshot` is a stable hash of the deterministic
feature snapshot (§2 fields), used by the "deterministic recommendation for
same input snapshot" required test.

## 7. Conflicts / open questions to resolve before implementation

1. **A second "R0" already exists in this repository**:
   `benchmarks/results/retrieval_hygiene_r0_closeout.md` documents
   "Retrieval Hygiene and Reproducibility R0"
   (`RETRIEVAL_HYGIENE_AND_REPRODUCIBILITY_R0_COMPLETE`), which already
   covers low-relevance abstention, executed-route fingerprinting, cache
   freshness for the pre-cognitive cache, and duplicate suppression — the
   exact same primitives this preflight's "Required Reading" section asks
   me to identify. This is not a code conflict (different module, no
   overlapping files), but it is a **naming collision**: two unrelated,
   independently-closed "R0"s will both exist in `docs/` and `tests/`.
   I'm naming this effort's artifacts `evidence_admission_and_budgeting_r0_*`
   throughout (this file, the future module, future docs) to keep them
   textually distinct from `retrieval_hygiene_r0_*`. Flagging in case you
   want a different disambiguating name.
2. **Pre- vs. post-retrieval split** (§5): the spec's input list and
   architecture diagram are not fully consistent about whether
   route/budget recommendation happens before or after normal retrieval
   runs. I'm proceeding with the two-call split (`recommend_admission`
   before, `assess_sufficiency` after) since it's the only reading under
   which "no new retrieval calls" and "uses normal retrieval score spread"
   are both simultaneously true. Please confirm or redirect.
3. **No kill-switch env var is added.** E1/E2 both pair their request flag
   with a global `MNEMOS_*_ENABLED` kill switch because they can mutate the
   delivered response. R0 never mutates the delivered response (shadow-only,
   additive metadata block only), so I'm not adding one — the request flag
   alone is sufficient and matches the spec's literal "Service Integration"
   section, which specifies only `evidence_admission_shadow`. Flagging
   since this is a deviation from the E1/E2 precedent, in case you want
   parity for ops/runbook consistency.
4. No other boundary conflicts found. `RetrievalRouter.search()`,
   governance, authority, disclosure, promotion, deletion, and MCP/MSF
   configuration are all untouched by this design.

## 8. Exact test files to add or extend

New:

- `tests/test_evidence_admission_r0.py` — policy rule table, budget
  defaults, sufficiency signals, determinism, redaction, candidate/context
  budget bounds, no-durable-write, no-governance-mutation (mirrors the
  structure of `tests/test_associative_routing_e2_expansion.py`'s
  `TestEngineIsolation` / `TestRequestLayerWiring` / `TestRuntimeWiring`
  split).

Extend (regression-safety only, no behavior change to existing assertions):

- `tests/test_service_hybrid_api.py` — add fakes/assertions for the new
  `evidence_admission_shadow` request flag, following the existing pattern
  used for `associative_routing_shadow`/`associative_candidate_expansion`.

Retained unchanged as regression gates:

- `tests/test_associative_routing_e0.py`
- `tests/test_associative_routing_e1_shadow.py`
- `tests/test_associative_routing_e2_expansion.py`
- `tests/test_retrieval_hygiene_r0.py`

---

Awaiting confirmation on §7 items 1–3 before writing any R0 code.
