# Phase 7 Spec — TurboMatryoshka Tiered Retrieval + Budget-Aware Routing

Date: June 10, 2026
Workstreams: W6 (Performance Economics) + W7 (SLO + Reliability)
Status: BudgetAwareRouter implemented (additive, unwired); Matryoshka lane spec'd, blocked on embedding pivot decision

---

## 1. Sequencing decision

**BudgetAwareRouter ships first, Matryoshka migration second.** Rationale:

1. The router is model-agnostic and composes knobs that already exist
   (conditional rerank + circuit breaker, oversample depth, HNSW ef).
2. The Phase 7 gate cannot be measured without per-stage latency accounting —
   the router's cost model *is* that instrumentation.
3. The Nomic re-embed is the expensive, semi-reversible step; it should follow
   the established cutover pattern (shadow → canary) once the orchestration
   exists to evaluate it.
4. `mnemos/retrieval/retrieval_router.py` carries uncommitted MemGraphRAG
   changes on the current branch; wiring waits for that work to land.

## 2. BudgetAwareRouter (implemented)

`mnemos/retrieval/budget_router.py` — pure planning logic, no retrieval calls.

- `latency_budget_ms` (request parameter, planned for `/v1/mnemos/search`)
  resolves to a `StagePlan` via a fixed degradation ladder:
  `drop_rerank → reduce_oversample (3.0→1.5) → reduce_hnsw_ef (128→64) →
  drop_rescore (prefetch-only)`.
- The plan never degrades below prefetch. An unmeetable budget returns a
  runnable plan flagged `budget_infeasible: true` — honest degradation
  (principle #7), not failure.
- `StageCostModel` holds EWMA per-stage estimates seeded with conservative
  priors and updated from observed latencies (the retrieval router already
  measures dense and rerank stage latency in its telemetry).
- Every plan serialises (`to_dict()`) for response meta: budget-degraded
  responses MUST carry `degraded: true` and the `degradation_steps` taken.

Tests: `tests/test_budget_router.py` (8) — ladder order, prefetch floor,
infeasible flagging, EWMA adaptation, serialization contract.

## 3. Matryoshka tiered retrieval (spec)

### Blueprint corrections (vs. June 10 proposal)

1. **The rescore stage is Qdrant-native, not TurboQuant.** TurboQuant 4-bit
   Recall@10 is ~84% vs float32 — a 4-bit rescore stage makes the 98%
   recall-retention criterion unreachable by construction. Architecture:
   one `query_points()` call with a 64-dim named-vector `prefetch`
   (oversampled) rescored against the Qdrant-stored 768-dim vectors
   (optionally Qdrant int8). No second network round trip. TurboQuant
   remains the application-layer at-rest compression — unchanged role.
2. **Embedding pivot is required.** BGE-base-en-v1.5 is not MRL-trained;
   truncated dims are not a usable coarse representation. Default engine for
   the new profile: `nomic-embed-text-v1.5` (MRL-trained, 64→768 nesting,
   already in the supported engine set as the long-context option).

### Profile

`core_memory_appliance_turbomatryoshka` — same 3-container topology as Core;
collection schema adds named vectors `{"dense_64": ..., "dense_768": ...}`.

### Migration tool (next implementation step)

`tools/mnemos_matryoshka_migrate.py`:
- Streams engrams from the existing collection, re-embeds with Nomic v1.5,
  writes both named vectors to a new collection; checkpointed and resumable.
- Non-destructive: source collection untouched (profile-migration pattern,
  whitepaper §13).
- Emits a migration manifest (counts, checksums, embed model/version) for the
  audit trail.
- Shadow validation step: replays a query set against old and new collections
  and reports overlap before any cutover.

## 4. Phase 7 gate

Baseline: current Core profile, BGE 768-dim single-stage, real-corpus
benchmark harness (`benchmarks/run_profile_benchmarks.py` family).

| Criterion | Threshold | Notes |
|---|---|---|
| Search p95 latency reduction | > 40% vs 768-dim baseline | full plan (rescore on), like-for-like top_k |
| Recall retention @10 | > 98% of float32-768 top-10 present after prefetch64 + rescore768 | measured at oversample 3.0; report 1.5 too (budget-degraded operating point) |
| Prefetch-only recall floor | reported, not gated | operators need the number that `drop_rescore` degrades to |
| Relevance parity | MRR / nDCG within noise of baseline | model swap changes semantics, not just speed — BGE→Nomic must not regress quality on the real corpus |
| Budget compliance | p95 of (actual latency ≤ budget) ≥ 0.95 across budget sweep | validates the cost model, not just the happy path |

The relevance-parity criterion is added because the embedding pivot is a
model change, not an optimization: a 40% latency win with a quality
regression is a failed gate.

## 5. Risks

- Nomic v1.5 requires prefix instructions (`search_query:` / `search_document:`);
  the embedding engine wrapper must apply them or quality silently degrades.
- Re-embed cost scales with corpus size; the migration tool must be resumable.
- EWMA priors are guesses until observations arrive; cold-start plans may
  over- or under-degrade. Mitigation: budget responses always disclose
  `estimated_total_ms` so operators can see the model's belief.
