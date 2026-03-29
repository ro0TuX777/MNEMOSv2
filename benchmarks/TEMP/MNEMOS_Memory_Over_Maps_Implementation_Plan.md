# MNEMOS Lightweight Memory Architecture Implementation Plan
## Memory-Over-Maps-Inspired Adaptation

## 1. Objective

Build a **lightweight memory architecture layer** inside MNEMOS that prioritizes:

- source-grounded memory over heavy precomputed structure
- cheap first-stage retrieval over expensive global reasoning
- selective expensive adjudication only on narrowed candidate sets
- on-demand synthesis instead of universal up-front synthesis

The goal is to reduce overbuilt intermediate representations and make MNEMOS more efficient, explainable, and scalable while preserving retrieval quality.

---

## 2. Product Thesis

### Current issue
MNEMOS has already learned, through benchmarking, that:
- Core/Qdrant is the practical default
- Governance/pgvector has not proven broad retrieval superiority
- reranking is still experimental/non-production
- hybrid mode exists but did not justify a global default switch

This suggests a product lesson:
- do not overbuild structure before it proves value

### Target state
MNEMOS should prefer:
- a **lightweight source-grounded memory bank**
- strong indexing
- staged retrieval
- expensive reasoning only where justified
- targeted synthesis of answers or memory views on demand

---

## 3. Scope

### In scope
- memory-bank-first architecture inside existing MNEMOS profiles
- source-grounded artifact retention
- staged candidate narrowing
- selective expensive reasoning on small candidate sets
- on-demand synthesis of derived memory views
- retrieval cost and storage trade-off benchmarks
- telemetry on narrowing effectiveness

### Out of scope
- robotics-specific RGB-D localization
- navigation policies
- rebuilding Track 2 reranker as a default path
- replacing governance work from the MemArchitect plan
- mandatory global graph construction or scene-style reconstruction

---

## 4. Design Principle

The guiding principle is:

> **Store the minimum useful source-grounded memory, retrieve a narrow candidate set quickly, and only then spend expensive reasoning or synthesis on those candidates.**

This means MNEMOS should resist converting all content into heavy permanent derived structures unless benchmarks prove they are worth the build/storage cost.

---

## 5. Architecture Concept

Implement a layered memory architecture with four stages:

### A. Source Memory Bank
The primary store keeps:
- original documents
- chunks
- source metadata
- embeddings
- provenance links

This remains the default truth substrate.

### B. Fast Candidate Retrieval
A cheap retrieval stage narrows the working set using:
- vector search
- lexical search where enabled
- profile-aware filters
- optional hybrid retrieval if configured

### C. Selective Adjudication
Only the top candidate pool receives:
- governance checks
- relevance veto
- expensive scoring
- optional fusion or synthesis
- optional multi-step reasoning

### D. On-Demand Derived Views
Instead of permanent universal structuring, create temporary or cached derived views only when needed:
- task summary
- contradiction view
- preference snapshot
- evidence bundle
- answer-support pack

---

## 6. Proposed Modules

```text
mnemos/lightweight_memory/
├── memory_bank.py
├── candidate_retriever.py
├── candidate_narrower.py
├── adjudication_router.py
├── on_demand_synthesizer.py
├── view_cache.py
├── policies/
│   ├── narrowing_policy.py
│   ├── synthesis_policy.py
│   ├── cache_policy.py
│   └── source_grounding_policy.py
├── models/
│   ├── source_artifact.py
│   ├── candidate_set.py
│   ├── derived_view.py
│   └── adjudication_trace.py
└── telemetry/
    ├── retrieval_cost_metrics.py
    └── synthesis_metrics.py
```

---

## 7. Core Data Model Additions

Each memory artifact should explicitly distinguish between:

### Source-grounded artifacts
- original document or chunk
- canonical provenance
- durable storage object

### Derived views
- summary
- semantic fact
- contradiction report
- entity snapshot
- answer-support pack

Recommended fields:

- `artifact_id`
- `artifact_type` (`source`, `chunk`, `derived_view`, `summary`, `evidence_pack`)
- `source_id`
- `derived_from`
- `creation_mode` (`stored`, `on_demand`, `cached`)
- `persistence_policy`
- `cost_to_build_ms`
- `reuse_count`
- `cache_state`
- `grounding_required: bool`

This keeps the system honest about what is original memory vs what is synthesized.

---

## 8. Phase 1 Capabilities

## Capability 1 — Source-grounded memory bank
Goal:
- keep canonical memory anchored to original sources

Implementation:
- ensure answerable artifacts can always be traced back to original chunks/documents
- avoid making derived summaries the only accessible representation

### Output
- stable source-grounded storage model
- provenance-first retrieval support

---

## Capability 2 — Candidate narrowing
Goal:
- reduce the amount of memory subjected to expensive logic

Implementation:
- first-stage vector/lexical retrieval returns top-N
- dedupe redundant candidates
- optionally enforce diversity across source/document/entity
- emit a narrowed candidate pool for adjudication

### Output
- `candidate_pool`
- `narrowing_trace`
- `dedupe_stats`

---

## Capability 3 — Selective expensive reasoning
Goal:
- only use expensive governance/synthesis on the narrowed pool

Implementation:
- expensive passes are blocked from operating on the full corpus by default
- only run on:
  - top-N candidates
  - top candidates per source or entity
  - explicit evaluation/debug modes

### Output
- bounded adjudication workload
- cost-aware reasoning path

---

## Capability 4 — On-demand synthesis
Goal:
- avoid permanent heavy structure when temporary synthesis will do

Implementation:
- generate derived views only when triggered by task type or query type
- examples:
  - contradiction bundle
  - preference summary
  - timeline summary
  - evidence pack
  - policy view

### Output
- `derived_view`
- `derived_view_metadata`
- cache and reuse behavior

---

## Capability 5 — Lightweight caching
Goal:
- preserve useful synthesized outputs without turning them into default permanent structure

Implementation:
- derived views can be cached
- cache expiry depends on:
  - source churn
  - freshness
  - reuse count
  - governance invalidation

### Output
- `cache_hit`
- `cache_miss`
- `cache_invalidated`

---

## 9. Retrieval Flow Design

### Input
- query
- retrieval mode
- optional governance mode
- optional synthesis hint
- token budget

### Flow
1. retrieve source-grounded candidates
2. dedupe and narrow pool
3. send narrowed pool to governance and/or adjudication
4. if task requires it, synthesize an on-demand derived view
5. return governed answer context plus optional derived artifact

### Output
- source-backed context set
- optional derived view
- adjudication trace
- synthesis trace

---

## 10. On-Demand Synthesis Types

Recommended initial derived views:

### View A — Evidence bundle
A compact set of supporting chunks with provenance and ranking rationale.

### View B — Contradiction bundle
A grouped view of conflicting memory claims plus current winner/suppressed items.

### View C — Preference snapshot
A synthesized state view for user preferences or profile-like data.

### View D — Timeline summary
A compact temporal summary assembled only when temporal reasoning is requested.

### View E — Task answer-support pack
A transient pack that explains why the selected memories were used.

These should not all be permanently stored by default.

---

## 11. Cost Controls

The main value of this architecture is cost discipline.

### Build-time controls
- avoid universal graph construction
- avoid universal summary generation
- avoid permanent precomputed heavy views without justification

### Query-time controls
- expensive synthesis only after narrowing
- optional timeout or budget cap per query
- configurable maximum adjudication pool size

### Cache controls
- store only high-reuse derived views
- invalidate on source mutation or delete cascade

---

## 12. API Surface Changes

### `/search`
Add:
- `candidate_pool_limit`
- `allow_on_demand_synthesis: true | false`
- `derived_view_type`
- `include_adjudication_trace: true | false`

### `/memory/view`
New endpoint or equivalent internal API to request an on-demand derived view.

### `/memory/cache`
Optional introspection endpoint for:
- cached derived views
- invalidation state
- reuse counts

### `/capabilities`
Expose:
- source-grounded memory bank support
- available derived view types
- synthesis support
- cache support

---

## 13. Benchmark Plan

### Benchmark question
Does a lightweight memory-bank-first architecture reduce build/storage cost while preserving or improving task-quality compared to heavier always-derived structures?

### Compare
- source-grounded retrieval only
- source-grounded + selective adjudication
- source-grounded + on-demand synthesis
- heavy precomputed view baseline (where available)

### Query classes
1. direct fact lookup
2. preference/state lookup
3. contradiction-heavy queries
4. temporal-summary queries
5. explanation/evidence-needed queries
6. long-document multi-hop queries

### Metrics
- build time
- storage footprint
- retrieval latency
- adjudication latency
- synthesis latency
- answer accuracy
- source-grounding rate
- cache hit rate
- unnecessary synthesis rate
- cost per successful answer

### Success criteria
The architecture passes if it:
- lowers build/storage cost relative to heavier always-derived approaches
- keeps acceptable answer quality
- demonstrates that expensive synthesis can be confined to a small candidate pool
- provides clearer source grounding than heavy precomputed abstraction-first workflows

---

## 14. LoE Estimate

For one strong AI Dev:

### Phase 1 — source-grounded memory bank cleanup and candidate narrowing
1.5 to 2 weeks

### Phase 2 — on-demand synthesis engine + derived view model
1 to 1.5 weeks

### Phase 3 — lightweight cache and invalidation rules
4 to 6 days

### Phase 4 — benchmark track and trade-off reporting
1 to 1.5 weeks

### Phase 5 — docs and hardening
3 to 5 days

### Total estimated LoE
~4 to 5.5 weeks for a strong v1.

---

## 15. Risks

### Risk 1 — too little structure
If the system stays too raw, some tasks may suffer from repeated synthesis cost.

Mitigation:
- cache high-value derived views
- benchmark reuse

### Risk 2 — candidate narrowing removes needed evidence
Mitigation:
- keep candidate pool limits configurable
- benchmark false-negative narrowing failures

### Risk 3 — on-demand synthesis becomes a hidden expensive default
Mitigation:
- enforce explicit synthesis triggers
- log synthesis rate
- monitor unnecessary synthesis

### Risk 4 — source-grounded approach feels less “intelligent” than always-structured systems
Mitigation:
- emphasize explainability and provenance
- provide targeted derived views where they actually help

---

## 16. Recommended Implementation Order

### Wave 1
- source-grounded artifact model
- candidate retriever
- candidate narrower
- adjudication router hooks

### Wave 2
- on-demand synthesizer
- evidence bundle
- contradiction bundle
- preference snapshot

### Wave 3
- lightweight cache
- cache invalidation on source changes and deletes
- synthesis telemetry

### Wave 4
- benchmark track
- trade-off evaluation
- operator guidance

---

## 17. Acceptance Criteria

Memory-Over-Maps-inspired v1 is complete when:

- MNEMOS distinguishes source-grounded artifacts from derived views
- retrieval narrows a bounded candidate pool before expensive reasoning
- on-demand synthesis can generate at least three useful derived view types
- expensive synthesis is not required for broad default operation
- cache rules exist for reusable derived views
- benchmark artifacts show build/storage/query trade-offs
- docs explain when to use raw source-grounded retrieval vs on-demand synthesis

---

## 18. Product Posture After This Phase

If successful, MNEMOS can be described as:

- **Core retrieval substrate**: fast, source-grounded memory access
- **Governance layer**: adjudicates what should surface
- **Lightweight memory architecture**: avoids overbuilt intermediate structure
- **On-demand synthesis**: creates just enough derived structure only when needed

That is a cleaner and more benchmarkable story than trying to precompute every possible semantic structure.

---

## 19. Recommendation

Build this **after** the MemArchitect-inspired governance layer.

Reason:
- governance decides what is trustworthy and admissible
- lightweight memory architecture then decides how little structure MNEMOS can get away with while preserving quality

That is the right order.
