# MNEMOSv2 Memory Over Maps — Task Board and Sprint Breakdown

## Purpose

This task board translates the Memory Over Maps architecture plan into implementation-ready work for the AI Dev. It assumes the following are already accepted as fixed:

- Governance is the strongest validated differentiator.
- Core/Qdrant remains the operational default.
- Reranker remains experimental / non-production.
- Hybrid retrieval remains available but not the broad default.
- Memory Over Maps is being adopted as a structural architecture pattern, not as another retrieval-feature experiment.

This board is organized into epics, sprints, ticket groups, acceptance criteria, dependencies, and exit gates.

---

## Program Structure

### Epic 1 — Source-Grounded Artifact Model
Create the canonical memory model that distinguishes source truth from indexed fragments and derived outputs.

### Epic 2 — Bounded Candidate Envelope
Introduce deterministic narrowing before expensive reasoning and synthesis.

### Epic 3 — On-Demand Derived Views
Add query-scoped, source-grounded views such as evidence bundles and contradiction bundles.

### Epic 4 — Cache and Invalidation
Add safe reuse of expensive derived views without allowing cache to become silent source-of-truth.

### Epic 5 — Benchmarks and E2E Proof
Validate correctness, bounded-cost behavior, and promotion readiness with reproducible artifacts.

---

## Sprint 0 — Alignment and Scaffold

### Goal
Prepare the codebase and interface contracts so the next sprints can land cleanly.

### Tickets
- Review existing engram schema, governance metadata, lineage fields, and retrieval path.
- Identify where source, chunk, and derived representations are currently conflated.
- Create feature flag namespace for Memory Over Maps rollout.
- Define temporary package/module structure for new memory model code.
- Add stub stats surfaces for future counters.

### Deliverables
- Architecture delta note
- File/module placement proposal
- Feature flag list
- Empty stats keys added behind safe defaults

### Acceptance
- No behavior changes in existing retrieval path
- No regression in current governance tests
- New feature flags compile and default to off

### Exit Gate
- Clean scaffold merged
- All current tests still passing

---

## Sprint 1 — SourceArtifact and Chunk Lineage

### Goal
Make source-grounded memory explicit.

### Ticket Group A — SourceArtifact Schema
- Create `SourceArtifact`
- Required fields:
  - `artifact_id`
  - `source_uri`
  - `source_type`
  - `content_hash`
  - `created_at`
  - `updated_at`
  - `metadata`
  - `lifecycle_state`
  - `version`
- Define immutability expectations for source artifacts

### Ticket Group B — Chunk Schema
- Create `Chunk`
- Required fields:
  - `chunk_id`
  - `artifact_id`
  - `chunk_index`
  - `text`
  - `embedding_ref`
  - `metadata`
  - `provenance_span`
  - `created_at`
- Ensure every chunk traces back to exactly one `SourceArtifact`

### Ticket Group C — Lineage Contract
- Define lineage fields for all retrieval results
- Ensure every current engram-hit can surface:
  - source artifact ID
  - chunk ID
  - provenance/source URI
- Add contract tests

### Deliverables
- New schemas
- Serialization/deserialization layer
- Lineage contract tests
- Migration note for legacy engrams

### Acceptance
- Every retrieved chunk can point back to a source artifact
- No orphan chunks
- Existing retrieval still works under compatibility shim

### Exit Gate
- Schema and lineage tests green
- Regression suite green

---

## Sprint 2 — DerivedView Base Model

### Goal
Introduce the explicit distinction between canonical stored memory and computed views.

### Ticket Group A — DerivedView Schema
- Create `DerivedView`
- Required fields:
  - `view_id`
  - `view_type`
  - `input_artifact_ids`
  - `input_chunk_ids`
  - `query_fingerprint`
  - `governance_state_hash`
  - `generated_at`
  - `cacheable`
  - `reproducible`
  - `invalidated_at`
  - `metadata`

### Ticket Group B — View Reproducibility Rules
- Define how to reconstruct a view from source artifacts + chunks + query + governance state
- Ensure derived views are not treated as canonical memory records

### Ticket Group C — Compatibility Layer
- Keep existing read path stable while introducing derived views in parallel
- Add explain/debug output indicating whether a response element is source, chunk, or derived

### Deliverables
- DerivedView base model
- Reproducibility contract
- Compatibility layer
- Debug/explain surfaces

### Acceptance
- Derived views can be generated without mutating canonical source/chunk records
- Derived views can be discarded without data loss
- Existing search/index behavior unaffected when feature flags are off

### Exit Gate
- DerivedView contract merged
- Explain/debug tests green

---

## Sprint 3 — Candidate Pool Limiter and Dedupe

### Goal
Bound cost before expensive reasoning.

### Ticket Group A — Candidate Pool Limiter
- Add `candidate_pool_limit`
- Add `pre_governance_top_n`
- Add stats for retrieved vs narrowed pool sizes

### Ticket Group B — Dedupe and Redundancy Reduction
- Add configurable similarity threshold for duplicate suppression
- Collapse near-identical chunk candidates
- Preserve at least one representative candidate from each duplicate cluster

### Ticket Group C — Max Per SourceArtifact
- Prevent narrowed pool domination by one artifact
- Add `max_per_source_artifact`

### Deliverables
- Narrowing module
- Dedupe module
- Source-family balancing logic
- Stats counters

### Acceptance
- Narrowed pool is smaller than raw retrieved pool on representative workloads
- Narrowing is deterministic under same input/state
- Evidence-supporting candidates are not unintentionally eliminated in benchmark fixtures

### Exit Gate
- Narrowing benchmark baseline saved
- No regression in answer-support retention tests

---

## Sprint 4 — Diversity-Aware Narrowing and Bounded Adjudication Envelope

### Goal
Improve narrowing quality without turning it into hidden reranking complexity.

### Ticket Group A — Diversity Policy
- Optional diversity-aware narrowing
- Candidate selection across:
  - source artifact families
  - chunk neighborhoods
  - metadata groups
- Keep deterministic behavior

### Ticket Group B — Bounded Adjudication Envelope
- Formalize retrieval flow:
  - retrieve
  - narrow
  - govern/adjudicate
  - synthesize on demand
- Prevent synthesis from operating on unbounded raw pools

### Ticket Group C — Instrumentation
- Add:
  - compression ratio
  - duplicate suppression rate
  - max-per-source hit rate
  - pre/post-governance reorder stats

### Deliverables
- Diversity-aware narrowing mode
- Bounded adjudication routing implementation
- Instrumentation output

### Acceptance
- Envelope is enforced in the read path when enabled
- Bounded set size is visible in stats and explain output
- Governance still operates on a meaningful candidate set

### Exit Gate
- Envelope tests pass
- Stats visible and stable

---

## Sprint 5 — Evidence Bundle v1

### Goal
Ship the first on-demand derived view.

### Ticket Group A — EvidenceBundle Schema
- Create `EvidenceBundle` as a typed `DerivedView`
- Include:
  - supporting artifact IDs
  - supporting chunk IDs
  - ranked evidence list
  - governance decisions used
  - contradiction notes if present

### Ticket Group B — Query-Scoped Assembly
- Generate evidence bundle only when requested or policy-enabled
- Keep it source-grounded and reproducible

### Ticket Group C — Explain Output
- Return evidence bundle references in explain/debug mode
- Ensure no permanent storage required unless explicitly cached

### Deliverables
- EvidenceBundle model
- Query-time assembly path
- Explain-mode support

### Acceptance
- Bundle references only source/chunk records that actually participated in the answer path
- Bundle can be regenerated deterministically
- No silent persistence by default

### Exit Gate
- Evidence bundle fixtures pass
- Explain output verified

---

## Sprint 6 — Contradiction Bundle and Preference Snapshot

### Goal
Add higher-value query-time views using existing governance truth.

### Ticket Group A — ContradictionBundle
- Capture contradiction clusters
- Show winner/loser state
- Include reasons/modifiers used

### Ticket Group B — PreferenceSnapshot
- Query-time view over preference-like records or high-utility signals
- Must remain source-grounded
- No autonomous canonization

### Ticket Group C — Validation Fixtures
- Build bundle-specific fixtures from governance validation scenarios
- Ensure contradiction losers cannot appear as silent co-equal support

### Deliverables
- ContradictionBundle
- PreferenceSnapshot
- Bundle validation fixtures

### Acceptance
- Bundle reflects actual contradiction policy state
- Snapshot is reproducible from underlying records
- No contradiction incoherence in derived outputs

### Exit Gate
- Contradiction view tests green
- Governance pack still accurate

---

## Sprint 7 — Timeline Summary

### Goal
Create a useful temporal derived view without introducing permanent summary debt.

### Ticket Group A — TimelineSummary
- Build query-scoped ordered timeline over source/chunk evidence
- Require provenance and timestamps
- Surface uncertainty where ordering is incomplete

### Ticket Group B — Time-Based Grouping Rules
- Group by artifact timestamps or record timestamps
- Document tie and missing-time behavior

### Ticket Group C — Explain and Guardrails
- Summary must cite underlying source/chunk IDs internally
- Must not replace source evidence

### Deliverables
- TimelineSummary view
- Time-ordering rules
- Explain surfaces

### Acceptance
- Timeline summary is reproducible
- Missing timestamps handled explicitly
- No hidden source elision

### Exit Gate
- Temporal fixtures pass
- No provenance regression

---

## Sprint 8 — Cache and Invalidation v1

### Goal
Reuse expensive derived views safely.

### Ticket Group A — Cache Key Design
- Cache key must include:
  - query fingerprint
  - input artifact IDs
  - input chunk IDs
  - governance state hash
  - view type

### Ticket Group B — Invalidation Rules
- Invalidate on:
  - source artifact update
  - chunk set change
  - contradiction state change
  - lifecycle change
  - deletion/tombstone
  - governance state hash change

### Ticket Group C — Dry-Run Invalidation
- Add dry-run mode
- Emit invalidation trace logs
- No destructive operations

### Deliverables
- Cache layer
- Invalidation engine
- Dry-run audit mode

### Acceptance
- Cached view never survives state changes that should invalidate it
- Dry-run logs are understandable and deterministic
- Cache can be disabled cleanly

### Exit Gate
- Cache correctness fixtures pass
- Invalidation audit output reviewed

---

## Sprint 9 — Benchmarks and E2E Proof Pack

### Goal
Create the promotion evidence needed for wider rollout.

### Ticket Group A — Benchmarks
- Candidate narrowing benchmark
- Derived view generation latency benchmark
- Cache hit/miss benchmark
- Invalidation correctness benchmark

### Ticket Group B — E2E Scenarios
- Source-grounded answer generation
- Contradiction bundle under active conflicts
- Timeline summary with mixed evidence freshness
- Cache invalidation after source update

### Ticket Group C — Promotion Report
- Summarize:
  - cost reduction
  - answer-support retention
  - contradiction correctness
  - cache safety
  - known limits

### Deliverables
- Benchmark artifacts
- E2E artifact pack
- Promotion decision memo

### Acceptance
- Evidence pack complete
- No hidden benchmark regressions
- Known gaps explicitly named

### Exit Gate
- Ready for advisory rollout expansion

---

## Cross-Cutting Requirements

### Mandatory Guardrails
- No destructive delete behavior in this program lane
- No hidden permanent structure creation
- No silent cache canonization
- No bypass around governance when envelope is enabled
- No promotion based on intuition without benchmark evidence

### Required Stats Additions
- `candidate_pool_raw_count`
- `candidate_pool_narrowed_count`
- `candidate_duplicate_suppressed_count`
- `candidate_source_cap_applied_count`
- `derived_view_generated_count`
- `derived_view_cache_hit_count`
- `derived_view_cache_miss_count`
- `derived_view_invalidated_count`

### Required Explain/Debug Fields
- source artifact IDs
- chunk IDs
- narrowing reason
- dedupe reason
- governance state hash
- derived view type
- cache hit/miss
- invalidation reason if applicable

---

## Risks to Watch

### Risk 1 — Narrowing becomes stealth reranking
Mitigation:
- keep deterministic policies first
- keep explain fields visible
- avoid model-heavy narrowing in this phase

### Risk 2 — Derived views become de facto canonical state
Mitigation:
- explicit schema separation
- cache invalidation
- reproducibility requirements
- no permanent persistence by default

### Risk 3 — Envelope reduces recall too aggressively
Mitigation:
- answer-support retention benchmark
- source-cap tuning
- diversity-aware fallback

### Risk 4 — Cache creates stale truth
Mitigation:
- governance state hash in cache keys
- lifecycle/contradiction invalidation triggers
- dry-run audit mode

---

## Definition of Done for the Program

The Memory Over Maps lane is done for v1 when:

1. Source artifacts, chunks, and derived views are explicitly separated.
2. Candidate narrowing is deterministic, benchmarked, and visible in stats.
3. Evidence bundle, contradiction bundle, and at least one temporal or preference-oriented view exist.
4. Cache/invalidation works with dry-run auditability.
5. E2E proof artifacts show bounded-cost synthesis without loss of source grounding.
6. Whitepaper claims can be updated from evidence, not architecture intent alone.
