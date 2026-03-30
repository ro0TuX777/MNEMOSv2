# MNEMOSv2 — Memory Over Maps Phased Approach

**Audience:** AI Dev / project lead  
**Purpose:** executable phased roadmap for Memory Over Maps integration into MNEMOSv2.

---

## Phase 1 — Artifact Model Refactor

### Goal
Create a clean source-grounded memory model where source truth, retrieval units, and synthesized views are explicitly separated.

### Why first
Without this, MNEMOS risks continuing to treat chunks or summaries as semi-canonical representations. That would undermine the intended architectural shift.

### Deliverables

- `SourceArtifact` root schema
- `SourceArtifactVersion` or version metadata contract
- `Chunk` schema with stable source lineage
- `DerivedView` base schema
- `EvidenceBundle` schema
- lineage contract covering:
  - source artifact IDs
  - chunk IDs
  - governance decision IDs / hashes
  - synthesis parameters
- audit extensions for view generation events

### Required fields

#### SourceArtifact
- `artifact_id`
- `artifact_type`
- `source_uri`
- `content_hash`
- `created_at`
- `updated_at`
- `source_authority`
- `lifecycle_state`
- `metadata`

#### Chunk
- `chunk_id`
- `artifact_id`
- `artifact_version`
- `chunk_index`
- `chunk_hash`
- `embedding_ref`
- `lexical_ref`
- `governance_ref`
- `metadata`

#### DerivedView
- `view_id`
- `view_type`
- `inputs`
- `query_fingerprint`
- `governance_state_hash`
- `cacheable`
- `created_at`
- `invalidated_at`
- `reproducible`

### Acceptance criteria

- Every chunk is traceable to exactly one source artifact version.
- Derived views cannot exist without declared inputs.
- Audit ledger records derived-view creation.
- Existing search/index operations still function.
- Backward compatibility story documented for legacy engrams.

### Risks

- legacy data migration complexity
- accidental duplication of source/chunk semantics
- breaking read-path assumptions in existing APIs

### Exit gate

Phase 1 is complete only when the system can answer:

> For this returned chunk or summary, which source artifact version did it come from?

without ambiguity.

---

## Phase 2 — Bounded Candidate Envelope

### Goal
Introduce candidate narrowing before expensive reasoning and synthesis while preserving answer support and governance quality.

### Why second
This is the highest-value transferable pattern from Memory Over Maps. It bounds cost, improves interpretability, and prevents expensive post-retrieval logic from touching the full memory pool.

### Deliverables

- candidate pool limiter
- dedupe/redundancy suppression
- max-per-source-artifact limiter
- optional diversity-aware narrowing
- bounded adjudication route
- instrumentation for pre/post narrowing diagnostics

### Key controls

- `candidate_pool_limit`
- `dedupe_similarity_threshold`
- `max_per_source_artifact`
- `diversity_policy`
- `bounded_adjudication_enabled`

### Candidate-envelope flow

1. retrieve initial candidates
2. apply dedupe
3. apply source balancing
4. apply diversity policy if enabled
5. produce bounded envelope
6. pass bounded envelope into governance
7. synthesize only from post-governance survivors

### Acceptance criteria

- bounded candidate envelope is deterministic given fixed inputs and config
- narrowed set retains answer-supporting evidence above threshold target
- governance behavior remains explainable after narrowing
- stats surface exposes pre/post counts and suppression reasons

### Risks

- over-narrowing reduces recall
- duplicate suppression hides legitimate multi-chunk evidence
- diversity logic adds too much complexity too early

### Exit gate

Phase 2 is complete only when benchmark runs show:

- meaningful candidate reduction,
- bounded latency/cost improvement,
- acceptable evidence retention,
- no silent governance bypass.

---

## Phase 3 — On-Demand Derived Views

### Goal
Create richer outputs only when needed and only from source-grounded, governed inputs.

### Initial derived views

1. `EvidenceBundle`
2. `ContradictionBundle`
3. `PreferenceSnapshot`
4. `TimelineSummary`

### Deliverables

- view builder interface
- per-view schema definitions
- reproducibility contract
- source/input declaration on every view
- view-generation audit events

### View specifics

#### EvidenceBundle
Purpose: collect supporting source artifacts/chunks for an answer.

#### ContradictionBundle
Purpose: show winner/loser evidence and contradiction reasoning.

#### PreferenceSnapshot
Purpose: summarize currently favored memory state for a user/entity/topic.

#### TimelineSummary
Purpose: synthesize ordered change/revision signals from source-grounded artifacts.

### Acceptance criteria

- every derived view can be regenerated from declared inputs
- no derived view becomes mandatory for retrieval correctness
- users can inspect input lineage for each view
- governance decisions used in synthesis are surfaced

### Risks

- overproducing views
- hidden business logic in view builders
- accidental dependence on cached views for correctness

### Exit gate

Phase 3 is complete only when a generated view can be deleted and fully regenerated with no data loss.

---

## Phase 4 — Cache + Invalidation

### Goal
Safely reuse expensive derived views without letting cache become permanent structure.

### Deliverables

- cache-key specification
- invalidation engine
- event-based invalidation hooks
- TTL policy
- dry-run invalidation traces
- cache metrics and miss/hit visibility

### Invalidation triggers

- source artifact updated
- source artifact deleted/tombstoned
- chunk set changed
- contradiction cluster changed
- governance state hash changed
- lifecycle state changed (stale/prune_candidate/etc.)
- synthesis config changed

### Acceptance criteria

- invalidation is deterministic and explainable
- stale views never survive invalidation events undetected
- dry-run mode shows what would be invalidated and why
- cache hit rate is visible, but correctness never depends on cache presence

### Risks

- invalidation gaps create stale outputs
- too many triggers destroy cache value
- hidden coupling between cache and UI/API behavior

### Exit gate

Phase 4 is complete only when a source mutation reliably invalidates all dependent derived views in both real and dry-run modes.

---

## Phase 5 — Wave 5 Bridge / Semantic Reflect Evolution

### Goal
Use the Memory Over Maps structure to support future semantic improvements without exploding cost.

### Intended scope

- semantic reflect-path verification on bounded candidates
- proper-noun/entity sensitivity improvements
- trust recovery scenarios
- enforced-mode drift variants
- concurrent reflect stress testing

### Why later
Governance Validation Pack v1 explicitly documented these as remaining gaps. They should land on top of the bounded candidate envelope rather than on top of unbounded retrieval.

### Exit gate

Phase 5 is complete only when semantic reflect improvements can be benchmarked on bounded candidates with clear cost and quality trade-offs.

---

## Rollout Policy

### Implementation posture
- advisory-first
- dry-run where applicable
- benchmark before default promotion
- no destructive lifecycle actions in initial rollout

### Promotion posture
A phase is not “done” because code exists. A phase is done only when:

- schemas are stable,
- instrumentation exists,
- benchmarks run,
- failure modes are documented,
- acceptance gates pass.

