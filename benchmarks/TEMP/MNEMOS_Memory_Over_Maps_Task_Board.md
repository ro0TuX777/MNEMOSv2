# MNEMOS Memory-Over-Maps Implementation Task Board

## Sprint Goal

Deliver a **source-grounded, lightweight memory-bank-first architecture** for MNEMOS that:
- narrows candidates before expensive reasoning
- supports on-demand derived views
- avoids unnecessary permanent precomputed structure
- benchmarks build/storage/query trade-offs against heavier approaches

---

## Epic 1 — Source-Grounded Artifact Model

### Task 1.1 — Distinguish source artifacts from derived views
**Description**
Extend the memory model so MNEMOS can explicitly represent source-grounded artifacts vs synthesized derived views.

**Dependencies**
- none

**Acceptance criteria**
- artifacts include `artifact_type`
- source and derived artifacts are queryable distinctly
- provenance fields are required for derived views

### Task 1.2 — Add source-grounding invariants
**Description**
Prevent derived views from becoming orphaned from their source artifacts.

**Dependencies**
- Task 1.1

**Acceptance criteria**
- derived views must reference source or parent IDs
- source grounding can be inspected in explain/debug mode
- tests fail when derived artifacts lack lineage

---

## Epic 2 — Candidate Narrowing Layer

### Task 2.1 — Build candidate retriever wrapper
**Description**
Wrap existing retrieval so raw source-grounded candidates are returned in a consistent format for narrowing.

**Dependencies**
- Task 1.1

**Acceptance criteria**
- retriever returns candidate list with source IDs, scores, and metadata
- works across current retrieval modes

### Task 2.2 — Implement dedupe and redundancy reduction
**Description**
Reduce repeated or near-duplicate candidates before adjudication.

**Dependencies**
- Task 2.1

**Acceptance criteria**
- duplicate or near-duplicate candidates can be suppressed
- dedupe stats are emitted
- unit tests cover redundant-source cases

### Task 2.3 — Implement candidate pool limiter
**Description**
Bound the pool that can flow into expensive governance/synthesis logic.

**Dependencies**
- Task 2.1

**Acceptance criteria**
- candidate pool size is configurable
- full-corpus expensive passes are blocked by default
- explain/debug mode shows final narrowed pool size

### Task 2.4 — Add diversity-aware narrowing
**Description**
Optionally diversify candidates across source/document/entity so one source does not dominate the pool.

**Dependencies**
- Task 2.3

**Acceptance criteria**
- diversity mode can be enabled/disabled
- narrowed pool can include multi-source coverage
- tests cover over-dominance suppression

---

## Epic 3 — Adjudication Router

### Task 3.1 — Build adjudication router
**Description**
Route narrowed candidates to governance or synthesis only when required.

**Dependencies**
- Epic 2
- MemArchitect governance layer work

**Acceptance criteria**
- router can invoke governance-only, synthesis-only, or combined paths
- raw retrieval path still works without adjudication
- decision path is logged

### Task 3.2 — Add per-query adjudication budget controls
**Description**
Prevent expensive downstream work from becoming an uncontrolled default.

**Dependencies**
- Task 3.1

**Acceptance criteria**
- max candidates and max synthesis calls per query are enforced
- budget exhaustion is visible in telemetry
- fallback behavior is documented

---

## Epic 4 — On-Demand Synthesizer

### Task 4.1 — Build derived view framework
**Description**
Implement a shared interface for generating on-demand derived views.

**Dependencies**
- Task 1.1
- Task 3.1

**Acceptance criteria**
- synthesizer can produce typed derived views
- outputs include provenance and generation metadata
- generation mode is marked as `on_demand`

### Task 4.2 — Implement evidence bundle view
**Description**
Create a compact evidence pack showing why a result was selected.

**Dependencies**
- Task 4.1

**Acceptance criteria**
- evidence bundle lists supporting source artifacts
- ordering and rationale are preserved
- explain/debug consumers can request it

### Task 4.3 — Implement contradiction bundle view
**Description**
Create an on-demand grouped contradiction view with winners and suppressed candidates.

**Dependencies**
- Task 4.1
- MemArchitect contradiction policy

**Acceptance criteria**
- contradiction bundle can be generated from a conflict group
- active and suppressed memories are visible
- provenance and timestamps are included

### Task 4.4 — Implement preference snapshot view
**Description**
Create a compact state view for user preference-like memory.

**Dependencies**
- Task 4.1
- MemArchitect governance metadata

**Acceptance criteria**
- snapshot aggregates current winning state
- conflicting stale entries are suppressed or marked
- snapshot remains grounded to source artifacts

### Task 4.5 — Implement timeline summary view
**Description**
Create an on-demand temporal summary for time-sensitive questions.

**Dependencies**
- Task 4.1

**Acceptance criteria**
- summary is built only when requested
- timeline output references source artifacts
- stale or superseded items are marked

---

## Epic 5 — Lightweight Cache

### Task 5.1 — Build derived-view cache
**Description**
Cache high-value synthesized outputs without making them universal permanent structure.

**Dependencies**
- Epic 4

**Acceptance criteria**
- cached derived views retain lineage
- cache hit/miss metrics are recorded
- cache entries are typed by derived-view type

### Task 5.2 — Add cache invalidation rules
**Description**
Invalidate cached views when source memories change or delete cascades occur.

**Dependencies**
- Task 5.1
- MemArchitect provenance/delete work

**Acceptance criteria**
- cache invalidates on source mutation
- cache invalidates on delete cascade
- invalid entries are not served

### Task 5.3 — Add reuse-based persistence policy
**Description**
Only highly reused derived views should persist longer.

**Dependencies**
- Task 5.1

**Acceptance criteria**
- reuse count affects cache retention
- low-reuse entries expire faster
- policy is configurable

---

## Epic 6 — API and Capability Surface

### Task 6.1 — Extend `/search` for lightweight memory controls
**Description**
Add:
- `candidate_pool_limit`
- `allow_on_demand_synthesis`
- `derived_view_type`
- `include_adjudication_trace`

**Dependencies**
- Epics 2–4

**Acceptance criteria**
- `/search` accepts and validates new controls
- defaults preserve current broad behavior
- invalid combinations fail clearly

### Task 6.2 — Add derived-view retrieval API
**Description**
Expose a direct way to request an on-demand derived view.

**Dependencies**
- Epic 4

**Acceptance criteria**
- callers can request derived view generation explicitly
- returned view is typed and grounded
- response includes generation metadata

### Task 6.3 — Extend `/capabilities`
**Description**
Advertise source-grounded memory, derived-view types, and cache availability.

**Dependencies**
- Epics 4–5

**Acceptance criteria**
- capabilities accurately reflect enabled features
- operators can inspect whether synthesis is available

---

## Epic 7 — Telemetry and Cost Metrics

### Task 7.1 — Add retrieval cost telemetry
**Description**
Track:
- raw candidates
- narrowed candidates
- adjudication pool size
- total query-time cost

**Dependencies**
- Epic 2
- Epic 3

**Acceptance criteria**
- candidate narrowing telemetry is emitted
- cost is attributable to retrieval vs adjudication vs synthesis

### Task 7.2 — Add synthesis telemetry
**Description**
Track:
- synthesis calls per query
- synthesis latency
- unnecessary synthesis rate
- derived-view reuse

**Dependencies**
- Epic 4
- Epic 5

**Acceptance criteria**
- synthesis metrics are logged and visible
- high-frequency unnecessary synthesis can be detected

### Task 7.3 — Add source-grounding rate metric
**Description**
Measure how often returned answers remain traceable to source artifacts.

**Dependencies**
- Epic 1
- Epic 4

**Acceptance criteria**
- source-grounding rate is computable
- reports can distinguish grounded vs weakly grounded outputs

---

## Epic 8 — Benchmark Track

### Task 8.1 — Create lightweight-memory benchmark dataset
**Description**
Build a benchmark set covering:
- direct fact lookup
- contradiction-heavy queries
- temporal summaries
- evidence-needed questions
- long-document multi-hop
- preference/state snapshots

**Dependencies**
- none

**Acceptance criteria**
- dataset is committed to repo
- each case is tagged by class
- expected value of synthesis is documented

### Task 8.2 — Build lightweight-memory benchmark runner
**Description**
Compare:
- source-grounded retrieval only
- source-grounded + adjudication
- source-grounded + on-demand synthesis
- heavy precomputed derived-view baseline (where available)

**Dependencies**
- Epics 2–5
- Task 8.1

**Acceptance criteria**
- benchmark produces raw JSON and summary artifacts
- all required modes are compared
- reports include build/storage/query trade-offs

### Task 8.3 — Add trade-off reporting
**Description**
Report:
- build time
- storage footprint
- retrieval latency
- adjudication latency
- synthesis latency
- answer accuracy
- source grounding rate
- cache hit rate
- unnecessary synthesis rate
- cost per successful answer

**Dependencies**
- Task 8.2

**Acceptance criteria**
- report contains all required metrics
- success/failure criteria are explicit
- heavy vs lightweight trade-offs are interpretable

---

## Epic 9 — Docs and Operator Guidance

### Task 9.1 — Document source-grounded memory model
**Description**
Explain why source artifacts remain primary and derived views are secondary.

**Dependencies**
- Epic 1

**Acceptance criteria**
- docs distinguish source vs derived clearly
- examples show why provenance matters

### Task 9.2 — Document on-demand synthesis rules
**Description**
Explain when synthesis should or should not be used.

**Dependencies**
- Epic 4
- Epic 8

**Acceptance criteria**
- docs explain triggers for each derived view type
- docs warn against turning synthesis into an uncontrolled default

### Task 9.3 — Write operator guidance
**Description**
Document recommended usage:
1. start source-grounded by default
2. enable adjudication where needed
3. request synthesis only for query classes that justify it
4. cache selectively

**Dependencies**
- Task 8.3

**Acceptance criteria**
- one concise operator guidance section exists
- class-based recommendations are documented when benchmark evidence exists

---

# Recommended Execution Order

## Wave 1
- Task 1.1
- Task 1.2
- Task 2.1
- Task 2.2
- Task 2.3

## Wave 2
- Task 2.4
- Task 3.1
- Task 3.2
- Task 4.1
- Task 4.2

## Wave 3
- Task 4.3
- Task 4.4
- Task 4.5
- Task 5.1
- Task 5.2
- Task 5.3

## Wave 4
- Task 6.1
- Task 6.2
- Task 6.3
- Task 7.1
- Task 7.2
- Task 7.3

## Wave 5
- Task 8.1
- Task 8.2
- Task 8.3
- Task 9.1
- Task 9.2
- Task 9.3

---

# Sprint Exit Criteria

The implementation phase is complete when:

- MNEMOS distinguishes source-grounded artifacts from derived views
- candidate narrowing is bounded and enforced before expensive reasoning
- at least three on-demand derived view types exist
- cache and invalidation rules work
- source-grounding metrics are available
- benchmark artifacts show build/storage/query trade-offs
- docs explain when to stay source-grounded vs when to synthesize on demand

---

# Key Product Questions This Work Should Answer

1. Can MNEMOS reduce build/storage cost by avoiding heavy permanent derived structure?
2. Can expensive synthesis be confined to a small candidate pool?
3. Which query classes actually benefit from on-demand synthesis?
4. Does lightweight source-grounded architecture preserve or improve answer trust?
5. Does this give MNEMOS a clearer architectural advantage over always-structured memory systems?
