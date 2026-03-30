# MNEMOSv2 Memory Over Maps — Sprint Ticket Board

## Sprint A — Artifact Model Foundations

### A1 — SourceArtifact base schema
**Type:** backend/schema  
**Priority:** P0  
**Dependencies:** none

**Tasks**
- Define SourceArtifact model
- Add serialization/deserialization
- Add lifecycle/version fields
- Add content hash field

**Acceptance**
- SourceArtifact instances created and validated through schema tests
- Existing code unaffected when feature flag is off

---

### A2 — Chunk lineage schema
**Type:** backend/schema  
**Priority:** P0  
**Dependencies:** A1

**Tasks**
- Define Chunk model
- Require artifact_id linkage
- Add provenance span support
- Add migration shim for legacy engrams

**Acceptance**
- Every chunk resolves to exactly one source artifact
- No orphan chunk allowed in tests

---

### A3 — Retrieval lineage exposure
**Type:** backend/api  
**Priority:** P0  
**Dependencies:** A1, A2

**Tasks**
- Add source artifact ID + chunk ID to explain/debug payloads
- Add provenance URI/path exposure

**Acceptance**
- Retrieval explain payload exposes lineage for every hit

---

## Sprint B — Candidate Narrowing Envelope

### B1 — Candidate pool limiter
**Type:** backend/retrieval  
**Priority:** P0  
**Dependencies:** A3

**Tasks**
- Add candidate_pool_limit
- Add pre_governance_top_n
- Record raw and narrowed counts

**Acceptance**
- Narrowed count always <= raw count
- Stats emitted

---

### B2 — Duplicate suppression
**Type:** backend/retrieval  
**Priority:** P0  
**Dependencies:** B1

**Tasks**
- Add similarity threshold
- Suppress near-duplicate candidates
- Preserve representative candidate

**Acceptance**
- Duplicate suppression count visible
- Fixture shows removal of redundant chunk cluster

---

### B3 — Max per source artifact
**Type:** backend/retrieval  
**Priority:** P1  
**Dependencies:** B1

**Tasks**
- Add per-source cap
- Prevent single artifact domination

**Acceptance**
- Narrowed set respects per-source cap in tests

---

### B4 — Diversity-aware narrowing
**Type:** backend/retrieval  
**Priority:** P2  
**Dependencies:** B2, B3

**Tasks**
- Add optional diversity policy
- Balance across source families / metadata groups

**Acceptance**
- Deterministic selection under same inputs
- Improves source spread in fixture set

---

## Sprint C — Derived Views v1

### C1 — DerivedView base schema
**Type:** backend/schema  
**Priority:** P0  
**Dependencies:** A2

**Tasks**
- Define DerivedView fields
- Add reproducibility and cacheability flags
- Add governance_state_hash field

**Acceptance**
- DerivedView schema validated in unit tests

---

### C2 — EvidenceBundle
**Type:** backend/synthesis  
**Priority:** P0  
**Dependencies:** C1, B1

**Tasks**
- Create query-scoped evidence bundle
- Include source and chunk references
- Include governance decisions used

**Acceptance**
- Bundle reproducible
- No silent persistence by default

---

### C3 — ContradictionBundle
**Type:** backend/synthesis  
**Priority:** P1  
**Dependencies:** C1

**Tasks**
- Render contradiction clusters
- Surface winner/loser reasoning

**Acceptance**
- Bundle reflects actual contradiction policy state

---

### C4 — TimelineSummary
**Type:** backend/synthesis  
**Priority:** P2  
**Dependencies:** C1

**Tasks**
- Assemble timestamped source evidence into ordered summary
- Document missing-time handling

**Acceptance**
- Summary handles missing timestamps explicitly

---

## Sprint D — Cache and Invalidation

### D1 — Cache key implementation
**Type:** backend/cache  
**Priority:** P0  
**Dependencies:** C2

**Tasks**
- Implement cache key over query + inputs + governance hash + view type

**Acceptance**
- Same logical input returns same key
- Governance changes alter key

---

### D2 — Invalidation engine
**Type:** backend/cache  
**Priority:** P0  
**Dependencies:** D1

**Tasks**
- Invalidate on source update
- Invalidate on chunk change
- Invalidate on contradiction/lifecycle/deletion state changes

**Acceptance**
- Tests prove invalidation on each trigger

---

### D3 — Dry-run invalidation audit
**Type:** backend/ops  
**Priority:** P0  
**Dependencies:** D2

**Tasks**
- Add dry-run mode
- Emit invalidation traces

**Acceptance**
- No destructive effect in dry-run
- Logs explain invalidation cause

---

## Sprint E — Benchmarks and E2E

### E1 — Narrowing benchmark
**Type:** benchmark  
**Priority:** P0  
**Dependencies:** B4

**Tasks**
- Measure raw->narrowed compression ratio
- Measure answer-support retention
- Measure per-source domination reduction

**Acceptance**
- Artifact produced in markdown + JSON

---

### E2 — Derived view latency benchmark
**Type:** benchmark  
**Priority:** P1  
**Dependencies:** C4

**Tasks**
- Measure bundle generation latency
- Measure timeline summary latency
- Measure with and without cache

**Acceptance**
- p50/p95 captured for each view type

---

### E3 — Invalidation correctness benchmark
**Type:** benchmark  
**Priority:** P1  
**Dependencies:** D3

**Tasks**
- Update source artifacts
- Confirm cache invalidation
- Confirm stale cached view not reused

**Acceptance**
- No false reuse across invalidation triggers

---

### E4 — E2E proof pack
**Type:** benchmark/e2e  
**Priority:** P0  
**Dependencies:** E1, E2, E3

**Tasks**
- Build end-to-end scenarios
- Produce evidence pack
- Produce promotion memo

**Acceptance**
- Artifact pack complete
- Known gaps explicitly listed

---

## Suggested Delivery Order

1. A1, A2, A3
2. B1, B2, B3
3. C1, C2
4. D1, D2, D3
5. B4, C3, C4
6. E1, E2, E3, E4

---

## Promotion Gates

### Gate 1 — Schema Readiness
- SourceArtifact / Chunk / DerivedView schemas complete
- Lineage visible in explain payloads

### Gate 2 — Envelope Readiness
- Candidate narrowing active
- Stats visible
- No severe answer-support loss in fixtures

### Gate 3 — Derived View Readiness
- EvidenceBundle and ContradictionBundle working
- Reproducibility proven

### Gate 4 — Cache Safety
- Invalidation works
- Dry-run logs understandable

### Gate 5 — E2E Promotion
- Benchmark artifacts complete
- Whitepaper-safe claims available
