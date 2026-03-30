# MNEMOSv2 — Memory Over Maps Technical Details

**Purpose:** implementation details and engineering contract for the Memory Over Maps lane.

---

## 1. Data Model Additions

### A. SourceArtifact
Canonical representation of stored source truth.

```text
SourceArtifact
- artifact_id
- artifact_type
- source_uri
- source_authority
- content_hash
- version_id
- created_at
- updated_at
- lifecycle_state
- metadata
```

### B. Chunk
Retrieval-oriented derivative of a SourceArtifact.

```text
Chunk
- chunk_id
- artifact_id
- version_id
- chunk_index
- chunk_hash
- text_span_ref
- embedding_ref
- lexical_ref
- governance_ref
- metadata
```

### C. DerivedView
Ephemeral or cacheable query-scoped synthesis object.

```text
DerivedView
- view_id
- view_type
- inputs
- query_fingerprint
- governance_state_hash
- synthesis_policy
- created_at
- invalidated_at
- cacheable
- reproducible
- metadata
```

### D. EvidenceBundle
```text
EvidenceBundle extends DerivedView
- supporting_artifact_ids[]
- supporting_chunk_ids[]
- support_roles[]
- exclusions[]
```

### E. ContradictionBundle
```text
ContradictionBundle extends DerivedView
- contradiction_cluster_id
- winner_ids[]
- loser_ids[]
- comparison_factors[]
- resolution_trace
```

### F. PreferenceSnapshot
```text
PreferenceSnapshot extends DerivedView
- subject_id
- preferred_memory_ids[]
- suppressed_memory_ids[]
- rationale_trace
```

### G. TimelineSummary
```text
TimelineSummary extends DerivedView
- timeline_subject
- ordered_event_refs[]
- source_artifact_ids[]
- temporal_confidence
```

---

## 2. Read Path Changes

### Current-state concept
retrieve -> govern -> answer

### Target-state concept
retrieve -> narrow -> govern -> synthesize on demand -> answer

### Expanded read path

1. initial retrieval
2. candidate dedupe
3. source balancing
4. optional diversity policy
5. bounded candidate envelope emitted
6. governance scoring and contradiction handling
7. post-governance survivor set
8. optional on-demand view generation
9. optional cache read/write
10. response assembly with lineage

---

## 3. Candidate Envelope Contract

### Envelope object

```text
CandidateEnvelope
- query_id
- initial_candidate_count
- final_candidate_count
- candidates[]
- suppression_reasons[]
- source_distribution
- similarity_distribution
- diversity_policy_applied
- config_snapshot
```

### Suppression reasons

- `duplicate_similarity`
- `source_cap_exceeded`
- `low_rank_after_diversity`
- `bounded_limit_exceeded`
- `policy_excluded`

### Required stats

- initial candidate count
- post-dedupe count
- post-source-cap count
- post-diversity count
- final envelope count
- source concentration ratio
- average pairwise similarity

---

## 4. API Implications

### Search request additions

```json
{
  "retrieval_mode": "semantic",
  "bounded_envelope": {
    "enabled": true,
    "candidate_pool_limit": 40,
    "dedupe_similarity_threshold": 0.90,
    "max_per_source_artifact": 3,
    "diversity_policy": "off"
  },
  "derive_views": ["evidence_bundle"],
  "explain": true
}
```

### Search response additions

```json
{
  "candidate_envelope": {
    "initial_candidate_count": 120,
    "final_candidate_count": 24,
    "suppression_summary": {
      "duplicate_similarity": 55,
      "source_cap_exceeded": 31,
      "bounded_limit_exceeded": 10
    }
  },
  "derived_views": [
    {
      "view_id": "...",
      "view_type": "evidence_bundle",
      "inputs": {
        "artifact_ids": ["..."],
        "chunk_ids": ["..."]
      }
    }
  ]
}
```

---

## 5. Governance Interaction Rules

1. Narrowing happens **before** expensive synthesis.
2. Governance remains authoritative for post-narrowing quality ordering.
3. Derived views must only use post-governance survivors unless explicitly marked otherwise.
4. Contradiction bundles must reuse contradiction policy outputs rather than duplicate winner logic.
5. Lifecycle-aware hygiene states must be visible to derived view builders.

---

## 6. Cache Contract

### Cache key components

- normalized query fingerprint
- source artifact set hash
- chunk set hash
- governance state hash
- view type
- synthesis policy version

### Cache object minimums

- cache key
- view ID
- created_at
- last_verified_at
- invalidation status
- dependency refs

### Invalidation dependency graph

```text
SourceArtifact -> Chunk -> CandidateEnvelope -> GovernedResultSet -> DerivedViewCache
```

Any mutation upstream must invalidate dependent downstream caches.

---

## 7. Dry-Run Requirements

The following paths require dry-run support:

- invalidation engine
- candidate envelope config experiments
- synthesis policy changes
- hygiene-linked lifecycle changes when they affect view validity

Dry-run output must show:

- what would change
- why it would change
- what dependencies were traversed
- whether cache artifacts would be evicted

---

## 8. Test Requirements

### Phase 1 tests
- source->chunk lineage integrity
- derived-view input declaration enforcement
- audit logging for view generation

### Phase 2 tests
- deterministic narrowing under fixed inputs
- dedupe suppression correctness
- max-per-source balancing behavior
- bounded envelope evidence retention

### Phase 3 tests
- reproducibility of derived views
- contradiction bundle correctness
- timeline summary source trace coverage

### Phase 4 tests
- invalidation on source change
- invalidation on contradiction update
- cache miss/hit correctness
- dry-run invalidation parity

---

## 9. Non-Goals

These are intentionally out of scope for the first Memory Over Maps push:

- making hybrid the default
- promoting reranker back into production path
- building permanent summary stores as first-class truth
- destructive deletion flows
- major new retrieval backends

---

## 10. Engineering Guardrails

- Prefer simpler deterministic policies before adaptive ones.
- Instrument every narrowing stage before optimizing it.
- Preserve explainability even when adding cost controls.
- Do not trade source traceability for convenience.
- Do not let performance shortcuts break epistemic hierarchy.

