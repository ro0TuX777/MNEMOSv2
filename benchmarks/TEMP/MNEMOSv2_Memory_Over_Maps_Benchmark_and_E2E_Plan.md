# MNEMOSv2 — Memory Over Maps Benchmark and E2E Plan

**Purpose:** benchmark protocol, acceptance gates, and end-to-end validation plan for the Memory Over Maps lane.

---

## 1. Benchmark Philosophy

Memory Over Maps features should not be promoted because they sound elegant. They should be promoted only if they produce measurable product value while preserving source grounding and governance integrity.

This benchmark plan focuses on four proof questions:

1. Does source-grounded structure improve explainability and reproducibility?
2. Does candidate narrowing reduce cost without breaking answer-support retention?
3. Do on-demand views remain reproducible and safely invalidated?
4. Does the full E2E path preserve governance quality while lowering unnecessary reasoning cost?

---

## 2. Benchmark Tracks

### Track M1 — Artifact Lineage Integrity

**Question:** Can every returned chunk/view trace cleanly to source truth?

**Metrics:**
- lineage completeness rate
- source trace resolution latency
- % of responses with full source artifact coverage
- % of derived views with complete input declarations

**Pass criteria:**
- 100% lineage completeness on benchmark set
- 0 orphan derived views

---

### Track M2 — Candidate Envelope Efficiency

**Question:** Does bounded narrowing reduce cost while retaining support evidence?

**Metrics:**
- initial candidate count
- final candidate count
- compression ratio
- duplicate suppression rate
- source concentration ratio
- latency before/after narrowing
- answer-support retention rate
- governed reorder stability

**Pass criteria:**
- meaningful envelope compression
- acceptable answer-support retention threshold
- no unexplained governance degradation

---

### Track M3 — On-Demand View Reproducibility

**Question:** Can derived views be regenerated from declared inputs?

**Metrics:**
- reproducibility success rate
- time to regenerate
- input completeness rate
- regeneration mismatch count

**Pass criteria:**
- 100% reproducibility on frozen test set
- 0 unexplained regeneration mismatches

---

### Track M4 — Cache + Invalidation Correctness

**Question:** Are cached views safely invalidated when upstream truth changes?

**Metrics:**
- invalidation trigger coverage
- stale cache survival rate
- dry-run/real-run invalidation parity
- cache hit rate
- false invalidation rate

**Pass criteria:**
- 0 stale cached views surviving required invalidation events
- dry-run parity with live invalidation behavior

---

### Track M5 — E2E Governed Selective Synthesis

**Question:** Does the full path work as intended?

**E2E path:**
retrieve -> narrow -> govern -> synthesize -> cache/invalidate -> answer

**Metrics:**
- end-to-end response latency
- derived-view generation rate
- governance decision consistency
- evidence bundle completeness
- contradiction bundle correctness
- source-grounded answer coverage

**Pass criteria:**
- E2E path operates with bounded latency
- source-grounded answer trace preserved
- governance remains intact after narrowing

---

## 3. Benchmark Datasets / Test Sets

### A. Frozen internal evaluation set
Should include:
- multi-source overlapping evidence
- contradiction clusters
- stale-but-retrievable artifacts
- short generic distractors
- long-answer overlap cases
- timeline-like event sequences
- preference-like memory conflicts

### B. Synthetic structure stress set
Used for:
- duplicate-heavy source families
- source dominance collapse tests
- cache invalidation fanout tests

### C. Governance-linked set
Reuses scenarios from Governance Validation Pack v1 where applicable, especially for:
- contradiction priority
- ignored-memory behavior
- precision/recall threshold documentation

---

## 4. Required E2E Scenarios

### Scenario E2-1 — Evidence Bundle Generation

**Flow:**
- ingest source artifacts
- retrieve relevant chunks
- narrow candidates
- govern survivors
- generate evidence bundle

**Validate:**
- evidence bundle only uses declared governed inputs
- source artifact trace visible
- no orphan support references

---

### Scenario E2-2 — Contradiction Bundle

**Flow:**
- ingest contradiction cluster
- retrieve conflicting candidates
- narrow without collapsing true contradiction coverage
- govern winner/loser
- generate contradiction bundle

**Validate:**
- winner/loser identities correct
- bundle cites contradiction factors
- loser never presented as unresolved equal evidence

---

### Scenario E2-3 — Timeline Summary

**Flow:**
- ingest sequential source artifacts
- retrieve time-linked chunks
- narrow and govern
- synthesize timeline summary

**Validate:**
- timeline order matches source evidence
- summary can be regenerated from declared inputs
- source mutation invalidates cached timeline

---

### Scenario E2-4 — Preference Snapshot

**Flow:**
- retrieve preference-related memory set
- govern competing memories
- synthesize preference snapshot

**Validate:**
- preferred vs suppressed memories are explicit
- snapshot is lineage-traceable
- later governance change invalidates cached snapshot

---

### Scenario E2-5 — Source Update Invalidation

**Flow:**
- create derived view and cache it
- mutate source artifact
- run invalidation
- request same view again

**Validate:**
- old cached view invalidated
- new derived view regenerated
- invalidation reason logged

---

## 5. Metrics Dashboard Requirements

Expose these in stats/reporting:

### Candidate envelope stats
- `candidate_envelope_initial_total`
- `candidate_envelope_final_total`
- `candidate_dedupe_suppressed_total`
- `candidate_source_cap_suppressed_total`
- `candidate_diversity_suppressed_total`

### Derived view stats
- `derived_view_generated_total`
- `derived_view_cache_hit_total`
- `derived_view_cache_miss_total`
- `derived_view_invalidated_total`
- `derived_view_regeneration_total`

### Governance-linked stats
- `governed_post_envelope_total`
- `governed_contradiction_bundle_total`
- `governed_evidence_bundle_total`
- `governed_source_trace_complete_total`

---

## 6. Acceptance Gates by Phase

### Phase 1 gate
- lineage integrity benchmark passes
- zero orphan derived views

### Phase 2 gate
- narrowing benchmark demonstrates useful compression
- answer-support retention within agreed threshold
- governance order remains explainable

### Phase 3 gate
- all initial derived views reproducible from declared inputs
- contradiction/timeline/preference view tests pass

### Phase 4 gate
- invalidation benchmark passes
- stale cached view survival rate is zero on required cases

### E2E gate
- all five E2 scenarios pass
- benchmark report and engineering memo written

---

## 7. E2E Benchmark Output Artifacts

Every benchmark cycle should produce:

1. raw JSON metrics artifact
2. markdown summary report
3. decision memo
4. known limitations section
5. promotion recommendation: pass / hold / fail

Suggested filenames:

- `memory_over_maps_<timestamp>_raw.json`
- `memory_over_maps_<timestamp>_report.md`
- `memory_over_maps_<timestamp>_decision.md`

---

## 8. Promotion Rules

### Promote to default behavior only if:
- benchmarks show clear operational value,
- source grounding remains intact,
- governance quality does not degrade,
- invalidation correctness is verified,
- known limitations are documented.

### Hold if:
- cost decreases but evidence retention weakens materially,
- reproducibility is partial,
- invalidation confidence is incomplete.

### Fail if:
- derived views cannot be reliably traced to source,
- narrowing silently drops needed evidence,
- cache returns stale views after source mutations,
- governance is effectively bypassed.

---

## 9. Benchmark Sequence Recommendation

1. run M1 lineage integrity first
2. run M2 candidate envelope efficiency second
3. run M3 reproducibility third
4. run M4 invalidation fourth
5. run M5 end-to-end selective synthesis last

Rationale: do not validate E2E behavior before the underlying lineage and narrowing contracts are proven.

---

## 10. Final Benchmark Success Statement

The Memory Over Maps push is successful only when MNEMOSv2 can demonstrate:

> It keeps source truth lightweight and primary, narrows before expensive reasoning, preserves governed memory quality, and synthesizes richer outputs only on demand with reproducible lineage and safe invalidation.

