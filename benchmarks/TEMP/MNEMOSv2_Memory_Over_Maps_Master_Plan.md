# MNEMOSv2 — Memory Over Maps Master Plan

**Date:** 2026-03-30  
**Status:** Planning baseline  
**Primary lane:** Memory Over Maps integration  
**Strategic objective:** Evolve MNEMOSv2 from a retrieval-centric memory stack into a **governed, source-grounded memory system** with bounded reasoning cost and selective, on-demand synthesis.

---

## 1. Executive Summary

MNEMOSv2 should adopt **Memory Over Maps** as the main architectural lane.

This does **not** mean copying the robotics specifics of the paper. It means adopting the paper's transferable system pattern:

1. keep the primary stored representation lightweight,
2. retrieve and narrow quickly,
3. apply expensive reasoning only to a bounded candidate set,
4. synthesize richer views only on demand,
5. avoid silently turning derived structure into permanent truth.

For MNEMOSv2, that means:

- **Source artifacts become the canonical root objects**.
- **Chunks remain derived indexing units**, not the product's conceptual truth.
- **Governance operates on bounded candidate envelopes**, not broad unbounded retrieval result sets.
- **Evidence bundles, contradiction bundles, preference snapshots, and timeline summaries** are generated on demand.
- **Cache exists only as an optimization layer**, with explicit invalidation and no accidental canonization.

This lane complements the frozen benchmark conclusions:

- Core/Qdrant remains the operational default.
- Governance/pgvector remains valid but not a broad correctness winner.
- Reranker remains experimental/non-production.
- Hybrid retrieval remains available but did not justify default promotion.
- Governance is the strongest validated differentiator.

The product thesis therefore becomes:

> **MNEMOSv2 is a governed, source-grounded memory system that narrows before it reasons and synthesizes only when needed.**

---

## 2. Why This Is the Right Strategic Move

MNEMOSv2 already learned that simply adding more retrieval machinery does not automatically produce a better product outcome.

### What benchmark evidence already showed

- Qdrant wins on ingest speed, latency, and QPS.
- pgvector did not prove broad retrieval-quality superiority.
- ColBERT/reranking did not show safe production uplift.
- Hybrid retrieval did not justify becoming the default.

### What governance evidence already showed

Governance Validation Pack v1 established a more important differentiator:

- repeated winners strengthen with bounded convergence,
- repeated distractors weaken,
- contradiction winners and losers separate over time,
- contradiction state outranks lexical overlap,
- threshold tuning is a documented precision dial,
- known precision limitations are explicit.

### Strategic conclusion

MNEMOSv2 should **stop trying to win by piling on retrieval complexity** and instead win by combining:

- source-grounded memory,
- bounded adjudication,
- selective synthesis,
- governance-backed memory quality maintenance.

---

## 3. Architectural Target State

The future architecture should be understood in five layers.

### Layer 1 — Source Truth
Canonical, durable objects:

- `SourceArtifact`
- immutable or versioned source payload
- metadata, provenance, source authority, timestamps
- deletion/tombstone lifecycle

### Layer 2 — Indexable Derivatives
Operational retrieval units:

- `Chunk`
- embeddings
- lexical index projections
- governance metadata attachment
- lineage back to source artifact

### Layer 3 — Candidate Envelope
Bounded set used for adjudication:

- retrieval output
- deduplication
- redundancy collapse
- max-per-source balancing
- optional diversity-aware narrowing
- bounded adjudication route

### Layer 4 — Governed Read Path
Quality and policy layer:

- veto/freshness/trust/utility modifiers
- contradiction handling
- advisory/enforced modes
- hygiene state awareness

### Layer 5 — On-Demand Synthesis
Ephemeral or cacheable views:

- `EvidenceBundle`
- `ContradictionBundle`
- `PreferenceSnapshot`
- `TimelineSummary`
- future semantic synthesis views

**Critical rule:** Layer 5 must never quietly replace Layer 1 as canonical truth.

---

## 4. Design Principles

1. **Source-first:** source artifacts are the durable truth.
2. **Derived-not-canonical:** chunks and summaries are derivatives.
3. **Bounded reasoning:** expensive reasoning must operate on narrowed candidate sets.
4. **Govern before synthesize:** governance must stay in the core read path.
5. **Ephemeral rich views:** richer abstractions are generated on demand.
6. **Reproducible synthesis:** every derived view must list its source inputs.
7. **Cache is subordinate:** caching improves performance, not epistemic authority.
8. **Advisory before enforced:** new behavior is benchmarked in advisory posture first.
9. **No destructive hygiene first:** state transitions before deletion.
10. **Evidence-backed promotion:** no default promotion without benchmark support.

---

## 5. Program Structure

This plan is broken into four major phases and one follow-on phase.

- **Phase 1 — Artifact Model Refactor**
- **Phase 2 — Bounded Candidate Envelope**
- **Phase 3 — On-Demand Derived Views**
- **Phase 4 — Cache + Invalidation**
- **Phase 5 — Wave 5 Bridge / Semantic Reflect Evolution**

Detailed phase execution is in:

- `MNEMOSv2_Memory_Over_Maps_Phased_Approach.md`
- `MNEMOSv2_Memory_Over_Maps_Technical_Details.md`
- `MNEMOSv2_Memory_Over_Maps_Benchmark_and_E2E_Plan.md`

---

## 6. Delivery Rules

### Must-have rules

- Dry-run support for all new lifecycle-affecting paths.
- Before/after fixtures for any state transition feature.
- Explicit counters in stats surfaces.
- No destructive operations in initial hygiene-connected Memory Over Maps work.
- Every derived view must declare lineage inputs.

### Must-not-do rules

- Do not promote permanent summary structures before source artifact refactor lands.
- Do not bypass governance to save latency.
- Do not allow cache to act as a hidden source of truth.
- Do not introduce new default retrieval complexity unless benchmark evidence supports it.

---

## 7. Product Positioning Outcome

When these phases are complete, MNEMOSv2 should be positioned as:

> A governed, source-grounded memory system that maintains memory quality over time, narrows candidate sets before expensive reasoning, and constructs richer evidence-backed views only when needed.

That is a materially stronger contender story than:

> A memory platform with multiple retrieval modes and optional reranking.

---

## 8. Definition of Success

This plan succeeds if MNEMOSv2 can demonstrate all of the following:

1. **Every answer can trace back to source artifacts and chunks.**
2. **Candidate narrowing reduces cost without silently dropping answer-supporting evidence.**
3. **Governance still operates as the authoritative quality layer after narrowing.**
4. **Derived views are reproducible, query-scoped, and safely invalidated.**
5. **No part of the system quietly turns synthesized artifacts into canonical truth.**
6. **Benchmarks show bounded reasoning cost and acceptable evidence retention.**
7. **Whitepaper and product narrative can honestly claim source-grounded selective synthesis.**

---

## 9. Immediate Next Actions

1. Review and approve the phased approach.
2. Approve Phase 1 schemas and lineage contract.
3. Require instrumentation definitions before Phase 2 implementation begins.
4. Lock the benchmark protocol for narrowing and on-demand synthesis before promotion.
5. Keep governance evidence and Memory Over Maps evidence in the same proof stack.

