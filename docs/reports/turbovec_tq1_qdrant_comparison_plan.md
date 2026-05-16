# Turbovec TQ-1.3 Qdrant Comparison Plan

## Objective
Compare MNEMOS’s current Qdrant-style retrieval path against the new `TurbovecTier` (Turbovec + SQLite) on the same real PDF corpus, identical chunks, exact embeddings, strict query set, and consistent metadata/governance filters.

The core question is:
**Can Turbovec + SQLite provide a compact, local-first retrieval profile with acceptable quality, latency, traceability, filtering, persistence, and hybrid behavior compared to Qdrant?**

## Boundary Conditions
- **Corpus Constraints**: A single extraction run must produce canonical chunk and embedding artifacts (`tq13_real_pdf_chunks.jsonl` and `tq13_real_pdf_embeddings.npz`). Both Qdrant and Turbovec will ingest exactly from these artifacts to eliminate parsing/embedding drift.
- **Goal Scope**: The goal is NOT to unconditionally replace Qdrant. The goal is to determine if Turbovec qualifies as a `TURBOVEC_PORTABLE_PROFILE_CANDIDATE` for local-first, low-footprint operations while Qdrant serves as the Core Memory Appliance.

## Required Benchmark Modes
1. **Qdrant dense-only**
2. **Qdrant hybrid** (If available locally, otherwise fallback)
3. **Turbovec dense-only**
4. **Turbovec + SQLite FTS hybrid**
5. **SQLite FTS-only baseline**

## Execution Phases
- **Pass A**: ToLearn dataset subset
- **Pass B**: SIGINT dataset subset
- **Pass C**: Combined subset
- **Pass D**: Full combined corpus (Operational constraint permitting)

## Deterministic Retrieval Quality Probes
A dedicated JSON suite (`tq13_real_pdf_queries.json`) will enforce 50 static query paths:
- 10 Semantic recall
- 10 Acronym/exact-term
- 10 Source-specific
- 10 Policy/procedure
- 10 Hard-negative/evidence-gap

### Scoring Model
```python
retrieval_quality_score = (
    0.25 * expected_source_hit_rate +
    0.20 * expected_term_hit_rate +
    0.20 * expected_dataset_hit_rate +
    0.15 * metadata_filter_correctness +
    0.10 * hard_negative_handling +
    0.10 * source_traceability
)

portable_profile_score = (
    0.30 * retrieval_quality_score +
    0.20 * latency_score +
    0.20 * storage_score +
    0.15 * persistence_score +
    0.15 * operational_simplicity_score
)
```

## Potential Decision Gates
- `QDRANT_COMPARISON_BLOCKED`
- `TURBOVEC_REJECT`
- `TURBOVEC_DENSE_ONLY_SHADOW`
- `TURBOVEC_PORTABLE_PROFILE_CANDIDATE` (Target)
- `TURBOVEC_QDRANT_REPLACEMENT_CANDIDATE`
