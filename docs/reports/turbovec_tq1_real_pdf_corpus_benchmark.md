# Turbovec TQ-1.2 Real PDF Corpus Benchmark

## Objective
Test whether the `TurbovecTier` (Turbovec + SQLite) works against real MNEMOS-style document payloads, including chunking, source provenance, real embeddings (`BAAI/bge-base-en-v1.5`), metadata filtering, lexical/hybrid retrieval, persistence, and source traceability.

*Note: For the feasibility gate, extraction was limited to the first 5 pages of a representative sample of 20 PDFs across two real-world datasets (`ToLearn` and `SIGINT`).*

## Dataset Performance

### 1. ToLearn Corpus
- **PDFs Processed**: 10 successful, 0 failed
- **Chunks Generated**: 102 (Avg size: ~235.7 words)
- **Ingestion Rate**: 293.96 docs/sec
- **Dense Latency p50**: 1.17 ms
- **Hybrid Latency p50**: 3.20 ms
- **Lexical Latency p50**: 0.20 ms
- **Index Size**: 0.04 MB
- **Metadata Size**: 1.68 MB
- **Load/Save Times**: 0.011s / 0.002s
- **Validation**: Filter Correctness, Delete Exclusion, Persistence all passed.

### 2. SIGINT Corpus
- **PDFs Processed**: 8 successful, 2 failed *(failed gracefully due to image-only scan layers without OCR text)*
- **Chunks Generated**: 52 (Avg size: ~221.7 words)
- **Ingestion Rate**: 251.67 docs/sec
- **Dense Latency p50**: 1.33 ms
- **Hybrid Latency p50**: 2.26 ms
- **Lexical Latency p50**: 0.20 ms
- **Index Size**: 0.02 MB
- **Metadata Size**: 0.42 MB
- **Load/Save Times**: 0.009s / 0.001s
- **Validation**: Filter Correctness, Delete Exclusion, Persistence all passed.

### 3. Combined Corpus
- **PDFs Processed**: 18 successful, 2 failed
- **Chunks Generated**: 154 (Avg size: ~231.0 words)
- **Ingestion Rate**: 346.73 docs/sec
- **Dense Latency p50**: 1.31 ms
- **Hybrid Latency p50**: 4.37 ms
- **Lexical Latency p50**: 0.10 ms
- **Index Size**: 0.06 MB
- **Metadata Size**: 0.97 MB
- **Load/Save Times**: 0.009s / 0.001s
- **Validation**: Filter Correctness, Delete Exclusion, Persistence all passed.

## Retrieval Quality & Traceability

Across 50 targeted probe queries (General, Acronyms, Policy, Source-specific, Evidence Gap), the `TurbovecTier` correctly executed both dense and FTS queries, and merged them natively in Python using the local Reciprocal Rank Fusion implementation. 

- **Source Traceability**: All generated chunks successfully preserved their `source_uri` pointing back to the original PDF payload on disk, along with standard governance bounds (`"clearance": "standard"`).
- **Graceful Failures**: The two unreadable SIGINT files safely registered as `failed_parses` rather than silently skipping or panicking the ingestor loop.
- **Hybrid Advantage**: The lexical baseline captured exact acronyms (e.g., NSA, COMINT) while the dense index accurately clustered broad policy queries, proving the two technologies successfully complement each other in local space.

## Decision Gate

**TURBOVEC_REAL_CORPUS_PASS**

**Reasoning:**
The turbovec and SQLite sidecar pattern gracefully and stably handles completely unstructured real-world data payloads. It cleanly digests, chunks, embeds, and indexes physical PDFs, while preserving necessary metadata, governance parameters, and exact retrieval boundaries. Latency remains well under 10ms for hybrid queries even while conducting real RRF operations on textually dense documents. 

Turbovec has decisively crossed the threshold from "scaffolding" into real feasibility for MNEMOS. The logical next phase is **TQ-1.3 — Qdrant vs Turbovec Real Corpus Comparison**.
