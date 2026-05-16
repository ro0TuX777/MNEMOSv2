# TQ-1 Phase Closeout: Turbovec Retrieval Tier

## Phase Summary
The TQ-1 engineering phase successfully prototyped, implemented, and benchmarked the `TurbovecTier` for the MNEMOS memory service. This involved integrating the Rust-native `turbovec` SIMD index alongside an SQLite FTS5 metadata sidecar, and demonstrating local Reciprocal Rank Fusion (RRF). 

## TQ-1 Decision: `TURBOVEC_PORTABLE_PROFILE_CANDIDATE`

### Rationale
Turbovec + SQLite matched Qdrant Dense retrieval quality on the canonical real-PDF diagnostic subset (154 semantic chunks), preserved `source_uri` / `dataset` / `engram_uuid` traceability, and achieved lower local p50 retrieval latency with an incredibly compact disk footprint (~1.5 MB). The tier is highly suitable for further shadow/profile integration as a local-first portable memory appliance.

### Boundary Constraints
**This does not replace Qdrant.** Qdrant remains the Core Memory Appliance backend for service-oriented, containerized, and scalable deployments. Turbovec advances purely as a portable/offline profile candidate pending full-corpus repeatability, cross-platform validation, and operator-runbook gates.

## Benchmark Artifacts
- **TQ-1 (10K Synthetic)**: Correctness & performance scaling verified.
- **TQ-1.1 (100K Repeatability)**: High volume, cold-start, and persistence validated. 
- **TQ-1.2 & 1.3 (Real PDF Corpus vs Qdrant)**: Apples-to-apples evaluation confirming equivalent semantic retrieval fidelity to Qdrant without HTTP overhead.

## Next Engineering Phase: TQ-2
**Phase TQ-2 — Portable Memory Appliance Profile Integration** is officially open.

### TQ-2 Objectives:
1. Formalize installer profile support.
2. Implement runtime profile selection based on `MNEMOS_PROFILE`.
3. Add health/capabilities reporting for the `turbovec` dependency.
4. Establish backup/restore commands for `index.tvim`, `metadata.sqlite`, and `manifest.json`.
5. Run full-corpus benchmark gates and cross-platform (Linux/Windows) verifications.
