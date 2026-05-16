# Turbovec TQ-1 Real Adapter Benchmark

## Environment
- **OS**: Windows
- **Rust Toolchain**: cargo 1.84.1 (with `rustup default nightly`)
- **Python**: 3.12 (Anaconda)
- **Turbovec Module**: `<module 'turbovec' from 'C:\\Users\\vin\\anaconda3\\Lib\\site-packages\\turbovec\\__init__.py'>`

## Configuration
- **Corpus Size**: 10000
- **Embedding Dimension**: 768
- **Bit Width**: 4

## Metrics
- **Ingestion Rate**: 167.27 docs/sec (Batched vector initialization)
- **Dense Latency (Avg)**: 103.37 ms
- **Hybrid Latency (Avg)**: 10.14 ms
- **Index Size on Disk** (`index.tvim`): ~3.9 MB
- **Metadata Size on Disk** (`metadata.sqlite`): ~11.0 MB

## Validation Outcomes
- **Delete Exclusion Result**: PASS (Soft-deleted Enrgrams successfully excluded from dense/hybrid results)
- **Persistence Result**: PASS (Index reloading verified to hold exact original count and mapping)
- **Filter Correctness Result**: PASS (Metadata sidecar filtering properly slices target pools)
- **Hybrid RRF Correctness Result**: PASS (RRF merging successfully re-ranks dual matches to top ranks natively in python)

## Decision

**Decision:** `TURBOVEC_REAL_ADAPTER_READY`

We successfully established local Python bindings using the nightly Rust toolchain. The `turbovec` crate efficiently compiles and maps directly into the TurbovecTier local RRF architecture.

With persistence, delete exclusion, and basic operational metrics successfully recorded against a 10K dataset with an extremely efficient disk footprint (~15MB total), we are clear to advance to the repeatable 100K scaling drills and Qdrant comparison benchmarks.
