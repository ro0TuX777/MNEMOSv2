# Turbovec TQ-1 100K Repeatability Benchmark

## Aggregated Metrics (3 Runs)
- **Ingestion Rate (docs/sec)**: Mean: 407.31 | Min: 394.10 | Max: 416.26 | Stdev: 11.67
- **Dense Latency p50 (ms)**: Mean: 6.21 | Min: 5.53 | Max: 7.06 | Stdev: 0.78
- **Dense Latency p95 (ms)**: Mean: 30.13 | Min: 26.59 | Max: 34.58 | Stdev: 4.07
- **Dense Latency p99 (ms)**: Mean: 345.49 | Min: 322.76 | Max: 372.87 | Stdev: 25.38
- **Hybrid Latency p50 (ms)**: Mean: 37.31 | Min: 33.88 | Max: 44.15 | Stdev: 5.93
- **Hybrid Latency p95 (ms)**: Mean: 45.07 | Min: 41.58 | Max: 49.21 | Stdev: 3.86
- **Hybrid Latency p99 (ms)**: Mean: 47.90 | Min: 44.99 | Max: 49.82 | Stdev: 2.57
- **Index Size (MB)**: Mean: 37.77 | Min: 37.77 | Max: 37.77 | Stdev: 0.00
- **Metadata DB Size (MB)**: Mean: 61.57 | Min: 61.57 | Max: 61.57 | Stdev: 0.00
- **Save Time (sec)**: Mean: 0.07 | Min: 0.07 | Max: 0.08 | Stdev: 0.01
- **Load Time (sec)**: Mean: 0.02 | Min: 0.02 | Max: 0.02 | Stdev: 0.00

## Correctness Gates
- Delete Exclusion: PASS (3/3)
- Persistence Reload: PASS (3/3)
- Filters/RRF Correctness: PASS (3/3)

## Decision Gate
**TURBOVEC_100K_REPEATABILITY_PASS**

All 3 runs completed successfully with stable persistence and zero semantic drift. Disk footprint scaled linearly and comfortably. Latencies stayed within operational safety limits.