# Turbovec TQ-1.3 Qdrant vs Turbovec Real Corpus Benchmark

## Official Decision Gate
**TURBOVEC_PORTABLE_PROFILE_CANDIDATE**

## Performance Metrics
### Ingestion
- Qdrant: 0.224928617477417s
- Turbovec: 0.7072792053222656s

### Latency (p50)
- qdrant_dense: 30.38 ms
- turbovec_dense: 11.48 ms
- turbovec_hybrid: 13.30 ms
- sqlite_fts_only: 0.59 ms

### Retrieval Quality
- qdrant_dense: 0.296
- turbovec_dense: 0.296
- turbovec_hybrid: 0.296
- sqlite_fts_only: 0.160
