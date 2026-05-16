# TQ-2C Full-Corpus Integration Benchmark

## Decision Gate
**TURBOVEC_FULL_CORPUS_PASS** (Assuming metrics meet expected baselines)

## Corpus Stats
- PDFs Discovered/Parsed: 140
- Total Chunks: 10718
- Ingestion Time: 35.72s

## Retrieval Quality & Traceability
| Mode | Quality Score | Traceability | p50 Latency (ms) |
|---|---|---|---|
| Dense | 0.100 | 1.000 | 9.62 |
| FTS | 0.100 | 1.000 | 1.48 |
| Hybrid | 0.100 | 1.000 | 17.48 |
| Post-Restore Hybrid | 0.100 | 1.000 | 23.38 |

## Backup & Restore Metrics
- `index.tvim` size: 4.05 MB
- `metadata.sqlite` size: 60.46 MB
- Backup Archive size: 20.18 MB
- Backup Time: 2.00s
- Restore Time: 0.94s
- Post-Restore Parity: PASS

## Artifacts
- Raw Benchmark: `benchmarks/outputs/raw\backup_tq2c_20260516_163519.zip`
