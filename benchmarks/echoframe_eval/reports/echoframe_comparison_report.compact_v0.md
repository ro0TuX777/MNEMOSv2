# EchoFrame vs MNEMOS Baseline Comparison Report (compact_v0)

## Summary Metrics
- **Total Queries Evaluated**: 20
- **Avg Baseline Tokens**: 892.35
- **Avg JSON_v0 Tokens**: 946.05
- **Avg Compact_v0 Tokens**: 716.00
- **JSON_v0 Token Reduction Ratio**: 1.0602
- **Compact_v0 Token Reduction Ratio**: 0.8024
- **Queries where compact < baseline**: 8
- **Queries where compact > baseline**: 12
- **Queries where compact < json_v0**: 20

## Governance & Provenance Violations (compact_v0)
- Dropped Provenance: 0
- Dropped Evidence Gaps: 0
- Dropped Contradictions: 0
- Dropped Approval Requirements: 0
- Lost Unknown Preservation: 0
- Invalid/Empty Packets: 0

## Promotion Gates
- **Gate A Fidelity**: N/A
- **Gate B Provenance**: PASS
- **Gate C Governance Preservation**: PASS
- **Gate D Token Efficiency**: PASS
- **Gate E Context Stability**: N/A
- **Gate F Explainability**: PASS

## Category Breakdown (compact_v0 vs baseline)
### exact fact retrieval
- Count: 2
- Reduction Ratio: 0.3888
- Dropped Signals: 0
### multi-hop source resolution
- Count: 2
- Reduction Ratio: 0.5603
- Dropped Signals: 0
### contradiction handling
- Count: 2
- Reduction Ratio: 1.1169
- Dropped Signals: 0
### code/API recall
- Count: 2
- Reduction Ratio: 0.5573
- Dropped Signals: 0
### policy/obligation recall
- Count: 2
- Reduction Ratio: 0.6766
- Dropped Signals: 0
### low-risk general query
- Count: 2
- Reduction Ratio: 2.8146
- Dropped Signals: 0
### high-risk governance query
- Count: 2
- Reduction Ratio: 1.1542
- Dropped Signals: 0
### repeated query stability
- Count: 2
- Reduction Ratio: 2.0968
- Dropped Signals: 0
### stale evidence
- Count: 2
- Reduction Ratio: 2.3414
- Dropped Signals: 0
### insufficient evidence / unknown preservation
- Count: 2
- Reduction Ratio: 0.7385
- Dropped Signals: 0