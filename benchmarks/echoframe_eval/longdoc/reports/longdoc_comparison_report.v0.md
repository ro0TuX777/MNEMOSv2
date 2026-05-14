# EchoFrame vs MNEMOS Long-Document Benchmark (compact_safe_v0)

## Summary Metrics
- **Queries Evaluated**: 30
- **Avg Baseline Tokens**: 1819.90
- **Avg json_v0 Tokens**: 1394.43 (Ratio: 0.7662)
- **Avg compact_v0 Tokens**: 1215.37 (Ratio: 0.6678)
- **Avg compact_safe_v0 Tokens**: 1226.83 (Ratio: 0.6741)
- **Best Case Token Reduction**: 0.4016
- **Worst Case Token Regression**: 1.9888
- **Queries where safe beats baseline**: 26
- **Queries where safe loses to baseline**: 4

## Packet Characteristics
- Multiple Sources: 30
- Contradiction Flags: 2
- Approval Required: 2
- Unknown Preservation: 2

## Safety Check Failures
- dropped_provenance: 0
- dropped_gaps: 0
- dropped_contradictions: 0
- dropped_approval: 0
- lost_unknown_preservation: 0
- lost_numeric_threshold: 0
- lost_date: 0
- lost_negation: 0
- lost_exception: 0
- unsupported_synthesis: 0

## Promotion Gates
- **Gate B Provenance**: PASS
- **Gate C Governance Preservation**: PASS
- **Gate D Token Efficiency**: PASS
- **Numeric Threshold Preservation**: PASS
- **Date Preservation**: PASS
- **Negation Preservation**: PASS
- **Exception Language Preservation**: PASS
- **No Unsupported Synthesis**: PASS

**Recommendation**: PROMOTE. EchoFrame achieves stretch targets on long documents safely.

## Category Breakdown (compact_safe_v0 vs baseline)
### exact date retrieval
- Count: 3
- Reduction Ratio: 0.4861
### definition lookup
- Count: 3
- Reduction Ratio: 0.5093
### multi-hop definition dependency
- Count: 3
- Reduction Ratio: 0.5472
### policy exception retrieval
- Count: 3
- Reduction Ratio: 0.4768
### obligation / must / shall retrieval
- Count: 2
- Reduction Ratio: 0.5498
### contradiction across sections
- Count: 2
- Reduction Ratio: 1.9418
### stale vs current section resolution
- Count: 2
- Reduction Ratio: 0.4444
### large excerpt compression
- Count: 2
- Reduction Ratio: 0.5994
### multi-source synthesis
- Count: 2
- Reduction Ratio: 0.4929
### high-risk governance query
- Count: 2
- Reduction Ratio: 1.9566
### unknown / insufficient evidence preservation
- Count: 2
- Reduction Ratio: 0.5645
### API or config key recall
- Count: 2
- Reduction Ratio: 0.5293
### exact numeric threshold retrieval
- Count: 2
- Reduction Ratio: 0.6480