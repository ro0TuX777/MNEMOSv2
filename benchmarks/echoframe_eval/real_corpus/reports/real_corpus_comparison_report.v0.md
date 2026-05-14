# EchoFrame vs MNEMOS Real-Corpus Benchmark (compact_safe_v0)

## Summary Metrics
- **Queries Evaluated**: 30
- **Avg Baseline Tokens**: 2064.97
- **Avg compact_safe_v0 Tokens**: 1854.27 (Ratio: 0.8980)
- **Best Case Token Reduction**: 0.6522
- **Worst Case Token Regression**: 2.2730
- **Queries where safe beats baseline**: 16
- **Queries where safe loses to baseline**: 14

## Packet Characteristics
- Multiple Sources: 30
- Contradiction Flags: 0
- Approval Required: 3
- Unknown Preservation: 0

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
- fabricated_sources: 0

## Promotion Gates
- **Gate B Provenance**: PASS
- **Gate C Governance Preservation**: PASS
- **Gate D Token Efficiency**: FAIL
- **Numeric Threshold Preservation**: PASS
- **Date Preservation**: PASS
- **Negation Preservation**: PASS
- **Exception Language Preservation**: PASS
- **No Unsupported Synthesis**: PASS
- **No Fabricated Sources**: PASS

**Recommendation**: CONTINUE EXPERIMENT. EchoFrame is 100% safety-preserving on real documents, but token efficiency (0.8980) failed to beat the 0.85 target. Additional compression intelligence is required.

## Category Breakdown (compact_safe_v0 vs baseline)
### architecture
- Count: 4
- Reduction Ratio: 0.8701
### governance
- Count: 2
- Reduction Ratio: 0.7353
### policy
- Count: 3
- Reduction Ratio: 0.8285
### config
- Count: 3
- Reduction Ratio: 1.1857
### multi-hop
- Count: 2
- Reduction Ratio: 0.6598
### contradiction
- Count: 2
- Reduction Ratio: 1.2502
### stale vs current
- Count: 2
- Reduction Ratio: 0.8919
### obligation
- Count: 3
- Reduction Ratio: 0.7024
### high-risk
- Count: 3
- Reduction Ratio: 1.3441
### unknown
- Count: 3
- Reduction Ratio: 1.0435
### definition
- Count: 2
- Reduction Ratio: 0.8699
### multi-source
- Count: 1
- Reduction Ratio: 0.7079