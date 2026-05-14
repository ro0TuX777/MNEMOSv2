# EchoFrame vs MNEMOS Real-Corpus Benchmark (compact_selective_v0)

## Summary Metrics
- **Queries Evaluated**: 30
- **Avg Baseline Tokens**: 2064.97
- **Avg compact_safe_v0 Tokens**: 1854.27
- **Avg compact_selective_v0 Tokens**: 1839.13 (Ratio: 0.8906)
- **Queries where selective beats safe**: 7
- **Fallback Count**: 21
- **High-Risk Fallbacks**: 3
- **Contradiction Fallbacks**: 2
- **Total Facts Pinned**: 555

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
- **No Fabricated Sources**: PASS
- **Selective Fact Drops Documented**: PASS
- **High Risk Do Not Overcompress**: PASS
- **Unknown Do Not Become Confident**: PASS

**Recommendation**: CONTINUE EXPERIMENT. Safety passes but token ratio (0.8906) remains > 0.85.