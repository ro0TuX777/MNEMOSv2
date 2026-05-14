# EchoFrame vs MNEMOS Real-Corpus Benchmark (compact_semantic_minEvidence_v0)

## Summary Metrics
- **Queries Evaluated**: 30
- **Avg Baseline Tokens**: 2064.97
- **Avg compact_semantic_v0 Tokens**: 1770.67
- **Avg compact_semantic_minEvidence_v0 Tokens**: 1324.83 (Ratio: 0.6416)
- **Queries where minEvidence beats semantic**: 22
- **Fallback Count**: 0
- **High-Risk Fallbacks**: 0
- **Contradiction Fallbacks**: 0
- **Total Facts Pinned**: 287
- **E0 (Pointer Only) Renderings**: 548
- **E2 (Window) Renderings**: 127
- **E3 (Excerpt) Renderings**: 225

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
- **Gate D Token Efficiency**: PASS
- **Numeric Threshold Preservation**: PASS
- **Date Preservation**: PASS
- **Negation Preservation**: PASS
- **Exception Language Preservation**: PASS
- **No Fabricated Sources**: PASS
- **Semantic Fact Drops Documented**: PASS
- **High Risk Do Not Overcompress**: PASS
- **Unknown Do Not Become Confident**: PASS
- **No Unsupported Synthesis**: PASS

**Recommendation**: PROMOTE TO SHADOW RUNTIME CANDIDATE. compact_semantic_minEvidence_v0 passes all safety gates and token ratio <= 0.85.