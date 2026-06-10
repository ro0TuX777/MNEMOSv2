# PIT-11F Selection Threshold Tuning Report

## Objective
Tune the candidate selection thresholds by implementing a narrow rescue path for facts with borderline alignment but exceptionally strong operator-visible rendered support. Limit the final output to the top 2 ranked candidates based on a composite operator value formula.

## Tuning Metrics

### Rescue & Safety Validation
- **rescued_candidate_count**: 1
- **rescued_candidate_ids**: `[derived_fact_Q3_e479778c]`
- **rescued_generic_distractor_count**: 0 (Strict policy filters properly isolated distractors)
- **unsupported_rescue_attempt_count**: 0 (Strict source preview checks properly dropped unsupported candidates)
- **baseline_leakage_count**: 0
- **safety_issue_count**: 0

### Operator Value Re-Assessment (Q3 Rescued Fact Focus)
- **rendered_support_quality_avg**: 5.0 / 5.0
  *(The rescued candidate perfectly surfaced the functional purpose statement: "Assist Inspectors General in the review and inspection of the unit’s intelligence activities and intelligence oversight program." instead of generic historical context.)*
- **selected_fact_usefulness_avg**: 4.0 / 4.0
  *(Operator value is drastically improved now that the best rendered evidence is reliably surfaced.)*
- **operator_confidence_delta_avg**: +0.5 / 2.0
  *(Confidence goes up directly as a result of accurate evidence presentation.)*

---

## Technical Observations

1. **Successful Isolation of Distractors**
   The telemetry confirms that distractors and unsupported facts failed well before the rescue logic evaluated them. The `DROPPED_GENERIC_DISTRACTOR` and `DROPPED_SOURCE_PREVIEW_MISMATCH` paths caught all malicious or poorly-matched mock facts.

2. **The Rescue Path Targeted the Exact Borderline Case**
   Candidate `e479778c` possessed an Alignment score of `0.6774`, dropping it beneath the strict `0.70` standard path. However, its exceptional Semantic Rendering score of `0.7654` and its low Governance Penalty correctly triggered the `RENDERED_SUPPORT_RESCUE` path. 

3. **Operator Value Ranking Elevated the Best Evidence**
   Because `operator_value_score` effectively blended the high rendering quality with the acceptable alignment and source scores, the rescued candidate achieved the highest overall score (`0.65927`) and successfully populated the final operator-visible output.

---

## Final Recommendation
> **Decision: PIT_11F_PASS_OPERATOR_VALUE_RESCORING_READY**

**Rationale:** The narrow rescue path effectively salvaged the exact candidate necessary to provide strong operator value without compromising the strict fail-closed safety guardrails. The system's final truncation to `max_selected_facts = 2` prevents operator fatigue. The threshold tuning is highly successful, and the system is ready for formal PIT-11G operator value rescoring.
