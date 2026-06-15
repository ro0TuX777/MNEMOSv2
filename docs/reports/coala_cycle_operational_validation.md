# CoALA Cycle Operational Validation

Date: June 14, 2026

## Summary Decision

PHASE_1_COALA_CYCLE_VALIDATION_READY

The validation posture is implemented as a deterministic local harness:

```bash
python tools/run_coala_cycle_validation.py --summary-only
```

The harness builds representative `CognitiveCycleRecord` instances without an LLM or live service dependency, then validates the contract gates that matter for MNEMOS v3.2 cognitive-cycle transparency.

## Representative Query Set

- `CLASS_A_DIRECT_LOOKUP`
- `CLASS_B_MULTI_HOP`
- `CLASS_C_GLOBAL_SYNTHESIS`
- `CONTRADICTION_RECONCILIATION`
- `HIGH_VOLATILITY_GOVERNANCE`
- `FORECAST_TRIGGERED_PULSE`
- `PRE_COGNITIVE_SHADOW_SEARCH`
- `DERIVED_VIEW_EVIDENCE_BUNDLE`

This covers the primary MNEMOS cognitive paths: direct lookup, multi-hop retrieval, global synthesis, contradiction handling, high-volatility governance, forecast-triggered pulse behavior, pre-cognitive shadow search, and derived evidence bundles.

## Gates

| Gate | Validation |
| --- | --- |
| `attention_faithfulness` | Every attention decision must include a runtime/config `policy_source`; LLM or generated-explanation sources are rejected. |
| `bounded_record` | Serialized cycle records must remain under the bounded size limit and keep `query_or_event` capped at 240 characters. |
| `redaction` | Cycle and forecast records must not expose sensitive keys such as secrets, tokens, raw prompts, private reasoning, or raw engram content. |
| `sam_compatibility` | Records must include the stable SAM-facing keys and operation-type labels on action records. |
| `forecast_resolution` | Forecast-triggered cycles must link to a resolved `ForecastOutcomeRecord` lifecycle. |
| `learning_boundary` | Learning writes must declare an explicit write class; candidates remain advisory and non-authoritative. |

Key invariant:

> The cycle record must be evidence-derived, not explanation-generated.

In implementation terms, attention decisions are validated from runtime, router, feature-flag, forecast, and governance metadata rather than from generated summary text.

## Semantic/Procedural Learning Boundary

The write classes are explicit in `mnemos.cognitive.learning_boundary`:

- `episodic_write`
- `semantic_write`
- `semantic_candidate_write`
- `governance_score_update`
- `forecast_outcome_write`
- `cache_write`
- `audit_write`
- `procedural_change_candidate`
- `blocked_procedural_mutation`

`audit_write` is append-only and non-authoritative for retrieval. Pattern candidates are split by behavioral risk:

| Pattern type | Boundary class |
| --- | --- |
| Descriptive pattern candidate | `semantic_candidate_write` |
| Operational recommendation candidate | `procedural_change_candidate` |
| Policy, routing, or template mutation | `blocked_procedural_mutation` until approved |

## PatternEngramCandidate

`PatternEngramCandidate` is implemented in advisory mode only. Candidates include:

- `candidate_id`
- `pattern_summary`
- `supporting_cycle_ids`
- `supporting_engram_ids`
- `contradicting_engram_ids`
- `confidence_score`
- `support_score`
- `contradiction_score`
- `promotion_status`
- `governance_review_required`
- `pattern_type`
- `recommended_scope`
- `applies_when`
- `does_not_apply_when`
- `risk_if_wrong`
- `source_cycle_count`
- `first_seen_at`
- `last_seen_at`
- `proposed_learning_class`

High-confidence gates may set `promotion_status=promotion_recommended`, but they do not promote a candidate to an authoritative `PatternEngram`. Promotion requires explicit governance approval in this first implementation.

## Test Coverage

```bash
python -m pytest tests/test_coala_cycle_validation.py tests/test_learning_boundary.py
```

The tests verify representative path coverage, gate pass/fail behavior, evidence-derived attention, bounded and redacted records, forecast lifecycle resolution, advisory-only pattern candidates, and explicit governance approval for promotion.
