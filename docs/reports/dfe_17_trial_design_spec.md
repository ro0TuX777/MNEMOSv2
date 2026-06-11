# DFE-17 Limited Operator Trial Design Specification

## Overview
Phase `DFE-16` produced a positive single-operator validation signal. The Derived Fact lane improved reviewer confidence and reduced review burden in the reviewed sample without producing override, safety, or claim-strength issues. This specification defines the bounds for a controlled, live Limited Operator Trial.

## Trial Constraints & Scope

### Operational Scope
- **Operators:** 2–5 approved operators.
- **Duration:** 1–2 weeks.
- **Mode:** Limited live shadow/cockpit trial. Operators may view Derived Fact candidates in an explicitly labeled evaluation panel, but Derived Facts remain excluded from production default retrieval and blocked from default `/api/v1/query` responses.
- **Corpus:** Approved real PDFs only (e.g., from the DFE-12B real-corpus domains).
- **Automatic Promotion:** Prohibited.

### Minimum Trial Activity
A valid execution must meet the following baseline usage metrics to avoid low-N statistical errors:
- At least 2 operators complete review.
- At least 25 total evaluated queries.
- At least 5 queries per evaluation category where possible.
- At least 10 surfaced Derived Fact events or clear documentation that the lane correctly failed closed when no candidates qualified.

If these minimums are not met, the trial must halt with: `DFE_18_HOLD_INSUFFICIENT_OPERATOR_SAMPLE`.

## User Interface Presentation Rules
- **Labeling:** All Derived Facts must be visibly flagged as `[CANDIDATE]` or `[DERIVED_FACT_LANE]`.
- **Rendered Support:** All selected facts must prominently display the exact rendered support excerpt from the source document.
- **Feedback Mechanism:** Operator feedback capture is required per query or session.

## Telemetry & Success/Failure Thresholds

### Success Thresholds (Required to Pass Trial)
- `operator_confidence_delta_avg` > 0
- `review_burden_delta_avg` < 0
- `rendered_support_quality_avg` >= 4.0
- `source_support_quality_avg` >= 4.0
- `selected_fact_usefulness_avg` >= 2.5 / 4
- `operator_override_rate` <= 5%
- `claim_strength_issue_rate` <= 1%
- `safety_issue_count` = 0

### Required Hard Gates
- `baseline_derived_fact_count` = 0
- `default_retrieval_unchanged` = true
- Derived Facts remain blocked from default `/api/v1/query`
- `authority_label_missing_count` = 0
- `safety_issue_count` = 0
- `kill_switch_success` = true
- Human feedback capture required
- All Derived Facts visibly labeled as derived/candidate
- All selected facts require rendered support excerpts

### Operator Override Taxonomy
When logging an override (`operator_override_yes_no = YES`), the reason must be systematically captured using one of the following:
- `UNSUPPORTED_BY_SOURCE`
- `WEAK_RENDERED_SUPPORT`
- `MISLEADING_CLAIM_STRENGTH`
- `WRONG_SOURCE`
- `LOW_USEFULNESS`
- `CONFUSING_PRESENTATION`
- `AUTHORITY_LABEL_UNCLEAR`
- `OTHER`

## Post-Trial Decision Outcomes (DFE-18)
Upon completion, the trial must be formally evaluated against the `DFE-18` gates, resulting in exactly one of the following:
- `DFE_18_PASS_RECOMMEND_EXTENDED_OPERATOR_TRIAL`
- `DFE_18_PASS_KEEP_LIMITED_SHADOW_PANEL`
- `DFE_18_REVISE_UI_PRESENTATION`
- `DFE_18_REVISE_DERIVED_FACT_SELECTION`
- `DFE_18_REVISE_RENDERED_SUPPORT_POLICY`
- `DFE_18_REVISE_OPERATOR_FEEDBACK_WORKFLOW`
- `DFE_18_HOLD_INSUFFICIENT_OPERATOR_SAMPLE`
- `DFE_18_STOP_DERIVED_FACT_LIVE_TRIAL_PATH`
