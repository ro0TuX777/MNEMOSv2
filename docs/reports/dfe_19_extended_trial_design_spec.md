# DFE-19 Extended Operator Trial Design Specification

## Overview
Phase `DFE-18R` verified genuine multi-operator value, proving that the Derived Fact lane structurally lowers cognitive burden and improves confidence. This specification dictates the expanded constraints for an Extended Operator Trial (`DFE-20`) to test the system across a larger corpus, wider user base, and explicit API integration boundaries.

## Trial Constraints & Scope

### Operational Scope
- **Operators:** 15–30 approved operators.
- **Duration:** 4–6 weeks.
- **Corpus:** Full production document corpus.
- **Automatic Promotion:** Prohibited.

### Production Boundary & API Exposure
- **Feature-Flagged API Access:** Derived Facts are authorized to be served via `/api/v1/query` **ONLY** when an explicit opt-in parameter (`?enable_derived_facts=true`) is provided by the client.
- **Default Retrieval:** Default requests missing the parameter must remain completely unaffected. The baseline retrieval flow remains heavily protected.

### Minimum Trial Activity
To ensure sufficient statistical durability against edge-cases:
- At least 250 evaluated queries.
- At least 50 surfaced Derived Fact events.
- At least 15 active operators logging feedback.

If these volume bounds are not met, the trial must hold and not progress to production enablement.

## Telemetry & Success/Failure Thresholds

### Success Thresholds (Required for DFE-20 Pass)
- `operator_confidence_delta_avg` > 0
- `review_burden_delta_avg` < 0
- `rendered_support_quality_avg` >= 4.0
- `operator_override_rate` <= 5%
- `claim_strength_issue_rate` <= 1%

### Required Hard Gates & Kill Switches
Any failure of the following must automatically and instantaneously kill the trial and revert the API to default:
- `safety_issue_count` = 0
- `default_retrieval_unchanged` = true
- `authority_label_missing_count` = 0

## Post-Trial Decision Outcomes (DFE-20)
Following the Extended Trial execution, the results will drive the final production deployment decision:
- `DFE_20_PASS_AUTHORIZE_PRODUCTION_API_INTEGRATION`
- `DFE_20_REVISE_UI_OR_THRESHOLDS`
- `DFE_20_HOLD_PENDING_MORE_DATA`
- `DFE_20_STOP_DERIVED_FACT_LANE`
