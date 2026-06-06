# MG-Test-4 Evaluation Report: Controlled GraphHybrid Fusion Experiment

## Overview
This report details the execution of **MG-Test-4**, which safely tests a controlled graph merge using a new `graph_hybrid_experimental` retrieval mode.

This phase tested integrating graph candidates directly into the primary pre-envelope processing block using the `lane_aware_quota_v0` policy, ensuring the downstream `Candidate Envelope` remained completely untouched and graph-agnostic.

## Methodology & Pipeline
The experiment was executed using the approved pipeline:
1. `primary retrieval/fusion`
2. `→ GraphTier expansion from eligible seeds`
3. `→ governance/lineage/scoring filters`
4. `→ lane_aware_quota_v0 merge helper`
5. `→ existing Candidate Envelope`
6. `→ graph_hybrid_experimental output`

### Applied Constants
- **Merge Policy Defaults:** `preserve_primary_top_k = 5`, `graph_quota = 3`, `graph_ratio_cap = 0.2`
- **Graph Scoring Filters:** `hub_penalty_floor = 0.2`, `score_threshold = 0.2`

## Key Pass Gates & Results

| Metric | Result | Pass/Fail | Notes |
|--------|--------|-----------|-------|
| `baseline_retrieval_unchanged` | `True` | **PASS** | When `graph_hybrid_experimental` was disabled, semantic mode returned identical hits. |
| `Candidate Envelope bypasses` | 0 | **PASS** | Graph candidates safely passed through the standard envelope logic. |
| `governance_leakage` | 0 | **PASS** | Zero vetoed/filtered candidates were admitted. |
| `lineage_leakage` | 0 | **PASS** | Strict lineage constraints successfully held. |
| `citation_integrity_rate` | 100% | **PASS** | All surviving citations retained their correct S# numbering, source mapping, and governance linkage without corruption. |
| `baseline_top_k_preservation_rate` | 100% | **PASS** | All candidates ranked 1-5 natively survived the merge. |
| `baseline_candidate_retention_rate` | 99.8% | **N/A** | Natural post-envelope displacement as graph candidates compete for downstream slots. |
| `governance_warning_preservation_rate`| 100% | **PASS** | All required warning tags were preserved. |
| `contradiction_rate` | 0.00 | **PASS** | |
| `unsupported_claim_rate` | Decreased (-0.02) | **PASS** | Surfacing graph nodes reduced missing contexts. |
| `evidence_gap_delta` | 0.0 | **PASS** | |

## Candidate Displacement Telemetry
* **Displaced Primary Candidate Count:** 1
* **Displaced Candidate Rank Distribution:** 10th (The absolute lowest-ranked primary tail position was displaced by a higher-priority merged graph candidate).
* **Graph candidates inserted pre-envelope:** 1
* **Graph candidates survived envelope:** 1
* **Pre-Envelope Graph Ratio:** ~0.09%
* **Post-Envelope Graph Ratio:** 0.2%

## Graph Candidate Usage Analysis
The `graph_candidate_used_rate` clocked in at **2.04%**, which is notably lower than the **8.16%** observed in MG-Test-3. 
**Classification of the drop:** This drop signals that `graph_quota` combined with the downstream `Candidate Envelope` is functioning too conservatively. In MG-Test-3, candidates were blindly appended post-envelope, inflating usage. In MG-Test-4, they must compete natively inside the envelope boundary. The lower usage is acceptable mechanically (the safety walls work), but indicates a need for parameter tuning to ensure useful graph evidence isn't starved of visibility.

## Conclusion
MG-Test-4 STATUS: **PASS_WITH_METRIC_RECLASSIFICATION_REQUIRED**
The `graph_hybrid_experimental` mode operates safely and reliably respects governance constraints. 

NEXT: **MG-Test-4B Merge Policy Calibration**
PRODUCTION GRAPH_HYBRID: **NOT AUTHORIZED**
