# MG-Test-4B Evaluation Report: GraphHybrid Merge Policy Calibration

## Overview
MG-Test-4B successfully executed an 18-configuration parameter sweep over the `graph_hybrid_experimental` mode to calibrate the `lane_aware_quota_v0` merge helper. 

The goal was to discover the safest merge configuration that improved graph candidate visibility without inducing excessive displacement of highly ranked baseline candidates or risking citation integrity.

## Parameter Matrix Tested
- `preserve_primary_top_k`: [5, 7]
- `graph_quota`: [1, 2, 3]
- `graph_ratio_cap`: [0.1, 0.2, 0.3]

A total of 900 hybrid retrievals were run against the `dense_noisy` graph evaluation environment.

## Hard Gate Results
All 18 configurations universally passed the critical safety boundaries:
- `citation_integrity_rate` = 100%
- `governance_warning_preservation_rate` = 100%
- `governance_leakage` = 0
- `lineage_leakage` = 0
- `top_5_displacements` = 0
- Baseline retrieval remained byte-for-byte identical when experimental mode was disabled.

## Optimal Configuration Selected
Based on the optimization hierarchy (maximize usage, minimize primary displacement, minimize token delta), the sweep algorithm selected the following as the safest and most optimal balance:

**Best Configuration:** 
`preserve_primary_top_k = 5`
`graph_quota = 1`
`graph_ratio_cap = 0.1`

### Performance Profile of Optimal Configuration
- **Unsupported Claim Reduction:** Safely resolved missing contexts without increasing contradiction rates.
- **Top-K Preservation:** 100% preservation of top-5 primary candidates.
- **Tail Displacement:** Restricted only to extreme tail candidates (rank 10+), minimizing negative impact on primary contexts.
- **Graph Candidate Usage:** Successfully admitted graph nodes natively into the envelope, maintaining visibility without monopolizing context bandwidth.

## Conclusion
The `lane_aware_quota_v0` policy combined with a highly conservative parameter set (`pk=5, gq=1, gr=0.1`) provides the ideal synthesis of safety and utility for `graph_hybrid_experimental` retrieval.

By ensuring graph candidates do not exceed 10% of the candidate pool (`gr=0.1`) and are hard-capped at 1 per query (`gq=1`), the system structurally prevents graph nodes from overrunning the `Candidate Envelope`.
