# Goal Description
The objective of MG-Test-4B is a small calibration phase designed to determine the optimal balance for the `lane_aware_quota_v0` policy. 

Because MG-Test-4 indicated that the initial graph quotas were functioning correctly but were slightly too conservative (dropping candidate usage compared to offline tests), this test will sweep a matrix of configuration values to find the safest merge policy that provides graph candidates sufficient visibility without displacing important dense/lexical evidence.

## Experimental Scope
We will sweep the following parameter matrix across the `graph_hybrid_experimental` retrieval mode:
* `preserve_primary_top_k`: [5, 7]
* `graph_quota`: [1, 2, 3]
* `graph_ratio_cap`: [0.1, 0.2, 0.3]

The baseline Candidate Envelope limits remain exactly the same.

## Proposed Changes

### [MODIFY] [run_mg_test_4b.py](file:///g:/MNEMOS/benchmarks/run_mg_test_4b.py)
* Duplicate `run_mg_test_4.py` and modify it into an iterative grid search.
* The script will iterate through all 18 combinations of the matrix above against the dense/noisy graph profile.
* For each combination, the script will record the following metrics:
  - `citation_integrity_rate`
  - `baseline_top_k_preservation_rate`
  - `baseline_candidate_retention_rate`
  - `graph_candidate_used_rate`
  - `unsupported_claim_rate_delta`
  - `contradiction_rate_delta`
  - `evidence_gap_delta`
  - `primary_candidates_displaced_count`
  - `graph_candidates_survived_envelope`

### [MODIFY] [retrieval_router.py](file:///g:/MNEMOS/mnemos/retrieval/retrieval_router.py)
* Modify the `merge_graph_candidates_for_experiment` helper to dynamically accept `preserve_primary_top_k`, `graph_quota`, and `graph_ratio_cap` arguments from the benchmark script rather than hardcoding them, allowing the harness to iterate properly.

## User Review Required
Are there any other specific bounds or edge cases we should track during this parameter sweep? 

## Verification Plan

### Output Artifacts
1. **`mg_test_4b_calibration_results.json`**: Matrix of all 18 configurations and their resultant metrics.
2. **`mg_test_4b_evaluation_report.md`**: Analysis of the sweep, recommending the optimal configuration that maximizes `graph_candidate_used_rate` without negatively impacting `baseline_top_k_preservation_rate`.
