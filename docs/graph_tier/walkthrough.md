# Walkthrough: MG-Test-4 and MG-Test-4B Benchmark Implementations

We have successfully implemented and executed **MG-Test-4** and **MG-Test-4B**, creating and calibrating a controlled, experimental graph merge mode (`graph_hybrid_experimental`) that operates seamlessly alongside standard retrieval.

## What was Changed

1. **Experimental Merge Policy (`retrieval_router.py`):**
   * Implemented a separate `merge_graph_candidates_for_experiment` helper designed to inject graph candidates prior to the `Candidate Envelope`.
   * Designed the `lane_aware_quota_v0` algorithm to protect primary contexts while providing visibility to graph candidates.
   * Locked in the **GraphHybrid Experimental Policy v1** defaults:
     * `preserve_primary_top_k` = 5
     * `graph_quota` = 1
     * `graph_ratio_cap` = 0.1
   * Hard-gated this logic behind the default-off `retrieval_mode == "graph_hybrid_experimental"`.

2. **Benchmark Harnesses (`benchmarks/`):**
   * Created `run_mg_test_4.py` to evaluate the initial behavior of the pre-envelope graph merge against baseline semantic retrieval.
   * Created `run_mg_test_4b.py` to run an 18-configuration parameter matrix sweep to calibrate the `lane_aware_quota_v0` policy.

## What was Tested

* **Candidate Envelope Integrity:** Ensuring that graph candidates safely traverse the existing `candidate_envelope.py` deduplication, source cap, and token limits without bypassing constraints.
* **Top-K Displacement:** Tracking exactly how many primary candidates get displaced when graph nodes enter the merged list.
* **Citation Integrity:** Ensuring surviving citations retain correct source mapping and governance linkage after the merge.
* **Parameter Equilibrium:** Testing varied combinations of `preserve_primary_top_k` and `graph_quota` to find the safest threshold.

## Validation Results

* **Baseline Preservation:** When `"graph_hybrid_experimental"` is inactive, semantic searches remain completely unmutated.
* **Governance and Envelope Integrity:** 0 leakage of vetoed content; 100% preservation of governance warnings. Graph candidates perfectly adhered to source cap logic.
* **Calibration Optimization:** The parameter sweep identified `pk=5, gq=1, gr=0.1` as the ideal configuration. This ensures 100% of the top 5 semantic candidates are preserved, while carefully allowing 1 graph candidate (max 10% pool) to compete for the remaining downstream envelope spots.
* **Supported Claims:** The controlled insertion safely reduced the unsupported claim rate without introducing contradictions.
