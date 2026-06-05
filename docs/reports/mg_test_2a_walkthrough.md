# Walkthrough: MG-Test-2A Synthetic Evaluation

## Overview
We executed **MG-Test-2A: Synthetic Offline Graph Candidate Evaluation** to validate the `GraphTier` edge-traversal mechanics, latency, and telemetry using a heavily controlled mock graph. This test ensures the boundary constraints hold up and accurately measures filtering behavior under simulated stress without affecting production components or real datasets.

## Artifacts Created
- [run_mg_test_2.py](file:///g:/MNEMOS/benchmarks/run_mg_test_2.py): The benchmark script injecting the synthetic graph model based on 10 queries from `gate_b_sanity_queries.json`.
- [mg_test_2_metrics.json](file:///g:/MNEMOS/benchmarks/mg_test_2_metrics.json): The raw JSON telemetry captured across the 10 query runs.
- [mg_test_2_evaluation_report.md](file:///C:/Users/vin/.gemini/antigravity/brain/959e7894-a3d3-4736-9b89-d33119d6997f/mg_test_2_evaluation_report.md): The final quantitative report breaking down utility, latency, and hub saturation.

## Implementation Details
1. **Synthetic Corpus Definition**:
   - For each query, 2 target semantic "seed" nodes were mocked.
   - The seeds correctly expanded out into 5 categories of neighbors:
     - **Useful Edge**: Contains known missing support facts.
     - **Useless Edge**: Semantically drifting / distracting node.
     - **Governance Blocked Edge**: Simulates a conflict-vetoed node.
     - **Lineage Incomplete Edge**: Simulates a chunk lacking a valid `source_uri`.
     - **Hub Node**: A highly-connected, generic architectural document linked by every other query.
2. **Execution Parameters**:
   - `RetrievalRouter` was executed with `graph_shadow_enabled=True`.
   - The returned hits were verified to exactly match the baseline semantic hits (meaning graph candidates were strictly restricted to telemetry boundaries).

## Validation Outcomes
- **Utility Tracking**: The test correctly tracked `utility_useful` and `utility_useless` candidates separately in the JSON metrics. 40% of returned graph candidates were specifically tailored facts, while 60% were useless distractions or generic hubs.
- **Hub Saturation**: The architectural hub node (`hub_node_01`) manifested exactly as intended, dominating 5 out of the 10 queries without penalized ranking.
- **Filtering Strictness**: 
  - `graph_governance_filtered_count`: Blocked exactly 10 instances of the vetoed candidate.
  - `graph_lineage_filtered_count`: Blocked exactly 10 instances of the candidate missing a `source_uri`.
- **Latency**: P99 traversal latency was an exceptional **0.54ms**.

## Next Steps
As reported in the evaluation, we recommend a **REVISE** decision. Before integrating graph hits into the `Candidate Envelope`, we must implement:
1. Hub-penalization to naturally suppress overly-connected nodes.
2. Cosine similarity edge scoring to discard distractor nodes before telemetry accumulation.
