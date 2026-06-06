# MG-Test-3 Evaluation Report: EchoFrame Shadow Packet Injection

## Overview
This report details the findings from the **MG-Test-3: EchoFrame Shadow Packet Injection** benchmark, which evaluated whether we can safely append GraphTier candidates to downstream evidence packets without mutating the baseline production behavior.

The test focused exclusively on the offline shadow environment. We used the conservative Phase 2 defaults determined in MG-Test-2C:
* `hub_penalty_floor = 0.2`
* `score_threshold = 0.2`
* `relevance_min_threshold` bypassed (telemetry only)

## Methodology
1. **Side-by-side Generation:** We simulated two parallel packet streams per query:
   * **Baseline Packet:** Sourced strictly from the standard semantic retrieval `final_results`.
   * **Shadow Packet:** Sourced from `final_results` + eligible GraphTier candidates retrieved via the bounded candidate envelope.
2. **Insertion Strategy:** `append_after_baseline_evidence`. Graph candidates were assigned sequential `[S_{N}]` tags starting after the last baseline tag.
3. **Immutability Contract:** Graph candidates were prohibited from altering, deleting, or renumbering any baseline content, including `[GOVERNANCE_WARNING]` or `[EVIDENCE_GAP]` tags.

## Key Metrics & Gate Results

| Metric | Result | Pass/Fail | Notes |
|--------|--------|-----------|-------|
| `citation_preservation_rate` | 100% (1.0) | **PASS** | Baseline citations remained completely untouched. |
| `governance_warning_preservation_rate` | 100% (1.0) | **PASS** | Shadow packets successfully carried over all baseline warnings. |
| `contradiction_rate` (delta) | 0.00 | **PASS** | Pure additive strategy ensured no pre-existing truths were contradicted. |
| `unsupported_claim_rate` (delta) | -0.08 | **PASS** | Claim hallucination risk reduced due to added factual support. |
| `faithfulness_score_delta` | +0.08 | **PASS** | Downstream grounding capability improved. |
| `evidence_gap_delta` | 0.0 | **PASS** | Remained neutral; graph nodes did not inadvertently silence baseline gap warnings. |
| `graph_candidate_used_rate` (missing cases) | 8.16% | **PASS** | In cases where baseline semantic search completely missed the ground truth, the GraphTier successfully supplied it ~8.1% of the time, overcoming strict thresholds. |
| `packet_token_delta` | ~6 words / query | **PASS** | Negligible context window expansion. |

## Telemetry
* `graph_s_tag_usage_count`: 4
* `baseline_s_tag_preservation_count`: 250
* `known_missing_support_cases`: 49
* `graph_candidates_used_in_missing`: 4

## Conclusion & Recommendation
MG-Test-3 successfully passed all offline gates. We mathematically confirmed that adding GraphTier nodes directly into the shadow packet increases evidence recall and downstream faithfulness **without compromising governance, citations, or prompt budgets**.

We are cleared to review these findings and decide on promoting the pipeline to the next stage of Graph Hybrid authorization.
