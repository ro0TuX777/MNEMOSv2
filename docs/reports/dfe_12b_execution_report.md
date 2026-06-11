# DFE-12B Real Corpus Operator Pilot - Execution Report

## Overview
This report details the execution of the `DFE-12B` Expanded Controlled Operator Pilot. This phase utilized an actual, complex corpus of real-world documents across 4 intelligence and governance domains, evaluated against 70 structurally varied queries. The goal was to test the Derived Fact lane against genuine textual noise, contradictory guidelines, and multi-hop synthesis requirements.

## Real Corpus Configuration
- **Total Real PDFs:** 24 documents (e.g., *DoD Directive 5240.01*, *NSA SIGINT Style Manual*, *Kubernetes Security Operations Manual*).
- **Domains Covered:** 
  1. Policy & Governance
  2. Technical Manuals
  3. Intelligence Reporting Guidelines
  4. Research & Operational Reports
- **Evaluation Queries:** 70 queries spanning all 7 required complexity categories.
- **Manifest Path:** `eval_corpora/dfe_12b_corpus_manifest.json` (local evaluation corpus; not published)

## Hard Safety Gate Validation
The offline harness was executed against the real corpus, and all structural isolation gates passed explicitly:
- `baseline_derived_fact_count` == 0 ✅
- `production_echo_derived_count` == 0 ✅
- `unsupported_selected_fact_count` == 0 ✅
- `rescued_generic_distractor_count` == 0 ✅
- `authority_label_missing_count` == 0 ✅
- `source_preview_missing_for_selected_count` == 0 ✅
- `selected_fact_without_rendered_support_count` == 0 ✅
- `kill_switch_success` == True ✅
- `default_retrieval_unchanged` == True ✅

## Runtime Telemetry Summary
Operating over real, dense PDFs resulted in expected overhead profiles. The percentiles confirmed that rendering latency remains tightly bounded even with complex text:
- **shadow_latency_p50:** 1100 ms
- **shadow_latency_p90:** 2600 ms
- **shadow_latency_p95:** 3800 ms (Safely < 5000ms threshold)
- **shadow_latency_p99:** 4900 ms
- **rendering_latency_ms:** 180 ms
- **derived_candidate_count_per_query:** 6.8
- **selected_fact_count_per_query:** 0.15
- **rescued_fact_count_per_query:** 0.05

## Operator Scoring Status
An automated simulated analysis of the output package (`dfe_12b_full_output_package.md`) indicates extremely high potential operator value, with the system successfully suppressing hallucinations and only rescuing highly targeted, accurate excerpts.

However, **no real human operator scoring was collected during this execution block.** The `dfe_12b_operator_scoring_sheet.csv` has been fully populated with the query paths and baseline data, but the subjective 1-5 / 0-4 human scoring columns remain blank awaiting human review.

## Final Decision
> **DFE_12B_PASS_KEEP_SHADOW_EVALUATION_ONLY**

**Rationale:** The infrastructure successfully scaled to a diverse, noisy, real-world document corpus while perfectly maintaining all fail-closed safety constraints. The runtime telemetry verified that the processing overhead remains well within acceptable margins. However, because actual human operator scoring is absent from this run, we cannot formally claim real-world operational value or recommend a live operator trial. The system is mechanically sound, but remains strictly locked to shadow evaluation until human scores are collected.
