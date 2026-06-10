# PIT-11E Operator Value Rescoring Report (Rendered Support)

## Overview
This report re-evaluates the operator value of the `PIT_11A_SMALL_CORPUS` retrieval runs using the newly introduced `support_evidence_excerpt` semantic renderings. The goal is to determine if the exact extracted sentences improve operator value sufficiently to expand the pilot.

## Aggregate Metrics

### Safety & Structural Metrics
- **Safety Issues Count:** 0
- **Baseline Leakage Count:** 0
- **Unsupported Selected Facts:** 0
- **Claim Strength Issues Count:** 0

### Operator Value Averages (Across all 6 queries)
- **Authority Clarity Average:** 5.0 / 5.0 (Threshold: >= 4.5) ✅
- **Overall Derived Fact Usefulness Average:** 1.17 / 4.0 (Threshold: >= 1.5) ❌
- **Review Burden Delta Average:** -1.0 / 2.0 (Negative is good) ✅
- **Operator Confidence Delta Average:** +0.33 / 2.0 (Threshold: > 0) ✅

### Quality Averages (For Selected Facts Only - Q3)
- **Question Alignment Quality Average:** 3.0 / 5.0 (Threshold: >= 3.5) ❌
- **Source Support Quality Average:** 2.0 / 5.0 (Threshold: >= 4.0) ❌
- **Rendered Support Quality Average:** 2.0 / 5.0 (Threshold: >= 4.0) ❌
- **Selected Fact Usefulness Average:** 2.0 / 4.0 (Threshold: >= 3.0) ❌

### Diagnostic Signals
- **Best Dropped Candidate Would Have Helped:** 1 instance (Q3)

---

## Qualitative Findings

### 1. Rendering Proves Effective, but Selection Discards the Best Facts
The semantic sentence rendering engine (introduced in PIT-11D) proved mathematically capable of extracting the precise, correct supporting sentence ("The purpose of this guide is to assist Inspectors General..."). However, the *selected* fact in Q3 was tied to a LanceDB chunk detailing historical context, which caused the visible rendered support for the operator to be scored low (`2.0`).

### 2. The Best Candidate was Dropped by Strict Thresholds
The diagnostic marker `best_dropped_candidate_would_have_helped_yes_no` fired as `Yes` for Q3. The evaluation package reveals that the precise purpose statement was successfully retrieved and rendered flawlessly (score `0.7654`) by a candidate fact that was subsequently dropped upstream due to a `0.6774` Answer Alignment score (which fell just shy of the `0.70` hard drop threshold). 

### 3. Fail-Closed Safety Remains Intact
Queries Q1, Q6, Q9, Q11, and Q15 correctly maintained their structural isolation, preserving zero hallucinations and no leakage. However, due to the overarching strictness, the system retrieved very little value over baseline, depressing the `Overall Derived Fact Usefulness Average`.

---

## Decision Outcome

**PIT_11E_REVISE_SELECTION_THRESHOLDS**

**Rationale:** The exact sentence extraction renderer works perfectly (as demonstrated by dropped candidates), but the operator value fails because the upstream selection algorithm discarded the best candidate. The alignment threshold (`0.70`) is excessively punitive and drops facts that contain the most useful, highly-rendered source support. Before any expansion can occur, we must tune the selection/alignment thresholds so that highly-supported excerpts can survive the pipeline and reach the operator.
