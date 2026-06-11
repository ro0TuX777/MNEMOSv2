# PIT-11C Operator Value Rescoring Report

## Overview
This report provides the operator-value rescoring of the `PIT_11A_SMALL_CORPUS` 6-question subset, evaluating the practical utility, safety, and source support of the `Derived Fact` lane compared to the baseline retrieval. 

## Aggregate Metrics

### Safety & Structural Metrics
- **Safety Issues Count:** 0
- **Baseline Leakage Count:** 0
- **Unsupported Selected Facts:** 0
- **Claim Strength Issues Count:** 0

### Operator Value Averages (Across all 6 queries)
- **Authority Clarity Average:** 4.83 / 5.0 (Threshold: 4.5) ✅
- **Derived Fact Usefulness Average:** 1.0 / 4.0 (Threshold: 2.5) ❌
- **Review Burden Delta Average:** -1.16 / 2.0 (Negative is good) ✅
- **Operator Confidence Delta Average:** 0.5 / 2.0 (Threshold: > 0) ✅

### Quality Averages (For Selected Facts Only - Q3)
- **Question Alignment Quality Average:** 5.0 / 5.0 (Threshold: 3.5) ✅
- **Source Support Quality Average:** 2.0 / 5.0 (Threshold: 4.0) ❌
- **Selected Fact Usefulness Average:** 3.0 / 4.0 (Provides clear value over baseline despite weak preview)

---

## Qualitative Findings

### 1. Fail-Closed Safety Proven, But Suppresses Usefulness
The system's new fail-closed architecture successfully blocked unsupported and loosely aligned facts (Q1, Q6, Q9, Q11, Q15). This yielded an exceptionally high `Authority Clarity` (4.83) and reduced the `Review Burden` (-1.16) because operators did not have to read hallucinated garbage. However, by safely returning `Derived_FactNodes: 0` for 5 out of 6 queries, the overall `Derived Fact Usefulness Average` plummeted to 1.0. **Safety was achieved at the cost of utility.**

### 2. Source Support Preview Misalignment
For Q3, the system correctly selected an aligned fact (`[GOLD_ALIGNED] The Intelligence Oversight Guide assists IGs in preparing, executing, and completing Intelligence Oversight inspections.`). However, the `support_evidence_preview` fetched dynamically from LanceDB discussed historical contexts (e.g., the Pike Committee) rather than the strict functional purpose of the guide. 

This resulted in a low `Source Support Quality` score (2.0) despite passing the mathematical `> 0.65` cosine similarity threshold. The mathematical vector overlap was sufficient, but the human-readable text span was inadequate for verifying the specific claim.

---

## Decision Outcome

**PIT_11C_REVISE_SOURCE_SUPPORT_RENDERING**

**Rationale:** The Derived Fact lane is structurally safe and no longer hallucinates or leaks. However, it cannot proceed to an expanded operator pilot because the overall usefulness is too low, and the single positive selection (Q3) presented a misleading evidence preview. The rendering of the exact supporting span needs to be revised (e.g., highlighting or extracting the exact matching sentence rather than just the start of the chunk), or the semantic threshold (0.65) must be tuned to demand tighter evidentiary overlap.
