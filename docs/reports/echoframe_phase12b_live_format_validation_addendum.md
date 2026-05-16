# Phase 12B.1 — Live Format Validation Addendum

## Objective
Run the Phase 12B format benchmark against the same local model used for Phase 12A to evaluate if any format variation of Layout D improves token efficiency and readability without regressions in fail-closed safety constraints.

## 1. Runtime/Model Used
- **Provider:** Ollama
- **Model:** `hf.co/WSDW/Qwen2.5-7B-Instruct-Q4_K_M-GGUF:Q4_K_M`
- **Endpoint:** `http://localhost:7777`

## 2. Execution Summary
- **Number of Cases Executed:** 6 cases across 8 format permutations (48 total calls)

## 3. Format-Level Quality Scores
- **1_baseline:** 0.790
- **2_layout_d:** 0.890
- **3_ultra_compact:** 0.873
- **4_yaml_lite:** 0.893 (Highest raw score)
- **5_minified_json:** 0.783
- **6_markdown_table:** 0.783
- **7_toon_rows:** 0.783
- **8_source_table_facts:** 0.850

## 4. Format-Level Safety Pass/Fail
*Note: Due to the zero-tolerance safety thresholds, all formats registered a FAIL due to dropping source attribution or failing to acknowledge evidence gaps against our strict metrics across the 6 test cases.*

## 5. Detailed Metric Preservation Breakdown (Pass Rates)
- **Source Attribution Accuracy by Format:** Suboptimal across the board. The 7B model frequently failed to explicitly cite the exact `S#` identifiers in the response, regardless of formatting (YAML, JSON, or Markdown).
- **Numeric/Date Preservation by Format:** High preservation across all formats, particularly in `4_yaml_lite` and `2_layout_d`.
- **Negation/Exception Preservation by Format:** Moderate. Formats like `5_minified_json` and `1_baseline` struggled more than `2_layout_d` and `4_yaml_lite` to convey negative constraints effectively.
- **Contradiction/Evidence-Gap Handling by Format:** Poor across all formats. The model frequently attempted to synthesize an answer instead of explicitly acknowledging the gap or contradiction, causing zero scores in this metric.
- **Parser Reliability by Format:** 100% (No catastrophic JSON/YAML structural rendering failures).

## 6. Token Count and Latency
- As expected, `3_ultra_compact` and `5_minified_json` reduced token overhead slightly, but latency remained relatively constant across the local inference calls.

## 7. Decision
**Decision:** `FAIL_FORMAT_SAFETY_REGRESSION`

**Rationale:**
While `4_yaml_lite` and `2_layout_d` achieved the highest composite placement scores, **every single format** dropped critical source pointers or failed to halt on evidence gaps in at least one test case. The stringent fail-closed promotion boundary requires absolute zero source pointer loss and zero gap suppression. 

Since the live model evidence shows regressions across all format DSLs (likely due to the reasoning capabilities of the 7B parameter local model on strict citation constraints), we cannot safely promote any shadow variant.

**Next Steps:**
- Do not promote any format to shadow mode or default.
- We must run this test with a larger reasoning model or iterate the prompt structure to ensure 100% citation fidelity. 
- The existing EchoFrame Layout D placement remains our best candidate in shadow, but the format syntax should remain `KEEP_CURRENT_TAGGED_D` (standard Markdown) until we can clear the safety gates.
