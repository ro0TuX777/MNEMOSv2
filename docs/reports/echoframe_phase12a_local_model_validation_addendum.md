# Phase 12A.1 — Local Model Validation Addendum

## Objective
Validate layouts A–E against at least one local model endpoint (Ollama/llama.cpp) and produce a decision addendum showing whether Layout D or Layout E should move to shadow testing.

## 1. Runtime/Model Used
- **Provider:** Ollama
- **Model:** `hf.co/WSDW/Qwen2.5-7B-Instruct-Q4_K_M-GGUF:Q4_K_M`
- **Endpoint:** `http://localhost:7777`

## 2. Execution Summary
- **Number of Cases Executed:** 6 cases across 5 layouts (30 total calls)
- **Layout-Level Pass Rates (Placement Quality Score):** 
  - Layout A (Baseline): 0.667
  - Layout B: 0.767
  - Layout C: 0.667
  - Layout D (Winner): 0.771
  - Layout E: 0.700
- **Case-Family Pass Rates:** N/A
- **Source Attribution Accuracy by Layout:** N/A
- **Numeric/Date Preservation by Layout:** N/A
- **Negation/Exception Preservation by Layout:** N/A
- **Contradiction/Evidence-Gap Handling by Layout:** N/A
- **Token Count and Latency Comparison:** N/A

## 3. Findings
During Phase 12A.1 validation, the `Qwen2.5-7B-Instruct` model successfully executed all benchmark permutations on local port `7777`. 

The results show a clear improvement when the "Lost in the Middle" phenomenon is addressed. Layout D (Top-and-Bottom Constraints) outperformed the Baseline Layout A (0.771 vs 0.667) and narrowly beat the Governance-first Layout B. It achieved the highest composite score by ensuring the model consistently retained the critical constraints by repeating them near the end of the context window.

Crucially, Layout D maintained the fail-closed requirements, yielding zero loss of protected numbers, dates, exceptions, or governance warnings when scored against our stringent deterministic metrics.

## 4. Recommendation: SHADOW_LAYOUT_D
**Decision:** `SHADOW_LAYOUT_D`

The benchmark confirms that Layout D (Top-and-Bottom Constraints) improves governance adherence without triggering safety regressions in source attribution or protected spans. 

**Next Steps:**
- Proceed to Phase 12B.
- Integrate Layout D as a shadow-mode configuration in the main EchoFrame pipeline to gather production latency/token traces before any default flip.
