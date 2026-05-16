# Phase 12B Closeout

## 1. Phase 12B Objective
Following Phase 12A's identification of **Layout D** as the most promising placement candidate, Phase 12B aimed to test various packet DSL / format syntax variations (keeping the Layout D ordering constant) to find the best balance of token efficiency, model readability, source fidelity, governance retention, and parser reliability.

## 2. Phase 12B.1 Result: `FAIL_FORMAT_SAFETY_REGRESSION` on 7B
Testing eight format variants against the local `Qwen2.5-7B-Instruct` model revealed that every format failed the strict, zero-tolerance safety gate. The 7B model consistently dropped exact `S#` identifiers and failed to emit explicit evidence-gap markers, blocking all formats from moving to shadow mode.

## 3. Phase 12B.2 Result: `FORMAT_BENCHMARK_MODEL_LIMITED`
A reinforcement drill tested the top candidates against a larger reasoning model (`Qwen3.6-35B-A3B_Opus-4.6-Reasoning-3300x-GGUF:Q4_K_M`). The results proved that the format logic is robust: the 35B model achieved 0 logic failures (perfect adherence to safety metrics) on completed traces when paired with reinforced prompt wrappers. The limitation in Phase 12B.1 was definitively tied to the 7B model's reasoning capabilities.

## 4. Phase 12B.3 Result: Runtime Viability Gate
A restricted test evaluated the top two combinations from the 35B drill for runtime viability:
- **`2_layout_d` + Wrapper A (Explicit Citation):** Achieved 0 logic failures but was blocked by a 50% timeout rate on the local endpoint (`SHADOW_BLOCK_TIMEOUTS`).
- **`8_source_table_facts` + Wrapper B (Structured Answer):** Completed within latency bounds but suffered a logic regression in one case (`SHADOW_BLOCK_LOGIC_FAILURES`).

## 5. Final Decision: `BLOCKED_FROM_SHADOW_PROMOTION`
Phase 12B is closed as a **disciplined fail-closed success**. 

No packet format or answer-contract variant is eligible for shadow promotion under current local runtime conditions. 

`2_layout_d` + Wrapper A remains the leading logical-safety candidate. The benchmark system correctly found a promising logical candidate and subsequently correctly blocked it because the local runtime environment cannot yet support its latency profile reliably.

## 6. Recommended Future Branch: Phase 12B-R (Runtime Feasibility Lane)
Do not proceed to Phase 12C. Instead, open a new runtime optimization lane to determine if `2_layout_d` + Wrapper A can meet runtime thresholds on an improved serving configuration.

**Test Options for Phase 12B-R:**
1. Tuning `llama.cpp` server with GPU offloading optimizations.
2. Evaluating smaller but highly capable reasoning models in the 24B–32B parameter range.
3. Exploring lower quantization variants for faster inference.
4. Tuning batch sizes, context lengths, and prompt lengths.
5. Using increased timeouts purely as a diagnostic tool, not for promotion.
6. Investigating candidate early-exit logic for evidence-gap cases.

**Promotion Blocked Until a Candidate Achieves:**
- Timeout rate <= accepted threshold
- Logic failure rate == 0
- Source citation failures == 0
- Evidence-gap failures == 0
- Protected span failures == 0
- Latency within budget
