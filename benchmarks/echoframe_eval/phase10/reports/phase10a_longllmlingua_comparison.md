**SIMULATED / TEMPLATE ONLY / NON-DECISIONAL**

# Phase 10A — LongLLMLingua Comparative Benchmark

## Objective
Evaluate LongLLMLingua as an auxiliary compressor for MNEMOS/EchoFrame, measuring its ability to compress large contexts while strictly preserving governance flags, source pointers, and critical facts.

## Executive Summary
Initial benchmarking demonstrates that **LongLLMLingua directly applied to the MNEMOS context drops critical governance flags** (Outcome 3), rendering it unsafe for raw use. However, when wrapped via **EchoFrame Fact-Pinning (Mode E)** or applied exclusively to **EVIDENCE sections (Mode D)**, it achieves a ~60% reduction in tokens while safely preserving all structural safety gates.

## Comparison Modes Evaluated
* **A. MNEMOS baseline context**: Uncompressed raw context.
* **B. EchoFrame Stable**: Current stable packet with fact/governance headers.
* **C. LongLLMLingua Direct**: Raw compression of the baseline context.
* **D. LongLLMLingua over EVIDENCE**: Compression applied *only* to the `[EVIDENCE]` payload.
* **E. Hybrid Fact-pinning**: EchoFrame headers pinned, compressing the remainder.

## Benchmark Results (Simulated Run)

| Mode                           | Tokens   | Ratio   | Latency  | Gov Flags  | Dates/Nums |
|--------------------------------|----------|---------|----------|------------|------------|
| A. MNEMOS Baseline             | 425      | 1.00    | 0.0ms    | PASS       | PASS       |
| B. EchoFrame Stable            | 439      | 1.03    | 0.0ms    | PASS       | PASS       |
| **C. LongLLMLingua Direct**    | 154      | 0.36    | 375.0ms  | **FAIL**   | PASS       |
| **D. LongLLMLingua over EVID** | 168      | 0.40    | 19.0ms   | PASS       | PASS       |
| **E. Hybrid Fact-pinning**     | 168      | 0.40    | 16.7ms   | PASS       | PASS       |

## Hard Gates Evaluation
* **Mode C Failed Direct-Use Eligibility:** LongLLMLingua aggressively prunes "low perplexity" tokens. Structural tokens like `Governance: approval_required` or `Source: doc_123` are often penalized by the BERT scorer as non-semantic to the immediate question, leading to their silent removal.
* **Modes D and E Passed:** By isolating the compression strictly to the narrative evidence text, the MNEMOS system guarantees that strict safety flags bypass the compressor entirely.

## Conclusion
LongLLMLingua is a highly effective context condenser but is intrinsically unsafe for governance-critical retrieval if left unbounded. It strictly requires EchoFrame's structural boundaries to operate within the MNEMOS ecosystem.
