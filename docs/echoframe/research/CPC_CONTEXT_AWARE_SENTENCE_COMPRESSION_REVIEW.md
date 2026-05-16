# Phase 11A — Context-Aware Sentence Compression (CPC) Review

## 1. Paper Summary
"Prompt Compression with Context-Aware Sentence Encoding for Fast and Improved LLM Inference" introduces CPC, a sentence-level compression technique. Instead of pruning individual tokens, CPC uses a context-aware sentence encoder to compute the cosine similarity between each sentence in the context and the user's query. It then selects only the top $T$ relevant sentences. The authors claim CPC preserves human readability, outperforms state-of-the-art token-based methods (like LongLLMLingua) on LongBench and ZeroSCROLLS, and operates up to 10.93x faster during inference.

## 2. Why Token-Level Compression Failed in Phase 10
In Phase 10, LLMLingua aggressively pruned context to achieve high compression. While it succeeded on generic text, it failed catastrophically on dense SIGINT and DoD manuals. LongLLMLingua optimizes for semantic flow by deleting "less informative" tokens. However, in high-stakes evidence, these "less informative" tokens often happen to be specific operational dates, numeric thresholds, exception clauses, and negations embedded deep within sentences. Deleting them destroyed the factual reliability of the intelligence payload.

## 3. How CPC Differs from LLMLingua
CPC operates strictly at the **sentence level**. It either includes a sentence in its entirety or removes it entirely. LLMLingua operates at the **token level**, actively modifying the internal structure of sentences.

## 4. Sentence-Level Compression Benefits
- **Atomicity of Facts**: If a sentence is selected because it answers the query, all internal operational dates, numbers, and negations within that sentence remain perfectly intact.
- **Auditable Failures**: It is much easier to detect if a specific source sentence was omitted than to audit whether an LLM stealthily deleted the word "not" from inside a preserved sentence.
- **Latency**: Sentence embedding and ranking is highly parallelizable and significantly faster than auto-regressive token-level perplexity calculation.

## 5. Risks for Governance and High-Stakes Evidence
CPC is not inherently immune to failure. While it won't mutilate a sentence, it might **drop an entire sentence** that contains a critical threshold, exception, or classification marker if that sentence's vector similarity to the query is low (e.g. an overarching "KILL SWITCH" sentence might not seem semantically relevant to a query about "target coordinates", but must be included).

## 6. MNEMOS Integration Opportunities
CPC aligns perfectly with the EchoFrame architecture. We can deploy CPC as a bounded compressor exclusively for the `[EVIDENCE]` block. Because it operates on sentences, we can apply a **Protected Sentence Policy**: before ranking, any sentence containing a classification, threshold, or negation is flagged as protected and automatically bypasses the CPC removal phase.

## 7. Required Benchmark Plan
We will build a Phase 11 prototype containing:
1. A robust sentence/clause segmenter that handles military structures (e.g. 1.2.3, (a)).
2. A protected sentence policy that shields high-risk sentences.
3. A CPC-style sentence ranker.
4. An evaluation suite comparing Stable EchoFrame against CPC-selected sentences (Modes A-F).

## 8. Recommendation
**PROCEED WITH PHASE 11 PROTOTYPE.**
CPC is a highly viable candidate for an evidence-window compressor inside EchoFrame. By combining EchoFrame's structural governance, our deep protected-span rules, and CPC's sentence-level atomicity, we can potentially achieve safe compression on massive E3/E4 evidence payloads without risking the token-level hallucination seen in Phase 10.
