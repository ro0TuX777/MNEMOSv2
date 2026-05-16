# MNEMOS AI Dev Plan: Phase 12A — EchoFrame Evidence Placement Benchmark

## 1. Objective
Implement a benchmark suite that evaluates how different EchoFrame packet layouts affect LLM answer quality, source attribution, governance warning retention, and protected-span fidelity.

## 2. Core Research Question
Does EchoFrame’s packet layout influence whether the LLM correctly uses critical facts, governance warnings, source pointers, contradictions, and evidence gaps?

## 3. Scope
**In scope:**
- Build a deterministic benchmark harness for EchoFrame packet layouts.
- Test five packet layouts (A through E).

**Out of scope:**
- Do not replace EchoFrame’s production default.
- Do not integrate new token/sentence-level compressors.
- Do not change retrieval ranking or admissibility gates.
- Do not weaken fail-closed behavior.

## 4. Benchmark Layouts
- **A**: Current-style baseline ([FACTS] → [GOVERNANCE] → [EVIDENCE])
- **B**: Governance-first ([GOVERNANCE] → [FACTS] → [EVIDENCE])
- **C**: Governance-recap ([FACTS] → [EVIDENCE] → [GOVERNANCE_RECAP])
- **D**: Top-and-bottom constraints ([GOVERNANCE] + [FACTS] → [EVIDENCE] → [CRITICAL_CONSTRAINTS])
- **E**: Source-local (Fact → Source → Governance repeated per evidence unit)

## 5. Decision Process
The final deliverable will be a decision artifact to determine if Layout D or Layout E should move into shadow testing, based on strict safety and performance criteria.
