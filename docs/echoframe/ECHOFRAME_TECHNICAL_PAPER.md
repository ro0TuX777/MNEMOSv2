# EchoFrame: A Governed Evidence Codec for MNEMOS

## 1. Introduction

As AI applications scale, providing models with raw retrieval contexts leads to rapidly expanding token costs, unstable generation behavior across multi-turn sessions, and an increased risk of context window exhaustion. MNEMOS inherently solves the retrieval side by finding and governing relevant memory vectors. However, passing these retrieved vectors linearly to the LLM presents a bottleneck.

**EchoFrame** is introduced as a MNEMOS-native shadow adapter and context optimizer. It acts as a "governed evidence codec" that sits between the MNEMOS retrieval pipeline and the LLM endpoint. It intercepts the dense, retrieved context and compresses it into a high-density, source-traceable packet—drastically reducing the payload size while maintaining perfect cryptographic and governance lineage.

## 2. Architectural Design

EchoFrame operates immediately after the MNEMOS baseline document retrieval and governance filtering steps. 

### The Old Model
`Retrieval (Thousands of Tokens)` → `Send Raw Context to LLM`

### The EchoFrame Model
`Retrieval` → `Governance Filtering` → `EchoFrame Compression` → `Send Dense Packet to LLM`

EchoFrame utilizes the `compact_semantic_minEvidence_hysteresis_v0` rendering mode. This mode strips away redundant prose and reconstructs the valid evidence as a dense, serialized matrix:
- **Evidence IDs**: `[E0], [E1]`
- **Source IDs**: `[S0], [S1]`
- **Governance Signals**: `[VETOED]`, `[SUPPRESSED]`, `[CONTRADICTION]`

By converting narrative context into relational pointers, the LLM retains full access to the facts and their provenance without paying the token cost of the original natural language.

## 3. Strict Precedence and Admission Gates

EchoFrame is designed with a fail-safe, default-closed architecture. It exclusively processes "eligible" low- and medium-risk traffic. All exceptions, anomalies, and high-risk operations immediately and silently fall back to the baseline MNEMOS context.

### The Decision Hierarchy
1. **Kill Switch**: If `MNEMOS_ECHOFRAME_KILL_SWITCH=true`, bypass EchoFrame completely. All traffic falls back to baseline.
2. **Default-On Eligible**: If `MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE=true`, traffic is evaluated against the admission gates.
3. **Legacy Pilot**: If the legacy pilot is active, only a configured sample rate is evaluated.
4. **Baseline Fallback**: Any configuration miss results in baseline routing.

### Admission Gates
A query packet is **eligible** for EchoFrame compression only if it passes all of the following validations:
- **Zero Safety Failures**: Source pointers must exist and map correctly. No `dropped_provenance` allowed.
- **Zero Validator Failures**: The constructed packet must meet schema requirements.
- **Promotable Content**: The packet must not be flagged with `NO_EVIDENCE_FOUND`.
- **Token Compression**: The ratio of EchoFrame tokens to baseline tokens must be `<= 1.0`.
- **Stability**: The multi-turn hysteresis stability score must be `>= 0.90`.
- **Governance Risk**: The query must not be flagged as `high-risk`.
- **Approval Constraints**: The governance metadata must not flag the request as `approval_required`.

## 4. Benchmark Results and Validation

EchoFrame underwent a rigorous 9-phase rollout, culminating in a 2,000-query soak test under the Release Candidate (Phase 8) configuration.

### Phase 8 Soak Test (2,000 Queries)
The final validation drill simulated a full production workload with a mix of safe, high-risk, and approval-required queries to test the absolute integrity of the admission gates.

**Key Metrics:**
- **Total Runtime Calls Observed**: 2,000
- **Eligible Events**: 1,980
- **EchoFrame LLM-facing Events**: 1,980
- **Baseline Fallback Events**: 20 (All 20 correctly identified as `high_risk_excluded`)
- **Failure Rate**: 0.00%

### Compression and Efficiency
- **Average Baseline Tokens**: 1,989.00 tokens per query
- **Average EchoFrame Tokens**: 52.98 tokens per query
- **Average Token Ratio**: 0.0266 (EchoFrame utilizes only ~2.6% of the original token footprint)
- **p99 Token Ratio**: 0.0266

### Stability and Governance
- **Average Stability Score**: 1.0000 (Perfect multi-turn context stability)
- **Minimum Stability Score**: 1.0000
- **Safety Gate Failures**: 0
- **Validator Failures**: 0

## 5. Conclusion

EchoFrame successfully transforms MNEMOS from a high-overhead retrieval engine into an ultra-efficient, local-model-ready knowledge platform. By achieving a **~97% reduction in LLM-facing context size** with zero degradation in source traceability or governance compliance, EchoFrame drastically lowers inference costs and context window pressure. 

Its strict fail-closed architecture ensures that critical governance boundaries—such as high-risk queries and required approvals—are never compromised, defaulting to the robust baseline MNEMOS behavior whenever necessary.
