# MNEMOS Support Matrix

This matrix states what MNEMOS currently supports, what is available for
controlled evaluation, and what remains research or specification-only work.
It is the public boundary for product claims.

## Status Definitions

| Status | Meaning |
|---|---|
| Supported | Suitable for normal deployment when the documented profile and operating assumptions apply. |
| Beta / pilot | Implemented and bounded, but requires operator review before broad production use. |
| Experimental | Available for evaluation, shadow runs, or targeted pilots. Do not present as default production behavior. |
| Research / spec only | Design or benchmark lane only. No production runtime claim. |
| Blocked | Intentionally not promoted because a gate, evidence condition, or safety boundary is unmet. |

## Capability Status

| Capability | Status | Production claim | Evidence / notes |
|---|---|---|---|
| Core Memory Appliance profile | Supported | Default semantic retrieval profile using Qdrant, PostgreSQL audit storage, and MNEMOS API. | See [deployment profiles](deployment_profiles.md), [INSTALL](../INSTALL.md), and [benchmark results](benchmark.md). |
| Governance Native profile | Supported | PostgreSQL/pgvector-oriented profile for provenance-heavy and metadata-filtered deployments. | Supported as a profile; relevance and correctness advantages are workload-dependent and must be benchmarked per corpus. |
| REST API and Boundary SDK | Supported | Stable integration surface for health, capabilities, index, search, engram lookup, audit, stats, and warmup. | Contract behavior is validated by health and CI tooling. |
| Chat evidence contract | Supported | Search responses expose citation-ready per-result evidence and grouped source summaries for downstream chat systems. | See [chat integration evidence contract](chat_integration_evidence_contract.md). |
| Forensic audit ledger | Supported | Mutation and operational events can be recorded for audit and review. | PostgreSQL-backed audit is preferred in deployed profiles; SQLite fallback exists for local operation. |
| Summary isolation | Supported | Summary engrams are isolated from default factoid retrieval by server-managed controls. | Summary sentinels are server-managed and rejected when supplied directly by clients. |
| Resolution Engrams | Supported | Additive contradiction resolution can preserve parent lineage and receive governed read-path priority. | Governed use only; Phase 10 gate evidence is linked from [README](../README.md) and [benchmark results](benchmark.md). |
| Semantic / hybrid retrieval mode | Beta / pilot | Semantic remains the default. Hybrid is available for targeted evaluation where exact-term failures are suspected. | Gate C did not justify a broad hybrid default. See [benchmark results](benchmark.md). |
| Governance advisory / enforced modes | Beta / pilot | Governance can score, explain, and optionally suppress candidates when explicitly enabled. | Default remains off until corpus-specific thresholds and policy choices are validated. |
| TimesFM predictive pulse | Experimental | Forecast signals may inform advisory operational decisions. | Controlled by `MNEMOS_TIMESFM_ENABLED` and `MNEMOS_PULSE_ACTIONS`; advisory posture is the production default. |
| Graph Tier | Experimental | Read-only graph/hybrid work may be evaluated outside the public default retrieval surface. | See [graph tier docs](graph_tier/). |
| Derived facts lane | Experimental | Derived evaluation outputs are shadow/bounded and must not become authoritative retrieval facts by default. | Promotion and reliance are controlled by explicit gates and audit boundaries. |
| EBIR-R1 refinement | Experimental / shadow only | Offline evidence refinement can be benchmarked against contradiction fixtures. | Authoritative promotion remains blocked. See [EBIR-R1 acceptance](ebir_r1_acceptance.md). |
| EBIR-R2 reviewer harness | Beta / pilot | Reviewer packet generation, blinding, compilation, and scoring fields are frozen for full R2 setup. | Human-value claim remains pending independent full reviewer trial. See [EBIR-R2 protocol](ebir_r2_trial_protocol.md). |
| ColBERT / reranker path | Blocked for production | No production relevance claim. | Reference-fidelity gate has not justified promotion. Keep experimental until the gate passes. |
| Context Atlas P0 | Research / spec only | No runtime production claim. | See [Context Atlas spec](context_atlas_spec.md). |
| Associative Retrieval A1 | Research / spec only | No runtime production claim. | See [Associative Retrieval A1 spec](associative_retrieval_a1_spec.md). |
| Multimodal evidence extensions | Research / spec only | No production claim. | Future contract extension only. |

## Claim Rules

- Do not describe experimental, blocked, or spec-only capabilities as production
  features.
- Do not claim EBIR improves human review outcomes until independent blinded
  reviewers complete the full R2 protocol and scoring analysis.
- Do not claim a retrieval-quality advantage from synthetic benchmarks alone.
  Synthetic benchmarks are valid for regression and invariant checks.
- Do not claim governance suppression is safe for a corpus until thresholds and
  failure modes are validated on that corpus.
- Keep shadow-only lanes shadow-only unless a documented gate explicitly changes
  status.
