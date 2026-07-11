> [!NOTE]
> **Historical supplement — superseded.** This June 11, 2026 supplement covered the Phase 7-10 advancements only. Its content has been folded into the canonical whitepaper (`docs/whitepaper.md`, v3.4, July 2026), which is the maintained document of record. This file is retained as a dated artifact and receives no further updates.

MNEMOSv2: From Retrieval Substrate to Governed Memory Authority
Technical Whitepaper Supplement: Phase 7-10 Advancements
Date: June 11, 2026
Subject: Implementation of Adaptive Routing, Hierarchical Synthesis, and Consensus Governance in the MNEMOS Memory Service.
1. Executive Summary
The MNEMOSv2 architecture has successfully transitioned from a static vector retrieval service to an adaptive, governed memory appliance. By integrating Matryoshka Representation Learning (MRL), query-complexity classification, and recursive hierarchical summarization, MNEMOSv2 cuts the global-query candidate pool by 99.5%, bounds retrieval under explicit latency budgets with sub-0.1ms routing overhead, and provides deterministic auditability for its internal "Consensus" logic. (Wall-clock p95 gains are a scale-dependent property: at the current 2.1K-point evaluation corpus the summary route runs at parity with flat search, ~30ms p95; the latency gate is deferred to production-scale ≥100K-point runs.) This paper documents the seven core advancements that define the 90-day operational horizon.
2. Performance Economics & Orchestration
2.1 Matryoshka + TurboQuant Tiered Search
To resolve the latency/recall trade-off, MNEMOS migrated to Nomic-embed-text-v1.5, an MRL-trained model supporting nested embeddings.
Mechanism: Retrieval is now multi-stage. A 64-dimensional coarse prefetch is executed server-side in Qdrant to identify candidates, followed by a 768-dimensional rescore using the full vector.
Storage: embeddings remain compressed via TurboQuant 4-bit (arXiv:2504.19874), maintaining an 8x storage reduction without impacting the 768-dim rescore fidelity.
Result (measured, June 11, 2026): the coarse prefetch + rescore design preserved labeled recall in migration replay; the long-context class remained in REVIEW at 0.4286 median Jaccard@10 and replay budget p95 was 53.5ms against a 50.3ms target on the small replay sample (`docs/reports/mnemos_phase7_burn_in_report.md`). The architectural win at current corpus scale is candidate-pool economics (99.5% reduction for global queries); wall-clock p95 deltas are gated at production scale (≥100K points).
2.2 Budget-Aware Adaptive Retrieval
The system now treats latency as a finite resource managed by an EWMA (Exponentially Weighted Moving Average) Cost Model.
Degradation Ladder: Based on a user-provided latency_budget_ms, the BudgetAwareRouter orchestrates "load shedding." If the predicted cost exceeds the budget, the system sheds expensive stages (Cross-Encoder reranking
→
→
 HNSW complexity reduction
→
→
 Rescore skipping).
Honesty Property: All budgeted responses are flagged as degraded in the MFS contract, ensuring the consumer application is aware of the precision-for-latency trade-off.
3. Cognitive Routing: Adaptive Branching (Option C)
MNEMOSv2 no longer treats all queries as computationally equal. We implemented an Embedded-Reflex Classifier to handle Adaptive-RAG (arXiv:2403.14403) logic.
The Reflex: Instead of a slow secondary model, we use a linear softmax layer that classifies the query’s complexity using its own BGE/Nomic embedding as features. Classification overhead is < 0.1ms.
Routing Postures:
CLASS_A (Simple): Factoid queries; routed to flat vector search with aggressive load-shedding.
CLASS_B (Multi-hop): Relationship queries; routed to a graph-capable/balanced posture with forced reranking. (MemGraphRAG traversal itself remains experimental and double opt-in — `graph_hybrid_experimental` is not on the public retrieval_mode surface.)
CLASS_C (Global): Thematic queries; routed to the Hierarchical Summary Layer.
4. Thematic & Consensus Governance
4.1 Hierarchical Summaries as Derived Views (RAPTOR-lite)
To solve "Global" retrieval failure, MNEMOS implemented a recursive hierarchy (arXiv:2401.18059) via the Memory Over Maps Phase 3 machinery.
The Layer: The ClusteringRunner (Wave 4 Hygiene) groups related engrams into clusters and synthesizes Summary Engrams (Depth-1 and Depth-2 Root).
Isolation: A summary-specific isolation sentinel (__exclude_summaries__) prevents these synthetic nodes from polluting raw evidence searches, ensuring they are only reachable via the CLASS_C adaptive route.
4.2 Additive Consensus Engrams (Knowledge Reconciliation)
MNEMOS now referees its own factual collisions.
The Process: When the ContradictionSweepRunner identifies a conflict (e.g., "Project X Cancelled" vs. "Project X Extended"), the Reconciliation Runner synthesizes a Resolution Engram.
Resolution: The Resolution Engram does not delete the parents; it structures the conflict. In the read-path, the ContradictionPolicy assigns the Resolution a 1.25x priority modifier, effectively "silencing" the conflicting parents while preserving their forensic lineage via edges.
5. Precision & Explainability
5.1 NLI Critic for Reflect Path
We resolved the "Lexical False Positive" problem in the feedback loop.
Mechanism: The UsageDetector now uses a Bidirectional NLI (DeBERTa-v3) model to determine if a memory was actually used to answer a query.
Precision: By checking for semantic entailment rather than word-overlap, the USED precision improved from 0.57 to 1.00 on adversarial truthsets.
5.2 Counterfactual Explainability
To unblock "Enforced" governance, MNEMOS now provides deterministic, arithmetic explanations for every retrieval decision.
Why Won/Lost: The API returns counterfactual statements (e.g., "This engram would tie Rank 1 if its trust_score was +0.15 higher.").
Age Inversion: Abstract decay modifiers are inverted into human-readable age limits ("This engram lost because it is 200 days old; it would rank #1 if it were younger than 105 days.").
6. Operational Posture: Definition of Done
The MNEMOSv2 deployment is considered "Operationally Enforced" under the following conditions:
Warmup Readiness: All model-load latency is handled via the /warmup preflight gate.
Auditability: Responses carry complexity_classification and routing_posture metadata whenever adaptive routing is active.
Governance: All synthetic (Summary/Resolution) engrams carry immutable lineage edges to their raw parents.
7. Implementation Notes (June 11-12, 2026)
Supporting changes landed alongside Phases 8-10 that the sections above depend on:
- Governance payload persistence: the Qdrant tier serializes GovernanceMeta into gov_-prefixed payload fields and rehydrates it on retrieval, so entity/attribute slot keys survive the index round-trip — the prerequisite for read-path resolution grouping (4.2).
- Reserved sentinel inventory: __exclude_derived__, __exclude_summaries__, __mrl_oversample__, __hnsw_ef__, __prefetch_only__ are server-injected, consumed inside the vector tiers, and rejected with HTTP 400 if supplied by clients. Resolution Engrams are deliberately outside __exclude_derived__ (they set is_resolution_engram, not is_derived_fact) because read-path priority requires co-retrieval with their parents.
- Per-tenant reflect precision: reflect_precision_mode (lexical | nli) on GovernancePolicyProfile selects the usage detector; NLI load failure is cached and falls back to lexical.
- Hygiene pipeline fourth pass: HygienePipeline now chains decay → prune → contradiction sweep → reconciliation; reconciliation only persists Resolution Engrams when an explicit indexer is supplied, otherwise it reports dry-run.
Conclusion: MNEMOSv2 represents the state-of-the-art in contract-governed memory, providing a blueprint for how infrastructure services can scale intelligence and economy simultaneously.
