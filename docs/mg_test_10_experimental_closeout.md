# MG-Test-10: GraphHybrid Experimental Closeout & Limited Pilot Readiness

## 1. Summary of Completed Phases
The GraphHybrid experimental track has successfully completed a rigorous 9-phase evaluation:
* **MG-Test-1:** Telemetry-only GraphTier scaffold implemented to prove topological integration without mutations.
* **MG-Test-2A/2B/2C/2R:** Comprehensive scoring calibration, hub penalty design, and synthetic/offline evaluations of semantic graph retrieval algorithms.
* **MG-Test-3:** EchoFrame Shadow Packet comparison, verifying payload shape matching without downstream disruption.
* **MG-Test-4 / 4B:** Calibration of experimental candidate merging (`preserve_primary_top_k=5`, `graph_quota=1`) ensuring 0 critical semantic evidence displacement.
* **MG-Test-5:** Enforcement of double opt-in controls (`enable_experimental_graph_hybrid = True` and mode toggle) guarding the production mainline.
* **MG-Test-6:** Controlled Operator Shadow Trial proving operator usefulness.
* **MG-Test-7A/7B:** Targeted Use-Case Pilot establishing high utility (87.8%) for specific query types, alongside an architectural review confirming Qdrant payload readiness.
* **MG-Test-8:** Implementation of the read-only, batched `QdrantEngramResolver` mapping network queries dynamically.
* **MG-Test-9:** Live Qdrant Resolver Validation, verifying `6.8ms` p95 retrieval latency under live load with 0 governance leaks and 0 write-path mutations.

## 2. What Is Proven
* It is entirely possible to read and deserialize raw graph edges safely from existing Qdrant payloads in $O(1)$ batch time without building a dedicated Graph Database.
* The `CandidateEnvelope` seamlessly isolates topological expansions, retaining 100% citation and governance integrity.
* Graph expansions materially improve unsupported claim delta and evidence gaps in specific targeted query vectors.
* Batched retrievals definitively mitigate N+1 scaling collapse.

## 3. What Remains Unproven
* Broad viability across unstructured, low-quality source corpora.
* Live generation of edges (the system currently relies solely on structural edges).
* The safety, reliability, and hallucination footprint of dynamic model-generated fact/passage extraction.

## 4. Final Recommended Operating Boundary
* **Mode Name:** Strictly locked as `graph_hybrid_experimental`. Do not rename to production `graph_hybrid`.
* **State:** Experimental, isolated, read-only.
* **Network Behavior:** Mandatory deduplication and batched retrieval limited to $<10$ seeds per query.

## 5. Targeted Allowed Use Cases
The experimental feature yields maximum efficacy exclusively for:
* `evidence_gap_investigation`
* `multi_hop_dependency_question`

## 6. Prohibited Use Cases
It is explicitly NOT recommended for:
* Fast factual lookup queries.
* Uncontrolled live-user traffic APIs.
* Low-latency SLA critical paths.
* Workloads requiring complex reasoning over highly contradictory inputs without human-in-the-loop oversight.

## 7. Required Double Opt-In Configuration
Operators wishing to pilot the mode must explicitly configure:
1. System/Global level: `enable_experimental_graph_hybrid = True`
2. Request level: `retrieval_mode = "graph_hybrid_experimental"`

## 8. Live Qdrant Resolver Telemetry Requirements
The system must actively monitor and log:
* `qdrant_resolver_latency_p95` (must stay under 15ms bound)
* `missing_neighbor_id_count`
* `qdrant_resolver_failure_count`
* `n_plus_one_retrieval_behavior` (Hard alert on value > 0)
* Any downstream governance or lineage filtering counts.

## 9. Rollback Procedure
In the event of failure or unacceptable latency spikes, rollback is entirely decoupled from deployment.
**Immediate Rollback:** Turn off the global system flag (`enable_experimental_graph_hybrid = False`) or strip `retrieval_mode` from the request. The router will instantly intercept the request and silently redirect to the stable `hybrid` or `semantic` baseline. No data cleanup is required since no memory writes exist.

## 10. Remaining Risks
* **Stale Edges:** Missing IDs are currently ignored safely, but heavy upstream document deletion could artificially dilute graph expansion pools.
* **P99 Instability:** Under catastrophic Qdrant cluster load, p99 batched retrieval could spike, though telemetry fallback catches this gracefully.

> [!CAUTION]
> ## 11. Schema / Fact Extraction Blocked
> Schema, fact, and passage extraction capabilities remain structurally and explicitly **BLOCKED**. The system does not possess authorization to construct or persist model-generated facts into semantic memory. Transitioning to write-path extraction requires a separate, dedicated architectural review and authorization track.
