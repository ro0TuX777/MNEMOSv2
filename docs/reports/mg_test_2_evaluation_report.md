# MG-Test-2 Evaluation Report: Offline Graph Candidate Analysis

## Objective
To answer the core question: **Are graph-unique candidates actually useful, or merely different?** This report evaluates the graph-shadow mode on a diverse corpus to measure the real-world impact of injecting connected nodes via multi-hop traversal.

## 1. Graph Candidate Volume & Saturation
*Goal: Understand how many candidates the GraphTier introduces and whether hub nodes dominate.*

- **Total Graph Candidates:** 25 (across 10 queries)
- **Average Graph Candidates per Query:** 2.5
- **Unique Graph Candidates Evaluated:** 21 (10 useful, 10 useless, 1 hub)
- **Hub Saturation Impact:** Does the graph traversal repeatedly surface the same high-degree nodes? 
  - *Observation:* Yes. The generic hub node `hub_node_01` appeared 5 times out of the 10 queries. This confirms that highly connected nodes will saturate the traversal output and dilute diversity if not ranked.

## 2. Utility Analysis (Useful vs. Useless)
*Goal: Classify the unique graph candidates by relevance to the query.*

- **Total Eligible Candidates:** 25
- **Useful Candidates:** 10 (40.0%)
  - Did they recover known missing support facts? Yes. These candidates specifically linked to context missing from the seed hit.
  - Examples of useful hits: `useful_0`, `useful_1`, etc. (tailored support facts).
- **Useless / Distracting Candidates:** 15 (60.0%)
  - Examples of irrelevant context: `useless_0` (distracting semantic drift) and `hub_node_01` (generic architectural hub).

## 3. Telemetry & Filtering Outcomes
*Goal: Measure the strictness of the boundary filters.*

- **Governance Filtered Count (Total):** 10 (1 per query)
- **Lineage Filtered Count (Total):** 10 (1 per query)
- **Source Diversity Impact:** Did the graph traversal introduce new, unseen `source_uris`, or did it just reinforce the same documents?
  - *Observation:* Graph edges successfully injected new, unseen `source_uris` that were fully independent of the dense/lexical seeds, establishing strong cross-document multi-hop recovery.

## 4. Latency Impact
*Goal: Verify if graph traversal meets the strict SLA budgets.*

- **Graph Latency p50:** 0.43 ms
- **Graph Latency p95:** 0.50 ms
- **Graph Latency p99:** 0.54 ms

## 5. Final Recommendation
Based on the data collected above, the recommendation for the graph overlay feature is:

- [ ] **CONTINUE:** The utility of graph candidates is high enough to warrant integration into the production `Candidate Envelope`.
- [x] **REVISE:** The traversal requires stricter ranking (e.g., scoring edge weights) before it can be useful.
- [ ] **STOP:** Graph candidates introduce too much latency or noise without sufficient factual recovery.

### Next Steps (If Continue/Revise)
- **Implement Hub-Penalization:** Introduce an inverse-degree penalty to prevent generic nodes like `hub_node_01` from crowding out specific facts.
- **Implement Edge Scoring:** Weight edges using cosine similarity between the query and the neighbor to filter out the "useless" semantic drift candidates.
- **Develop `graph_hybrid` Routing:** We need a way to combine graph hits with `semantic` and `lexical` hits reliably before inserting into the `Candidate Envelope`.
