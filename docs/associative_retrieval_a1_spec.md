# MNEMOS Associative Retrieval A1 Specification

Date: 2026-06-19

Status: **Accepted specification. Offline, projection-based benchmark only.
Implementation remains deferred until EBIR-R2 protocol and reviewer corpus are
frozen.**

## Executive Boundary

Associative Retrieval A1 is an offline, projection-based, benchmark-first
research lane for evaluating whether bounded graph traversal can improve
CLASS_B evidence-chain discovery.

Governing invariant:

```text
Graph traversal may improve evidence discovery, but provenance, authority,
contradiction handling, and human review remain entirely within existing MNEMOS
governance boundaries.
```

A1 is not a production retrieval path, Context Atlas endpoint, durable relation
store, OpenIE authority layer, Graph Tier behavior change, Resolution Engram
input, EBIR input, or default search ranking change.

Explicitly blocked in A1:

- durable OpenIE or derived-relation stores
- live `POST /v1/mnemos/context/explore` behavior
- default `/v1/mnemos/search` ranking changes
- Graph Tier production behavior changes
- trust, utility, freshness, or contradiction-score mutation from graph
  centrality
- Resolution Engram or EBIR influence
- authority claims from graph edges
- automatic graph update paths
- graph-only answer paths
- recognition filtering in the baseline A1 path

## Research Question

A1 evaluates this retrieval experiment:

```text
CLASS_B query
  -> existing semantic/hybrid seed retrieval
  -> bounded source-linked graph projection
  -> deterministic PPR seeded from query-linked source/entity/claim nodes
  -> rank source engrams, never relations as facts
  -> existing candidate envelope and governance evaluation
  -> shadow benchmark comparison
```

Required questions:

- Does bounded PPR improve all-required-supporting-evidence recall for CLASS_B
  queries?
- Does it outperform current Graph Tier neighbor expansion?
- Does it avoid increasing stale, contradicted, low-authority, or misleading
  candidate evidence?
- Does it fall back correctly when graph signal is weak or no usable graph path
  exists?
- Can every surfaced graph path be traced back to source-grounded parent
  evidence?

A1 is split into two evidence phases:

| Phase | Purpose | Threshold Posture |
|---|---|---|
| `A1-SMOKE` | Validate projection correctness, fallback behavior, artifact shape, and safety invariants. | No superiority claim; safety gates only. |
| `A1-BENCHMARK` | Evaluate comparative retrieval value on a balanced CLASS_B truthset. | Requires retrieval improvement plus no safety regression. |

## Comparators

A1 benchmark comparisons must include:

| Comparator | Meaning |
|---|---|
| `semantic_hybrid_baseline` | Existing semantic or hybrid retrieval behavior, unchanged. |
| `graph_tier_neighbor_baseline` | Current Graph Tier neighbor expansion behavior in its existing experimental/shadow posture. |
| `bounded_ppr_projection` | A1 ephemeral graph projection with deterministic PPR ranking. |
| `oracle_path_analysis` | Optional analysis against gold evidence paths where fixture labels provide them. |

Comparators must run without changing production retrieval defaults.

## Projection Data Sources

A1 graph projections may be built only from existing MNEMOS artifacts:

- source-grounded engrams
- existing engram `edges`
- lineage and source metadata
- entity/attribute slot metadata where available
- contradiction records
- temporal and supersession metadata
- optional shadow-derived relation candidates, labeled as non-authoritative

Projection construction must not persist new storage identities. Every projected
node and edge must reference an existing artifact ID, existing metadata field, or
benchmark-local synthetic ID scoped to the run artifact only.

## Node Schema

Projected nodes use this normalized shape:

```json
{
  "node_id": "run-local stable id",
  "node_type": "source_engram",
  "artifact_ref": {
    "artifact_type": "engram",
    "artifact_id": "ENG-123"
  },
  "tenant_scope": "acme",
  "authority_class": "source_grounded",
  "source_lineage": {},
  "governance_state": {},
  "temporal_state": {},
  "review_state": "none",
  "redaction_state": "metadata_only"
}
```

Allowed node types:

| Node Type | Authority Posture | Notes |
|---|---|---|
| `source_engram` | `source_grounded` | Only node type that may be ranked as answer evidence. |
| `entity` | `navigation_only` | Derived from existing entity metadata or source-linked slots. |
| `attribute_slot` | `navigation_only` | Entity/attribute grouping key where already present. |
| `source_artifact` | `navigation_only` | Source document or artifact-level grouping. |
| `contradiction_cluster` | `audit_navigation_only` | Existing contradiction cluster metadata. |
| `resolution_engram` | `synthetic_resolution` | May connect to parent evidence, but cannot be promoted or used as source truth. |
| `time_period` | `navigation_only` | Normalized temporal bucket from existing metadata. |
| `shadow_relation_candidate` | `shadow_only` | Optional, non-authoritative, benchmark-local input only. |

Only `source_engram` nodes may be counted as retrieved supporting evidence.
Non-source nodes may seed, connect, explain, or route traversal, but must never
be treated as factual answer authority.

## Edge Schema

Projected edges use this normalized shape:

```json
{
  "edge_id": "run-local stable id",
  "edge_type": "mentions_entity",
  "source_node_id": "node-a",
  "target_node_id": "node-b",
  "source_artifact_id": "ENG-123",
  "source_span_ref": null,
  "weight": 1.0,
  "authority_class": "navigation_only",
  "provenance": "existing_metadata",
  "review_state": "none",
  "created_from": "projection_builder",
  "redaction_state": "metadata_only"
}
```

Allowed edge types:

| Edge Type | Source |
|---|---|
| `engram_edge` | Existing engram `edges`. |
| `mentions_entity` | Existing entity metadata or source-linked slots. |
| `has_attribute_slot` | Existing entity/attribute slot metadata. |
| `same_source_artifact` | Existing source lineage. |
| `same_time_window` | Existing temporal metadata. |
| `supersedes` | Existing temporal/supersession metadata. |
| `contradicts` | Existing contradiction records. |
| `resolved_by` | Existing resolution metadata. |
| `derived_from` | Existing lineage metadata. |
| `shadow_relation_hint` | Optional shadow-derived relation candidate. |

Edge weights are retrieval hints only. They must not alter trust, utility,
freshness, contradiction status, promotion status, or human-review state.

## PPR Seed Rules

Baseline A1 uses deterministic and structured seeds only. No LLM recognition
filter is included in the baseline path.

Seed construction:

1. Run existing semantic or hybrid retrieval for the CLASS_B query.
2. Select up to `seed_top_k` source-grounded seed engrams from the baseline
   result set after existing access and governance checks.
3. Add entity, attribute-slot, source-artifact, and temporal nodes directly
   linked to those seed engrams when present.
4. Add query-linked entity or attribute nodes only when matching is deterministic
   and auditable, such as exact normalized slot match or existing metadata key
   match.
5. Exclude shadow relation candidates from baseline seeding unless the benchmark
   variant explicitly enables `include_shadow_relation_hints=true`.

Default seed limits:

```text
seed_top_k = 10
max_seed_nodes = 40
max_query_linked_nodes = 20
```

If no source-grounded seed engrams survive authorization, governance, and
redaction checks, the A1 path must fall back to the baseline comparator.

## Traversal Limits

A1 traversal is bounded to prevent graph overreach and hub domination.

Default limits:

```text
max_projection_nodes = 2000
max_projection_edges = 8000
max_hops_from_seed = 2
max_neighbors_per_node = 25
max_source_engrams_ranked = 200
max_graph_candidates_returned = 20
hub_degree_cap = 100
ppr_damping = 0.85
min_ppr_score = 0.0
```

Traversal rules:

- Expansion starts from seed nodes only.
- Traversal may include non-source nodes, but output candidates must be
  source-grounded engrams.
- Contradiction and supersession edges are allowed for path awareness, not
  authority resolution.
- High-degree hub nodes above `hub_degree_cap` are either excluded or downweighted
  according to a deterministic benchmark parameter.
- Missing neighbor IDs are ignored and counted.
- Any candidate lacking source lineage is excluded from answer-evidence ranking
  and counted as a lineage failure.

## Candidate Ranking And Envelope

PPR ranks projected nodes, but only source-grounded engrams may enter the A1
candidate list.

Candidate output fields:

```json
{
  "engram_id": "ENG-123",
  "ppr_score": 0.0123,
  "graph_rank": 4,
  "path_refs": ["path:run-local-1"],
  "authority_class": "source_grounded",
  "governance_state": {},
  "temporal_state": {},
  "source_lineage": {},
  "redaction_state": "metadata_only"
}
```

A1 candidates must then pass through the existing candidate envelope and
governance evaluation used by the benchmark harness. If a source engram is
blocked by existing governance, it remains blocked. PPR score must not override
that decision.

Benchmark artifacts must preserve both stages:

- `ppr_ranked_source_engram_ids`: source-grounded engrams ranked by PPR before
  envelope and governance filtering.
- `governance_eligible_source_engram_ids`: PPR-ranked source engrams that remain
  eligible after existing envelope and governance evaluation.
- `governance_blocked_source_engram_ids`: PPR-ranked source engrams blocked by
  existing governance, access, lineage, redaction, temporal, or envelope rules.
- `blocked_reasons`: deterministic reason codes for blocked source engrams.

This split is required so A1 can distinguish graph discovery quality from
governance eligibility. A PPR result may locate relevant evidence that is
properly blocked; that must not be misclassified as graph-ranking failure.

## Fallback Criteria

The bounded PPR lane must fall back to `semantic_hybrid_baseline` when any of
the following occurs:

- no authorized source-grounded seed engrams are available
- projection construction exceeds node, edge, or latency budget
- no source-grounded candidate survives projection ranking
- graph candidates fail lineage completeness requirements
- graph candidates are dominated by stale, contradicted, or blocked evidence
  above configured safety thresholds
- PPR produces only candidates already present in the baseline and provides no
  evidence-chain gain
- projection construction errors or graph traversal errors occur

Fallback must be explicit in the benchmark artifact:

```json
{
  "fallback": {
    "occurred": true,
    "reason": "no_source_grounded_graph_candidates",
    "fallback_comparator": "semantic_hybrid_baseline"
  }
}
```

## Fixture And Truthset Design

A1 fixtures must focus on CLASS_B evidence-chain retrieval, not answer fluency.

Required query classes:

- multi-hop policy lineage
- cross-document entity relationships
- temporal supersession
- authority-conflict cases
- path-finding questions
- false-association traps
- cases where dense or hybrid retrieval should win
- cases where graph traversal should abstain or fall back

Each truthset item should include:

```json
{
  "query_id": "class_b_001",
  "query": "Which policy change affected contractor access review for the legacy identity provider?",
  "query_class": "CLASS_B",
  "tenant_scope": "acme",
  "required_supporting_engram_ids": ["ENG-101", "ENG-205"],
  "optional_supporting_engram_ids": ["ENG-309"],
  "misleading_engram_ids": ["ENG-404"],
  "stale_engram_ids": ["ENG-088"],
  "contradicted_engram_ids": ["ENG-077"],
  "low_authority_engram_ids": ["ENG-066"],
  "gold_path": [
    {"from": "ENG-101", "edge_type": "mentions_entity", "to": "entity:legacy-idp"},
    {"from": "entity:legacy-idp", "edge_type": "has_attribute_slot", "to": "slot:access-review"},
    {"from": "slot:access-review", "edge_type": "mentions_entity", "to": "ENG-205"}
  ],
  "expected_best_lane": "bounded_ppr_projection",
  "fallback_expected": false
}
```

Truthsets must distinguish:

- required evidence needed for a complete answer
- optional supporting evidence
- misleading evidence
- stale evidence
- contradicted evidence
- low-authority evidence
- expected fallback cases

`A1-BENCHMARK` should include at least 30-40 CLASS_B cases, balanced across the
case families above, before making any comparative retrieval-value claim.
Smaller fixture packs are allowed only for `A1-SMOKE`.

## Metrics

Primary metrics:

| Metric | Meaning |
|---|---|
| `all_required_evidence_recall_at_k` | Fraction of queries where all required supporting engrams appear in top-k. |
| `supporting_parent_recall_at_k` | Recall of required supporting engrams across all labels. |
| `ppr_stage_recall_at_k` | Required supporting evidence found by PPR before governance and envelope filtering. |
| `governance_eligible_recall_at_k` | Required supporting evidence remaining after governance and envelope filtering. |
| `graph_path_precision` | Fraction of surfaced graph paths that connect query seeds to relevant supporting evidence. |
| `authority_safe_candidate_rate` | Fraction of returned graph candidates that are source-grounded, authorized, and governance-eligible. |
| `fallback_correctness` | Whether fallback occurred exactly when expected by fixture labels or safety criteria. |

Seed-ceiling diagnostics:

| Metric | Meaning |
|---|---|
| `required_evidence_present_in_baseline_seed_set` | Required evidence already present in the top-k baseline seed set. |
| `required_evidence_recovered_only_via_graph_expansion` | Required evidence absent from the baseline seed set but recovered by graph expansion. |
| `required_evidence_unreachable_from_seed_set` | Required evidence not reachable from the seed projection under A1 traversal bounds. |
| `reachable_required_evidence_missed_by_ppr` | Required evidence structurally reachable from the projection but missed by PPR ranking. |

Seed-ceiling diagnostics are required to distinguish graph expansion failure,
seed recall failure, projection coverage failure, and true graph-added value.

Safety metrics:

| Metric | Meaning |
|---|---|
| `contradiction_leakage_rate` | Contradicted evidence returned without appropriate warning or suppression. |
| `stale_source_leakage_rate` | Stale evidence returned where fresher superseding evidence exists. |
| `low_authority_leakage_rate` | Low-authority evidence returned above source-grounded authoritative evidence. |
| `false_association_rate` | Graph candidates connected by misleading or irrelevant paths. |
| `lineage_failure_rate` | Returned graph candidates lacking required source lineage. |
| `shadow_relation_authority_violation_count` | Any shadow relation treated as answer evidence. Must be zero. |

Operational metrics:

| Metric | Meaning |
|---|---|
| `p50_latency_ms` | Median benchmark retrieval latency for each comparator. |
| `p95_latency_ms` | p95 benchmark retrieval latency for each comparator. |
| `projection_node_count` | Nodes in the ephemeral projection. |
| `projection_edge_count` | Edges in the ephemeral projection. |
| `missing_neighbor_id_count` | Existing edge references not resolvable in projection inputs. |
| `token_cost` | Zero for baseline A1 unless later ablations add LLM recognition filtering. |

## A1 Pass/Fail Thresholds

A1-SMOKE passes only if:

- projection construction uses only allowlisted existing artifacts and
  benchmark-local IDs
- fallback behavior is deterministic and explicitly recorded
- artifact shape is stable
- all zero-tolerance safety invariants pass
- no writes occur to engrams, graph stores, governance state, EBIR artifacts,
  Resolution Engrams, pattern stores, indexes, or Context Atlas artifacts

A1-SMOKE must not claim retrieval superiority.

A1-BENCHMARK may advance to a deeper offline prototype only if:

- `bounded_ppr_projection` improves `all_required_evidence_recall_at_k` over
  `semantic_hybrid_baseline` on CLASS_B cases by at least 10% relative.
- `bounded_ppr_projection` improves or ties `graph_tier_neighbor_baseline` on
  `all_required_evidence_recall_at_k`.
- `authority_safe_candidate_rate >= 0.98`.
- `lineage_failure_rate == 0` for returned graph candidates.
- `shadow_relation_authority_violation_count == 0`.
- `contradiction_leakage_rate` does not exceed the baseline comparator.
- `stale_source_leakage_rate` does not exceed the baseline comparator.
- `false_association_rate <= 0.05`, unless a stricter fixture-specific bound is
  defined.
- `fallback_correctness >= 0.95`.
- seed-ceiling diagnostics show that improvements are not solely explained by
  required evidence already appearing in the baseline seed set.
- no writes occur to engrams, graph stores, governance state, EBIR artifacts,
  Resolution Engrams, pattern stores, indexes, or Context Atlas artifacts.

Failure to meet any zero-tolerance safety invariant blocks advancement even if
recall improves.

## Safety Invariants

A1 benchmark execution must assert:

- Graph artifacts are retrieval hints only.
- Only source-grounded engrams may count as answer evidence.
- PPR score never changes trust, utility, freshness, governance state, conflict
  status, promotion status, or review state.
- Contradiction records can inform path awareness but cannot be resolved by A1.
- EBIR artifacts and Resolution Engrams cannot be modified or promoted by A1.
- Shadow-derived relation candidates, when enabled, remain `shadow_only`.
- The baseline `/v1/mnemos/search` behavior remains unchanged.
- Context Atlas P0 remains read-only and does not expose A1 exploration behavior.
- All benchmark output is redacted according to existing source and tenant rules.

## Separation From Context Atlas P0

Context Atlas P0 remains a read-only observability and namespace facade with only
these approved endpoint classes:

- `tree`
- `resolve`
- `trajectory`

A1 must not implement or rely on:

- `POST /v1/mnemos/context/explore`
- Atlas-backed semantic search
- Atlas-backed graph traversal
- Atlas-generated summaries
- Atlas write paths
- MCP exposure
- agent-facing memory writes

Future Context Atlas exposure of A1 results would require a separate post-EBIR-R2
review and must preserve Atlas authority labeling and redaction rules.

## Separation From EBIR-R2

EBIR-R2 remains the active proof obligation for human review value. A1 is
deferred behind EBIR-R2 protocol and reviewer-corpus freeze.

A1 must not:

- alter EBIR packets
- influence EBIR promotion status
- change EBIR benchmark pass/fail criteria
- provide authoritative reconciliation output
- select contradiction winners
- claim reviewer-value improvement without a separate human review trial

## Recognition Filtering Ablation

Recognition filtering is explicitly out of the baseline A1 path.

A later ablation may compare:

- deterministic structured seeds only
- deterministic seeds plus recognition filter

That ablation must report:

- precision
- recall
- latency
- token cost
- fallback behavior
- useful-link removal rate
- misleading-link removal rate

Recognition filtering decisions must be recorded as benchmark metadata and must
not silently steer production retrieval.

## CI Gate

Suggested test files:

```text
tests/test_associative_retrieval_a1_projection.py
tests/test_associative_retrieval_a1_ppr.py
tests/test_associative_retrieval_a1_fallback.py
tests/test_associative_retrieval_a1_invariants.py
```

Suggested benchmark command after implementation:

```bash
python tools/run_associative_retrieval_a1_benchmark.py
python -m pytest tests/test_associative_retrieval_a1_*.py
python tools/run_ebir_refinement_benchmark.py
```

Required CI assertions:

| Gate ID | Area | Required Assertion |
|---|---|---|
| AR-A1-001 | Deferred posture | A1 code is not reachable from production search or Context Atlas endpoints. |
| AR-A1-002 | Read-only behavior | A1 benchmark causes no writes to source artifacts, engrams, indexes, governance, EBIR, resolutions, pattern stores, or Atlas artifacts. |
| AR-A1-003 | Projection inputs | Projection uses only allowlisted existing artifacts and benchmark-local IDs. |
| AR-A1-004 | Source-only evidence | Only source-grounded engrams are counted as retrieved supporting evidence. |
| AR-A1-005 | Shadow relation isolation | Shadow-derived relation candidates remain non-authoritative and never answer evidence. |
| AR-A1-006 | PPR non-authority | PPR score does not mutate governance, trust, utility, freshness, contradiction, promotion, or review state. |
| AR-A1-007 | Fallback determinism | Weak, missing, unsafe, or failed graph signal falls back to semantic/hybrid baseline with explicit reason. |
| AR-A1-008 | Comparator stability | Semantic/hybrid baseline and current Graph Tier comparator remain behaviorally unchanged. |
| AR-A1-009 | Safety thresholds | Leakage, false association, lineage, and authority metrics meet pass/fail thresholds. |
| AR-A1-010 | EBIR preservation | EBIR-R1/R2 blocked-promotion posture is unchanged. |

## Artifact Format

The benchmark should write a raw JSON artifact:

```text
benchmarks/results/associative_retrieval_a1_benchmark.json
```

Top-level shape:

```json
{
  "benchmark_id": "associative_retrieval_a1",
  "status": "pass",
  "phase": "A1-SMOKE",
  "generated_at": "2026-06-19T00:00:00Z",
  "deferred_behind": "EBIR-R2 protocol and reviewer-corpus freeze",
  "comparators": {
    "semantic_hybrid_baseline": {},
    "graph_tier_neighbor_baseline": {},
    "bounded_ppr_projection": {},
    "oracle_path_analysis": {}
  },
  "metrics": {},
  "safety_invariants": {
    "write_mutation_count": 0,
    "shadow_relation_authority_violation_count": 0,
    "default_search_delta": false,
    "context_atlas_endpoint_delta": false,
    "ebir_promotion_delta": false
  },
  "fallback_summary": {},
  "query_results": []
}
```

Each query result should include:

```json
{
  "query_id": "class_b_001",
  "query_class": "CLASS_B",
  "comparators": {
    "semantic_hybrid_baseline": {
      "retrieved_engram_ids": [],
      "metrics": {}
    },
    "graph_tier_neighbor_baseline": {
      "retrieved_engram_ids": [],
      "metrics": {}
    },
    "bounded_ppr_projection": {
      "retrieved_engram_ids": [],
      "ppr_ranked_source_engram_ids": [],
      "governance_eligible_source_engram_ids": [],
      "governance_blocked_source_engram_ids": [],
      "blocked_reasons": {},
      "path_refs": [],
      "fallback": {
        "occurred": false,
        "reason": null
      },
      "projection_stats": {
        "node_count": 0,
        "edge_count": 0,
        "missing_neighbor_id_count": 0
      },
      "metrics": {}
    }
  },
  "seed_ceiling_diagnostics": {
    "required_evidence_present_in_baseline_seed_set": [],
    "required_evidence_recovered_only_via_graph_expansion": [],
    "required_evidence_unreachable_from_seed_set": [],
    "reachable_required_evidence_missed_by_ppr": []
  },
  "safety_flags": [],
  "redaction_state": "metadata_only"
}
```

The artifact must not include raw source text, raw query text where prohibited,
private prompts, chain-of-thought, unrestricted source spans, or full ledger
payloads unless an existing compliant benchmark policy already permits them.

## Sequencing

1. Freeze EBIR-R2 protocol and reviewer corpus.
2. Keep Context Atlas P0 spec and implementation separate from A1.
3. Approve this A1 specification.
4. Build fixture and truthset pack for CLASS_B evidence-chain evaluation.
5. Run `A1-SMOKE` first against a small fixture pack to validate artifact shape,
   fallback behavior, and safety invariants.
6. Expand to a balanced `A1-BENCHMARK` truthset before making comparative
   retrieval-value claims.
7. Implement benchmark-local projection builder with no production route.
8. Implement deterministic PPR comparator.
9. Run semantic/hybrid, Graph Tier, bounded PPR, and optional oracle/path
   analysis comparators.
10. Enforce CI gates and write benchmark artifact.
11. Decide whether a deeper offline prototype is justified.

No runtime integration, durable relation store, recognition filtering ablation,
or Context Atlas exposure begins until A1 benchmark evidence passes and a
separate review authorizes the next track.
