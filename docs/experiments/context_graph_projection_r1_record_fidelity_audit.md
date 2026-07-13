# Context Graph Projection R1 Record Fidelity Audit

Date: 2026-07-13

Status: **Research-only artifact. No implementation authorized.**

## Executive Summary

This audit evaluates which Context Graph Projection nodes and edges can support
the top-ranked R1 use case:

```text
Evidence-to-decision traceability
```

The audit asks whether each proposed node and edge can be projected
deterministically from existing MNEMOS records without inference, LLM judgment,
new storage semantics, retrieval changes, governance changes, promotion
changes, or authority changes.

The smallest viable future JSON-only projection path is:

```text
source_artifact
  -> source_engram
  -> retrieval_result_set
  -> evaluation_result or decision record
  -> handoff_package or context_package
```

R1 should prefer exclusion over speculation. Anything not backed by stable IDs,
lineage metadata, package metadata, evaluation artifacts, or audit-safe records
is not eligible for the first projection.

## Boundary

This audit does not authorize:

- code implementation
- graph storage
- graph database adoption
- GraphRAG
- graph-assisted retrieval
- retrieval reranking
- governance mutation
- promotion mutation
- context assembly changes
- Engram schema changes

The graph remains an explanation layer, not a retrieval or authority layer.

## Classification Labels

| Label | Meaning |
|---|---|
| `READY_NOW` | Documented records appear sufficient for deterministic projection with stable backing fields. |
| `PARTIALLY_READY` | The concept exists, but R1 needs a field audit or stronger stable references before implementation. |
| `NOT_READY` | The relationship may be valuable, but current records are not documented enough for deterministic projection. |
| `EXCLUDED_FROM_R1` | The candidate requires inference, LLM judgment, speculative semantics, or authority-risky interpretation. |

## Evidence Base

This audit is grounded in the current documentation, especially:

- `docs/architecture.md`
- `docs/whitepaper.md`
- `docs/experiments/context_graph_projection_r0_design_note.md`
- `docs/experiments/context_graph_projection_r1_use_cases.md`

The audit does not inspect runtime data, database rows, or source-code field
definitions. A later implementation proposal must verify these findings against
actual models and persisted artifacts before code is authorized.

## Node Fidelity Audit

| Node type | Memory class | Backing artifact | Required fields | Classification | Risk | R1 eligible? |
|---|---|---|---|---|---|---|
| `source_artifact` | Evidence | Existing source metadata on Engrams, indexed source records, intake artifacts | Stable source identifier, source URI or source label, tenant/scope if applicable | `PARTIALLY_READY` | Source identity may be represented as labels or URIs rather than normalized artifact IDs in all lanes | Yes, if stable source IDs or deterministic source keys exist |
| `source_engram` | Evidence | Engram model | Engram ID, source reference, metadata, lineage-ready fields | `READY_NOW` | Low, assuming indexed Engrams have stable IDs | Yes |
| `summary_engram` | Evidence | Existing summary Engrams | Engram ID, synthetic label, parent Engram refs or edges | `PARTIALLY_READY` | Useful for later trace views, but not required for the smallest evidence-to-decision path | Not in minimal R1 path |
| `resolution_engram` | Evidence/decision | Existing Resolution Engrams | Engram ID, synthetic label, parent conflict refs or edges, governance metadata | `PARTIALLY_READY` | Strong lineage exists conceptually, but R1 should not make contradiction-resolution behavior central yet | Not in minimal R1 path |
| `search_request` | Working | Search API event, cognitive-cycle record, audit event | Query/request ID, timestamp, caller/scope, retrieval metadata | `PARTIALLY_READY` | Search events are documented, but stable request IDs must be verified across API, ledger, and cycle records | Optional for minimal path |
| `retrieval_result_set` | Working | Search response, cognitive-cycle record, evidence receipt, audit event | Result-set ID or deterministic request-derived key, returned Engram IDs, rank/score metadata, timestamp | `PARTIALLY_READY` | Result sets may not be durable as first-class records in all flows | Yes, if represented as projection-local node keyed from request/audit record |
| `context_package` | Working | Session Context Assembler package | Package ID or digest, parent Engram IDs, parent source IDs, lineage completeness, safety labels | `READY_NOW` | Limited to the accepted local shadow milestone and artifacts that preserve package metadata | Yes |
| `handoff_package` | Working | Handoff artifact or summarized handoff workflow | Handoff ID or artifact path, included evidence refs, included context refs, timestamp | `PARTIALLY_READY` | Handoff representation may be workflow-dependent rather than a uniform persisted record | Yes, if artifact-local refs are present |
| `session_artifact` | Working | Session observation, task state, or workflow artifact | Artifact ID, session ID, source refs or Engram refs, timestamp | `PARTIALLY_READY` | Session artifacts may vary by consumer workflow | Not in minimal R1 path |
| `cognitive_cycle_record` | Decision | CoALA cognitive-cycle record | Cycle ID, actions, governance summaries, forensic ledger refs, retrieval/grounding refs | `READY_NOW` | Low for audit-safe metadata; raw reasoning remains excluded | Yes, as a decision-record candidate |
| `action_record` | Decision | CoALA action record | Action ID or stable position in cycle, operation type, associated refs | `PARTIALLY_READY` | Stable action identity must be verified if actions are only embedded in cycle records | Optional |
| `governance_eval_summary` | Decision | Governance summary in cognitive cycle or response metadata | Mode, veto/suppression counts, blocked candidates, reason codes where available | `PARTIALLY_READY` | Useful for explanation, but may not identify every candidate edge uniformly | Optional |
| `contradiction_cluster` | Decision | Contradiction records, governance hygiene artifacts | Cluster ID, member Engram IDs, status, lineage refs | `PARTIALLY_READY` | Valuable later, but not needed for primary evidence-to-decision trace | Not in minimal R1 path |
| `pattern_candidate` | Decision | PatternEngramCandidate store | Candidate ID, source cycle refs, promotion status, advisory flags | `EXCLUDED_FROM_R1` | Advisory learning records are not part of the top-ranked evidence-to-decision trace | No |
| `pattern_engram` | Decision | Approved PatternEngram | Pattern ID, approved candidate ref, governance review ID | `EXCLUDED_FROM_R1` | Promotion semantics are outside the first projection and invite authority confusion | No |
| `evaluation_result` | Decision | Derived-fact shadow evaluation, benchmark/evidence outcome, formal evaluation artifact | Evaluation ID, accepted/rejected status, source refs, result status, timestamp | `PARTIALLY_READY` | Evaluation artifacts exist, but their schema varies by lane | Yes, if the selected evaluation lane has stable IDs and source refs |
| `claim_record` | Decision | Formal claim record when represented by an artifact lane | Claim ID, status, evidence refs, evaluation ref | `PARTIALLY_READY` | Claims may be embedded in reports rather than structured records | Optional |
| `ledger_event` | Audit | Forensic ledger | Transaction/event ID, operation type, target refs, timestamp | `READY_NOW` | Low, but disclosure constraints must be enforced | Yes |
| `lineage_digest` | Audit/control | Context package digest, handoff digest, verification artifact | Digest value, artifact ID, verification scope | `READY_NOW` | Low where digest artifacts exist; not all records may have digests | Yes for packages with digests |

## Edge Fidelity Audit

| Edge type | Backing artifact | Required fields | Classification | Risk | R1 eligible? |
|---|---|---|---|---|---|
| `derived_from_source_artifact` | Engram source metadata | Source artifact ID or deterministic source key, Engram ID | `PARTIALLY_READY` | Source may be a URI/string label rather than a normalized source artifact ID | Yes, if deterministic source keying is accepted |
| `related_to_engram` | Existing Engram `edges` | Source Engram ID, target Engram ID | `READY_NOW` | Existing edge semantics may be broad and should not be treated as evidence-to-decision support | Not in minimal R1 path |
| `summarizes` | Summary Engram metadata and parent edges | Summary Engram ID, parent Engram IDs | `PARTIALLY_READY` | Synthetic summary semantics require careful labels | Not in minimal R1 path |
| `resolves` | Resolution Engram metadata and parent edges | Resolution Engram ID, contradicted parent Engram IDs | `PARTIALLY_READY` | Resolution has governance implications; avoid in first trace path | Not in minimal R1 path |
| `supersedes` | Temporal/supersession metadata | Newer artifact ID, earlier artifact ID, timestamp/status | `PARTIALLY_READY` | Supersession fields may vary by lane | Not in minimal R1 path |
| `retrieved` | Search response, cycle record, audit event | Search/request ID or projection-local event ID, result-set key | `PARTIALLY_READY` | Retrieval result set may need projection-local identity derived from audit or response metadata | Yes |
| `includes_retrieved_engram` | Search response, evidence receipt, cycle record | Result-set key, Engram ID, rank/score metadata where available | `PARTIALLY_READY` | Ready if result set is captured; otherwise transient responses may be unavailable after the fact | Yes |
| `packages` | Session Context Assembler package | Package ID or digest, included Engram/source artifact IDs | `READY_NOW` | Limited to packages that retain artifact-local provenance | Yes |
| `includes_handoff_artifact` | Handoff artifact | Handoff ID/path, included artifact IDs or package refs | `PARTIALLY_READY` | Handoff records may not be uniform across workflows | Yes, if refs are explicit |
| `observed_in_session` | Session artifact | Session artifact ID, source or Engram refs | `PARTIALLY_READY` | Session observations can drift toward weak provenance | Not in minimal R1 path |
| `recorded_in_cycle` | CognitiveCycleRecord | Cycle ID, action/governance entry ref | `READY_NOW` | Stable identity for embedded actions may need projection-local keys | Optional |
| `used_evidence` | CognitiveCycleRecord, evaluation artifact, evidence receipt | Decision/evaluation/cycle ID, source Engram IDs | `PARTIALLY_READY` | Must not infer use from semantic similarity; only explicit refs qualify | Yes, if explicit evidence refs exist |
| `blocked_by_governance` | Governance eval summary | Candidate Engram ID, governance summary ID, reason code | `PARTIALLY_READY` | Useful but may be incomplete for all result records | Optional |
| `contradicts` | Contradiction records | Contradiction cluster ID, member Engram IDs | `PARTIALLY_READY` | High value later, but not required for the first trace path | Not in minimal R1 path |
| `promoted_from_candidate` | Pattern candidate and PatternEngram records | PatternEngram ID, candidate ID, governance review ID | `EXCLUDED_FROM_R1` | Promotion semantics should remain outside first trace path | No |
| `evaluated_as` | Claim/evaluation artifact | Claim ID, evaluation result ID, accepted/rejected status | `PARTIALLY_READY` | Claim records may not be structured uniformly | Yes, if the selected evaluation lane has explicit claim IDs |
| `justified_by` | Evaluation result, decision record, context package | Decision/evaluation ID, source Engram or context package refs | `PARTIALLY_READY` | Must not infer justification from text; only explicit evidence refs qualify | Yes, if explicit refs exist |
| `audited_by` | Forensic ledger | Artifact ID, ledger event ID | `READY_NOW` | Disclosure rules still required | Yes |
| `verified_by_digest` | Package digest or verification artifact | Package/handoff ID, digest value | `READY_NOW` | Only available where digest artifacts exist | Yes |

## Explicitly Excluded From R1

The following relationship families are excluded from R1 even if they may be
useful later:

- causal influence edges
- semantic equivalence edges
- model-inferred entity equivalence
- "reasoned because" edges
- hidden reasoning or raw chain-of-thought edges
- graph-mined explanation edges
- centrality, similarity, or connectivity-derived authority edges
- graph-assisted retrieval edges
- graph-based reranking signals

These are excluded because they require inference, invite authority drift, or
would need a separate evaluation lane.

## Smallest Viable JSON-Only Projection Path

The smallest viable future projection should avoid optional or high-risk nodes
and focus on explicit evidence-to-decision traceability.

### Minimal node set

| Node type | Classification | Required for minimal path? |
|---|---|---|
| `source_artifact` | `PARTIALLY_READY` | Yes |
| `source_engram` | `READY_NOW` | Yes |
| `retrieval_result_set` | `PARTIALLY_READY` | Yes |
| `evaluation_result` | `PARTIALLY_READY` | Yes, if evaluating formal outcomes |
| `cognitive_cycle_record` | `READY_NOW` | Yes, if representing decision records |
| `handoff_package` | `PARTIALLY_READY` | Optional but useful |
| `context_package` | `READY_NOW` | Optional but useful |
| `ledger_event` | `READY_NOW` | Recommended |
| `lineage_digest` | `READY_NOW` | Recommended for packages |

### Minimal edge set

| Edge type | Classification | Required for minimal path? |
|---|---|---|
| `derived_from_source_artifact` | `PARTIALLY_READY` | Yes |
| `retrieved` | `PARTIALLY_READY` | Yes |
| `includes_retrieved_engram` | `PARTIALLY_READY` | Yes |
| `used_evidence` | `PARTIALLY_READY` | Yes, if explicit refs exist |
| `evaluated_as` | `PARTIALLY_READY` | Yes for formal evaluation trace |
| `justified_by` | `PARTIALLY_READY` | Yes, if explicit refs exist |
| `packages` | `READY_NOW` | Optional but useful |
| `includes_handoff_artifact` | `PARTIALLY_READY` | Optional but useful |
| `audited_by` | `READY_NOW` | Recommended |
| `verified_by_digest` | `READY_NOW` | Recommended for packages |

### Proposed minimal path

```text
source_artifact
  -> source_engram
  -> retrieval_result_set
  -> cognitive_cycle_record or evaluation_result
  -> context_package or handoff_package
```

### Minimal path caveat

The path is viable only where retrieval, decision/evaluation, and package
records carry explicit Engram IDs, source refs, ledger refs, or digest refs.
If a record only contains prose or inferred semantic references, it must be
omitted or labeled `lineage_incomplete`.

## R1 Eligibility Summary

| Candidate group | R1 posture |
|---|---|
| Source-to-Engram lineage | Eligible if source keys are stable |
| Retrieval inclusion | Eligible if result sets can be reconstructed or explicitly captured |
| Decision/evaluation evidence use | Eligible only with explicit evidence refs |
| Package and handoff inclusion | Eligible where artifact-local provenance exists |
| Audit and digest refs | Eligible where existing refs are present |
| Contradiction/resolution paths | Defer from minimal R1 projection |
| Pattern promotion paths | Exclude from R1 |
| Speculative explanation paths | Exclude from R1 |
| Graph-assisted retrieval paths | Exclude from R1 |

## Required Follow-Up Checks Before Any Future Implementation

A future implementation proposal must verify:

- source artifact key normalization
- durable or reconstructable retrieval result-set identity
- explicit evidence refs in selected decision or evaluation artifacts
- explicit artifact refs in selected handoff packages
- package digest availability and verification scope
- authorization and disclosure behavior for relationship edges
- omission or labeling rules for lineage-incomplete records

If any of these fail, the future implementation must narrow scope rather than
invent relationships.

## Acceptance Statement

This audit is acceptable only under the following interpretation:

```text
R1 may proceed only with nodes and edges backed by existing stable records.

Partial readiness means "research further before implementation,"
not "fill the gap with inference."

The smallest future projection should show evidence-to-decision traceability
without creating graph-derived truth or changing MNEMOS behavior.
```

## Closeout Labels

```text
CONTEXT_GRAPH_PROJECTION_R1_RECORD_FIDELITY_AUDIT_COMPLETE
RESEARCH_ONLY
EVIDENCE_TO_DECISION_TRACEABILITY_ANCHORED
READY_NOW_AND_PARTIALLY_READY_SEPARATED
PREFER_EXCLUSION_OVER_SPECULATION
NO_IMPLEMENTATION_AUTHORIZED
NO_GRAPH_STORAGE
NO_GRAPHRAG
NO_RETRIEVAL_GOVERNANCE_PROMOTION_OR_CONTEXT_ASSEMBLY_CHANGE
```
