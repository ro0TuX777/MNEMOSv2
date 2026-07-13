# Context Graph Projection R0 Design Note

Date: 2026-07-13

Status: **Design-only artifact. No implementation authorized.**

## Executive Summary

Context Graph Projection R0 is a **read-only projection** over existing MNEMOS
records. It is not a new storage system, not a new authority model, not a graph
database initiative, and not a retrieval-path change.

R0 exists to give MNEMOS a boundary-safe way to describe and inspect connected
memory across evidence, working context, and decision records without changing
what is true, what is retrievable, what is promotable, or what is authoritative.

The projection posture is:

```text
R0 projects existing truth.
R0 does not create new truth.
```

## Purpose

MNEMOS already contains source-grounded lineage, retrieval artifacts,
governance state, bounded context assembly, cognitive-cycle records, pattern
candidate records, contradiction/supersession structures, and audit references.
Those records are currently described across different feature lanes and
artifacts.

R0 proposes a single **read-only context graph projection** that exposes the
relationships already present in those records for inspection, visualization,
handoff tracing, and evidence-to-decision auditability.

This is a product-language and observability lane, not a memory-authority lane.

## Vocabulary

R0 uses the following three-memory vocabulary:

### Evidence memory

- source-grounded long-term evidence from PDFs, docs, code, research artifacts,
  notes, and indexed project material

### Working memory

- bounded session context
- current task state
- handoff packages
- consumer-neutral context packages
- retrieved evidence packages

### Decision memory

- audit-safe decision records
- accepted or rejected claims
- supersession
- contradiction status
- handoff inclusion
- evaluation outcomes
- tool/action lineage
- bounded outcome justifications

## Core Boundary

R0 does **not** adopt a “single graph brain” model.

MNEMOS keeps the following concerns distinct:

- source evidence
- retrieval candidates and retrieval results
- bounded working-context packages
- decision and evaluation records
- governance state
- audit and ledger references

The context graph projection connects those concerns without collapsing them
into one authority surface.

## Non-Goals

R0 does not include any of the following:

- graph database dependency
- Neo4j dependency
- GraphRAG
- graph-first retrieval
- graph-based reranking
- new write path
- new durable graph store
- new truth-creation path
- LLM-generated edges treated as authoritative
- raw chain-of-thought capture
- hidden model reasoning capture
- unverifiable internal trace capture
- changes to Engram structure
- changes to retrieval ranking
- changes to governance behavior
- changes to context assembly behavior
- changes to promotion logic
- changes to contradiction resolution behavior

## Existing MNEMOS Grounds For This Projection

R0 is grounded in existing MNEMOS records and boundaries already described in
the current documentation:

- Engrams already carry source, metadata, tags, confidence, and lineage-ready
  fields.
- Engrams already expose related `edges`.
- Governance already tracks contradiction state, lifecycle state, and
  suppressions.
- Resolution and summary artifacts already preserve parent lineage rather than
  replacing it.
- The CoALA cognitive cycle already records audit-safe per-request cognitive and
  action metadata without changing retrieval behavior.
- Session Context Assembler already defines bounded, non-authoritative context
  packages with provenance labels and explicit abstention behavior.
- The forensic ledger already provides audit correlation for operations and
  mutations.

R0 therefore begins as a projection over records MNEMOS already owns.

## 1. Node Types Projectable From Existing MNEMOS Records

R0 should expose only node types that can be derived from existing records or
existing compliant artifacts.

### Evidence-memory nodes

- `source_artifact`
  - document, PDF, code artifact, note, research file, or indexed project
    material already represented through existing source metadata
- `source_engram`
  - an indexed source-grounded engram
- `summary_engram`
  - existing summary engram, labeled synthetic and lineage-preserving
- `resolution_engram`
  - existing additive contradiction-resolution artifact, lineage-preserving and
    explicitly synthetic

### Working-memory nodes

- `search_request`
  - a bounded retrieval invocation or retrieval event
- `retrieval_result_set`
  - the set of returned evidence candidates or finalized returned results for a
    specific query/event
- `context_package`
  - a bounded consumer-neutral Session Context Assembler package
- `handoff_package`
  - a session handoff or summarized handoff artifact
- `session_artifact`
  - current-task or prior-session observation/state artifact already captured by
    an existing MNEMOS-compatible workflow

### Decision-memory nodes

- `cognitive_cycle_record`
  - an existing CoALA cognitive-cycle record
- `action_record`
  - an audit-safe action entry referenced from a cognitive cycle
- `governance_eval_summary`
  - an existing governance outcome summary
- `contradiction_cluster`
  - an existing contradiction cluster or contradiction grouping artifact
- `pattern_candidate`
  - an advisory PatternEngramCandidate
- `pattern_engram`
  - an approved PatternEngram
- `evaluation_result`
  - an accepted/rejected evaluation artifact or benchmark/evidence outcome
- `claim_record`
  - an accepted or rejected formal claim record when already represented by an
    existing artifact lane

### Audit and control nodes

- `ledger_event`
  - a forensic-ledger reference or correlated audit event
- `lineage_digest`
  - a digest or verification artifact that proves package or artifact integrity

R0 must not invent node types whose only source is inferred model reasoning.

## 2. Edge Types Derivable Deterministically

R0 should expose only deterministic edges backed by existing records, metadata,
or artifact-local references.

### Evidence edges

- `derived_from_source_artifact`
  - source artifact → source engram
- `related_to_engram`
  - source engram → source engram via existing `edges`
- `summarizes`
  - summary engram → parent source engram(s)
- `resolves`
  - resolution engram → contradicted parent engram(s)
- `supersedes`
  - newer artifact or decision → earlier artifact or decision via existing
    temporal/supersession metadata

### Retrieval and context edges

- `retrieved`
  - search request → retrieval result set
- `includes_retrieved_engram`
  - retrieval result set → source engram
- `packages`
  - context package → source engram or supporting artifact
- `includes_handoff_artifact`
  - handoff package → source engram, context package, session artifact, or
    evaluation result
- `observed_in_session`
  - session artifact → source artifact or source engram when already captured by
    the session record

### Decision and governance edges

- `recorded_in_cycle`
  - action record or governance summary → cognitive cycle record
- `used_evidence`
  - cognitive cycle record or evaluation result → source engram
- `blocked_by_governance`
  - retrieval result or candidate → governance eval summary
- `contradicts`
  - contradiction cluster → source engram or resolution engram
- `promoted_from_candidate`
  - pattern engram → pattern candidate
- `evaluated_as`
  - claim record → evaluation result
- `justified_by`
  - evaluation result or decision record → source engram or context package

### Audit edges

- `audited_by`
  - projected artifact → ledger event
- `verified_by_digest`
  - context package or handoff package → lineage digest

No R0 edge may be produced solely from an LLM judgment.

## 3. Audit-Safe Versus Speculative Relationships

R0 must explicitly separate **audit-safe** relationships from **speculative**
relationships.

### Audit-safe relationships

These are allowed in R0:

- lineage links already present in Engram, governance, or package metadata
- explicit parent-child relationships
- retrieval inclusion relationships
- context-package inclusion relationships
- handoff inclusion relationships
- contradiction, supersession, and resolution references already present in
  governed records
- approved pattern promotion lineage
- cognitive-cycle references to recorded actions, governance summaries, and
  forensic-ledger refs
- digest and integrity-verification references

### Speculative relationships

These are not part of R0 unless separately labeled and separately authorized:

- model-inferred causal claims
- model-inferred entity equivalence without existing deterministic resolver
  output
- model-inferred “reasoned because” links with no audit-safe backing
- graph edges mined from free-form reasoning text
- graph edges inferred from hidden chain-of-thought
- unstored or unverifiable “this influenced that” assumptions

If a future lane needs speculative links, they must be explicitly labeled
`speculative`, `non_authoritative`, and `excluded_from_authority`, and they are
out of scope for R0.

## 4. Metadata That Proves Lineage

R0 should rely on metadata already available in MNEMOS records or explicitly
allowed artifact-local metadata. Lineage-proof metadata includes:

- source artifact identifiers
- engram identifiers
- parent engram identifiers
- parent source identifiers
- artifact type labels
- source URI or source classification markers
- tenant or scope markers where applicable
- timestamps and supersession markers
- contradiction-cluster identifiers
- governance-state summaries
- package digest values
- redaction-state labels
- non-authoritative or non-promotable labels
- forensic-ledger reference IDs
- promotion-review IDs for approved pattern promotion
- evaluation artifact IDs and result status

R0 should not claim lineage when the underlying artifact cannot provide stable
IDs or verifiable references.

## 5. Projection Export Shape

R0 should export a **read-only graph projection artifact**, not a database or
new storage surface.

Recommended export shape:

```json
{
  "projection_id": "context_graph_projection_r0",
  "generated_at": "2026-07-13T00:00:00Z",
  "scope": {
    "tenant": "example",
    "session": "optional",
    "query_id": "optional",
    "artifact_ids": ["optional"]
  },
  "labels": [
    "read_only",
    "non_authoritative",
    "projection_only",
    "no_truth_creation"
  ],
  "nodes": [
    {
      "node_id": "engram:ENG-123",
      "node_type": "source_engram",
      "memory_class": "evidence_memory",
      "authority_class": "source_grounded",
      "artifact_ref": {
        "artifact_type": "engram",
        "artifact_id": "ENG-123"
      },
      "lineage": {
        "source_artifact_id": "SRC-1",
        "parent_engram_ids": []
      },
      "safety": {
        "audit_safe": true,
        "non_authoritative": false,
        "non_promotable": false,
        "speculative": false
      },
      "metadata": {}
    }
  ],
  "edges": [
    {
      "edge_id": "edge-1",
      "edge_type": "derived_from_source_artifact",
      "from": "source:SRC-1",
      "to": "engram:ENG-123",
      "derivation": "deterministic",
      "audit_safe": true,
      "authority_effect": "none",
      "evidence_ref": {
        "artifact_type": "engram",
        "artifact_id": "ENG-123",
        "field": "source"
      }
    }
  ],
  "prohibitions": [
    "does_not_change_retrieval_ranking",
    "does_not_change_governance",
    "does_not_change_promotion",
    "does_not_create_truth",
    "does_not_capture_raw_chain_of_thought"
  ]
}
```

### Export requirements

- JSON-first for auditability and portability
- optional visualization layer may consume the JSON artifact
- every node and edge must include a derivation class
- every node and edge must include an authority-effect field or equivalent
  neutral marker
- speculative relationships are excluded in R0
- missing lineage must cause omission or explicit `lineage_incomplete` labeling,
  not silent fabrication

## 6. What Is Explicitly Read-Only

R0 is explicitly read-only in all of the following senses:

- no writes to Engrams
- no writes to graph stores
- no writes to governance state
- no writes to contradiction state
- no writes to resolution artifacts
- no writes to pattern candidates or pattern promotions
- no writes to context-package content
- no writes to handoff-package content
- no writes to evaluation outcomes
- no write-back into source metadata
- no mutation of audit records beyond existing normal audit generation

R0 may read and project existing compliant artifacts only.

## 7. What Is Prohibited From Affecting Retrieval, Governance, Promotion, Or Authority

The R0 projection must have **zero effect** on:

- retrieval mode selection
- retrieval ranking
- candidate-envelope composition
- semantic or hybrid search behavior
- Graph Tier behavior
- governance scoring
- contradiction adjudication
- suppression logic
- freshness or lifecycle state
- promotion eligibility
- pattern approval rules
- context-assembly policy
- authorization or disclosure policy
- truth status of any record

The following are specifically prohibited:

- treating a projected edge as a fact unless the backing artifact is already a
  fact-bearing record
- allowing projected graph centrality or connectivity to affect ranking or
  authority
- allowing decision-memory records to mutate source truth
- allowing audit links to imply truth beyond their backing artifact
- allowing handoff inclusion to imply factual correctness
- allowing speculative or model-inferred links to enter R0 as audit-safe

## 8. Minimal Demo

The minimal R0 demo should show:

```text
source artifact
  → source engram
  → retrieval result set
  → decision or evaluation record
  → handoff package or context package
```

### Minimal demo scenario

1. A source PDF, document, code file, or note is indexed into one or more
   Engrams.
2. A search request retrieves one or more source-grounded Engrams.
3. A cognitive-cycle or evaluation artifact records that those results were
   used, accepted, rejected, contradicted, or superseded.
4. A handoff package or context package includes the selected supporting
   artifacts.
5. The exported projection shows the end-to-end deterministic trace.

### Demo acceptance

The demo is sufficient if it can answer all of the following:

- what source shaped this retrieval result
- what retrieved evidence was included in the downstream package
- what decision or evaluation referenced that evidence
- whether the referenced result was accepted, rejected, contradicted, or
  superseded
- what handoff or bounded context package included it

The demo is not sufficient if it depends on inferred edges, graph reranking, or
new storage semantics.

## Recommended R0 Labels

```text
CONTEXT_GRAPH_PROJECTION_R0
READ_ONLY
NON_AUTHORITATIVE
PROJECTION_ONLY
EVIDENCE_MEMORY
WORKING_MEMORY
DECISION_MEMORY
NO_TRUTH_CREATION
```

## Suggested Phasing

R0 should remain design-only until separately authorized.

If a future implementation lane is opened, it should still remain:

- read-only by default
- JSON projection first
- audit-safe only
- lineage-complete where possible
- visualization-friendly
- authority-neutral

Any future retrieval, governance, or graph-assisted ranking lane must be
proposed separately and cannot inherit approval from this note.

## Acceptance Statement

This design note is acceptable only under the following interpretation:

```text
MNEMOS may expose a read-only context graph projection over existing evidence,
working-context, decision, and audit records.

This projection is an observability and product-language layer only.

It does not become a new authority path, does not create new truth, and does
not change retrieval, governance, promotion, or durable memory behavior.
```

## Closeout Labels

```text
CONTEXT_GRAPH_PROJECTION_R0_DESIGN_NOTE_COMPLETE
READ_ONLY_PROJECTION_ONLY
NO_AUTHORITY_SURFACE_CHANGE
NO_RETRIEVAL_RERANKING
NO_GRAPH_DATABASE_DEPENDENCY
NO_RAW_CHAIN_OF_THOUGHT_CAPTURE
NO_CORE_MEMORY_MUTATION
```
