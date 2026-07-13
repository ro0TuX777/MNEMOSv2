# Context Graph Projection R1 Field Validation Plan

Date: 2026-07-13

Status: **Research-only validation plan. No implementation authorized.**

## Executive Summary

This plan defines how to validate whether the `PARTIALLY_READY` nodes and edges
identified in `docs/experiments/context_graph_projection_r1_record_fidelity_audit.md`
are actually backed by stable fields in real MNEMOS runtime records and
persisted artifacts.

The validation is empirical, but still research-only. It asks:

```text
Do real MNEMOS artifacts contain the fields required
to project evidence-to-decision traceability safely?
```

The future JSON-only projection remains conditional. This plan does not
authorize code, graph storage, GraphRAG, retrieval changes, governance changes,
promotion changes, context assembly changes, or authority changes.

## Accepted Prior Decision

The record fidelity audit is accepted under:

```text
CONTEXT_GRAPH_PROJECTION_R1_RECORD_FIDELITY_AUDIT_ACCEPTED
RESEARCH_ONLY
EVIDENCE_TO_DECISION_TRACEABILITY_ANCHORED
NO_IMPLEMENTATION_AUTHORIZED
FIELD_LEVEL_VALIDATION_REQUIRED_BEFORE_CODE
```

## Validation Goal

Verify whether the minimal evidence-to-decision trace path can be produced from
explicit fields already present in MNEMOS records:

```text
source_artifact
  -> source_engram
  -> retrieval_result_set
  -> evaluation_result or decision record
  -> handoff_package or context_package
```

The validation must prove that each link can be backed by explicit refs rather
than inference, semantic guessing, text matching, or new authority semantics.

## Go/No-Go Standard

Future JSON-only implementation may be considered only if:

```text
GO only if the minimal trace path can be produced from explicit refs
without inference, semantic guessing, or new authority semantics.
```

Any missing field must result in one of these outcomes:

- narrow the future projection scope
- mark the record or edge `lineage_incomplete`
- omit the edge from projection
- recommend `NO_GO`

Missing fields must not be filled by LLM judgment, embedding similarity,
free-text interpretation, or graph connectivity.

## Scope

This plan validates field availability and traceability for the top-ranked R1
use case: **evidence-to-decision traceability**.

In scope:

- source artifact keys
- Engram IDs and source refs
- retrieval result-set identity
- explicit evidence refs in cognitive-cycle or evaluation records
- explicit artifact refs in handoff packages
- package digests and lineage fields
- ledger-event correlation
- disclosure behavior for relationship edges
- `lineage_incomplete` labeling rules

Out of scope:

- code implementation
- database schema changes
- graph database evaluation
- GraphRAG
- graph-assisted retrieval
- graph reranking
- new governance behavior
- new promotion behavior
- context assembly changes
- speculative edges
- LLM-inferred relationship recovery

## Validation Inputs

The validation should inspect representative existing artifacts only.

Candidate artifact families:

- indexed Engrams
- search responses or retained evidence receipts
- CoALA `CognitiveCycleRecord` outputs
- derived-fact or benchmark evaluation artifacts
- Session Context Assembler packages
- handoff artifacts, where available
- forensic ledger events
- digest or verification artifacts

The validation should use redacted or synthetic fixtures if production data
would introduce disclosure risk.

## Validation Method

For each check, record:

- artifact family inspected
- sample artifact path or source
- fields required
- fields present
- field stability
- whether refs are explicit
- whether refs require inference
- disclosure concern
- result: `PASS`, `PARTIAL`, or `FAIL`
- impact on future JSON-only projection

Recommended table shape:

```text
check_id | artifact family | required fields | observed fields | explicit refs? | inference needed? | result | projection impact
```

## Field Validation Checks

### FV-001: Source Artifact Key Normalization

**Question**

```text
Can each source artifact be identified by a stable key
that can connect it to one or more source Engrams?
```

Required fields:

- source artifact ID, or deterministic source key
- source URI or source label
- source classification where applicable
- tenant or scope marker where applicable
- Engram ID references

Pass criteria:

- source-to-Engram links can be produced deterministically
- source keys are stable across repeated reads
- no semantic matching is needed

Failure impact:

- `source_artifact` remains `PARTIALLY_READY`
- `derived_from_source_artifact` cannot enter a future minimal projection

### FV-002: Retrieval Result-Set Identity

**Question**

```text
Can a retrieval result set be identified after the retrieval event
without relying on transient response text?
```

Required fields:

- request ID, query ID, audit ID, receipt ID, or deterministic event key
- timestamp
- returned Engram IDs
- rank or score metadata where available
- caller, tenant, or scope marker where applicable

Pass criteria:

- result sets can be reconstructed or referenced from explicit records
- included Engrams are listed by ID
- projection-local IDs can be derived without ambiguity if no durable result-set
  ID exists

Failure impact:

- `retrieval_result_set`, `retrieved`, and `includes_retrieved_engram` cannot
  support the minimal trace path

### FV-003: Explicit Evidence Refs In Cognitive-Cycle Records

**Question**

```text
Do cognitive-cycle records explicitly reference the evidence they used,
or only describe actions and governance metadata?
```

Required fields:

- cycle ID
- action or grounding records
- retrieved Engram IDs or evidence refs
- governance summary refs
- forensic ledger refs

Pass criteria:

- a decision-style cycle record can point to evidence Engrams or result sets by
  explicit ref
- raw reasoning and hidden chain-of-thought are absent or excluded
- no semantic guess is required to infer evidence use

Failure impact:

- `cognitive_cycle_record` may remain usable as metadata
- `used_evidence` must be omitted unless another explicit evidence-ref artifact
  exists

### FV-004: Explicit Evidence Refs In Evaluation Records

**Question**

```text
Do evaluation artifacts explicitly list the source Engrams, source artifacts,
or evidence packets behind accepted or rejected outcomes?
```

Required fields:

- evaluation ID
- result status
- claim ID or claim key where applicable
- source Engram IDs, source artifact IDs, or packet IDs
- timestamp
- scope or tenant marker where applicable

Pass criteria:

- `evaluation_result -> source_engram` or `evaluation_result -> context_package`
  can be projected from explicit refs
- accepted/rejected status is structured or deterministically extractable from
  artifact metadata

Failure impact:

- formal evaluation trace remains `PARTIALLY_READY`
- future minimal projection should prefer cognitive-cycle or package paths

### FV-005: Explicit Artifact Refs In Handoff Packages

**Question**

```text
Do handoff packages explicitly list included evidence, context packages,
evaluation artifacts, or source refs?
```

Required fields:

- handoff ID, path, or digest
- included Engram IDs, source IDs, context package IDs, or evaluation IDs
- timestamp
- author/session/scope metadata where applicable

Pass criteria:

- `handoff_package -> included artifact` edges can be projected from refs
- no prose-only interpretation is needed to infer inclusion

Failure impact:

- handoff inspection remains outside the minimal projection
- handoff edges must be omitted or labeled `lineage_incomplete`

### FV-006: Package Digest Availability

**Question**

```text
Do context packages and handoff packages carry digests
that can verify package identity and lineage integrity?
```

Required fields:

- package ID or artifact key
- digest value
- digest algorithm or verification scope
- parent Engram IDs
- parent source IDs
- lineage completeness marker

Pass criteria:

- package identity can be verified by digest
- package-to-parent links are explicit
- digest scope is clear enough to avoid false confidence

Failure impact:

- `verified_by_digest` is unavailable for affected packages
- package nodes may still be projectable, but with weaker integrity labeling

### FV-007: Ledger-Event Correlation

**Question**

```text
Can projected nodes and edges be correlated to forensic ledger events
without exposing unauthorized details?
```

Required fields:

- ledger event ID
- operation type
- target artifact refs
- timestamp
- tenant or scope marker where applicable
- redaction-safe event summary where applicable

Pass criteria:

- `audited_by` edges can be projected from explicit refs
- ledger refs do not require joining through unrestricted payloads
- disclosure policy can hide or redact event labels when needed

Failure impact:

- audit correlation remains optional
- projected artifacts must not imply audit support where refs are absent

### FV-008: Disclosure Behavior For Relationship Edges

**Question**

```text
Can a relationship edge be safely shown when one or both endpoint nodes
are restricted, redacted, or outside caller scope?
```

Required fields or policy inputs:

- endpoint authorization state
- tenant/scope markers
- redaction labels
- edge type
- edge authority class
- caller purpose or entitlement where applicable

Pass criteria:

- edge visibility rules are explicit
- restricted endpoints do not leak through visible relationship labels
- relationship-only leakage is handled

Failure impact:

- future projection must be scoped to single-user or synthetic validation only
- multi-tenant or compliance-facing graph views must be blocked

### FV-009: `lineage_incomplete` Labeling Rules

**Question**

```text
When required refs are missing, can the projection label or omit
the affected node or edge deterministically?
```

Required rules:

- which missing fields force omission
- which missing fields allow `lineage_incomplete`
- which missing fields force `NO_GO`
- how omitted edges are counted in evaluation artifacts

Pass criteria:

- missing lineage never silently fabricates a relationship
- `lineage_incomplete` has a clear, repeatable meaning
- omissions are auditable

Failure impact:

- future projection is blocked until omission and labeling behavior is specified

## Minimal Trace Validation Matrix

| Trace segment | Required checks | GO condition |
|---|---|---|
| `source_artifact -> source_engram` | FV-001 | Stable source key or source artifact ID connects to Engram ID |
| `source_engram -> retrieval_result_set` | FV-002 | Retrieval result set contains explicit returned Engram IDs |
| `retrieval_result_set -> cognitive_cycle_record` | FV-003 | Cycle record explicitly refs result set or evidence IDs |
| `retrieval_result_set -> evaluation_result` | FV-004 | Evaluation artifact explicitly refs evidence IDs or packet IDs |
| `evaluation_result -> context_package` | FV-004, FV-006 | Evaluation refs package or package refs evaluation/evidence |
| `cognitive_cycle_record -> context_package` | FV-003, FV-006 | Cycle or package contains explicit shared refs |
| `context_package -> handoff_package` | FV-005, FV-006 | Handoff contains explicit package or artifact refs |
| `any projected node -> ledger_event` | FV-007 | Ledger event refs projected artifact without disclosure violation |
| `any projected edge -> caller-visible graph` | FV-008, FV-009 | Edge visibility and incomplete-lineage rules are deterministic |

## Evidence Artifact Template

Field validation should produce a compact evidence artifact with this shape:

```json
{
  "validation_id": "context_graph_projection_r1_field_validation",
  "status": "pass|partial|fail",
  "generated_at": "2026-07-13T00:00:00Z",
  "scope": "research_only",
  "minimal_trace_path": {
    "status": "pass|partial|fail",
    "segments": []
  },
  "checks": [
    {
      "check_id": "FV-001",
      "artifact_family": "engram",
      "result": "pass|partial|fail",
      "required_fields": [],
      "observed_fields": [],
      "explicit_refs": true,
      "inference_needed": false,
      "projection_impact": ""
    }
  ],
  "go_no_go": {
    "recommendation": "GO|NO_GO|NARROW_SCOPE",
    "reason": ""
  },
  "prohibitions_verified": [
    "no_inference",
    "no_semantic_guessing",
    "no_new_authority_semantics",
    "no_retrieval_change",
    "no_governance_change",
    "no_promotion_change",
    "no_context_assembly_change"
  ]
}
```

The artifact must not include raw private source content, raw prompts, hidden
reasoning, unrestricted ledger payloads, secrets, or credentials.

## Go/No-Go Decision Rules

### GO

Recommend `GO` for a future JSON-only implementation only if:

- all minimal trace segments pass
- all projected edges are backed by explicit refs
- no inference or semantic guessing is required
- disclosure behavior is deterministic
- missing lineage behavior is deterministic
- no behavior changes are required in retrieval, governance, promotion, or
  context assembly

### NARROW_SCOPE

Recommend `NARROW_SCOPE` if:

- source-to-Engram and retrieval inclusion are valid
- one downstream record family is valid, such as context packages or evaluation
  artifacts
- handoff or ledger correlation remains partial
- the projection can still answer a narrower evidence traceability question

### NO_GO

Recommend `NO_GO` if:

- retrieval result sets cannot be identified or reconstructed
- explicit evidence refs are absent from all decision/evaluation/package paths
- relationship disclosure rules are unresolved
- lineage gaps would require inference
- any minimal trace edge would require new authority semantics

## Expected Output

The next completed research output after this plan should be a validation
result, not code. Suggested artifact:

- `docs/experiments/context_graph_projection_r1_field_validation_results.md`

That result should answer:

```text
Do real MNEMOS artifacts contain the fields required
to project this graph safely?
```

## Acceptance Statement

This plan is acceptable only under the following interpretation:

```text
Field validation is required before any code.

The future JSON-only projection is allowed to proceed only if the minimal trace
path can be produced from explicit refs without inference, semantic guessing,
or new authority semantics.
```

## Closeout Labels

```text
CONTEXT_GRAPH_PROJECTION_R1_FIELD_VALIDATION_PLAN_COMPLETE
RESEARCH_ONLY
FIELD_LEVEL_VALIDATION_REQUIRED_BEFORE_CODE
EVIDENCE_TO_DECISION_TRACEABILITY_ANCHORED
NO_IMPLEMENTATION_AUTHORIZED
NO_GRAPH_STORAGE
NO_GRAPHRAG
NO_RETRIEVAL_GOVERNANCE_PROMOTION_OR_CONTEXT_ASSEMBLY_CHANGE
```
