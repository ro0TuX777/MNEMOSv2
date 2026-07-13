# Context Graph Projection R1 Preregistration

Date: 2026-07-13

Status: Research-only preregistration. No implementation authorized.

This preregistration defines the testable standard a future JSON-only,
read-only Context Graph Projection would have to satisfy before any
implementation discussion. It consolidates the accepted R1 research artifacts
into a single pass/fail framework.

R1 does not authorize code, graph storage, GraphRAG, retrieval changes,
governance changes, promotion changes, context assembly changes, Engram schema
changes, or authority-surface changes.

## Core Pass Condition

A future implementation passes only if it projects explicit relationships
without inference, without unauthorized relationship leakage, and without
changing retrieval, governance, promotion, context assembly, Engram schema, or
authority behavior.

If that cannot be proven from existing records, the projection must be narrowed
or rejected.

## Accepted Inputs

This preregistration is grounded in the accepted research artifacts:

- `context_graph_projection_r0_design_note.md`
- `context_graph_projection_r1_use_cases.md`
- `context_graph_projection_r1_record_fidelity_audit.md`
- `context_graph_projection_r1_field_validation_plan.md`
- `context_graph_projection_r1_field_validation_results.md`
- `context_graph_projection_r1_disclosure_model.md`
- `context_graph_projection_r1_lineage_omission_semantics.md`

The anchor use case remains evidence-to-decision traceability.

## Allowed Artifact Families

Only artifact families with explicit, inspectable lineage fields are allowed.

### READY_FOR_NARROW_SCOPE_VALIDATION

- Retained evidence receipts with `citations[].engram_id`.
- Formal benchmark or evaluation records with `engram_id` and `source_path`.
- Session Context Assembler records with `selected_parent_engram_ids` and
  `selected_source_ids`.
- Structured review packets with `parent_engram_ids` and `parent_source_ids`.

### CONDITIONAL

- CognitiveCycleRecord outputs, only when they contain explicit evidence refs
  rather than prose-only rationale.
- Handoff packages, only when they contain explicit artifact refs, parent source
  IDs, parent Engram IDs, package IDs, and digest or verification fields.
- Forensic ledger events, only when they correlate to explicit artifact IDs or
  package IDs without guessing.
- Digest or verification artifacts, only when they bind directly to a projected
  package, source, or evaluation artifact.

### EXCLUDED_FROM_R1

- Prose-only notes, summaries, or rationale fields.
- Inferred conceptual relationships.
- LLM-generated relationship judgments.
- Pattern-promotion records that require authority interpretation.
- Any artifact family requiring new storage semantics to become projectable.

## Allowed Node Types

Allowed node types are projection labels over existing records only. They do
not create new memory authorities.

| Node type | Status | Required evidence |
| --- | --- | --- |
| `source_artifact` | Allowed | Stable source path, source ID, digest, URI, or equivalent explicit key. |
| `source_engram` | Allowed | Explicit Engram ID and source linkage. |
| `retrieval_result_set` | Conditional | Explicit retained search response, evidence receipt, or result-set identity. |
| `evaluation_result` | Allowed | Evaluation artifact with explicit `engram_id` and `source_path` refs. |
| `decision_record` | Conditional | Audit-safe decision or cognitive-cycle record with explicit evidence refs. |
| `context_package` | Allowed | Session Context Assembler package with explicit parent/source fields. |
| `handoff_package` | Conditional | Structured handoff or review packet with explicit parent artifact refs. |
| `ledger_event` | Conditional | Ledger event explicitly correlated to a projected artifact or package. |
| `digest_record` | Conditional | Digest or verification artifact explicitly bound to a projected record. |

Nodes must be omitted when their identity, lineage, or authorization state is
unknown.

## Allowed Edge Types

Edges are allowed only when both endpoints and the relationship are explicit,
authorized, and within scope.

| Edge type | Meaning | Required explicit refs |
| --- | --- | --- |
| `materialized_as` | Source artifact materialized into an Engram. | Source artifact key and Engram ID. |
| `returned_in` | Engram appeared in a retained retrieval result set. | Result-set identity and cited Engram ID. |
| `cited_by_evaluation` | Evaluation result cites an Engram or source. | Evaluation ID plus `engram_id` or `source_path`. |
| `included_in_context_package` | Engram or source included in context package. | Package ID plus selected parent/source IDs. |
| `included_in_handoff_package` | Engram or source included in handoff/review packet. | Packet ID plus parent Engram/source IDs. |
| `verified_by_digest` | Package or artifact bound to verification material. | Digest/verification artifact ID plus package or artifact ID. |
| `correlated_with_ledger_event` | Artifact or package has a matching ledger event. | Ledger event ID plus explicit artifact/package correlation key. |

Edges are not allowed when they depend on semantic similarity, prose
interpretation, temporal proximity, score thresholds alone, or model judgment.

## Explicit-Ref Requirements

A future projection must prove every emitted edge from fields already present
in existing records.

Minimum acceptable refs:

- Source identity: stable path, source ID, URI, digest, or artifact key.
- Engram identity: explicit Engram ID.
- Retrieval identity: retained receipt, search response ID, package-local result
  set ID, or deterministic artifact identity.
- Evaluation identity: evaluation artifact ID or path plus explicit evidence
  refs.
- Decision identity: decision or cognitive-cycle record ID plus explicit
  evidence refs.
- Package identity: context package, handoff package, or review packet ID.
- Verification identity: digest, verification artifact, or ledger event ID.

Absent refs cannot be reconstructed by inference. Prose-only references do not
qualify as explicit refs.

## Disclosure Rules

The disclosure model is mandatory for every future projection.

- Endpoint authorization is required before an edge can be shown.
- Relationship authorization is required before an edge can be shown.
- Label authorization is required before an edge label can be shown.
- If either endpoint or the relationship itself is not authorized, omit the
  edge by default.
- Do not show relationship stubs unless explicitly authorized.
- Redacted nodes must not reveal sensitive type, lineage, or relationship
  information unless that metadata is separately authorized.
- Omission counts must be content-free and suppressible when counts would leak
  restricted relationships.

Disclosure is evaluated before lineage-incomplete labeling.

## Lineage And Omission Semantics

Lineage handling must preserve the distinction between missing lineage and
unauthorized lineage.

- Authorization failure means omit by default.
- Incomplete lineage may be labeled only when the label itself is safe.
- `lineage_incomplete` is not a substitute for unauthorized disclosure.
- Missing fields produce omission, not inferred edges.
- Prose-only or inference-required records do not create edges.
- Unknown authorization, unknown identity, or unknown relationship state omits
  by default.
- Package-level lineage counters may explain incomplete local lineage but cannot
  fabricate missing graph edges.
- Omission counters must be content-free and must not reveal hidden endpoints,
  relationship labels, source categories, or decision categories.

## Prohibited Behaviors

A future projection fails preregistration if it requires or performs any of the
following:

- Creating graph-derived authority.
- Adding a graph database dependency.
- Adding GraphRAG.
- Changing retrieval ranking, filtering, reranking, or result construction.
- Changing governance, promotion, contradiction handling, or authority logic.
- Changing Engram schema or context assembly behavior.
- Treating projected graph edges as memory records.
- Inferring edges from prose, semantic similarity, temporal proximity, or model
  judgment.
- Showing relationship stubs by default.
- Using `lineage_incomplete` to reveal unauthorized relationships.
- Capturing raw chain-of-thought, hidden reasoning, or unverifiable internal
  traces.

## Minimal Candidate Path

The smallest viable future JSON-only projection remains:

```text
source_artifact
-> source_engram
-> retrieval_result_set
-> evaluation_result or decision_record
-> context_package or handoff_package
```

This path is viable only when every segment is backed by explicit refs and
passes disclosure checks.

If one segment is missing, unauthorized, or inference-required, that segment is
omitted. The projection may label lineage incomplete only when the label itself
does not leak restricted relationship information.

## GO / NARROW_SCOPE / NO_GO Criteria

### GO

GO is available only if all core checks pass:

- The minimal candidate path can be produced from explicit refs.
- Every emitted node has stable identity.
- Every emitted edge has explicit endpoint refs and relationship refs.
- Endpoint, relationship, and label disclosure checks pass.
- Omitted edges do not leak through stubs, labels, or unsafe counters.
- No inference, semantic guessing, LLM judgment, or prose interpretation is
  needed.
- No retrieval, governance, promotion, context assembly, Engram schema, or
  authority behavior changes are required.

### NARROW_SCOPE

NARROW_SCOPE is required when only a subset of the path can be safely projected.

Examples:

- Evidence receipts and evaluation records pass, but decision records are
  prose-only.
- Context packages expose explicit parent refs, but handoff artifacts do not.
- Ledger events exist but cannot be correlated without guessing.
- Omission counts are safe only at package level, not edge-label level.

In this case, the future projection must emit only the passing subset and mark
or omit incomplete lineage according to the omission rules.

### NO_GO

NO_GO is required if any of the following are true:

- The minimal path requires inference.
- Relationship-edge disclosure rules cannot be enforced.
- Unauthorized relationship leakage is possible through edges, labels, node
  types, stubs, or counters.
- Existing records lack stable explicit refs for the proposed projection scope.
- The implementation would require graph storage, GraphRAG, retrieval changes,
  governance changes, promotion changes, context assembly changes, Engram schema
  changes, or authority behavior changes.

## Validation Matrix

| Check ID | Requirement | Passing evidence | Failure mode |
| --- | --- | --- | --- |
| CGP-R1-001 | Artifact family is allowlisted. | Artifact belongs to an allowed or conditional family. | Artifact is prose-only, inferred, or outside R1 scope. |
| CGP-R1-002 | Node type is allowlisted. | Node label maps to an existing record family. | Node creates new authority or storage semantics. |
| CGP-R1-003 | Edge type is allowlisted. | Edge label appears in the R1 edge allowlist. | Edge is speculative, semantic, or model-judged. |
| CGP-R1-004 | Source identity is explicit. | Stable source path, source ID, URI, digest, or artifact key. | Source identity requires normalization by guessing. |
| CGP-R1-005 | Engram identity is explicit. | Existing Engram ID is present. | Engram inferred from text or search score alone. |
| CGP-R1-006 | Retrieval identity is explicit. | Retained receipt, response ID, or deterministic result-set artifact. | Result set reconstructed from logs or timing. |
| CGP-R1-007 | Decision/evaluation evidence refs are explicit. | Record contains `engram_id`, `source_path`, or equivalent refs. | Record cites evidence only in prose. |
| CGP-R1-008 | Package refs are explicit. | Package contains selected parent/source IDs. | Package lineage inferred from included text. |
| CGP-R1-009 | Endpoint authorization passes. | Caller may see both endpoints. | One endpoint is restricted or unknown. |
| CGP-R1-010 | Relationship authorization passes. | Caller may see the relationship itself. | Edge would reveal a restricted link. |
| CGP-R1-011 | Label authorization passes. | Caller may see the edge label and node type metadata. | Label reveals restricted source or decision relationship. |
| CGP-R1-012 | Unauthorized edges are omitted. | No stub, placeholder, or unsafe counter appears. | Projection reveals hidden relationship existence. |
| CGP-R1-013 | `lineage_incomplete` is safe. | Label is shown only after authorization checks. | Label substitutes for unauthorized disclosure. |
| CGP-R1-014 | Prose-only records are excluded. | No edge is emitted from prose-only rationale. | Edge inferred from natural-language text. |
| CGP-R1-015 | Package counters do not fabricate edges. | Counters only summarize safe local incompleteness. | Counter creates or implies missing endpoints. |
| CGP-R1-016 | Omission accounting is content-free. | Counts do not reveal hidden type, label, or endpoint details. | Count leaks restricted relationship structure. |
| CGP-R1-017 | Output is deterministic. | Same records produce same JSON projection. | Output depends on model judgment or runtime interpretation. |
| CGP-R1-018 | Core behavior is unchanged. | Retrieval, governance, promotion, context assembly, Engram schema, and authority behavior are unchanged. | Projection changes any core memory behavior. |
| CGP-R1-019 | No graph storage is required. | JSON artifact is derived read-only from existing records. | Implementation depends on graph database state. |
| CGP-R1-020 | Audit summary is safe. | Summary records redactions/omissions without restricted details. | Audit event leaks hidden relationships. |

## Expected Future Evidence Package

A future implementation proposal must include a validation package, not merely a
claim of compliance.

Required evidence:

- Input artifact list with paths and artifact families.
- Extracted explicit refs used for each node and edge.
- Omitted candidate relationships with safe omission reasons.
- Disclosure decision summary with content-free counts.
- `lineage_incomplete` labels and proof that each label is authorized.
- Determinism proof using repeated projection over the same records.
- Statement that retrieval, governance, promotion, context assembly, Engram
  schema, and authority behavior were not changed.

## Closeout Labels

```text
CONTEXT_GRAPH_PROJECTION_R1_PREREGISTRATION_COMPLETE
RESEARCH_ONLY
FUTURE_IMPLEMENTATION_STANDARD_ONLY
EXPLICIT_REFS_REQUIRED
NO_INFERENCE_OR_SEMANTIC_GUESSING
NO_UNAUTHORIZED_RELATIONSHIP_LEAKAGE
NO_RETRIEVAL_GOVERNANCE_PROMOTION_CONTEXT_ASSEMBLY_ENGRAM_SCHEMA_OR_AUTHORITY_CHANGE
NO_IMPLEMENTATION_AUTHORIZED
NO_GRAPH_STORAGE
NO_GRAPHRAG
```
