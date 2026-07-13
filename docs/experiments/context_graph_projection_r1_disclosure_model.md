# Context Graph Projection R1 Disclosure Model

Date: 2026-07-13

Status: **Research-only disclosure model. No implementation authorized.**

## Executive Summary

The R1 field validation result was accepted with `NARROW_SCOPE_RECOMMENDED`.
The next blocker is relationship-edge disclosure.

Context Graph Projection can leak information through relationships even when
node content is redacted. For example, a caller may be allowed to view a
decision record but not the source evidence behind it. Showing an edge from the
decision to a redacted evidence node can still reveal that restricted evidence
exists and influenced the decision.

This model defines when projected nodes and edges may be shown, redacted,
omitted, or labeled `lineage_incomplete` without leaking restricted
relationships.

Default-safe rule:

```text
If either endpoint or the relationship itself is not authorized,
omit the edge by default.

Do not show relationship stubs unless explicitly authorized.
```

## Accepted Prior Decision

```text
CONTEXT_GRAPH_PROJECTION_R1_FIELD_VALIDATION_RESULTS_ACCEPTED
RESEARCH_ONLY
NARROW_SCOPE_RECOMMENDED
IMPLEMENTATION_BLOCKED
DISCLOSURE_RULES_REQUIRED_BEFORE_CODE
NO_GRAPH_STORAGE
NO_GRAPHRAG
NO_AUTHORITY_SURFACE_CHANGE
```

## Scope

This model applies only to a future narrow, JSON-only, read-only projection over
artifact families with explicit refs:

- retained evidence receipts with `citations[].engram_id`
- formal benchmark/evaluation records with `engram_id` and `source_path`
- Session Context Assembler records with `selected_parent_engram_ids` and
  `selected_source_ids`
- structured review packets with `parent_engram_ids` and `parent_source_ids`

All other artifact families remain omitted, labeled `lineage_incomplete`, or
deferred.

## Non-Goals

This model does not authorize:

- graph code
- graph storage
- graph database adoption
- GraphRAG
- retrieval changes
- governance changes
- promotion changes
- context assembly changes
- Engram schema changes
- authority changes
- relationship stubs by default
- inferred or speculative edges

## Disclosure Principles

1. **Relationship visibility is separate from node visibility.**
   - Authorization to view two nodes does not automatically authorize viewing
     the relationship between them.

2. **Endpoint redaction is not enough.**
   - A redacted node can still leak information through its type, position,
     edge labels, degree, or neighboring nodes.

3. **Edges are disclosures.**
   - Every visible edge reveals at least one relationship claim, even if the
     edge is non-authoritative.

4. **Omission is safer than stubbing.**
   - A visible placeholder can reveal that restricted evidence, a restricted
     decision, or a restricted handoff exists.

5. **No edge may imply authority.**
   - A visible edge explains a recorded relationship only. It does not increase
     truth, ranking, promotion, governance, or decision authority.

6. **Missing or unauthorized lineage must not be fabricated.**
   - If lineage cannot be safely shown, omit the edge or label the visible
     projection aggregate, not the hidden relationship.

## Disclosure Decision Inputs

A future projection must evaluate these fields before showing any node or edge:

| Input | Meaning |
|---|---|
| `caller_scope` | Tenant, session, project, role, purpose, and entitlement context. |
| `node_authorized` | Whether the caller may know this node exists. |
| `node_content_authorized` | Whether the caller may see node content or only safe metadata. |
| `node_type_authorized` | Whether the caller may know the node type. |
| `edge_authorized` | Whether the caller may know this relationship exists. |
| `edge_label_authorized` | Whether the caller may see the relationship label. |
| `relationship_target_authorized` | Whether relationship direction and endpoint identity may be shown. |
| `lineage_complete` | Whether the relationship has complete explicit refs. |
| `redaction_state` | Whether content, labels, or identifiers require redaction. |
| `authority_class` | Source-grounded, retrieval observation, decision record, audit reference, or projection-only. |

If any required input is unknown, the default result is omission.

## Node Disclosure Rules

| Case | Rule | Result |
|---|---|---|
| Caller may view node and content | Show node with allowed fields. | `SHOW_NODE` |
| Caller may view node existence but not content | Show node only if node type disclosure is also authorized. | `SHOW_REDACTED_NODE` |
| Caller may view node existence but not node type | Show only a generic authorized placeholder if explicitly allowed. | `SHOW_GENERIC_NODE` |
| Caller may not know node exists | Omit node. | `OMIT_NODE` |
| Node exists but lineage is incomplete | Show only if otherwise authorized and label safely. | `SHOW_NODE_LINEAGE_INCOMPLETE` |

Default: `OMIT_NODE`.

## Edge Disclosure Rules

| Endpoint A | Endpoint B | Relationship | Default result |
|---|---|---|---|
| Authorized | Authorized | Authorized | Show edge with allowed label. |
| Authorized | Authorized | Label restricted | Show unlabeled edge only if explicitly authorized; otherwise omit. |
| Authorized | Restricted | Any | Omit edge. |
| Restricted | Authorized | Any | Omit edge. |
| Restricted | Restricted | Any | Omit edge. |
| Authorized | Redacted | Any | Omit edge unless relationship and redacted endpoint disclosure are explicitly authorized. |
| Redacted | Authorized | Any | Omit edge unless relationship and redacted endpoint disclosure are explicitly authorized. |
| Authorized | Authorized | Lineage incomplete | Omit edge or show aggregate `lineage_incomplete` count; do not invent edge. |

Default: `OMIT_EDGE`.

## Required Questions

### 1. Can an edge be visible if one endpoint is restricted?

Default answer:

```text
No.
```

If either endpoint is restricted, omit the edge by default. A future projection
may expose a redacted relationship only if a separate disclosure rule explicitly
authorizes:

- endpoint existence disclosure
- endpoint type disclosure
- relationship existence disclosure
- relationship label disclosure
- direction disclosure

Without all five, the edge is omitted.

### 2. Can a redacted node still reveal its type?

Default answer:

```text
Only if node type disclosure is explicitly authorized.
```

A label such as `source_artifact`, `decision_record`, or `evaluation_result`
can leak sensitive context. If the caller may know that an object exists but
not what kind of object it is, the projection must either omit the node or show
a generic authorized placeholder.

Allowed generic placeholder:

```json
{
  "node_id": "redacted:local-1",
  "node_type": "redacted",
  "redaction_state": "existence_only",
  "authority_effect": "none"
}
```

This placeholder is allowed only when existence disclosure itself is authorized.

### 3. Can relationship labels reveal sensitive source/decision links?

Default answer:

```text
Yes. Relationship labels are disclosures.
```

Labels such as `used_evidence`, `justified_by`, `evaluated_as`, `blocked_by`,
or `included_in_handoff` can reveal sensitive relationships even when endpoint
content is hidden.

If the edge label is not authorized, omit the edge unless an explicit rule
allows a generic relationship label.

Allowed generic edge label:

```text
related_record
```

This label is allowed only when relationship existence is authorized but the
specific relation type is not.

### 4. What happens when caller can see a decision but not the evidence behind it?

Default answer:

```text
Show the decision node only. Omit evidence nodes and evidence edges.
```

The projection may optionally include an aggregate omission count if explicitly
authorized:

```json
{
  "omissions": {
    "hidden_evidence_edge_count": 3,
    "reason": "endpoint_or_relationship_not_authorized"
  }
}
```

This count must be suppressed if the count itself would reveal restricted
relationship structure.

### 5. What happens when caller can see evidence but not the downstream decision?

Default answer:

```text
Show the evidence node only. Omit decision nodes and decision edges.
```

Do not show a stub that implies a hidden decision exists unless downstream
decision existence disclosure is explicitly authorized.

### 6. How are omitted edges counted?

Omitted edges should be counted only in aggregate and only when aggregate
counts are authorized.

Recommended counters:

| Counter | Meaning |
|---|---|
| `omitted_edge_count` | Total edges omitted from the caller-visible projection. |
| `omitted_restricted_endpoint_edge_count` | Edges omitted because one endpoint was restricted. |
| `omitted_relationship_not_authorized_count` | Edges omitted because the relationship itself was not authorized. |
| `omitted_lineage_incomplete_edge_count` | Edges omitted because explicit refs were incomplete. |
| `omitted_label_restricted_edge_count` | Edges omitted because the relationship label was not authorized. |

Counters must be suppressed when counts would reveal restricted structure.

### 7. What audit event records projection omission or redaction?

A future implementation must emit or reference a content-free projection audit
event before any caller-visible graph export is considered.

Recommended audit event shape:

```json
{
  "event_type": "context_graph_projection_disclosure_decision",
  "projection_id": "projection-local-id",
  "caller_scope_hash": "sha256:...",
  "policy_profile_id": "projection-policy-v1",
  "visible_node_count": 0,
  "visible_edge_count": 0,
  "omitted_node_count": 0,
  "omitted_edge_count": 0,
  "redacted_node_count": 0,
  "redacted_edge_count": 0,
  "lineage_incomplete_count": 0,
  "reason_codes": [
    "endpoint_not_authorized",
    "relationship_not_authorized"
  ],
  "content_free": true
}
```

The audit event must not include raw source text, raw prompts, hidden reasoning,
unrestricted ledger payloads, secrets, credentials, or unauthorized endpoint
identifiers.

### 8. What is the default-safe behavior?

Default-safe behavior is:

```text
Omit by default.
```

More specifically:

- unknown endpoint authorization means omit node and connected edges
- unknown relationship authorization means omit edge
- unknown label authorization means omit edge
- incomplete lineage means omit edge unless a safe `lineage_incomplete` rule
  exists
- restricted endpoint means omit connected edges
- redacted endpoint means omit connected edges unless explicitly authorized
- no relationship stubs unless explicitly authorized
- no aggregate omission counters unless explicitly authorized

## Projection Disclosure Outcomes

| Outcome | Meaning |
|---|---|
| `SHOW_NODE` | Node is visible with authorized fields. |
| `SHOW_REDACTED_NODE` | Node existence and type are visible, content is hidden. |
| `SHOW_GENERIC_NODE` | Node existence is visible, type/content are hidden. |
| `OMIT_NODE` | Node is absent from caller-visible projection. |
| `SHOW_EDGE` | Edge is visible with authorized label. |
| `SHOW_GENERIC_EDGE` | Relationship existence is visible, label is generic. |
| `OMIT_EDGE` | Edge is absent from caller-visible projection. |
| `LINEAGE_INCOMPLETE` | Visible record is labeled incomplete only when safe. |
| `SUPPRESS_COUNTERS` | Omission/redaction counts are hidden because counts leak structure. |

## Edge-Level Decision Procedure

For every candidate edge:

1. Verify both endpoint refs are explicit.
2. Verify endpoint A existence is authorized.
3. Verify endpoint B existence is authorized.
4. Verify the relationship existence is authorized.
5. Verify the edge label is authorized.
6. Verify edge direction disclosure is authorized.
7. Verify lineage is complete.
8. Apply redaction rules to endpoint metadata.
9. Emit `SHOW_EDGE`, `SHOW_GENERIC_EDGE`, `OMIT_EDGE`, or
   `LINEAGE_INCOMPLETE`.
10. Record a content-free audit summary.

Any failed or unknown check defaults to `OMIT_EDGE`, except incomplete lineage,
which may produce `LINEAGE_INCOMPLETE` only under explicit safe-label rules.

## Artifact Family Rules

### Retained evidence receipts

Allowed:

- show receipt-local evidence edges only when citation Engram IDs, source
  labels, and retrieval metadata are authorized for the caller

Default:

- omit citation edges if cited source identity is restricted

### Formal benchmark/evaluation records

Allowed:

- show evaluation-to-evidence edges only when query ID, result status,
  `engram_id`, and `source_path` are authorized

Default:

- omit evidence edges if source path or evaluation status is restricted

### Session Context Assembler records

Allowed:

- show package-to-parent edges when `selected_parent_engram_ids`,
  `selected_source_ids`, and package disclosure labels are authorized

Default:

- omit package edges when synthetic context labels, source IDs, or parent IDs
  are not authorized

### Structured review packets

Allowed:

- show packet-to-parent edges when `parent_engram_ids`, `parent_source_ids`,
  and non-authoritative labels are authorized

Default:

- omit edges for prose-only handoff notes or unstructured references

## Relationship Label Sensitivity

| Label family | Sensitivity | Default |
|---|---|---|
| `derived_from_source_artifact` | Reveals source lineage | Omit unless source lineage disclosure is authorized |
| `includes_retrieved_engram` | Reveals retrieved evidence | Omit unless retrieval/evidence disclosure is authorized |
| `used_evidence` | Reveals influence/support relationship | Omit unless evidence-use disclosure is authorized |
| `evaluated_as` | Reveals claim or outcome status | Omit unless evaluation disclosure is authorized |
| `justified_by` | Reveals support relationship | Omit unless justification disclosure is authorized |
| `packages` | Reveals context package composition | Omit unless package inclusion disclosure is authorized |
| `includes_handoff_artifact` | Reveals handoff composition | Omit unless handoff inclusion disclosure is authorized |
| `audited_by` | Reveals operational event existence | Omit unless audit reference disclosure is authorized |
| `verified_by_digest` | Reveals artifact identity/integrity relationship | Omit unless digest disclosure is authorized |

## Interaction With `lineage_incomplete`

`lineage_incomplete` is not a workaround for unauthorized disclosure.

Use `lineage_incomplete` only when:

- the caller is authorized to see the node or aggregate projection
- lineage fields are missing or incomplete
- the missing relationship itself is not sensitive
- the label does not reveal hidden endpoints

Do not use `lineage_incomplete` when:

- either endpoint is restricted
- the relationship existence is restricted
- the missing edge count would leak protected structure
- the label would reveal a hidden source, decision, evaluation, or handoff

## Acceptance Criteria For This Model

A future projection may move past the disclosure research blocker only if:

- edge visibility rules are implemented as explicit policy checks in a later
  authorized design
- both endpoint authorization and relationship authorization are required
- stubs are disabled by default
- omission counters are policy-controlled
- `lineage_incomplete` is defined separately from authorization failure
- content-free audit summaries are required
- no visible edge can reveal unauthorized source, decision, evaluation, or
  handoff relationships

## Next Gate

The next research artifact should define graph-specific lineage and omission
semantics:

- `docs/experiments/context_graph_projection_r1_lineage_omission_semantics.md`

That artifact should define:

- when to omit a node
- when to omit an edge
- when to label `lineage_incomplete`
- how omitted edges are counted
- how omissions appear in validation artifacts
- how omission semantics differ from authorization denial

## Acceptance Statement

This disclosure model is acceptable only under the following interpretation:

```text
Relationship visibility is an explicit disclosure decision.

If either endpoint or the relationship itself is not authorized,
omit the edge by default.

Do not show relationship stubs unless explicitly authorized.
```

## Closeout Labels

```text
CONTEXT_GRAPH_PROJECTION_R1_DISCLOSURE_MODEL_COMPLETE
RESEARCH_ONLY
DISCLOSURE_RULES_DEFINED_BEFORE_CODE
OMIT_EDGE_BY_DEFAULT
NO_RELATIONSHIP_STUBS_BY_DEFAULT
NO_IMPLEMENTATION_AUTHORIZED
NO_GRAPH_STORAGE
NO_GRAPHRAG
NO_AUTHORITY_SURFACE_CHANGE
```
