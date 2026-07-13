# Context Graph Projection R1 Lineage And Omission Semantics

Date: 2026-07-13

Status: **Research-only semantics artifact. No implementation authorized.**

## Executive Summary

This artifact defines what Context Graph Projection R1 should do when a
candidate node or edge is missing, incomplete, unauthorized, unsafe to show, or
only supported by prose.

It follows the accepted disclosure model:

```text
If either endpoint or the relationship itself is not authorized,
omit the edge by default.
```

The key operational rule is:

```text
lineage_incomplete is not a substitute for unauthorized disclosure.
```

Authorization failure and lineage incompleteness are different states:

- authorization failure means the caller may not see the node, edge, endpoint,
  type, label, or relationship
- lineage incomplete means the caller may see the visible record, but the
  projection lacks enough explicit refs to prove a candidate relationship

The projection must never use `lineage_incomplete` to hint that a hidden
restricted relationship exists.

## Prior Decisions

This artifact follows:

- `docs/experiments/context_graph_projection_r1_field_validation_results.md`
- `docs/experiments/context_graph_projection_r1_disclosure_model.md`

Accepted lane posture:

```text
RESEARCH_ONLY
IMPLEMENTATION_BLOCKED
DISCLOSURE_RULES_REQUIRED_BEFORE_CODE
NO_GRAPH_STORAGE
NO_GRAPHRAG
NO_AUTHORITY_SURFACE_CHANGE
```

## Scope

This model applies to a future narrow JSON-only projection over explicitly
referenced artifact families:

- retained evidence receipts with `citations[].engram_id`
- formal benchmark/evaluation records with `engram_id` and `source_path`
- Session Context Assembler records with `selected_parent_engram_ids` and
  `selected_source_ids`
- structured review packets with `parent_engram_ids` and `parent_source_ids`

Everything else must be omitted, labeled `lineage_incomplete` only when safe,
or deferred.

## Non-Goals

This artifact does not authorize:

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
- inference from prose-only records
- relationship stubs for unauthorized edges

## Core State Model

Every candidate node or edge must resolve to one of these states:

| State | Meaning | Caller-visible? |
|---|---|---|
| `SHOW` | Explicit refs exist and disclosure allows visibility. | Yes |
| `SHOW_REDACTED` | Existence and type are authorized, but content is redacted. | Yes |
| `LINEAGE_INCOMPLETE` | Visibility is authorized, but explicit refs are incomplete. | Yes, only when safe |
| `OMIT_UNAUTHORIZED` | Authorization or disclosure policy blocks visibility. | No |
| `OMIT_INCOMPLETE` | Required refs are missing and safe incomplete labeling is not allowed. | No |
| `OMIT_PROSE_ONLY` | Candidate relation depends on prose interpretation. | No |
| `OMIT_UNKNOWN` | Required fields or policy inputs are unknown. | No |
| `DEFERRED_OUT_OF_SCOPE` | Candidate is outside the narrow R1 projection lane. | No |

Default state:

```text
OMIT_UNKNOWN
```

## 1. When To Omit A Node

Omit a node when any of the following are true:

- node existence is not authorized
- node type disclosure is not authorized and no explicit generic placeholder is
  authorized
- required node identifier is missing
- node identity can only be inferred from prose
- node belongs to an artifact family outside the R1 narrow scope
- node label would reveal a restricted source, decision, evaluation, or handoff
- required disclosure inputs are unknown

Examples:

| Candidate node | Condition | Result |
|---|---|---|
| `source_artifact` | Source path is restricted. | `OMIT_UNAUTHORIZED` |
| `evaluation_result` | Evaluation existence is visible but status is restricted. | `SHOW_REDACTED` only if type/status disclosure rules allow it |
| `handoff_package` | Handoff note is prose-only with no artifact refs. | `OMIT_PROSE_ONLY` |
| `context_package` | Package has no stable package key or digest. | `LINEAGE_INCOMPLETE` or `OMIT_INCOMPLETE`, depending on safe visibility |

## 2. When To Omit An Edge

Omit an edge when any of the following are true:

- either endpoint is omitted
- either endpoint is unauthorized
- relationship existence is not authorized
- relationship label is not authorized and generic edge disclosure is not
  explicitly allowed
- relationship direction is not authorized
- required edge refs are missing
- edge depends on semantic similarity, text interpretation, or LLM judgment
- edge would reveal a restricted source, decision, evaluation, package, handoff,
  or audit event
- edge belongs to a deferred lane such as graph-assisted retrieval or
  speculative explanation
- required disclosure inputs are unknown

Default:

```text
OMIT_EDGE
```

No unauthorized edge should be replaced with a visible relationship stub.

## 3. When To Label `lineage_incomplete`

Use `lineage_incomplete` only when all of the following are true:

- the visible node or aggregate projection is authorized
- the relevant artifact family is in R1 scope
- the record has some explicit refs, but not enough to prove the full candidate
  relationship
- the missing relationship is not itself restricted
- the label does not reveal hidden endpoints
- the label does not reveal a hidden relationship type
- graph-specific incomplete-lineage labeling is explicitly allowed for that
  artifact family

Allowed examples:

| Case | Reason |
|---|---|
| A visible context package has parent source IDs but lacks a package digest. | The package may be shown, but integrity lineage is incomplete. |
| A visible retrieval receipt has source labels but lacks Engram IDs. | The receipt may be shown, but evidence-to-Engram linkage is incomplete. |
| A visible evaluation result has claim status but lacks source refs. | The result may be shown, but evidence lineage is incomplete. |

`lineage_incomplete` describes missing proof, not hidden authorization.

## 4. When `lineage_incomplete` Must Be Suppressed

Suppress `lineage_incomplete` when the label would reveal any of the following:

- a restricted endpoint exists
- a restricted relationship exists
- a restricted source was used
- a restricted decision depends on visible evidence
- a visible decision depends on hidden evidence
- a hidden handoff/package included visible evidence
- a restricted evaluation outcome exists
- the count of missing relationships reveals protected structure

In these cases, use omission without a visible incomplete-lineage label.

Default:

```text
authorization failure -> omit
lineage incomplete -> label only when safe
```

## 5. How Omitted Edges Are Counted

Omitted edges may be counted only in content-free aggregate form and only when
counts do not leak restricted structure.

Recommended counters:

| Counter | Meaning |
|---|---|
| `omitted_edge_count` | Total omitted edges visible to the validation artifact. |
| `omitted_node_count` | Total omitted nodes visible to the validation artifact. |
| `omitted_unauthorized_edge_count` | Edges omitted because endpoint or relationship authorization failed. |
| `omitted_incomplete_edge_count` | Edges omitted because explicit refs were incomplete. |
| `omitted_prose_only_edge_count` | Edges omitted because they required prose interpretation. |
| `omitted_unknown_policy_edge_count` | Edges omitted because disclosure policy inputs were unknown. |
| `lineage_incomplete_node_count` | Visible nodes labeled incomplete. |
| `lineage_incomplete_edge_count` | Visible edges labeled incomplete, if ever allowed. |
| `suppressed_counter_count` | Counters suppressed because counts would leak restricted structure. |

Counters must be suppressed when:

- the count reveals a hidden endpoint class
- the count reveals that a hidden decision or source exists
- the count reveals relationship degree for restricted records
- the count reveals handoff or package composition
- the caller is not authorized to know omissions occurred

## 6. How Omissions Appear In Validation Artifacts

Validation artifacts may record omission behavior, but must remain content-free.

Recommended shape:

```json
{
  "projection_validation": {
    "visible_node_count": 4,
    "visible_edge_count": 3,
    "omission_summary": {
      "omitted_edge_count": 2,
      "omitted_unauthorized_edge_count": 1,
      "omitted_incomplete_edge_count": 1,
      "omitted_prose_only_edge_count": 0,
      "suppressed_counter_count": 0
    },
    "lineage_summary": {
      "lineage_incomplete_node_count": 1,
      "lineage_incomplete_edge_count": 0
    },
    "content_free": true
  }
}
```

Validation artifacts must not include:

- raw source text
- raw prompts
- hidden reasoning
- unauthorized endpoint IDs
- unauthorized edge labels
- unrestricted ledger payloads
- secrets or credentials

## 7. How Omission Differs From Authorization Denial

Omission is a projection outcome. Authorization denial is one possible reason
for omission.

| Condition | Meaning | Visible label? |
|---|---|---|
| Authorization denial | Caller may not know the node, edge, type, label, or relationship exists. | No |
| Missing refs | Projection cannot prove the relation from explicit fields. | Maybe, if safe |
| Prose-only relation | Relation would require interpretation. | No |
| Unknown policy input | Projection cannot determine safe visibility. | No |
| Deferred lane | Relation is outside R1 scope. | No |

Do not expose `authorization_denied` labels in the caller-visible graph unless
a separate disclosure rule explicitly allows denial disclosure.

## 8. How Prose-Only Or Inference-Required Records Are Handled

Prose-only records are excluded from edge construction.

Examples:

- markdown handoff notes with no explicit artifact refs
- evaluation prose that says a claim was supported but does not list evidence
  IDs
- narrative descriptions of source influence
- free-form explanations of why a result was chosen

Allowed handling:

- omit candidate edges
- count as `omitted_prose_only_edge_count` in a content-free validation artifact
  if counters are authorized
- keep the prose artifact as a node only if node visibility and type disclosure
  are authorized

Prohibited handling:

- infer edges from prose
- infer evidence use from semantic similarity
- infer relationship type from wording
- infer authority from narrative confidence

## 9. How Package-Level Lineage Counters Map To Graph Edges

Session Context Assembler artifacts expose package-level counters such as:

- `selected_parent_engram_ids`
- `selected_source_ids`
- `source_lineage_loss_count`
- `decision_lineage_loss_count`
- `provenance_loss_count`
- `artifact_local_lineage_complete`
- `missing_required_artifact_ids`
- `silent_required_artifact_omission`
- `selection_abstention_reason`

Mapping rules:

| Package field | Graph meaning | Edge action |
|---|---|---|
| `selected_parent_engram_ids` | Explicit package-to-Engram candidates exist. | Build edges only for authorized IDs. |
| `selected_source_ids` | Explicit package-to-source candidates exist. | Build edges only for authorized IDs. |
| `artifact_local_lineage_complete=true` | Package-level lineage is complete. | Eligible for complete package lineage label. |
| `artifact_local_lineage_complete=false` | Package-level lineage is incomplete. | Label package `lineage_incomplete` only when safe. |
| `source_lineage_loss_count>0` | Source lineage loss occurred. | Do not invent missing source edges. |
| `decision_lineage_loss_count>0` | Decision lineage loss occurred. | Do not invent missing decision edges. |
| `provenance_loss_count>0` | Provenance loss occurred. | Do not claim full lineage. |
| `missing_required_artifact_ids` non-empty | Required artifacts were missing. | Omit missing edges; count only if safe. |
| `silent_required_artifact_omission=true` | Required omission occurred silently in source artifact. | Treat as validation failure for graph projection. |
| `selection_abstention_reason` present | Package abstained from complete context. | Show abstention only if authorized. |

Package-level counters are not edge lists. They can block or label graph
projection, but they cannot create edges for missing relationships.

## 10. Default Behavior When Required Fields Are Unknown

Unknown required fields default to omission.

| Unknown field | Default result |
|---|---|
| endpoint authorization | Omit node and connected edges. |
| relationship authorization | Omit edge. |
| edge label authorization | Omit edge. |
| node type authorization | Omit node or show generic node only if explicitly authorized. |
| source artifact key | Omit source-to-Engram edge or label visible node `lineage_incomplete` if safe. |
| Engram ID | Omit evidence edge. |
| retrieval result-set identity | Omit retrieval-result node and connected edges. |
| package digest | Omit digest edge; package may be `lineage_incomplete` if safe. |
| handoff artifact refs | Omit handoff inclusion edges. |
| ledger event ID | Omit `audited_by` edge. |

Default:

```text
unknown -> omit
```

## Decision Procedure

For each candidate relationship:

1. Confirm artifact family is in R1 scope.
2. Confirm both endpoint IDs are explicit.
3. Confirm relationship refs are explicit.
4. Confirm endpoint authorization.
5. Confirm relationship authorization.
6. Confirm label authorization.
7. Confirm lineage completeness.
8. If authorization fails, return `OMIT_UNAUTHORIZED`.
9. If refs are prose-only, return `OMIT_PROSE_ONLY`.
10. If refs are incomplete and safe labeling is allowed, return
    `LINEAGE_INCOMPLETE`.
11. If refs are incomplete and safe labeling is not allowed, return
    `OMIT_INCOMPLETE`.
12. If policy inputs are unknown, return `OMIT_UNKNOWN`.
13. Otherwise, return `SHOW`.

This order matters: authorization is evaluated before incomplete lineage is
shown.

## Acceptance Criteria

This semantics model is satisfied only if a future design can prove:

- unauthorized relationships are omitted without stubs
- incomplete lineage is separated from authorization denial
- prose-only records do not create edges
- missing fields do not create edges
- package-level counters do not fabricate edge-level relationships
- omission counters are content-free and suppressible
- unknown fields default to omission
- validation artifacts can record omissions without leaking restricted
  structure

## Next Gate

After this artifact, the next research gate should be a compact preregistration
or test matrix for the narrow JSON-only projection:

- `docs/experiments/context_graph_projection_r1_preregistration.md`

That artifact should combine:

- explicit-ref requirements
- disclosure rules
- lineage and omission semantics
- allowed artifact families
- GO / NARROW_SCOPE / NO_GO criteria

## Acceptance Statement

This artifact is acceptable only under the following interpretation:

```text
lineage_incomplete is not a substitute for unauthorized disclosure.

Authorization failure omits by default.

Prose-only or inference-required relationships do not produce graph edges.
```

## Closeout Labels

```text
CONTEXT_GRAPH_PROJECTION_R1_LINEAGE_OMISSION_SEMANTICS_COMPLETE
RESEARCH_ONLY
LINEAGE_INCOMPLETE_NOT_AUTHORIZATION_BYPASS
OMIT_UNAUTHORIZED_BY_DEFAULT
OMIT_UNKNOWN_BY_DEFAULT
PROSE_ONLY_RECORDS_DO_NOT_CREATE_EDGES
NO_IMPLEMENTATION_AUTHORIZED
NO_GRAPH_STORAGE
NO_GRAPHRAG
NO_AUTHORITY_SURFACE_CHANGE
```
