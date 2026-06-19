# MNEMOS Context Atlas P0 Specification

Date: 2026-06-19

Status: **Specification only. Implementation blocked until EBIR-R2 protocol and
reviewer corpus are frozen.**

## Executive Boundary

Context Atlas is a read-only observability and namespace facade over existing
MNEMOS artifacts. It is not a retrieval engine, summary system, memory substrate,
promotion path, execution system, policy system, or authority layer.

Approved P0 endpoints, in principle:

- `GET /v1/mnemos/context/tree`
- `GET /v1/mnemos/context/resolve?uri=...`
- `GET /v1/mnemos/context/trajectory/{cycle_id}`

Deferred until after EBIR-R2:

- `POST /v1/mnemos/context/explore`
- MCP exposure
- skill cards as indexed memory
- review-protocol persistence
- agent-facing writes
- automatic session-memory writes
- any execution authority

## P0 Behavior Classes

Context Atlas must distinguish four behaviors.

| Behavior | P0 Status | Meaning |
|---|---|---|
| `catalog` | Approved | Return tenant-scoped categories and counts/minimal hints only. |
| `browse` | Restricted | Enumerate concrete objects only for explicit authorized prefixes. |
| `resolve` | Approved | Return one artifact by URI only if the caller could already access it through existing boundaries. |
| `trajectory` | Approved | Return redacted route/attention/governance/audit trajectory derived from existing cycle and ledger records. |

`tree` defaults to catalog behavior. Concrete object enumeration is never the
default and requires an explicit prefix accepted by the URI allowlist and the
caller authorization context.

## URI Grammar

Atlas URIs are stable view identifiers over existing IDs. They must never become
a second storage identity or replace `Engram.id`, source artifact IDs, cycle IDs,
ledger transaction IDs, EBIR packet IDs, pattern IDs, or governance review IDs.

Grammar:

```text
mnemos://tenants/{tenant_id}/{category}[/{subcategory...}][/{object_id}]
```

Rules:

- Scheme must be exactly `mnemos://`.
- First path segment must be `tenants/{tenant_id}`.
- `{tenant_id}` must be derived from the caller's authorized tenant scope or an
  explicitly authorized admin scope.
- `{category}` must be in the P0 allowlist.
- `{object_id}` must map to an existing artifact ID. Atlas must not mint storage
  IDs.
- URI parsing must be structural, not string-split ad hoc routing.
- Unknown categories, unsupported subcategories, path traversal, empty segments,
  wildcard object IDs, and query-embedded filter directives must fail closed.

P0 category allowlist:

```text
sources
engrams/raw
engrams/source-grounded
summaries
contradictions/open
contradictions/resolved
resolutions/candidate
resolutions/promoted
evidence-packets/ebir
patterns/advisory
patterns/approved
sessions/cycles
```

Deferred categories:

```text
skill-cards
review-protocols
execution
policies/mutations
session-writes
```

## URI Mapping

| Atlas Prefix | Existing Identity | Existing Source |
|---|---|---|
| `sources/{artifact_id}` | `SourceArtifact.artifact_id` or engram lineage `artifact_id` | Memory Over Maps/source metadata |
| `engrams/raw/{engram_id}` | `Engram.id` | Existing engram read path |
| `engrams/source-grounded/{engram_id}` | `Engram.id` with lineage/source fields | Existing engram read path |
| `summaries/{engram_id}` | `Engram.id` where `metadata.is_summary_engram=true` | Phase 9 summary engrams |
| `contradictions/open/{conflict_group_id}` | `GovernanceMeta.conflict_group_id` or contradiction record | Governance contradiction state |
| `contradictions/resolved/{conflict_group_id}` | `ContradictionRecord.conflict_group_id` | Governance contradiction state |
| `resolutions/candidate/{engram_id}` | `Engram.id` or dry-run reconciliation record | Reconciliation artifacts |
| `resolutions/promoted/{engram_id}` | `Engram.id` with `metadata.is_resolution_engram=true` | Resolution Engram read path |
| `evidence-packets/ebir/{packet_id}` | `ReconciliationPacket.packet_id` | EBIR shadow artifacts/results |
| `patterns/advisory/{candidate_id}` | `PatternEngramCandidate.candidate_id` | Pattern candidate store |
| `patterns/approved/{pattern_id}` | `PatternEngram.pattern_id` | Promoted pattern store |
| `sessions/cycles/{cycle_id}` | `CognitiveCycleRecord.cycle_id` | Cognitive cycle ledger/API |

If an artifact cannot be resolved from existing stores, Atlas returns
`404 not_found` rather than synthesizing a placeholder.

## Artifact Types And Authority Classes

Every resolved object must include:

```text
artifact_type
authority_class
source_lineage
governance_state
promotion_status
retrieval_eligibility
human_review_requirement
tenant_scope
created_at
updated_at
redaction_state
```

Authority classes:

| Authority Class | Meaning | Retrieval Authority |
|---|---|---|
| `source_grounded` | Raw or normalized evidence tied to source lineage. | Eligible only through existing retrieval rules. |
| `synthetic_summary` | Hierarchical or derived overview. | Not source truth; summary-isolated. |
| `synthetic_resolution` | Additive contradiction reconciliation. | Only as existing Resolution Engram behavior allows. |
| `shadow_only` | Evaluation or refinement artifact with no authority path. | Never default retrieval authority. |
| `advisory` | Guidance or pattern not authoritative for retrieval/ranking. | Never affects ranking. |
| `approved_advisory` | Human-approved advisory semantic pattern. | Still not retrieval-authoritative unless a separate future gate changes that. |
| `audit_record` | Ledger/cycle trace. | Not answer evidence by itself. |
| `blocked_or_deferred` | Known concept outside P0. | No access except category notice. |

Artifact matrix:

| Artifact Type | Authority Class | Retrieval Eligibility | Human Review Requirement |
|---|---|---|---|
| `source_engram` | `source_grounded` | Existing retrieval rules only | No Atlas-specific review |
| `resource_engram` | `source_grounded` | Existing retrieval rules only | No Atlas-specific review |
| `summary_engram` | `synthetic_summary` | Summary-layer only; excluded from default factoid retrieval | Required for promotion beyond existing summary role |
| `resolution_engram` | `synthetic_resolution` | Existing Phase 10 co-retrieval behavior only | Required for any new promotion semantics |
| `ebir_evidence_packet` | `shadow_only` | Never authoritative | Always required |
| `derived_fact` | `shadow_only` or governed evaluation label | Never default retrieval | Always required outside approved evaluation lane |
| `pattern_candidate` | `advisory` | Never affects ranking | Required |
| `pattern_engram` | `approved_advisory` | Never affects ranking in current system | Already approved, still advisory |
| `cognitive_cycle` | `audit_record` | Not answer evidence | No, but redaction required |
| `forensic_ledger_ref` | `audit_record` | Not answer evidence | No, but redaction required |
| `skill_card` | `blocked_or_deferred` | Deferred | Deferred |
| `review_protocol` | `blocked_or_deferred` | Deferred | Deferred |

## L0/L1/L2 Progressive Disclosure

Atlas may expose a progressive-disclosure contract only by mapping to existing
data. It must not create a second summary system.

| Layer | Name | Purpose | Existing Sources |
|---|---|---|---|
| L0 | Routing Card | Decide whether an artifact is worth opening. | Existing metadata, lineage, governance flags, authority labels, counts. |
| L1 | Evidence Overview | Support navigation and planning. | Existing summary engrams, derived views, evidence bundles, contradiction bundles, cognitive-cycle summaries. |
| L2 | Evidence Payload | Support reasoning and human review. | Existing source engrams, source spans, parent links, EBIR packets, governance metadata, ledger refs. |

Rules:

- L0 and L1 are navigation aids only.
- L0/L1 must include `not_source_truth=true` when synthetic.
- L2 source evidence remains the basis for truth, review, and promotion.
- Atlas must not generate new L1 summaries. It can read existing summaries or
  report `layer_unavailable`.
- L2 payloads must respect the same access and redaction limits as existing
  engram, audit, derived-evaluation, and cognitive-cycle endpoints.

## Synthetic Artifact Labeling

Synthetic and non-source artifacts must be visibly distinct in every `tree`,
`resolve`, and `trajectory` response.

Required labels:

| Artifact | Required Labels |
|---|---|
| Summary Engram | `artifact_type=summary_engram`, `authority_class=synthetic_summary`, `not_source_truth=true`, `summary_isolation=required` |
| Resolution Engram | `artifact_type=resolution_engram`, `authority_class=synthetic_resolution`, `parent_ids`, `resolution_status`, `not_source_truth=true` |
| EBIR Packet | `artifact_type=ebir_evidence_packet`, `authority_class=shadow_only`, `shadow_only=true`, `auto_promoted=false`, `promotable=false` |
| Derived Fact | `artifact_type=derived_fact`, `authority_class=shadow_only`, `default_retrieval_eligible=false`, `display_label=[MNEMOS-DERIVED]` when rendered |
| Pattern Candidate | `artifact_type=pattern_candidate`, `authority_class=advisory`, `authoritative_for_retrieval=false`, `affects_ranking=false`, `mutates_policy=false` |
| Pattern Engram | `artifact_type=pattern_engram`, `authority_class=approved_advisory`, `authoritative_for_retrieval=false`, `affects_ranking=false`, `mutates_policy=false` |

## Tenant Isolation And Authorization

Atlas adds browsing risk because a namespace can reveal existence even when
payloads remain protected. P0 must therefore use metadata minimization by
default.

Authorization rules:

- All endpoints require the same service authorization as current protected API
  routes.
- Atlas authorization is delegated to the existing artifact-specific
  authorization path wherever one exists. Atlas may enforce additional
  restriction or redaction, but it must never grant access that the underlying
  artifact path would deny. Atlas must not maintain an independent allow/deny
  policy engine for artifact access.
- Tenant scope must be derived from authenticated caller context, existing
  request filters, or an explicit admin context.
- Non-admin callers may not request `mnemos://tenants/*` or another tenant ID.
- `tree` returns only category catalog data unless an explicit authorized prefix
  is supplied.
- Concrete object enumeration requires an explicit prefix and per-prefix
  authorization.
- `resolve` may return only artifacts the caller could already access through
  existing MNEMOS authorization boundaries.
- `trajectory` must redact sensitive source metadata according to the caller's
  principal and tenant scope.
- Access auditing may record endpoint, tenant scope, caller class, resolved
  artifact type, status, and latency. It must not record raw URI paths, raw
  source content, query content, L2 payload fields, or evidence spans unless an
  existing compliant audit policy already permits those fields.

Metadata minimization:

- Catalog responses return category names, availability, object-count hints only
  when safe, and authority-class descriptions.
- `include_counts=false` remains the default. Counts must be suppressed whenever
  they could reveal the existence of protected artifacts, investigations,
  conflicts, or activity within a tenant compartment.
- Catalog responses do not include raw query text, raw source text, evidence
  spans, prompt text, derived fact text, sidecar output text, or full ledger
  payloads.
- Object listings include IDs only when the caller can resolve those IDs.
- For sensitive artifacts, listings may return aggregate counts and
  `redaction_state=metadata_minimized`.

Redaction rules:

- Raw content is omitted unless the existing endpoint would already disclose it.
- `CognitiveCycleRecord.query_or_event` may be truncated or hashed for
  non-admin callers.
- Ledger metadata must drop raw query, prompt, derived fact text, sidecar output,
  canonical payloads, and unrestricted source snippets.
- Source URIs may be reduced to basename, hash, or artifact ID when full paths
  would leak tenant, user, filesystem, or repository information.
- EBIR parent evidence spans follow the same redaction level as source engram
  evidence spans.

## Endpoint Contracts

### `GET /v1/mnemos/context/tree`

Default behavior: tenant-scoped category catalog.

Query parameters:

```text
tenant_id optional; only honored for authorized admin/cross-tenant callers
prefix optional; must be in allowlist
include_counts optional boolean; default false
include_objects optional boolean; default false and requires explicit prefix
limit optional integer; default 50; max 200
```

Default response shape:

```json
{
  "behavior": "catalog",
  "tenant_scope": "acme",
  "prefix": "mnemos://tenants/acme",
  "categories": [
    {
      "uri": "mnemos://tenants/acme/engrams/source-grounded",
      "category": "engrams/source-grounded",
      "authority_classes": ["source_grounded"],
      "enumeration_requires_prefix": true,
      "redaction_state": "metadata_minimized"
    }
  ]
}
```

When `include_objects=true`, response behavior becomes `browse`, and all object
entries must include authority labels and redaction state.

### `GET /v1/mnemos/context/resolve?uri=...`

Resolves one allowlisted Atlas URI into a normalized artifact envelope.

Response shape:

```json
{
  "behavior": "resolve",
  "uri": "mnemos://tenants/acme/resolutions/promoted/resolution_conflict_123",
  "artifact_type": "resolution_engram",
  "authority_class": "synthetic_resolution",
  "promotion_status": "existing_phase10_resolution",
  "retrieval_eligibility": "existing_phase10_co_retrieval_only",
  "human_review_requirement": "required_for_new_promotion_semantics",
  "tenant_scope": "acme",
  "source_lineage": {
    "parent_ids": ["ENG-102", "ENG-208"]
  },
  "governance_state": {
    "conflict_group_id": "conflict:entity:attribute",
    "conflict_status": "winner"
  },
  "layers": {
    "l0": {},
    "l1": {},
    "l2": {}
  },
  "redaction_state": "none"
}
```

`resolve` must not fall back to semantic search if the object ID is unknown.

### `GET /v1/mnemos/context/trajectory/{cycle_id}`

Returns a redacted trajectory derived from `CognitiveCycleRecord` plus forensic
ledger references.

Response shape:

```json
{
  "behavior": "trajectory",
  "cycle_id": "...",
  "tenant_scope": "acme",
  "source": "cognitive_cycle_record",
  "redaction_state": "metadata_minimized",
  "steps": [
    {
      "stage": "query_classification",
      "decision": "CLASS_C",
      "reason": "global synthesis route selected",
      "source_field": "working_memory_snapshot.query_class"
    },
    {
      "stage": "summary_layer",
      "decision": "eligible",
      "reason": "summary layer active for global route",
      "source_field": "attention_decisions.summary_inclusion"
    },
    {
      "stage": "governance",
      "decision": "advisory",
      "reason": "governance read path evaluated",
      "source_field": "governance_evaluations"
    }
  ],
  "forensic_ledger_refs": ["ledger:123"],
  "sensitive_fields_redacted": ["query_or_event", "source_uri"]
}
```

Trajectory must never expose private chain-of-thought. It describes MNEMOS
operations and recorded decisions only.

## CI Gate And Test Matrix

Context Atlas cannot be accepted without automated gates. P0 gates should run
without changing production retrieval defaults.

| Gate ID | Test Area | Required Assertion |
|---|---|---|
| CA-P0-001 | Read-only behavior | `tree`, `resolve`, and `trajectory` cause no engram writes, governance mutations, pattern-store mutations, EBIR writes, or retrieval-index writes. |
| CA-P0-002 | No retrieval delta | Baseline `/v1/mnemos/search` responses are byte-stable or semantically identical before/after Atlas endpoints are called. |
| CA-P0-003 | Authority-label completeness | Every returned object includes `artifact_type`, `authority_class`, `promotion_status`, `retrieval_eligibility`, `human_review_requirement`, and `redaction_state`. |
| CA-P0-004 | Tenant catalog isolation | Non-admin tenant A cannot request tenant B catalog, prefixes, object IDs, counts, or trajectories. |
| CA-P0-005 | Browse prefix authorization | `include_objects=true` fails closed without an explicit allowlisted prefix and caller authorization. |
| CA-P0-006 | URI allowlist | Unknown category, wildcard tenant, path traversal, empty segment, unsupported deferred category, and injected filter syntax all return 400/403 without lookup side effects. |
| CA-P0-007 | Resolve access parity | `resolve` returns an artifact only when the caller could already access that artifact through existing APIs. |
| CA-P0-008 | Summary labeling | Summary engrams always return `synthetic_summary`, `not_source_truth=true`, and summary-isolation metadata. |
| CA-P0-009 | Resolution labeling | Resolution Engrams always return parent IDs, synthetic authority, and no new promotion claim. |
| CA-P0-010 | EBIR shadow invariant | EBIR artifacts always return `shadow_only=true`, `auto_promoted=false`, `promotable=false`; R1 benchmark promotion status remains blocked. |
| CA-P0-011 | Derived-fact isolation | Atlas calls do not increase `query.default_retrieval.derived_fact_count`; derived facts remain absent from default retrieval. |
| CA-P0-012 | Pattern advisory invariant | Pattern candidates and promoted patterns report `authoritative_for_retrieval=false`, `affects_ranking=false`, and `mutates_policy=false`. |
| CA-P0-013 | Trajectory redaction | `trajectory` redacts raw queries, sensitive source URIs, source spans, prompt text, derived fact text, sidecar output, and full ledger payloads when principal lacks access. |
| CA-P0-014 | Ledger reference minimization | Ledger refs are stable IDs or minimized summaries, not raw ledger payload dumps. |
| CA-P0-015 | L0/L1/L2 no-new-summary | Atlas never generates new summaries; unavailable existing layers return `layer_unavailable`. |
| CA-P0-016 | Snapshot contract stability | Golden JSON snapshots for catalog, resolve, and trajectory remain stable across runs except allowed timestamps/counts. |
| CA-P0-017 | Error contract | Authorization, URI parse, not-found, and redaction failures return deterministic error shapes. |
| CA-P0-018 | Performance budget | Catalog and resolve add bounded overhead; trajectory lookup is bounded by ledger/cycle lookup limits. |

Suggested focused test files:

```text
tests/test_context_atlas_uri.py
tests/test_context_atlas_tree.py
tests/test_context_atlas_resolve.py
tests/test_context_atlas_trajectory.py
tests/test_context_atlas_invariants.py
```

Suggested gate command after implementation:

```bash
python -m pytest tests/test_context_atlas_*.py
python tools/run_ebir_refinement_benchmark.py
```

The EBIR benchmark remains part of Atlas acceptance because Atlas must preserve
the blocked authoritative-promotion posture.

## Sequencing

1. Freeze EBIR-R2 protocol and reviewer corpus.
2. Review and approve this Context Atlas spec.
3. Implement URI parser and artifact envelope models with no route exposure.
4. Add tests for URI allowlist, authority labels, and read-only invariants.
5. Add `tree`, then `resolve`, then `trajectory`.
6. Run no-retrieval-delta and EBIR shadow invariant gates.
7. Only after P0 evidence passes, consider operator-facing documentation.

No work on `explore`, MCP, skill cards, review-protocol persistence, or agent
writes begins until EBIR-R2 is complete and separately reviewed.
