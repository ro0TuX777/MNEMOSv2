# Context Graph Projection R1 Research Plan

Date: 2026-07-13

Status: **Research-only artifact. No implementation authorized.**

## Executive Summary

Context Graph Projection R1 is the research phase that follows the accepted R0
design note in `docs/experiments/context_graph_projection_r0_design_note.md`.

R1 does **not** authorize code, storage changes, graph database adoption,
GraphRAG, retrieval reranking, or authority-surface changes. Its purpose is to
determine whether MNEMOS can expose a useful graph-shaped traceability layer
while preserving source-grounded evidence, governance boundaries, and
non-authoritative projection posture.

The core research question is:

```text
Can MNEMOS expose useful graph-shaped traceability
while preserving source-grounded authority boundaries?
```

## Starting Boundary

R1 inherits the R0 posture unchanged:

- read-only
- projection-only
- non-authoritative
- not a graph database initiative
- not GraphRAG
- not a retrieval-path change

R1 is therefore a **research package**, not an implementation package.

## Why R1 Exists

R0 establishes vocabulary and boundary rules. It does not yet prove:

- which graph questions are useful enough to justify implementation
- which node and edge types can be generated today from stable IDs and metadata
- which relationships are audit-safe versus speculative in actual MNEMOS data
- which storage/runtime shapes fit MNEMOS without dependency gravity
- which disclosure and authorization risks a graph-shaped projection creates
- which evaluation criteria should block or allow a narrow implementation

R1 exists to answer those questions before code is proposed.

## R1 Objective

Define a research package that determines whether a narrow, deterministic,
read-only context graph projection is useful and safe enough to justify a later
JSON-only implementation.

## Recommended R1 Outcome

The desired R1 outcome is a **go/no-go recommendation** for a future, narrow,
JSON-only, read-only projection over a trace path such as:

```text
source artifact
  → source engram
  → retrieval event or result set
  → decision or evaluation record
  → handoff package or context package
```

R1 is successful if it can justify or block that future step with evidence.

## Research Principles

- start with user and developer questions, not storage products
- prefer deterministic, ID-backed relationships over inferred relationships
- separate observability value from retrieval value
- separate explanation from authority
- define disclosure and authorization rules before UI claims
- require reproducibility from the same underlying records
- block implementation if stable IDs, lineage, or disclosure rules are missing

## Research Workstreams

R1 consists of seven research workstreams plus one decision memo.

### 1. Use-Case Ranking

**Research question**

```text
What user or developer question should the graph answer
better than current MNEMOS records?
```

Candidate jobs:

1. Evidence traceability
2. Decision lineage
3. Handoff inspection
4. Evaluation memory
5. Project continuity
6. Debugging and observability

Initial recommendation:

- prioritize **evidence-to-decision traceability**
- treat it as the strongest first use case because it fits MNEMOS’s existing
  purpose and avoids new authority claims

Research output:

- `docs/experiments/context_graph_projection_r1_use_cases.md`

Required contents:

- ranked use cases
- primary user questions
- why current records are insufficient or cumbersome
- why the graph shape helps
- why the proposed use case does not require authority expansion

### 2. Record Fidelity Audit

**Research question**

```text
Which nodes and edges can be generated today
from existing stable IDs and metadata?
```

This audit must separate three classes:

- **Ready now** — deterministic, already backed by IDs and metadata
- **Partially ready** — concept exists, but stable IDs, timestamps, digests, or
  parent refs are incomplete
- **Not ready** — would require inference, LLM judgment, or new records

Research output:

- `docs/experiments/context_graph_projection_r1_record_fidelity_audit.md`

Required audit table:

```text
edge_type | backing artifact | required fields | available now? | risk | R1 eligible?
```

Likely R1-eligible edges to test first:

- `derived_from_source_artifact`
- `retrieved`
- `includes_retrieved_engram`
- `packages`
- `includes_handoff_artifact`
- `used_evidence`
- `evaluated_as`
- `justified_by`
- `audited_by`
- `verified_by_digest`

Likely deferred edge families:

- causal influence
- semantic equivalence
- “reasoned because” links
- LLM-inferred relationship edges

### 3. Authority Model

**Research question**

```text
What does a projected node or edge mean,
and what authority does it not gain by being connected?
```

R1 must define a formal authority taxonomy for projected nodes and edges.

Initial authority classes:

- `source_grounded`
- `synthetic_summary`
- `retrieval_observation`
- `decision_record`
- `evaluation_result`
- `audit_reference`
- `non_authoritative_projection`
- `speculative_excluded`

Each projected node and edge should be evaluated for:

- `authority_class`
- `derivation_class`
- `audit_safe`
- `speculative`
- `authority_effect`
- `lineage_complete`

Research output:

- `docs/experiments/context_graph_projection_r1_authority_model.md`

Central rule:

```text
A projected graph edge may explain a relationship.
It may not increase the truth, ranking, promotion, or governance authority
of either endpoint.
```

### 4. Storage And Runtime Options

**Research question**

```text
What implementation shape fits MNEMOS,
without introducing dependency gravity too early?
```

R1 should compare, but not adopt, at least four options.

#### Option A: JSON projection only

Existing records → projection JSON → visualization or inspection

- Pros: lowest risk, easiest to audit, no dependency gravity
- Cons: limited interactive query capability at scale

#### Option B: in-memory graph

Existing records → in-memory graph object → query or export

- Pros: useful for demos and tests, no durable graph store
- Cons: limited scale, runtime memory costs

#### Option C: embedded graph or edge-table layer

Examples: lightweight local graph representation or SQLite-style edge table

- Pros: more queryable without a full graph database
- Cons: creates a second representation that must remain consistent

#### Option D: external graph database

Examples: Neo4j, Memgraph, ArangoDB, FalkorDB

- Pros: powerful query and visualization ecosystem
- Cons: dependency gravity, deployment burden, misleading authority perception

Initial recommendation:

- `R1 = JSON projection first`
- `R2 = optional in-memory or edge-table query layer`
- `R3 = only then consider external graph database`

Research output:

- `docs/experiments/context_graph_projection_r1_storage_options.md`

### 5. Disclosure Model

**Research question**

```text
How can a graph-shaped projection leak relationships,
even when node contents are redacted?
```

R1 must explicitly research:

- tenant isolation
- scope filters
- redacted node labels
- edge disclosure rules
- authorization checks
- relationship leakage
- audit logging
- export controls

This workstream is required because a graph can reveal that source A supported
decision B even if source A itself is not viewable.

Research output:

- `docs/experiments/context_graph_projection_r1_disclosure_model.md`

### 6. Visualization Requirements

**Research question**

```text
What is the minimum useful view,
and how do we avoid producing an unreadable hairball graph?
```

R1 should prioritize path-based views over global graph sprawl.

Minimum useful trace views:

1. source → chunk → retrieval → answer
2. source → decision → evaluation outcome
3. claim → accepted or rejected → supporting evidence
4. handoff package → included context → source artifacts
5. supersession chain
6. contradiction cluster

Preferred UX framing:

- show me the path from this answer back to its evidence
- show me what changed since the earlier decision
- show me which evidence entered this handoff

Research output:

- `docs/experiments/context_graph_projection_r1_visualization_requirements.md`

### 7. Evaluation And Preregistration

**Research question**

```text
What does “good graph” mean
before any implementation is approved?
```

R1 should define evaluation criteria for projection fidelity and safety, not
answer-quality gains.

Candidate metrics:

- lineage completeness
- edge correctness
- node coverage
- false edge rate
- missing edge rate
- authorization correctness
- redaction correctness
- export reproducibility
- query or path usefulness
- human traceability rating

The first formal evaluation should answer:

```text
Does the graph projection faithfully represent existing MNEMOS records
without inventing relationships or changing system behavior?
```

Initial acceptance criteria:

- 100% deterministic edge backing for R1-eligible edges
- 0 unauthorized nodes or edges exposed
- 0 retrieval, governance, or promotion behavior changes
- 0 LLM-only authoritative edges
- projection reproducible from the same records
- lineage-incomplete records omitted or explicitly labeled

Research output:

- `docs/experiments/context_graph_projection_r1_preregistration.md`

## Deliberately Deferred Lane

Graph-assisted retrieval is a separate, future research topic and must not be
opened as part of R1.

Possible future lane:

```text
GRAPH_ASSISTED_RETRIEVAL_R0
```

That future lane would need to answer:

- can graph structure improve retrieval without creating false authority?
- can it improve recall without hiding source evidence?
- can it avoid bounded-retrieval failure modes where required source evidence
  drops out?
- can it be evaluated against a frozen benchmark pack?

For R1, graph remains:

```text
explanation layer
not retrieval layer
```

## Recommended Research Package

R1 should be managed as:

```text
CONTEXT_GRAPH_PROJECTION_R1_RESEARCH_PACKAGE
```

Deliverables:

1. use-case ranking
2. record fidelity audit
3. authority model
4. storage and options review
5. disclosure model
6. visualization requirements
7. evaluation and preregistration template
8. go/no-go recommendation for JSON-only implementation

## Sequencing

Recommended order:

1. use-case ranking
2. record fidelity audit
3. authority model
4. disclosure model
5. storage and runtime options
6. visualization requirements
7. evaluation and preregistration
8. go/no-go recommendation

This order intentionally answers “what is the graph for?” and “what does a
relationship mean?” before asking “where would it run?”

## Go/No-Go Conditions

R1 should recommend **no-go** if any of the following remain unresolved:

- the highest-value use case is weak or duplicative of existing records
- stable IDs or lineage are missing for core trace paths
- disclosure rules cannot prevent relationship leakage
- authority classes remain ambiguous
- R1-eligible edges cannot be backed deterministically
- the likely implementation path introduces dependency gravity too early
- evaluation criteria cannot detect false edges or unauthorized exposure

R1 may recommend a narrow future R2 only if:

- at least one high-value traceability use case is clearly improved
- core R1-eligible edges are deterministically backed
- disclosure and authorization rules are specified
- authority boundaries remain intact
- JSON-only projection remains sufficient for the first implementation

## Expected Next Artifact

The immediate next artifact after this plan should be:

- `docs/experiments/context_graph_projection_r1_use_cases.md`

The first question to answer remains:

```text
What user or developer question should the graph answer
better than the current records?
```

## Acceptance Statement

This research plan is acceptable only under the following interpretation:

```text
R1 is a research phase, not an implementation phase.

Its job is to determine whether a narrow, deterministic, read-only graph-shaped
traceability layer is useful and safe enough to justify a future JSON-only
projection.

It does not authorize code, GraphRAG, graph database integration, retrieval
reranking, or authority changes.
```

## Closeout Labels

```text
CONTEXT_GRAPH_PROJECTION_R1_RESEARCH_PLAN_COMPLETE
RESEARCH_ONLY
NO_IMPLEMENTATION_AUTHORIZED
NO_GRAPH_DATABASE_DEPENDENCY
NO_GRAPH_ASSISTED_RETRIEVAL
NO_AUTHORITY_SURFACE_CHANGE
JSON_ONLY_IMPLEMENTATION_IS_FUTURE_AND_CONDITIONAL
```
