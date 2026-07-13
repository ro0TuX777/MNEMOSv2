# Context Graph Projection R1 Use Cases

Date: 2026-07-13

Status: **Research-only artifact. No implementation authorized.**

## Executive Summary

This document ranks the first candidate use cases for Context Graph Projection
R1 and recommends which one should anchor further graph research for MNEMOS.

The goal is not to ask whether graphs are generally useful. The goal is to ask
which user or developer question a graph-shaped projection can answer better
than current MNEMOS records while preserving MNEMOS’s evidence, governance, and
authority boundaries.

The recommended first use case is:

```text
Evidence-to-decision traceability
```

This use case aligns most closely with MNEMOS’s current purpose, requires the
fewest authority leaps, and provides the clearest observability value without
requiring graph-assisted retrieval or graph-derived truth.

## Starting Boundary

This use-case ranking inherits the accepted R0 and R1 boundaries:

- read-only
- projection-only
- non-authoritative
- not GraphRAG
- not a graph database initiative
- not a retrieval-path change
- not a governance or promotion change

All ranked use cases are evaluated under those constraints.

## Research Question

```text
What user or developer question should the graph answer
better than the current MNEMOS records?
```

## Ranking Criteria

Each candidate use case is ranked against the following criteria:

1. **Mission fit**
   - Does the use case match MNEMOS’s current role as a source-grounded,
     governed context layer?
2. **Authority safety**
   - Can the use case be delivered without encouraging users to treat graph
     relationships as new truth?
3. **Deterministic backing**
   - Is the use case likely to rely on existing IDs, lineage fields, audit refs,
     or other deterministic records rather than inference?
4. **Observability value**
   - Does the use case make the system easier to inspect, audit, or debug?
5. **Incremental implementability**
   - Could a later narrow JSON-only projection support the use case without new
     storage semantics?
6. **Disclosure risk**
   - Can the use case be bounded without immediately creating severe
     relationship-leakage problems?

## Ranked Use Cases

### Rank 1 — Evidence-To-Decision Traceability

**Core user question**

```text
What source evidence shaped this answer, decision, or formal outcome?
```

**Why this ranks first**

This is the strongest first use case because it maps directly to what MNEMOS
already claims to do well: preserve source grounding, lineage, evaluation
signals, and auditability around retrieved context and downstream use.

It also fits the existing boundary that a retrieved result is not automatically
a decision. A graph-shaped projection can expose the trace from source artifact
to source engram to retrieval event to decision or evaluation record without
changing which record is authoritative.

**Why current records are insufficient or cumbersome**

- the relevant records exist across multiple feature lanes
- lineage is present, but the trace is not yet unified as one inspectable path
- a user can inspect artifacts individually, but answering “what shaped this
  outcome?” may require hopping across engrams, retrieval outputs, governance
  metadata, cycle records, and handoff or evaluation artifacts

**Why a graph shape helps**

- the path itself is the product
- it makes evidence lineage visible as a connected trace rather than a set of
  isolated records
- it supports path-oriented inspection such as:
  - source → retrieval → decision
  - source → contradiction → resolution
  - source → evaluation → accepted or rejected claim

**Why this does not require authority expansion**

- the graph explains how records relate
- it does not turn relationships into truth
- it does not let connectivity increase ranking, trust, promotion, or
  governance authority

**Initial recommendation**

Use this as the primary anchor use case for the rest of R1.

### Rank 2 — Decision Lineage

**Core user question**

```text
What evidence supported, contradicted, superseded, or rejected a claim?
```

**Why it ranks second**

Decision lineage is closely related to evidence traceability, but it is
slightly broader and slightly riskier because it invites users to interpret
decision edges as stronger truth statements than the underlying artifacts may
justify.

It remains a high-value use case because MNEMOS already contains contradiction,
supersession, evaluation, and governance-related structures. A path-based
projection could make those structures easier to inspect.

**Why current records are insufficient or cumbersome**

- contradiction and supersession information may be available but not presented
  as a simple trace
- accepted or rejected outcomes may be documented in separate artifacts
- the inspection burden is higher when a user needs to compare multiple records
  manually

**Why a graph shape helps**

- it can surface claim-centered trace paths
- it can show how a decision changed over time
- it can show which evidence supported or weakened a result

**Why this is not ranked first**

- “decision lineage” can drift toward implicit causal claims
- users may over-read connected decision nodes as approved truth
- it requires especially careful authority labeling

**Initial recommendation**

Keep this in the primary R1 package, but subordinate it to evidence-to-decision
traceability rather than letting it define the graph on its own.

### Rank 3 — Handoff Inspection

**Core user question**

```text
What context package or handoff included which sources, and why?
```

**Why it ranks third**

This is a strong working-memory use case and fits MNEMOS’s bounded context
assembly and handoff posture well. It is likely to produce clear value for AI
developer workflows, collaboration, and audit-style review.

**Why current records are insufficient or cumbersome**

- a handoff may include evidence and context, but the inclusion path is not
  always presented as one inspectable trace
- a reviewer may want to know which specific source artifacts entered the
  package and how they got there

**Why a graph shape helps**

- it naturally represents package inclusion paths
- it supports trace questions such as:
  - which sources entered this handoff?
  - which context package carried this evidence?
  - which earlier decision was carried forward?

**Why this is not ranked higher**

- it is narrower than evidence-to-decision traceability
- it is more workflow-specific and less universal
- it may depend on how consistently handoff artifacts are represented today

**Initial recommendation**

Include it as a strong secondary use case, especially for later demos.

### Rank 4 — Evaluation Memory

**Core user question**

```text
What formal result accepted or rejected a feature claim?
```

**Why it ranks fourth**

This use case is valuable for development, benchmarking, and evidence-based
decision history. It fits MNEMOS’s growing emphasis on formal artifacts,
evaluation packs, technical closeouts, and bounded claims.

**Why current records are insufficient or cumbersome**

- evaluation artifacts may exist, but the path from source evidence to formal
  outcome is not always presented as one chain
- users may need to inspect multiple reports and supporting references manually

**Why a graph shape helps**

- it turns scattered evaluation artifacts into a single trace path
- it can connect evidence, decision records, and outcome artifacts cleanly

**Why this is not ranked higher**

- it is important, but more specialized than the top three
- its value depends on whether evaluation artifacts are consistently structured
  enough for deterministic projection

**Initial recommendation**

Keep it in scope for R1 as a likely secondary view once the primary trace path
is defined.

### Rank 5 — Debugging And Observability

**Core user question**

```text
Why did this assistant retrieve these records and not others?
```

**Why it ranks fifth**

This is potentially useful, but it is more likely to blur the line between
traceability and retrieval explanation. It may be valuable later, especially in
combination with cognitive-cycle and governance summaries, but it should not
anchor the first graph research phase.

**Why current records are insufficient or cumbersome**

- retrieval and governance explanations may exist, but they are distributed
- negative-space questions such as “why not this other record?” are harder to
  answer than inclusion traces

**Why a graph shape helps**

- it could connect retrieval observations, suppressions, and downstream use
- it could help explain why one path was taken instead of another

**Why this ranks below the top four**

- counterfactual and exclusion explanations are harder to back deterministically
- it increases pressure to expose ranking logic through graph relationships
- it risks drifting toward retrieval-theory claims before R1 is ready

**Initial recommendation**

Treat this as a follow-on observability use case, not the primary R1 driver.

### Rank 6 — Project Continuity

**Core user question**

```text
What prior decisions, artifacts, and retrieval events are connected to this task?
```

**Why it ranks sixth**

Project continuity is appealing and market-readable, but it is the broadest and
least bounded of the candidate jobs. It is also the most likely to drift toward
an implied “graph memory brain” model if pursued too early.

**Why current records are insufficient or cumbersome**

- project continuity is often assembled from many different artifacts
- continuity queries tend to be open-ended rather than trace-bounded

**Why a graph shape helps**

- it could help cluster related work over time
- it could connect tasks, prior decisions, handoffs, and evidence

**Why this ranks last**

- it is broad enough to invite authority drift
- it would likely require stronger assumptions about task identity and
  continuity than R1 should make
- it is better treated as a later product framing or UI composition problem
  rather than a first projection target

**Initial recommendation**

Defer this until narrower trace paths are proven useful and safe.

## Summary Ranking

| Rank | Use Case | Recommendation |
|---|---|---|
| 1 | Evidence-to-decision traceability | Primary R1 anchor |
| 2 | Decision lineage | Strong secondary lane |
| 3 | Handoff inspection | Strong secondary lane |
| 4 | Evaluation memory | Secondary formal-artifact lane |
| 5 | Debugging and observability | Follow-on lane |
| 6 | Project continuity | Defer |

## Why The Top Use Case Wins

Evidence-to-decision traceability should anchor R1 because it offers the best
combination of:

- strong fit with MNEMOS’s current mission
- likely deterministic backing from existing records
- high observability value
- low authority expansion pressure
- clean alignment with a future JSON-only read-only projection

It answers a question that MNEMOS users are likely to ask often:

```text
Show me the path from this outcome back to the source evidence.
```

That question is both useful and boundary-safe.

## Implications For The Rest Of R1

If evidence-to-decision traceability remains the top-ranked use case, the next
R1 workstreams should optimize for:

- evidence-path fidelity
- decision-record authority labeling
- package inclusion traceability
- disclosure-safe path rendering
- deterministic export reproducibility

They should **not** optimize first for:

- open-ended graph exploration
- graph-scale query performance
- graph-assisted retrieval
- semantic relationship mining
- global project-memory browsing

## Recommended Next Artifact

The next R1 artifact after this use-case ranking should be:

- `docs/experiments/context_graph_projection_r1_record_fidelity_audit.md`

That artifact should test whether the highest-ranked use cases can actually be
backed by stable IDs, lineage fields, digests, and audit-safe metadata.

## Acceptance Statement

This use-case ranking is acceptable only under the following interpretation:

```text
The graph is justified only if it answers a traceability question
better than current MNEMOS records without creating new authority.

The first and best candidate question is:
What source evidence shaped this answer, decision, or formal outcome?
```

## Closeout Labels

```text
CONTEXT_GRAPH_PROJECTION_R1_USE_CASES_COMPLETE
RESEARCH_ONLY
EVIDENCE_TO_DECISION_TRACEABILITY_PRIORITIZED
NO_IMPLEMENTATION_AUTHORIZED
NO_AUTHORITY_SURFACE_CHANGE
NO_GRAPH_ASSISTED_RETRIEVAL
```
