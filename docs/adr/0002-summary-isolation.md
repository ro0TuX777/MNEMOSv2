# ADR 0002: Summary Engram Isolation

Date: 2026-06-20

Status: Accepted

## Context

MNEMOS stores both raw/source-grounded engrams and synthetic summary engrams.
Summary engrams are useful for global and thematic retrieval, but they must not
silently replace source evidence in default factoid search.

## Decision

Summary engrams remain isolated from default factoid retrieval by server-managed
controls. Clients cannot directly supply reserved sentinel filters such as
`__exclude_summaries__`; those controls are owned by the service.

## Alternatives Considered

- Mix summaries into all retrieval results by default.
- Let clients manage summary exclusion directly.
- Store summaries outside the normal memory system.

## Invariants

- Raw evidence remains the citation authority for ordinary answer grounding.
- Summary engrams must retain lineage to their source parents.
- Reserved sentinel controls are server-managed and rejected when supplied by
  clients.
- Global/theme query paths may use summaries only when the retrieval posture
  explicitly allows them.

## Rollback

If summary isolation regresses, disable summary inclusion in read paths and rerun
the hierarchy and evidence-contract gates before re-enabling it.

## Evidence

- `docs/benchmark.md`
- `docs/whitepaper.md`
- `tests/test_hierarchy_lineage.py`
- `benchmarks/results/phase_9_hierarchy_sim.json`

