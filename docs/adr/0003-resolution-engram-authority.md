# ADR 0003: Resolution Engram Authority Is Additive And Governed

Date: 2026-06-20

Status: Accepted

## Context

Contradictory memories require a way to present the best current resolution
without deleting or mutating the original parent evidence. MNEMOS uses
Resolution Engrams to represent governed reconciliations while preserving parent
lineage.

## Decision

Resolution Engrams are additive governed artifacts. They may receive read-path
priority in a contradiction cluster, but they do not erase parent memories.
Parents remain available for audit and review.

## Alternatives Considered

- Mutate the winning parent in place.
- Delete or suppress losing parents permanently.
- Treat all contradictions as unresolved forever.

## Invariants

- Resolution Engrams must retain lineage edges to all conflicting parents.
- Parent evidence must remain auditable.
- Read-path priority is governed and explainable.
- Resolution authority is not equivalent to EBIR promotion.

## Rollback

If a Resolution Engram is unsupported or unsafe, remove its governed read-path
priority and leave parent evidence retrievable for audit.

## Evidence

- `docs/benchmark.md`
- `docs/whitepaper.md`
- `benchmarks/results/phase_10_consensus_gate.json`
- `tools/validate_phase10_resolution_gate.py`

