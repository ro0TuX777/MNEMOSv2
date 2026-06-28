# ADR 0010: GateMem G2 Is an Offline Retrieval/Disclosure Adapter

Date: 2026-06-24

Status: Accepted — offline evaluation only

## Context

GateMem G1 established a clean, deterministic input projection and an injected
shadow-observation boundary. It deliberately did not connect a retriever or
disclosure policy. G2 is needed to exercise that boundary with original MNEMOS
evaluation code and produce externally scoreable predictions without creating
a runtime integration or obscuring the deletion gap.

## Decision

Authorize `prototype/gatemem_g2/` as an original, standard-library-only offline
adapter that:

- consumes verified G1 projection rows exclusively;
- performs deterministic lexical retrieval over projected turns;
- applies a conservative visible-input disclosure policy;
- exposes only disclosed evidence in the answer/prompt-context audit;
- preserves exact used-record provenance;
- emits GateMem-compatible external predictions;
- freezes predictions before evaluator-only scoring annotations are joined; and
- explicitly refuses every visible deletion-relevant input.

The policy may use requester identity/role, turn identity/role, visible text,
adjacency, and G1-permitted metadata. It may not read GateMem scoring fields.

## Evaluator separation

The adapter receives only G1 projections. Aggregate measurement is a later,
separate phase. GateMem's offline rule scorer may join frozen predictions to
annotations, but no row-level scoring annotation may flow back into retrieval,
disclosure, answer construction, or prediction normalization.

GateMem's stub is prohibited because it consumes `expected_action`. No stub
metric is G2 evidence. No hosted judge is authorized.

## Deletion boundary

Deletion-relevant projections normalize to `refuse`, with empty prompt context
and no used record IDs. Refusal consistency may be measured. GateMem's deletion
leakage metric must not be presented as active-forgetting performance because
G2 neither mutates nor verifies durable memory.

A true governed deletion lifecycle remains a separate architectural decision
requiring its own ADR.

## Consequences

G2 provides a reproducible baseline for retrieval utility, disclosure denial,
redaction, cross-principal filtering, over-refusal, and provenance integrity.
Its policy is intentionally not production role enforcement. Weak utility or
privacy results remain visible as limitations rather than being optimized with
hidden annotations.

## Prohibited behavior

- MNEMOS runtime/service/SDK imports or calls;
- GateMem package imports or upstream edits;
- hosted models, hosted judges, or provider dependencies;
- leaderboard submission;
- production role/authorization claims;
- deletion, non-recoverability, or non-confirmation claims; and
- copying row-level GateMem data, predictions, diagnostics, or scores into
  MNEMOS.

## Evidence

- `docs/benchmarks/gatemem_g2_offline_adapter.md`
- `prototype/gatemem_g2/`
- `tools/run_gatemem_g2_offline.py`
- `tools/compile_gatemem_g2_report.py`
- `tools/run_gatemem_g2_gate.py`
- `tests/test_gatemem_g2.py`
- `benchmarks/results/gatemem_g2_offline_report.json`
- `benchmarks/results/gatemem_g2_offline_report.md`
- `benchmarks/results/gatemem_g2_gate.json`
- `benchmarks/results/gatemem_g2_gate.md`

