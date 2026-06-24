# ADR 0009: GateMem G1 Is Offline Benchmark Plumbing Only

Date: 2026-06-24

Status: Accepted — offline projection and normalization only

## Context

GateMem G0 found that MNEMOS can partially evaluate retrieval utility and can
shadow-test some disclosure mechanics, but does not have GateMem-grade active
forgetting. G0 also found that GateMem's native in-repository `Checkpoint`
object carries scoring-only annotations even though benchmark methods are not
permitted to consume them.

An isolated boundary is needed to prove that visible GateMem inputs can be
projected and normalized without editing GateMem, connecting to MNEMOS runtime,
or concealing the deletion gap.

## Decision

Authorize the original MNEMOS prototype under `prototype/gatemem_g1/` for:

- allowlist-only projection of episode identity, ordered visible turns through
  the checkpoint boundary, requester identity/role, query text, and separately
  supplied permitted metadata;
- deterministic projection fingerprints;
- injected offline retrieval and disclosure probes;
- normalization into GateMem-compatible `predictions.jsonl` rows; and
- explicit `unsupported` or `simulated_shadow` deletion observations that
  always normalize to `refuse`, never to a purported successful forget action.

All GateMem-derived projection and prediction files must remain outside the
MNEMOS repository. The implementation may read an external GateMem checkout
but must not import it, edit it, vendor it, or write outputs into it.

## Prohibited behavior

- importing from `mnemos`, `mnemos_sdk`, `service`, or GateMem's `bench` package;
- reading or branching on `query_type`, `attack_type`, `expected_action`,
  `judge_spec`, or `leak_targets`;
- production routing, consumer connection, or runtime retrieval;
- hosted-model or LLM-judge execution;
- treating simulation as deletion, non-recoverability, or forgetting evidence;
- writing GateMem-derived data into MNEMOS;
- editing the upstream GateMem clone; or
- public scoring or leaderboard submission.

## Deletion boundary

G1 may conservatively identify deletion relevance only from visible query and
turn language. This signal is benchmark plumbing, not a deletion policy and
not a correctness label. False positives and false negatives remain possible.

Any true governed deletion lifecycle requires a separate ADR covering
authorization, target resolution, tombstones, lineage cascade, cache
invalidation, re-ingestion defense, cross-tier negative verification, and an
answer-layer non-confirmation policy.

## Consequences

G1 can validate temporal projection, evaluator isolation, shadow disclosure
invariants, and output formatting. It cannot produce a MNEMOS benchmark-quality
score without a separately authorized offline retrieval/disclosure
implementation and cannot support a deletion capability claim under any mode.

## Evidence

- `docs/benchmarks/gatemem_g1_clean_projection.md`
- `prototype/gatemem_g1/`
- `tools/run_gatemem_g1_projection.py`
- `tools/normalize_gatemem_g1_predictions.py`
- `tools/run_gatemem_g1_gate.py`
- `tests/test_gatemem_g1.py`
- `benchmarks/results/gatemem_g1_gate.json`
- `benchmarks/results/gatemem_g1_gate.md`
