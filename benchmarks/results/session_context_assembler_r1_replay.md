# Session Context Assembler — R1 Replay Report

`OFFLINE_REPLAY_ONLY` `NO_HUMAN_VALUE_CLAIM` `NO_RUNTIME_INTEGRATION` `NO_PRODUCTION_READINESS_CLAIM`

This report measures replay behavior of three context-construction conditions against the frozen `session_context_assembler_r1` corpus. It is an offline prototype evaluation. It does not authorize consumer runtime integration, production use, or any claim that quality is non-inferior or that token reduction meets a Phase 4 target — see [ADR 0007](../../docs/adr/0007-session-context-assembler-shadow-only.md).

- Generated: 2026-06-21T21:28:26.294168+00:00
- Seed: 7
- Corpus manifest file_sha256: `9dc5682ec08ffad24a9c329ef8b581d3d68c3f83c92e078502f3d37c837e53dc`
- Cases replayed: 29

## Required safety/provenance gates (condition C only)

| Gate | Value | Required | Passed |
|---|---|---|---|
| source_id_preservation_rate | 1.0 | 1.0 | PASS |
| parent_engram_lineage_preservation_rate | 1.0 | 1.0 | PASS |
| provenance_loss_count | 0 | 0 | PASS |
| synthetic_context_label_coverage | 1.0 | 1.0 | PASS |
| unauthorized_memory_write_count | 0 | 0 | PASS |
| unauthorized_governance_mutation_count | 0 | 0 | PASS |

**All required safety/provenance gates: PASS**

A gate PASS authorizes review of Phase 4 gates only. It does not authorize consumer runtime integration or production use.

## Condition comparison (descriptive, not gated)

B and C use each case's binding R1 budget (50% of full history, with 15-token and largest-atomic-episode floors); A is unconstrained.

| Condition | Cases | Mean prior-decision recall | Decision-artifact retention | Mean source recall | Irrelevant-history selection rate | Mean prompt token reduction | Mean token estimate |
|---|---|---|---|---|---|---|---|
| A_full_history | 29 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 50.5862 |
| B_sliding_window | 29 | 0.4118 | 0.4118 | 0.1839 | 0.6981 | 0.5043 | 24.8276 |
| C_governed_episode_selected | 29 | 0.1176 | 0.1176 | 0.1379 | 0.5472 | 0.6126 | 20.1034 |

### Harness self-check: fixed stress budget (token_budget_override=20, not a corpus-budget replay)

This secondary pass forces B and C below their natural full-history size to confirm the truncation/selection machinery actually differentiates conditions when a budget binds. It exists only to show the harness is not silently inert — it is not a corpus-budget replay, is not gated, and is not a quality claim.

| Condition | Cases | Mean prior-decision recall | Decision-artifact retention | Mean source recall | Irrelevant-history selection rate | Mean prompt token reduction | Mean token estimate |
|---|---|---|---|---|---|---|---|
| A_full_history | 29 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 50.5862 |
| B_sliding_window | 29 | 0.1176 | 0.1176 | 0.0517 | 0.3208 | 0.6547 | 15.4138 |
| C_governed_episode_selected | 29 | 0.0000 | 0.0000 | 0.0690 | 0.3774 | 0.6614 | 16.5172 |

`NO_HUMAN_VALUE_CLAIM`: the above is replay-measured recall/token accounting against this corpus's own fixture labels, not a human or model judgment of answer quality. No Phase 4 evaluation has run.

## Observations (not conclusions)

Under the corpus-budget replay, C decision-artifact retention (0.1176) is lower than B (0.4118). No selector change was made during Phase 2R; this report measures the existing selector against revised inputs.

## Known measurement limitations in this run

- R1 repairs structural measurability but does not tune the Phase 1 selector. Low retention is an evaluation result, not a corpus-generation failure.

## Per-case results

Full per-case, per-condition records are in [session_context_assembler_r1_replay.json](session_context_assembler_r1_replay.json).
