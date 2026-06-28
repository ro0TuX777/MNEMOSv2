# Session Context Assembler — R0 Replay Report (Phase 3)

`OFFLINE_REPLAY_ONLY` `NO_HUMAN_VALUE_CLAIM` `NO_RUNTIME_INTEGRATION` `NO_PRODUCTION_READINESS_CLAIM`

This report measures replay behavior of three context-construction conditions against the frozen `session_context_assembler_r0` corpus. It is an offline prototype evaluation. It does not authorize consumer runtime integration, production use, or any claim that quality is non-inferior or that token reduction meets a Phase 4 target — see [ADR 0007](../../docs/adr/0007-session-context-assembler-shadow-only.md).

- Generated: 2026-06-21T20:58:14.878034+00:00
- Seed: 7
- Corpus manifest file_sha256: `1859b7b5a6c6f3786ec18ff5eed1d08babb3509e7519c376c56c62eebd1eae5a`
- Cases replayed: 24

## Phase 3 required gates (condition C only)

| Gate | Value | Required | Passed |
|---|---|---|---|
| source_id_preservation_rate | 1.0 | 1.0 | PASS |
| parent_engram_lineage_preservation_rate | 1.0 | 1.0 | PASS |
| provenance_loss_count | 0 | 0 | PASS |
| synthetic_context_label_coverage | 1.0 | 1.0 | PASS |
| unauthorized_memory_write_count | 0 | 0 | PASS |
| unauthorized_governance_mutation_count | 0 | 0 | PASS |

**All required Phase 3 gates: PASS**

A gate PASS authorizes review of Phase 4 gates only. It does not authorize consumer runtime integration or production use.

## Condition comparison (descriptive, not gated)

Each condition is capped at the case's own `expected_context_budget` (B and C; A is always unconstrained). r0's configured budgets are well above this corpus's short fixture turn counts (see Known measurement limitations below), so this primary comparison shows all three conditions selecting everything for every case — that is the actual, honest result of running this pass against r0, not a rendering error.

| Condition | Cases | Mean prior-decision recall | Mean source recall | Irrelevant-history selection rate | Mean prompt token reduction | Mean token estimate |
|---|---|---|---|---|---|---|
| A_full_history | 24 | 1.0000 | 0.2500 | 1.0000 | 0.0000 | 44.0000 |
| B_sliding_window | 24 | 1.0000 | 0.2500 | 1.0000 | 0.0000 | 44.0000 |
| C_governed_episode_selected | 24 | 1.0000 | 0.2500 | 1.0000 | 0.0000 | 44.0000 |

### Harness self-check: fixed stress budget (token_budget_override=20, not a corpus-budget replay)

This secondary pass forces B and C below their natural full-history size to confirm the truncation/selection machinery actually differentiates conditions when a budget binds. It exists only to show the harness is not silently inert — it is not a corpus-budget replay, is not gated, and is not a quality claim.

| Condition | Cases | Mean prior-decision recall | Mean source recall | Irrelevant-history selection rate | Mean prompt token reduction | Mean token estimate |
|---|---|---|---|---|---|---|
| A_full_history | 24 | 1.0000 | 0.2500 | 1.0000 | 0.0000 | 44.0000 |
| B_sliding_window | 24 | 0.1538 | 0.0000 | 0.4571 | 0.6282 | 15.2917 |
| C_governed_episode_selected | 24 | 0.0000 | 0.0833 | 0.4000 | 0.6572 | 14.2500 |

`NO_HUMAN_VALUE_CLAIM`: the above is replay-measured recall/token accounting against this corpus's own fixture labels, not a human or model judgment of answer quality. No Phase 4 evaluation has run.

## Observations flagged for Phase 4 review (not conclusions)

Under the stress-budget self-check (token_budget_override=20), condition C's mean required_prior_decision_recall is *lower* than condition B's on the cases that require a prior decision (e.g. the prior_architectural_decision_recall family): the relevance-scored episode selector can pick the episode whose wording most resembles the current task while excluding an earlier, smaller episode that actually contains the decision artifact, especially when only one episode fits a tight budget. This is a genuine selector-behavior signal at this artificial budget, not a measurement bug - recorded here for Phase 4 consideration of the relevance-scoring approach. No change to the Phase 1 selection algorithm was made under this authorization.

## Known measurement limitations in this run

- required_source_recall for sca_r0_urd_001 and sca_r0_urd_002 is 0.0 for ALL THREE conditions, including the unconstrained full-history condition A. This is an extractor/corpus ceiling, not a selection failure: those two cases' required source IDs are referenced only in case metadata and are never literally embedded in turn text, so no regex-based extractor can recover them regardless of how much history is included. See docs/session_context_assembler_phase_1_notes.md ('ID extraction is a stand-in'). A future corpus revision (r1) adding an explicit source-artifact pool separate from turn text would remove this ceiling.
- contradiction_awareness_result for sca_r0_urd_003 is 'omitted' for all three conditions for the same reason: its required source ID (SRC-SCA-explain-format-discussion) is never inline-extractable, so the all-required-ids-recovered rule cannot return 'mixed' even though the decision ID (DEC-SCA-113) is recoverable. Only the contradiction_aware_followup family (caf_001-003) has fully inline-extractable required IDs in r0, so it is the only family where contradiction_awareness_result currently differentiates conditions.

## Per-case results

Full per-case, per-condition records are in [session_context_assembler_r0_replay.json](session_context_assembler_r0_replay.json).
