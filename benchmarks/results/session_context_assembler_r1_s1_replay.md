# Session Context Assembler — R1 Phase 4R S1 Replay Report

`OFFLINE_REPLAY_ONLY` `NO_HUMAN_VALUE_CLAIM` `NO_RUNTIME_INTEGRATION` `NO_PRODUCTION_READINESS_CLAIM`

## Phase 4R S1 advancement requirements

| Requirement | Value | Required | Passed |
|---|---|---|---|
| decision_artifact_retention | 1.0000 | >= B (0.411765) | PASS |
| required_source_recall | 1.0000 | >= B (0.183908) | PASS |
| contradiction_awareness | 1.0000 | >= B (0.125000) | PASS |
| prompt_token_reduction | 0.4513 | >= 0.25 | PASS |
| provenance_loss_count | 0 | 0 | PASS |
| synthetic_context_label_coverage | 1.0000 | 1.0 | PASS |
| budget_compliance | 29 | 29 | PASS |
| silent_required_artifact_omission_count | 0 | 0 | PASS |

**Phase 4R S1 outcome: PASS**

This report measures replay behavior of four context-construction conditions against the frozen `session_context_assembler_r1` corpus. It is an offline prototype evaluation. It does not authorize consumer runtime integration, production use, or any claim that human value or production readiness is established — see [ADR 0007](../../docs/adr/0007-session-context-assembler-shadow-only.md).

- Generated: 2026-06-21T21:55:41.225225+00:00
- Seed: 7
- Corpus manifest file_sha256: `9dc5682ec08ffad24a9c329ef8b581d3d68c3f83c92e078502f3d37c837e53dc`
- Cases replayed: 29

## Required safety/provenance gates (C1_selector_s1_mandatory_preservation)

| Gate | Value | Required | Passed |
|---|---|---|---|
| source_id_preservation_rate | 1.0 | 1.0 | PASS |
| parent_engram_lineage_preservation_rate | 1.0 | 1.0 | PASS |
| provenance_loss_count | 0 | 0 | PASS |
| synthetic_context_label_coverage | 1.0 | 1.0 | PASS |
| unauthorized_memory_write_count | 0 | 0 | PASS |
| unauthorized_governance_mutation_count | 0 | 0 | PASS |

**All required safety/provenance gates: PASS**

A Phase 4R PASS authorizes Phase 5 human-review design only. It does not authorize consumer runtime integration or production use.

## Condition comparison (descriptive, not gated)

B, C0, and C1 use each case's identical binding R1 budget (50% of full history, with 15-token and largest-atomic-episode floors); A is unconstrained.

| Condition | Cases | Mean prior-decision recall | Decision-artifact retention | Mean source recall | Contradiction awareness | Irrelevant-history selection rate | Mean prompt token reduction | Mean token estimate |
|---|---|---|---|---|---|---|---|---|
| A_full_history | 29 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 50.5862 |
| B_sliding_window | 29 | 0.4118 | 0.4118 | 0.1839 | 0.1250 | 0.6981 | 0.5043 | 24.8276 |
| C_governed_episode_selected | 29 | 0.1176 | 0.1176 | 0.1379 | 0.0000 | 0.5472 | 0.6126 | 20.1034 |
| C1_selector_s1_mandatory_preservation | 29 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1509 | 0.4513 | 26.6207 |

### Harness self-check: fixed stress budget (token_budget_override=20, not a corpus-budget replay)

This secondary pass forces B and C below their natural full-history size to confirm the truncation/selection machinery actually differentiates conditions when a budget binds. It exists only to show the harness is not silently inert — it is not a corpus-budget replay, is not gated, and is not a quality claim.

| Condition | Cases | Mean prior-decision recall | Decision-artifact retention | Mean source recall | Contradiction awareness | Irrelevant-history selection rate | Mean prompt token reduction | Mean token estimate |
|---|---|---|---|---|---|---|---|---|
| A_full_history | 29 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 50.5862 |
| B_sliding_window | 29 | 0.1176 | 0.1176 | 0.0517 | 0.0000 | 0.3208 | 0.6547 | 15.4138 |
| C_governed_episode_selected | 29 | 0.0000 | 0.0000 | 0.0690 | 0.0000 | 0.3774 | 0.6614 | 16.5172 |
| C1_selector_s1_mandatory_preservation | 29 | 0.2941 | 0.2941 | 0.4310 | 0.2500 | 0.0755 | 0.8074 | 7.4138 |

`NO_HUMAN_VALUE_CLAIM`: the above is replay-measured recall/token accounting against this corpus's own fixture labels, not a human or model judgment of answer quality. No Phase 5 human review has run.

## Observations (not conclusions)

Under the corpus-budget replay, C1_selector_s1_mandatory_preservation decision-artifact retention (1.0000) is higher than B (0.4118).

## Known measurement limitations in this run

- S1 classifies contradiction candidates from deterministic runtime text signals and structured source links; no model-level semantic judgment is used.
- Five cases emit a conservative budget-insufficient abstention because an additional runtime-visible mandatory candidate cannot fit. The R1-scored artifact is nevertheless retained in each case; no omission is silent.

## Per-case results

Full per-case, per-condition records are in [session_context_assembler_r1_s1_replay.json](session_context_assembler_r1_s1_replay.json).
