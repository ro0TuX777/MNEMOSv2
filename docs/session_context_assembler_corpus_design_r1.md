# Session Context Assembler — R1 Corpus and Measurement Protocol

Status: frozen Phase 2R research fixture. Offline replay only; no external consumer or
runtime integration is authorized.

R1 preserves the frozen R0 file as an independent historical artifact. It
carries all 24 R0 cases forward without changing their turn text, required
IDs, irrelevant-turn labels, family, or notes, then adds five synthetic cases.
Additive R1 metadata is allowed and is covered by R1's own manifest.

## Binding budget protocol

For every case:

```text
expected_context_budget = max(
    round(0.50 × full_history_tokens),
    15,
    largest_segmented_episode_tokens
)
```

`full_history_tokens` is the replay harness's deterministic whitespace-token
estimate. The episode floor respects C's atomic selection unit, so neither B
nor C can exceed its nominal cap merely because one episode is larger than the
50% target. Every R1 budget remains strictly below full history. Condition A
is unconstrained; B and C receive the same case budget.

## Source-artifact measurement

Turns may declare `linked_source_ids`. Extraction uses the union of inline
`SRC-SCA-*` identifiers and these structured links, subject to the same
`eligible` gate. R1 adds links wherever an R0-required source was previously
present only in case metadata. Thus full history must recover every required
source; selection conditions can still lose a source by omitting its linked
turn.

## Decision retention and adversarial cases

`decision_artifact_retention` is the fraction of required decision IDs that
survive selection. It is intentionally explicit even though its R1 arithmetic
matches `required_prior_decision_recall`. New cases cover an old decisive
artifact, a recent irrelevant decision distractor, and a semantically similar
but incorrect distractor.

## Contradiction measurement

Applicable cases carry `expected_contradiction_status` (`resolved`,
`unresolved`, or `mixed`). A condition receives that classification only when
all required decision and source artifacts survive; otherwise it reports
`omitted`. The category is no longer inferred from prose notes, and structured
source links ensure absence of an inline source-ID string cannot by itself make
the full-history ceiling unmeasurable.

## Interpretation boundary

R1 revises inputs and measurement only. It does not tune the selector. A poor
R1 result is evidence about the existing policy, not permission to integrate,
write memory, mutate governance, or claim human-value non-inferiority.
