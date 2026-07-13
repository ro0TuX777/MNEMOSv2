# Session Context Assembler — Phase 2R Closeout

Status: `SESSION_CONTEXT_ASSEMBLER_PHASE_2R_COMPLETE`.

Phase 2R revised the corpus and measurement protocol without changing the
episode selector. R0 remains frozen and independently hash-validated. R1 is a
29-case frozen corpus with binding budgets, structured source links, explicit
contradiction categories, and three new decision-retention stress classes.

## What changed

- Every budget is the maximum of 50% of full history, 15 tokens, and the
  largest atomic episode; every result remains capped and the budget binds.
- `linked_source_ids` makes metadata-only source artifacts recoverable from
  selected eligible turns without requiring IDs in prose.
- `decision_artifact_retention` explicitly measures survival of required
  decision IDs.
- Applicable cases carry `expected_contradiction_status` (`resolved`,
  `unresolved`, or `mixed`).
- Five cases were added: three decision-retention adversaries and two
  structured-link contradiction cases.

Full-history condition A reaches 1.0 decision retention, 1.0 source recall,
and the authored category on all eight contradiction cases. This removes the
R0 source-recall ceiling and verifies that R1 is mechanically measurable.

## Binding-budget replay result (seed 7)

| Condition | Decision retention | Source recall | Irrelevant-history rate | Token reduction |
|---|---:|---:|---:|---:|
| A full history | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| B sliding window | 0.4118 | 0.1839 | 0.6981 | 0.5043 |
| C episode-selected | 0.1176 | 0.1379 | 0.5472 | 0.6126 |

C preserves provenance and labels perfectly and reduces estimated context by
61.26%, but it retains required decisions in only 11.76% of scored cases,
versus 41.18% for the naive sliding window. It
also returns `omitted` on all eight contradiction cases under their binding
budgets. On each of the three new decision-retention adversaries, C drops the
required decision and its source.

The six zero-tolerance safety/provenance gates still pass. That is necessary
but not sufficient: C is not non-inferior to full history on continuity or
contradiction awareness, and it does not outperform the naive window on
decision retention. The Phase 4 advancement gate therefore does not pass.

## Resulting state

```text
SESSION_CONTEXT_ASSEMBLER_PHASE_2R_COMPLETE
R1_CORPUS_FROZEN_AND_MEASURABLE
R1_BUDGETS_BINDING
R1_SOURCE_RECALL_CEILING_REMOVED
SELECTOR_DECISION_RETENTION_INADEQUATE
PHASE_4_GATE_REVIEW_NOT_PASSED
NO_SELECTOR_TUNING_PERFORMED
NO_CONSUMER_RUNTIME_INTEGRATION
NO_HUMAN_REVIEW_YET
```

The next technically plausible research step is a separately authorized
selector revision that combines task relevance with mandatory preservation of
eligible decision, source, and contradiction artifacts. R1 must remain frozen
for that comparison; tuning it does not authorize runtime integration.
