# Session Context Assembler — Phase 3 Notes (Offline Baseline Replay Harness)

Status: **complete**, scoped strictly to the Phase 3 authorization in
[docs/session_context_assembler_spec.md](session_context_assembler_spec.md)
(Phase 3) and [ADR 0007](adr/0007-session-context-assembler-shadow-only.md).
This is an offline replay evaluation only. No runtime integration, agent
wiring, durable-memory write, retrieval-ranking change, authority change,
governance mutation, or promotion behavior is authorized by this work. A
PASS on the Phase 3 gates below authorizes review of the Phase 4 gates
only — it does not authorize external consumer integration or production use, and it
does not claim that quality is non-inferior or that token reduction meets
the Phase 4 target.

## What was built

`prototype/session_context_assembler/replay.py` — pure replay logic, no I/O,
no model/LLM calls. Runs three conditions against every case in the frozen
[R0 corpus](session_context_assembler_corpus_design.md):

```text
A. Full conversation history       (run_condition_a — unconstrained ceiling)
B. Naive sliding-window history    (run_condition_b — recency-only, capped)
C. Governed episode-selected       (run_condition_c — wraps Phase 1's
   context package                  assemble_context_package unchanged)
```

B and C share the same token budget per case (`expected_context_budget` by
default), so any difference between them reflects selection strategy, not
budget asymmetry. A is always unconstrained.

No model dependency is needed because every required Phase 3 metric
(recall, token accounting, provenance, contradiction-awareness category) is
structurally computable from the frozen corpus's own ground-truth fields
(`required_prior_decision_ids`, `required_source_ids`,
`known_irrelevant_history_turn_ids`, `expected_context_budget`). This
satisfies "identical model configuration, where model evaluation is used"
vacuously, since no model is used.

`tools/run_session_context_assembler_replay.py` — thin CLI wrapper. Loads
and hash-validates the corpus, runs the replay, computes the required
gates and the descriptive condition comparison, and writes:

- `benchmarks/results/session_context_assembler_r0_replay.json` — full
  per-case, per-condition records plus gate/comparison summaries.
- `benchmarks/results/session_context_assembler_r0_replay.md` — human-
  readable report carrying the four required disclaimer labels
  (`OFFLINE_REPLAY_ONLY`, `NO_HUMAN_VALUE_CLAIM`, `NO_RUNTIME_INTEGRATION`,
  `NO_PRODUCTION_READINESS_CLAIM`).

Both paths are the only files the script writes; this is verified by
`tests/test_session_context_assembler_replay.py::test_tools_script_only_writes_inside_benchmarks_results`.

## Required per-case output fields

Every record (one per case × condition, 72 records for the 24-case r0
corpus) carries the full set specified in the Phase 3 authorization:
`session_id`, `task_id`, `condition`, `prototype_version`, `seed`,
`corpus_manifest_hash`, `case_hash`, `selected_turn_ids`,
`selected_episode_ids`, `selected_parent_engram_ids`, `selected_source_ids`,
`token_estimate`, `required_prior_decision_recall`,
`required_source_recall`, `contradiction_awareness_result`,
`known_irrelevant_history_selected`, `provenance_loss_count`,
`synthetic_context_label_coverage` — plus extra accounting fields
(`case_id`, `case_family`, `known_irrelevant_history_available`,
`decision_lineage_loss_count`, `source_lineage_loss_count`,
`synthetic_context_labels`, `selection_rationale`,
`prompt_token_reduction`, `contradiction_awareness_matches_expected`) for
auditability. Conditions A and B emit the same fields with empty
`selected_episode_ids`/labels, so all three conditions are comparable on
identical columns.

## Metric definitions as implemented

- `required_prior_decision_recall` / `required_source_recall` — fraction of
  the case's required IDs recovered by the condition's extracted decisions/
  sources. Returns `None` (not `1.0`) when the case requires none of that
  ID type, since recall-of-nothing is undefined rather than vacuously
  perfect (`replay._recall`).
- `contradiction_awareness_result` — `"not_applicable"` for cases outside
  the `contradiction_aware_followup` / `unresolved_vs_resolved_decision_distinction`
  families; otherwise the case's authored category
  (`resolved`/`unresolved`/`mixed`) if every one of that case's required IDs
  was recovered, else `"omitted"`. No partial-credit "mixed" result is
  synthesized from a partial recovery — see `replay.CONTRADICTION_GROUND_TRUTH`,
  derived from each case's frozen `notes` field, not a new corpus field.
- `known_irrelevant_history_selected` — count of selected turn IDs that
  intersect the case's `known_irrelevant_history_turn_ids`.
- `provenance_loss_count` — selected decision/source IDs that cannot be
  traced to a `synthetic_context` label's `parent_engram_ids`/
  `parent_source_ids`. Structurally `0` for A/B (no label layer exists for
  naive history dumps; every reported ID is extracted only from a selected
  turn by construction).
- `synthetic_context_label_coverage` — fraction of selected episodes that
  carry a label. `0.0` for A/B by construction (no episodes, no labels);
  this is the intended measurement, not a gap.
- `prompt_token_reduction` — `1 - (condition token_estimate / condition-A
  token_estimate)` for the same case.

## Required gates (condition C only)

```text
source_id_preservation_rate             = 1.0   PASS
parent_engram_lineage_preservation_rate = 1.0   PASS
provenance_loss_count                   = 0     PASS
synthetic_context_label_coverage        = 1.0   PASS
unauthorized_memory_write_count         = 0     PASS
unauthorized_governance_mutation_count  = 0     PASS
```

All six pass on the frozen r0 corpus (seed 7, and re-verified deterministic
under seed 0 in the test suite). `unauthorized_memory_write_count` and
`unauthorized_governance_mutation_count` are `0` by construction — this
module has no import path to `mnemos`/`service`/`mnemos_sdk` (AST-verified)
and performs no filesystem writes during replay (behaviorally verified by
patching `Path.write_text`/`write_bytes`/`os.remove`/`os.unlink`/
`os.rename` to raise and running a full replay).

**A gate PASS authorizes review of the Phase 4 gates only.** It does not
authorize external consumer integration or production use, and does not by itself claim
non-inferior quality or that token reduction meets the Phase 4 25% target.

## What the descriptive (non-gated) comparison shows

The primary comparison — each condition capped at its own case's
`expected_context_budget` — shows **identical** stats across A/B/C for
every one of the 24 cases. This is not a bug: r0's configured budgets
(200-420 tokens per case) are well above this corpus's short fixture
conversation token counts (24-71 tokens per case), so no condition ever
truncates anything under r0's own budgets. Reported as the honest primary
result, not smoothed over.

Because that primary pass cannot show whether the truncation/selection
machinery actually differentiates B and C, the harness includes a second,
clearly non-gated, non-corpus-budget **self-check stress pass**
(`token_budget_override`, default 20 tokens, chosen empirically from
testing 10/15/20 as the value yielding the most informative
differentiation — 14 of 24 cases differ between B and C). Under that
artificial pressure, B and C diverge meaningfully on several metrics.

## Observation flagged for Phase 4 review (not a conclusion)

Under the stress-budget self-check only, condition C's mean
`required_prior_decision_recall` (`0.0`) is *lower* than condition B's
(`0.1538`) on cases requiring a prior decision (the
`prior_architectural_decision_recall` family). Investigation confirmed
this is a genuine, reproducible selector-behavior signal, not a
measurement bug: the relevance-scored episode selector can prefer an
episode lexically closer to the current task over a smaller earlier
episode that actually contains the decision artifact, when only one
episode fits a tight budget. **No change to the Phase 1 selection
algorithm was made under this authorization** — Phase 3 is a replay
harness, not a tuning pass. This is recorded for Phase 4 consideration of
the relevance-scoring approach.

## Known measurement limitations in this run

- `required_source_recall` for `sca_r0_urd_001` and `sca_r0_urd_002` is
  `0.0` for **all three conditions**, including the unconstrained
  full-history condition A. This is an extractor/corpus ceiling, not a
  selection failure: those two cases' required source IDs are referenced
  only in case metadata and are never literally embedded in turn text
  (the same "ID extraction is a stand-in" limitation noted in
  [Phase 1 notes](session_context_assembler_phase_1_notes.md)), so no
  regex-based extractor can recover them regardless of how much history is
  included. A future r1 corpus revision adding an explicit source-artifact
  pool separate from turn text would remove this ceiling.
- `contradiction_awareness_result` for `sca_r0_urd_003` is `"omitted"` for
  all three conditions for the same reason: its required source ID
  (`SRC-SCA-explain-format-discussion`) is never inline-extractable, so the
  all-required-IDs-recovered rule cannot return its authored `"mixed"`
  category even though the decision ID (`DEC-SCA-113`) is recoverable.
  Only the `contradiction_aware_followup` family (`caf_001`-`caf_003`) has
  fully inline-extractable required IDs in r0, so it is currently the only
  family where `contradiction_awareness_result` differentiates conditions.

## Test coverage

`tests/test_session_context_assembler_replay.py` — 24 tests, all passing,
covering: required harness properties (shared corpus/manifest/case hash
across conditions, all three conditions present per case, required output
fields present, condition A is truly unbounded), determinism under a fixed
seed, the `None`-when-nothing-required recall convention, the
contradiction-awareness classification (including its `"omitted"` and
not-applicable branches), all six required gates passing on frozen r0 (plus
a gate-math sanity check that corrupting a label does make the gate fail,
proving the gate is not vacuous), the no-write invariant extended to
`replay.py` and the full replay pipeline, the AST-based no-forbidden-import
invariant extended to `replay.py` and
`tools/run_session_context_assembler_replay.py`, and a check that the CLI
script's only filesystem writes land on the two declared report paths.

Full repo suite: 910 passed, 1 skipped, 6 pre-existing failures unrelated
to this work (`test_hierarchy_lineage.py` x3, `test_retrieval_router*.py`
x2, `test_vfr7_api.py::test_gate_5_audit_log_integrity`) plus one
pre-existing collection error (`test_pit11b_support_scoring.py`, missing
optional `lancedb` dependency) — none introduced by this change.

## Canonical committed-artifact seed

The committed `benchmarks/results/session_context_assembler_r0_replay.{json,md}`
were generated with `--seed 7`. Seed choice has no effect on conditions A
or B (no stochastic component) and affects condition C only in the rare
event of an exact-score tie between candidate episodes; no tie occurred in
this run's case set at either seed 0 or seed 7 (confirmed identical
records in the test suite's determinism checks). Regenerate via:

```text
python tools/run_session_context_assembler_replay.py --seed 7
```

## What this phase does not authorize

No production integration, no authorized-consumer runtime connection, no Engram or
Resolution Engram writes, no retrieval-ranking change, no governance or
authority mutation, no claim that quality is non-inferior to full history,
and no claim that token reduction meets the Phase 4 25% target — r0's
configured budgets never bound in the primary pass, so this run cannot
speak to that target one way or the other. A PASS on the six required
gates above authorizes review of the Phase 4 gates only. See
[ADR 0007](adr/0007-session-context-assembler-shadow-only.md) for the full
boundary.
