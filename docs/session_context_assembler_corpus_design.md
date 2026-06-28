# Session Context Assembler — Phase 2 Corpus Design (R0)

Date: 2026-06-21

Status: **Frozen.** Corpus design only — no runtime or prototype code
changes. This document and the corpus it describes satisfy Phase 2 of
`docs/session_context_assembler_spec.md`. Phase 1 (offline prototype) and
Phase 3 (A/B/C baseline replay) remain not started.

## Scope note

Every case in this corpus is **illustrative and fixture-local**: synthetic
sessions written to exercise specific selection behaviors, not transcripts
of real conversations and not derived from real tenants or real Engrams. All
IDs use the `SCA-R0-*` / `DEC-SCA-*` / `SRC-SCA-*` fixture namespace. This
corpus produces no measured benchmark results by itself — it only defines
inputs and expected-recall labels for the Phase 3 replay that has not yet
run. Do not cite this corpus as evidence of assembler performance.

## Deliverables

| Deliverable | Location |
|---|---|
| 20-30 frozen long-session replay cases | `benchmarks/truthsets/session_context_assembler_r0.json` |
| Case-family coverage matrix | this document, below |
| Required prior-decision IDs | per-case `required_prior_decision_ids` field |
| Required source IDs | per-case `required_source_ids` field |
| Known irrelevant-history labels | per-case `known_irrelevant_history_turn_ids` field |
| Expected context-budget field | per-case `expected_context_budget` field |
| Corpus manifest hashes | `benchmarks/truthsets/session_context_assembler_r0.manifest.json` |

24 cases were built (within the 20-30 target), 3 per case family, covering
all 8 families named in `docs/session_context_assembler_spec.md` Phase 2.

## Case-family coverage matrix

| Case family | Case IDs | Count | What it stresses |
|---|---|---|---|
| `prior_architectural_decision_recall` | `sca_r0_pad_001..003` | 3 | Recalling a standing decision from an earlier, unrelated-seeming episode. |
| `contradiction_aware_followup` | `sca_r0_caf_001..003` | 3 | Preserving both sides of a contradiction (and the resolution status — resolved or explicitly unresolved) rather than collapsing to one claim. |
| `source_specific_followup` | `sca_r0_ssf_001..003` | 3 | Resolving a vague natural-language reference ("the audit doc") to the correct `source_id` without pulling in unrelated episodes. |
| `long_running_implementation_discussion` | `sca_r0_lrid_001..003` | 3 | Retaining a dense, single-topic technical thread (including still-open edge cases) across many turns, with and without distractor turns. |
| `topic_shift_and_return` | `sca_r0_tsr_001..003` | 3 | Resuming an earlier topic after a substantial, even high-salience, intervening episode without that episode bleeding into the resumed context. |
| `stale_session_material_exclusion` | `sca_r0_sse_001..003` | 3 | Excluding superseded facts/decisions entirely, not merely deprioritizing them. |
| `multiple_similar_prior_decisions` | `sca_r0_mspd_001..003` | 3 | Picking the final decision in a revision chain on the same parameter while keeping the chain traceable in selection rationale. |
| `unresolved_vs_resolved_decision_distinction` | `sca_r0_urd_001..003` | 3 | Not fabricating a resolved decision for a topic that was left open, including mixed cases where one sub-decision resolved and a related one didn't. |

Total: 24 cases, 8 families, 3 cases each.

## Per-case required fields

Every case includes the fields required by the spec:

- `session_id`, `task_id`, `current_task`
- `conversation_history` (turn-level: `turn_id`, `speaker`, `episode_hint`,
  `content`)
- `required_prior_decision_ids`
- `required_source_ids`
- `known_irrelevant_history_turn_ids`
- `expected_context_budget` (illustrative token estimate, not measured)
- `notes` (human-readable statement of what correct selection looks like,
  for Phase 5 human reviewers)

`episode_hint` labels are provided as ground-truth episode boundaries for
scoring the assembler's own clustering output against — they are not an
input the assembler is required to use, since Phase 1 must independently
derive episode boundaries from `conversation_history` alone.

## Why this ordering (corpus before prototype)

Per the user's stated rationale: without the corpus and its expected
decision/source IDs frozen first, a prototype implementation could shape
itself around its own behavior, scoring well against a benchmark that
implicitly mirrors its own assumptions rather than testing them
independently. Freezing R0 first means the Phase 1 prototype is built and
the Phase 3 A/B/C comparison is run against labels that already exist and
cannot be retroactively adjusted to fit the implementation.

## Freeze and versioning rule

`session_context_assembler_r0.json` is frozen as of 2026-06-21
(`file_sha256` recorded in the manifest). Any future correction or
expansion must produce a new corpus version (`r1`, `r2`, ...) with its own
manifest rather than silently editing R0's frozen cases — mirroring the
versioning discipline already used for EBIR truthsets
(`benchmarks/truthsets/ebir_r1_adversarial.json`).

## Next step

Per `docs/session_context_assembler_spec.md` sequencing, the next
authorization point is the Phase 1 offline prototype
(`prototype/session_context_assembler/`) implemented against this frozen
corpus, followed by the Phase 3 A/B/C replay harness. Neither is authorized
by this document.
