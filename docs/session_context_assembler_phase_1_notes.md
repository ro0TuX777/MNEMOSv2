# Session Context Assembler — Phase 1 Notes (Offline Prototype)

Status: **complete**, scoped strictly to the offline-prototype authorization in
[docs/session_context_assembler_spec.md](session_context_assembler_spec.md)
(Phase 1) and [ADR 0007](adr/0007-session-context-assembler-shadow-only.md).
No production route exists. No Phase 3 baseline comparison has run, and
nothing in this document is a benchmark claim.

## What was built

`prototype/session_context_assembler/` — a standalone package with zero
import path into `mnemos/`, `service/`, or `mnemos_sdk/`. It reads the frozen
[R0 corpus](session_context_assembler_corpus_design.md) and produces, per
case, the required 13-field output contract (`session_id`, `task_id`,
`prototype_version`, `seed`, `selected_episode_ids`, `selected_turn_ids`,
`selected_parent_engram_ids`, `selected_source_ids`,
`synthetic_context_labels`, `selection_rationale`, `token_estimate`,
`corpus_manifest_hash`, `case_hash`).

Pipeline: `corpus.py` (read-only, hash-validated load) →
`models.Turn`/`turn_from_dict` → `segmenter.segment_turns` (character-shingle
Jaccard continuity, content-only) → `selector.select_episodes` (relevance
score against `current_task`, seeded tie-break only, greedy under
`token_budget`) → `extractor.extract_ids_from_turn` (regex over turn text for
`DEC-SCA-*` / `SRC-SCA-*`) → `assembler.assemble_context_package` (assembles
the output contract and the `synthetic_context` label block).

`diagnostics.py` is the only module permitted to read `Turn.episode_hint`; it
scores predicted segmentation against the corpus's ground-truth episode
boundaries for human review and is not imported by the selection path.

## Test coverage

`tests/test_session_context_assembler_prototype.py` — 17 tests covering the
10 required categories:

- manifest hash validation (pass + tamper-rejection, file-level and per-case)
- deterministic output for a fixed seed (single case + full corpus)
- `episode_hint` not used as a selection input (segmentation is identical
  with `episode_hint` stripped; an AST-based scan confirms no module outside
  `diagnostics.py`/`models.py` reads the field in code — not just absent
  from a text search, which would have flagged the rule-explaining
  docstrings themselves as false positives)
- `synthetic_context` label coverage (every selected episode is labeled,
  every label is `non_authoritative`/`non_promotable`)
- parent-Engram-ID and source-ID preservation (IDs found in selected turns
  appear in output with no fabrication beyond what extraction found)
- no-write behavior (full corpus run with `Path.write_*`/`os.remove`/
  `os.unlink`/`os.rename` patched to raise; nothing fires)
- no governance/retrieval-ranking mutation (AST scan for real
  `import`/`from` statements referencing `mnemos`, `service`, or
  `mnemos_sdk`; output-contract key set checked against a forbidden-keys
  list such as `trust_score`, `authority_class`, `promotion_status`)
- blocked-artifact exclusion (`Turn.eligible = False` turns contribute zero
  IDs regardless of content)

All 17 pass. Six pre-existing failures elsewhere in the repo's test suite
(`test_hierarchy_lineage.py`, `test_retrieval_router*.py`,
`test_vfr7_api.py::test_gate_5_audit_log_integrity`) are unrelated to this
prototype and were not introduced by this work.

## Known simplifications (not benchmarked, not final)

- **ID extraction is a stand-in.** The R0 corpus has no separate
  `prior_decision_artifacts` / `eligible_source_linked_engrams` pool distinct
  from conversation text, so `extractor.py` regex-matches `DEC-SCA-*` /
  `SRC-SCA-*` tokens directly out of turn content. Several case families
  (e.g. `topic_shift_and_return`) intentionally reference required decision
  or source IDs only in case metadata, not inline in turn text — for those
  cases the extractor correctly returns no IDs. This is expected corpus
  design (see corpus design doc), not an extraction bug, and the
  preservation tests validate provenance-plumbing correctness on whatever
  the extractor finds, not recall against every corpus-labeled ID.
- **Segmentation threshold is untuned.** `DEFAULT_THRESHOLD = 0.03` over
  4-character shingles was picked to produce plausible behavior on a manual
  smoke run, not fit to the corpus's `episode_hint` boundaries — doing so
  would violate the Phase 1 boundary against using ground truth as a
  selection input. `diagnostics.pairwise_boundary_agreement` exists so a
  human reviewer can see how far off the untuned segmenter is from the
  corpus's authored boundaries, without that comparison feeding back into
  selection.
- **Relevance scoring is lexical only** (shingle-Jaccard against
  `current_task`), with no semantic/embedding component. Sufficient to
  demonstrate non-recency-biased ordering on the topic-shift-and-return
  smoke case, not validated as an accuracy claim.

## Manual verification performed

`assemble_context_package` was run against `sca_r0_tsr_001` (a
topic-shift-and-return case) with `seed=42`: the episode containing the
resumed reranker discussion scored highest relevance (0.6957) and was
selected ahead of intervening, more-recent billing-topic episodes —
consistent with the selector resisting pure recency bias on this case, not a
benchmarked accuracy result.

## What this phase does not authorize

No production integration, no authorized-consumer runtime connection, no Engram or
Resolution Engram writes, no retrieval-ranking change, no governance or
authority mutation. Phase 3 (baseline replay against full-history and
sliding-window baselines) and Phase 4 (evaluation-gate scoring) have not
run. See [ADR 0007](adr/0007-session-context-assembler-shadow-only.md) for
the full boundary.
