# Associative Routing E2 Commit Manifest

This manifest identifies the files included to close Associative Routing E2 as
an experimental, opt-in candidate-expansion capability.

Authorization:

```text
ASSOCIATIVE_ROUTING_E2_CLOSEOUT_AUTHORIZED
EXPERIMENTAL_OPT_IN_ONLY
DEFAULT_RETRIEVAL_UNCHANGED
NO_E3_WORK
NO_RETRIEVAL_TUNING
NO_CORPUS_EXPANSION
NO_CUE_OR_TAG_REGISTRY_CHANGE
NO_GOVERNANCE_OR_AUTHORITY_CHANGE
NO_MCP_ENVIRONMENT_CHANGE
```

## Included Files

| File | Type | Purpose |
| --- | --- | --- |
| `service/app.py` | code | Threads the `associative_candidate_expansion` request flag through `/v1/mnemos/search`; invokes expansion only after normal retrieval, governance, and abstention decisions; appends governed expansion candidates without suppressing or re-ranking normal results. Also carries E1 shadow wiring required by the focused regression suite. |
| `mnemos/retrieval/associative_expansion/__init__.py` | code | Public E2 module exports for the candidate-expansion engine, request flag, kill switch, fixtures path, and bounds. |
| `mnemos/retrieval/associative_expansion/config.py` | code | E2 request flag, global kill switch, fixture directory, and conservative expansion bounds. |
| `mnemos/retrieval/associative_expansion/engine.py` | code | Read-only, fail-closed, bounded E2 candidate-expansion engine. Resolves source-linked candidates through existing retrieval and never writes durable state. |
| `mnemos/retrieval/associative_expansion/fixtures/cue_registry.json` | fixture | Frozen E2 cue registry. No tuning or expansion authorized during closeout. |
| `mnemos/retrieval/associative_expansion/fixtures/tag_registry.json` | fixture | Frozen E2 tag registry. No tuning or expansion authorized during closeout. |
| `mnemos/retrieval/associative_expansion/fixtures/source_index.json` | fixture | Frozen E2 source index for candidate source linkage. |
| `tests/test_associative_routing_e2_expansion.py` | test | Focused E2 tests for flag-off identity, kill switch, no normal suppression, bounded and deduplicated additions, origin labeling, source lineage, governance rejection, abstention preservation, no authority-field injection, and no durable write. |
| `tests/test_service_hybrid_api.py` | test | Updates existing service API fakes to accept the new associative routing keyword arguments without changing hybrid-search expectations. |
| `docs/associative_routing_e2_design_note.md` | documentation | E2 design, comparison conditions, environment finding, evidence summary, limitations, and decision note. |
| `docs/associative_routing_e2_closeout.md` | documentation | Final closeout record required by the E2 closeout task. |
| `docs/associative_routing_e2_commit_manifest.md` | documentation | This included-file manifest. |
| `docs/experiments/associative_routing_e2_verification_pack.json` | evidence | Frozen 22-query verification pack used to scope E2 claims. It is disclosed as not independently authored and is not a broad-superiority artifact. |
| `docs/experiments/associative_routing_e2_development_pack.json` | evidence | Informal 33-query development pack retained for reproducibility of the design-note context; not used as final claim evidence. |
| `benchmarks/results/associative_routing_e2_live_comparison_run_001.json` | evidence | Recorded 22-query E2 comparison artifact: 2 correct-and-needed additions, 0 observed regressions in the frozen evaluation, and default-normal-result preservation. |

## E1 Prerequisite Files Included

The current `service/app.py` integration imports the E1 shadow adapter and the
required validation command includes `tests/test_associative_routing_e1_shadow.py`.
These files are included only as a prerequisite for reproducible E2 validation
and import stability; they do not expand E2 claims.

| File | Type | Purpose |
| --- | --- | --- |
| `mnemos/retrieval/associative_shadow/__init__.py` | code | Public E1 shadow module exports used by service import and E1 regression tests. |
| `mnemos/retrieval/associative_shadow/config.py` | code | E1 shadow request flag, kill switch, and fixture path. |
| `mnemos/retrieval/associative_shadow/adapter.py` | code | Read-only E1 shadow adapter used by the service import and E1 regression tests. |
| `mnemos/retrieval/associative_shadow/fixtures/cue_registry.json` | fixture | Frozen E1 cue registry required by E1 regression tests. |
| `mnemos/retrieval/associative_shadow/fixtures/tag_registry.json` | fixture | Frozen E1 tag registry required by E1 regression tests. |
| `mnemos/retrieval/associative_shadow/fixtures/source_index.json` | fixture | Frozen E1 source index required by E1 regression tests. |
| `tests/test_associative_routing_e1_shadow.py` | test | Required focused regression test named by the closeout task. |

## Excluded Files

The following dirty or untracked files are deliberately excluded from the E2
commit:

- `docs/experiments/demo_corpus/**` - user-supplied recording/demo PDFs; not
  E2 evidence or code.
- `tempwhy.md` - local working note.
- `tools/run_associative_routing_e1_comparison.py` - E1 comparison runner, not
  required to reproduce E2 candidate expansion or the E2 frozen artifact.
- `docs/associative_routing_e1_design_note.md` - E1 design note, reviewed as
  context but not required for the E2 committed evidence set.
- `docs/experiments/associative_routing_e1_comparison_pack.json` - E1
  comparison pack, reviewed as context but not included in E2 evidence.
- `docs/experiments/associative_routing_e1_fresh_verification_pack.json` - E1
  fresh-verification pack, reviewed as context but not included in E2 evidence.
- `benchmarks/results/associative_routing_e1_comparison_result_template.json`
  and E1 comparison/fresh-verification result files - E1 evidence, not E2
  closeout evidence.
- Any MCP/MSF dependency changes, global-environment artifacts, cache files,
  temporary logs, unrelated benchmarks, or generated local files not needed for
  replay.

## Claim Boundary

The included files support only this E2 claim:

```text
Associative Routing E2 is an experimental, opt-in, kill-switch-controlled
candidate-expansion path. It appends bounded, source-linked, origin-labeled
candidates after normal retrieval and governance. Default retrieval is
unchanged. The frozen 22-query comparison recorded 2 correct-and-needed
additions and 0 observed regressions on a small curated corpus. The evidence
does not support broad retrieval superiority, production readiness, or
authorization/security claims.
```
