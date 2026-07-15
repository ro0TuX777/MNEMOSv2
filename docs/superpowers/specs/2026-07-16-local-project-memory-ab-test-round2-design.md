# Local Project Memory A/B Test Round 2 Design

Date: 2026-07-16
Status: Proposed

## Goal

Run a tighter second A/B experiment to evaluate whether read-only MNEMOS
project memory reduces time and token cost to a correct fix for the same seeded
`path_prefix` retrieval bug, while preserving source-backed evidence discipline
and stale-memory boundaries.

This round is explicitly designed to remove the main contamination from round
one: the arm-visible repository must not contain the experiment design or plan
documents that described the seeded bug pattern.

## Scope

This round covers:

- reusing the same one-line `path_prefix` bug for comparability with round one;
- creating sanitized temporary arm worktrees where round-one and round-two
  experiment design/plan artifacts are absent from arm-visible repo context;
- building Arm B's packet from the same sanitized buggy snapshot;
- dispatching one arm without MNEMOS and one arm with read-only MNEMOS project
  memory;
- instrumenting wall-clock timing, command counts, and token usage;
- recording exact token totals when exposed by the harness, with a documented
  estimation fallback when exact usage is unavailable.

This round does not cover:

- merging any experiment branch back into `main`;
- changing default MNEMOS collections or runtime routes;
- introducing a VS Code extension;
- changing the packet contract beyond scope hygiene and measurement receipts;
- solving a different bug class.

## Round-One Lessons Applied

Round one established that the engineering flow worked: isolated worktrees,
seeded semantic fault, scoped packet build, read-only MCP boundary, human
approval gates, and stale-packet handling all functioned as intended.

Round one did not yet validate the MNEMOS-assisted SDLC hypothesis because:

- Arm B did not materially rely on packet-backed evidence;
- the repository exposed experiment design material that described the seeded
  bug pattern;
- elapsed time and token usage were not instrumented.

Round two exists to correct those weaknesses while keeping the bug itself
constant.

## Isolation Model

The controller may retain experiment design and planning artifacts in the main
repository, but the arm-visible worktrees used for the second round must not
contain those files. The A/B participants must not be able to discover the
seeded bug by reading design or plan docs from inside their own working repo.

Sanitization applies to both arms, not only Arm B, so the baseline remains
fair. Arm B's packet scope must also exclude those experiment files explicitly.

The second-round arm worktrees remain temporary and isolated under `%TEMP%`.
`G:\MNEMOS\main` remains untouched throughout.

## Bug Seed

The seeded fault is the same one-line semantic inversion used in round one:

- target area: `ProjectMemoryIndex.search` in
  `prototype/local_project_memory_r0/retrieval.py`
- bug class: invert the `path_prefix` eligibility gate so matching paths are
  excluded instead of admitted
- expected failure: `tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates`
  fails because `hits` becomes empty

Reusing the same bug preserves comparability between the first and second
rounds while allowing the cleaner isolation model to be the primary changed
variable.

## Arm Setup

Arm A:

- works from the sanitized buggy worktree only;
- receives the shared prompt and failing test command;
- has no project-memory packet or MCP sidecar.

Arm B:

- works from an equivalent sanitized buggy worktree from the same commit;
- receives the same prompt and same failing test command;
- gets a read-only `mnemos_project` packet sidecar built from that exact
  sanitized buggy snapshot;
- must treat the packet as stale immediately after any code mutation unless a
  new packet is explicitly rebuilt with approval.

Both arms stop after diagnosis and wait for approval before editing.

## Common Task Prompt

Both arms receive the same lightly guided task prompt:

```text
Scoped project-memory retrieval is behaving incorrectly. Use the failing test
command below to diagnose the cause and fix it. Keep the change minimal, verify
the result, and summarize the root cause and fix. Do not use internet resources.
```

Both arms receive the same exact failing test command:

```text
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q
```

Arm B receives one additional instruction:

```text
You may use the configured read-only mnemos_project packet/MCP for source-backed
evidence. After any code mutation, do not treat the packet as fresh unless it
has been explicitly rebuilt.
```

## Measurement Model

This round makes measurement a first-class outcome.

Per arm, record:

- wall-clock start time;
- diagnosis-complete time;
- fix approval time;
- fix-complete time;
- exact commands executed;
- count of commands and count of test commands;
- correctness outcome;
- evidence-discipline observations.

### Token Accounting

Primary path:

- record exact prompt tokens, completion tokens, and total tokens per arm if
  the execution harness exposes them.

Fallback path:

- estimate token usage from the exact controller-dispatched prompt text, any
  controller follow-up messages, and the arm's returned reports using one
  consistent tokenizer/model assumption across both arms.

Estimated token counts must be labeled as estimates and must not be presented
as exact usage. Even so, they remain useful for relative comparison when both
arms are measured by the same method.

## Success Criteria

Primary success criterion:

- Arm B reaches a correct fix with lower total time and/or lower token cost
  than Arm A.

Secondary success criteria:

- Arm B provides stronger source-backed evidence quality;
- Arm B preserves stale-memory boundary discipline after mutation;
- the second-round setup removes the design-doc contamination observed in
  round one.

The round is still informative if Arm B does not win all dimensions. The goal
is to measure whether the MNEMOS-assisted path yields meaningful SDLC benefit,
not to force a predetermined outcome.

## Execution Flow

1. Create a fresh temporary seed worktree from local `main`.
2. Prepare sanitized seed and arm-visible temporary worktrees where the
   experiment design/plan artifacts are absent.
3. Introduce the same one-line `path_prefix` bug in the sanitized seed and
   commit it.
4. Create sanitized Arm A and Arm B worktrees from that exact buggy commit.
5. Build Arm B's packet from the sanitized Arm B worktree with explicit scope
   that excludes experiment docs.
6. Dispatch both arms with the same prompt and same failing test command.
7. Stop both arms at diagnosis for approval.
8. Approve both fixes and allow both arms to verify.
9. Require Arm B to acknowledge packet staleness after mutation.
10. Write a second-round memo that compares round-two outcomes against round
    one.

## Approval Checkpoints

Human approval is required:

1. before seeding the bug in the sanitized seed worktree;
2. after each arm's diagnosis and before either fix is applied;
3. before any optional rebuild of Arm B's packet after mutation;
4. before cleanup of temporary worktrees, packet files, or receipts.

These gates preserve the same authority boundary established for the local
project-memory workflow: MNEMOS supplies evidence, not autonomous permission to
mutate code.

## Artifacts

Retain:

- this round-two design spec;
- the round-two implementation plan;
- sanitized temporary seed and arm worktree paths;
- Arm B packet path and snapshot ID;
- per-arm timing and command receipts;
- exact token totals if available, otherwise the token estimation worksheet;
- the round-two A/B memo with explicit comparison to round one.

Sensitive values, tokens, credentials, or unrelated local state must not appear
in any retained artifact.

## Constraints

The round must preserve all existing local project-memory boundaries:

- no mutation of the main checkout;
- no merge back to `main`;
- no runtime route changes;
- no default MNEMOS collection changes;
- no Research Intake dependency for code ingestion;
- no code mutation outside temporary experiment worktrees;
- no whole-repo default scope;
- no VS Code extension work;
- only Python and Markdown memory extraction;
- no internet resources for either arm.

## Expected Outcome

At the end of round two, we should be able to answer more rigorously than round
one whether read-only scoped MNEMOS project memory lowers SDLC cost for this
kind of debugging task, especially in time and token consumption, without
weakening boundary discipline.
