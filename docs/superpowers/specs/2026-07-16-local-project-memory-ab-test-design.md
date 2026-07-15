# Local Project Memory A/B Experiment Design

Date: 2026-07-16
Status: Proposed

## Goal

Run a controlled A/B experiment inside this Codex session to evaluate whether a
frontier coding agent performs better on a scoped MNEMOS debugging task when it
has access to a read-only local project-memory packet, compared with an
otherwise equivalent agent that does not.

The experiment should reflect the intended SDLC use case:

- the target project is the MNEMOS repository;
- the frontier agent remains responsible for diagnosis, editing, and test
  execution;
- MNEMOS project memory contributes only source-backed Python and Markdown
  evidence;
- all code mutation remains local to isolated temporary worktrees;
- the main checkout stays untouched.

## Scope

This experiment covers:

- seeding one narrow semantic bug into an isolated temporary worktree;
- creating two isolated child worktrees from the same buggy commit;
- giving both arms the same task prompt and failing test command;
- configuring only Arm B with the read-only `mnemos_project` MCP sidecar backed
  by a packet built from the exact buggy snapshot;
- collecting comparable debugging, fix, and verification outcomes.

This experiment does not cover:

- any merge back into `main`;
- any change to default MNEMOS collections, runtime routes, or Research Intake;
- any VS Code extension work;
- whole-repository ingestion defaults;
- internet-assisted debugging;
- long-running benchmark automation beyond this controlled trial.

## Experiment Structure

The experiment uses three temporary worktrees under `%TEMP%`:

1. a seed worktree created from local `main`;
2. an Arm A worktree created from the seed's buggy commit;
3. an Arm B worktree created from the same buggy commit.

The seed worktree is used only to introduce and commit the fault on an
experiment-only branch. Arm A and Arm B each receive an independent worktree so
they can diagnose and fix without sharing edits or runtime state.

The main repository at `G:\MNEMOS` remains unchanged throughout. No experiment
branch is merged, and all temporary packet outputs remain outside the repository.

## Bug Seed

The seeded fault must be:

- syntactically valid;
- semantically narrow;
- fixable with a small patch;
- capable of preserving general project operability;
- directly relevant to the local project-memory use case.

The selected bug class is a one-line semantic fault in
`ProjectMemoryIndex.search` affecting `path_prefix` filtering. The bug should
cause scoped retrieval behavior to become incorrect while preserving parser
validity and allowing both arms to run tests and inspect code normally.

This is preferred over a syntax error because syntax breakage would mostly test
packet completeness, parser failure, or hard abstention rather than whether
source-backed memory improves debugging and repair.

## Arm Setup

Arm A:

- receives the buggy worktree only;
- has no project-memory MCP configured;
- may use its normal local code and test tools within its own worktree.

Arm B:

- receives an equivalent buggy worktree from the same commit;
- gets the read-only `mnemos_project` MCP configured against a packet built from
  that exact buggy snapshot;
- may use both its normal local code and test tools and the project-memory MCP
  for source-backed evidence;
- must treat the packet as stale after any code mutation unless a new packet is
  explicitly rebuilt.

Both arms must be run by isolated sub-agents in this session to keep the
comparison controlled while avoiding cross-contamination of thought process and
tool history.

## Common Task Prompt

Both arms receive the same lightly guided task prompt:

```text
Scoped project-memory retrieval is behaving incorrectly. Use the failing test
command below to diagnose the cause and fix it. Keep the change minimal, verify
the result, and summarize the root cause and fix. Do not use internet resources.
```

Both arms also receive the same explicit failing test command. The test command
is provided up front so the experiment measures debugging and repair rather than
test-discovery luck.

Arm B receives one additional instruction:

```text
You may use the configured read-only project-memory MCP for source-backed
evidence. After any code mutation, do not treat the packet as fresh unless it
has been explicitly rebuilt.
```

## Approval Checkpoints

Human approval is required at the following points:

1. before the seed bug is introduced;
2. after each arm produces its diagnosis and before either arm applies a fix;
3. before any optional rebuild of Arm B's packet after mutation;
4. before any cleanup that removes temporary worktrees, packets, or receipts.

These checkpoints preserve the authority boundary established for the local
project-memory workflow: MNEMOS provides evidence, not autonomous permission to
mutate code or advance state.

## Execution Flow

1. Create the seed worktree from local `main` under `%TEMP%`.
2. Apply the one-line semantic bug in the seed worktree and commit it on an
   experiment-only branch.
3. Create Arm A and Arm B worktrees from that exact buggy commit.
4. Build one packet from Arm B's buggy snapshot and store it under `%TEMP%`.
5. Configure the read-only `mnemos_project` MCP sidecar for Arm B using the
   packet path, Arm B project root, and repo ID.
6. Launch two isolated sub-agents in this session, one for each arm.
7. Give both arms the same prompt and same failing test command.
8. Require both arms to stop after diagnosis and wait for approval before
   changing code.
9. After approval, allow each arm to edit only its own worktree, rerun focused
   verification, and report the result.
10. Record each arm's patch, command trail, test outcomes, elapsed time, and
    root-cause summary.
11. Leave `main` untouched and do not merge experiment branches.

## Scoring

The comparison should evaluate the following dimensions for both arms:

- correctness: the seeded failing test passes and nearby focused verification
  does not reveal an obvious regression;
- diagnosis quality: the arm identifies the actual root cause instead of merely
  producing a test-green patch;
- efficiency: elapsed time, number of search and test iterations, and time to
  first relevant file;
- evidence discipline: claims are tied to concrete files, spans, or test output;
- patch quality: the fix is minimal, focused, and avoids unrelated edits;
- boundary discipline: Arm B correctly recognizes and reports that the packet is
  stale after mutation unless rebuilt.

The experiment result does not require Arm B to outperform Arm A on every
metric. It is sufficient to determine whether the project-memory workflow adds
meaningful debugging value without violating the established authority boundary.

## Artifacts

Artifacts to retain from the experiment:

- this design specification;
- temporary seed, Arm A, and Arm B worktree paths under `%TEMP%`;
- the Arm B packet or report file under `%TEMP%`;
- a compact A/B results memo containing outcome, timings, diagnosis quality,
  patch quality, and stale-memory behavior;
- optional failing and passing test receipts if they materially support the
  evaluation.

Sensitive values, tokens, credentials, and unrelated local state must not be
captured in any experiment artifact.

## Constraints

The experiment must preserve the following constraints:

- no mutation of the main checkout;
- no merge back to `main` during the trial;
- no runtime route changes;
- no default MNEMOS collection changes;
- no Research Intake dependency for code ingestion;
- no code mutation outside the temporary experiment worktrees;
- no whole-repo default scope;
- no VS Code extension work;
- only Python and Markdown memory extraction;
- no internet resources for either arm.

## Success Criteria

The design is successful if it enables a controlled run where:

- both arms start from the exact same buggy code state;
- both arms receive the same task framing and failing test command;
- only Arm B has access to source-backed project memory;
- both arms can diagnose and apply a fix after approval;
- `main` remains untouched;
- the result produces a usable comparison memo about whether read-only project
  memory improved debugging and repair discipline for this SDLC task.
