# Local Project Memory A/B Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a controlled two-arm debugging experiment that compares a frontier coding agent with and without read-only local MNEMOS project memory on the same seeded scoped-retrieval bug.

**Architecture:** Create one temporary seed worktree from `main`, introduce a one-line semantic bug in the local project-memory retrieval path, and branch two isolated experiment worktrees from that exact buggy commit. Arm A debugs and fixes without project-memory MCP, while Arm B gets the same worktree plus a read-only `mnemos_project` packet sidecar built from the buggy snapshot.

**Tech Stack:** Git worktrees, PowerShell, Python, pytest, `tools/build_local_project_memory_packet.py`, `mcp_servers/mnemos_project/server.py`, Codex isolated sub-agents.

## Global Constraints

- no mutation of the main checkout
- no merge back to `main` during the trial
- no runtime route changes
- no default MNEMOS collection changes
- no Research Intake dependency for code ingestion
- no code mutation outside the temporary experiment worktrees
- no whole-repo default scope
- no VS Code extension work
- only Python and Markdown memory extraction
- no internet resources for either arm
- require human approval before seeding the bug
- require human approval after each arm's diagnosis and before either arm applies its fix
- require human approval before any optional rebuild of Arm B's packet after mutation
- require human approval before cleanup of temporary experiment artifacts

---

### Task 1: Establish Isolated Experiment Workspace

**Files:**
- Create: `%TEMP%\\mnemos-ab-seed\\` worktree
- Reserve for later creation: `%TEMP%\\mnemos-ab-arm-a\\` worktree
- Reserve for later creation: `%TEMP%\\mnemos-ab-arm-b\\` worktree
- Modify: none in `G:\\MNEMOS`
- Test: `git worktree list`

**Interfaces:**
- Consumes: local `main` checkout at `G:\\MNEMOS`
- Produces: one isolated seed worktree root and one experiment branch base

- [ ] **Step 1: Confirm the base checkout is safe to branch from**

Run:

```powershell
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

Expected:

```text
Only unrelated untracked paths such as benchmarks/reports/ and logs/ may appear.
Current branch is main.
HEAD resolves successfully.
```

- [ ] **Step 2: Remove any stale experiment worktrees only if they already exist and are known to belong to this A/B task**

Run:

```powershell
$seed = Join-Path $env:TEMP 'mnemos-ab-seed'
$armA = Join-Path $env:TEMP 'mnemos-ab-arm-a'
$armB = Join-Path $env:TEMP 'mnemos-ab-arm-b'
git worktree list
```

Expected:

```text
Existing worktrees, if any, are visible before cleanup.
Do not remove an unknown worktree without explicit user confirmation.
```

- [ ] **Step 3: Create the seed worktree from local `main`**

Run:

```powershell
$seed = Join-Path $env:TEMP 'mnemos-ab-seed'
git worktree add --detach $seed main
git -C $seed switch -c exp/local-project-memory-ab-seed
```

Expected:

```text
A detached worktree is created at %TEMP%\mnemos-ab-seed and switched to exp/local-project-memory-ab-seed.
```

- [ ] **Step 4: Record the seed worktree identity receipt**

Run:

```powershell
git -C $seed status --short
git -C $seed rev-parse --abbrev-ref HEAD
git -C $seed rev-parse HEAD
```

Expected:

```text
Working tree is clean.
Branch is exp/local-project-memory-ab-seed.
Commit matches the chosen local main base.
```

- [ ] **Step 5: Pause for approval before introducing the seeded bug**

Report:

```text
Seed worktree created and isolated under %TEMP%.
Main checkout remains untouched.
Await approval before seeding the semantic bug.
```

### Task 2: Seed the Scoped Retrieval Bug and Lock a Failing Test

**Files:**
- Modify in seed worktree: `prototype/local_project_memory_r0/retrieval.py`
- Modify in seed worktree: `tests/test_local_project_memory_retrieval.py`
- Test: `tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates`

**Interfaces:**
- Consumes: seed worktree from Task 1
- Produces: one committed buggy snapshot and one deterministic failing test command

- [ ] **Step 1: Add or confirm the scoped retrieval test that should fail under the seeded bug**

Code target in `tests/test_local_project_memory_retrieval.py`:

```python
def test_filters_are_eligibility_gates(packet) -> None:
    _, value = packet
    hits = ProjectMemoryIndex(value).search(
        "snapshot",
        path_prefix="docs/",
        artifact_types=("markdown_section",),
    )
    assert hits
    assert all(hit.artifact.file_path.startswith("docs/") for hit in hits)
    assert all(hit.artifact.artifact_type == "markdown_section" for hit in hits)
```

Expected:

```text
The test asserts that path and type filters behave as hard eligibility gates.
```

- [ ] **Step 2: Introduce the one-line semantic bug in `prototype/local_project_memory_r0/retrieval.py`**

Change this code in the seed worktree:

```python
        for artifact in self.packet.artifacts:
            if prefix and not (
                artifact.file_path == prefix
                or artifact.file_path.startswith(prefix + "/")
            ):
                continue
```

To this buggy variant:

```python
        for artifact in self.packet.artifacts:
            if prefix and (
                artifact.file_path == prefix
                or artifact.file_path.startswith(prefix + "/")
            ):
                continue
```

Expected:

```text
Scoped results are incorrectly excluded, producing a focused semantic failure while preserving syntax and general operability.
```

- [ ] **Step 3: Run the targeted test to verify the seeded bug fails**

Run:

```powershell
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q
```

Expected:

```text
FAIL because no eligible docs-scoped markdown hits remain after the inverted prefix gate.
```

- [ ] **Step 4: Commit the buggy experiment seed**

Run:

```powershell
git -C $seed add prototype/local_project_memory_r0/retrieval.py tests/test_local_project_memory_retrieval.py
git -C $seed commit -m "test: seed local project memory ab bug"
git -C $seed rev-parse HEAD
```

Expected:

```text
One experiment-only commit is created and its SHA becomes the shared base for both arms.
```

- [ ] **Step 5: Capture the failing test command to reuse verbatim in both arms**

Command:

```text
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q
```

Expected:

```text
This exact command is handed to both arms to remove test-discovery luck from the comparison.
```

### Task 3: Create Arm Worktrees and Build the Arm B Packet

**Files:**
- Create: `%TEMP%\\mnemos-ab-arm-a\\` worktree
- Create: `%TEMP%\\mnemos-ab-arm-b\\` worktree
- Create: `%TEMP%\\mnemos-project-memory-ab-<timestamp>.md`
- Modify: none in `G:\\MNEMOS`
- Test: `tools/build_local_project_memory_packet.py`, `tools/verify_mnemos_local_stack.py` only if connectivity validation is needed

**Interfaces:**
- Consumes: buggy seed commit SHA from Task 2
- Produces: two identical buggy arm worktrees and one Arm B packet path

- [ ] **Step 1: Create Arm A from the buggy seed commit**

Run:

```powershell
$armA = Join-Path $env:TEMP 'mnemos-ab-arm-a'
$buggy = git -C $seed rev-parse HEAD
git worktree add --detach $armA $buggy
git -C $armA switch -c exp/local-project-memory-ab-arm-a
```

Expected:

```text
Arm A starts from the exact seeded buggy commit on its own experiment branch.
```

- [ ] **Step 2: Create Arm B from the same buggy seed commit**

Run:

```powershell
$armB = Join-Path $env:TEMP 'mnemos-ab-arm-b'
git worktree add --detach $armB $buggy
git -C $armB switch -c exp/local-project-memory-ab-arm-b
```

Expected:

```text
Arm B starts from the same commit as Arm A on a separate branch.
```

- [ ] **Step 3: Verify both arms match the same buggy snapshot**

Run:

```powershell
git -C $armA rev-parse HEAD
git -C $armB rev-parse HEAD
git -C $armA status --short
git -C $armB status --short
```

Expected:

```text
Both HEAD SHAs match the buggy seed commit and both worktrees are clean.
```

- [ ] **Step 4: Build the Arm B packet from the buggy snapshot with explicit scope**

Run:

```powershell
$packet = Join-Path $env:TEMP ("mnemos-project-memory-ab-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".md")
python tools/build_local_project_memory_packet.py `
  --project-root $armB `
  --repo-id mnemos `
  --scope-root mnemos `
  --scope-root service `
  --scope-root mnemos_sdk `
  --scope-root mcp_servers/mnemos `
  --scope-root mcp_servers/mnemos_project `
  --scope-root prototype/local_project_memory_r0 `
  --scope-root tools `
  --scope-root tests `
  --scope-file README.md `
  --scope-file docs/architecture.md `
  --scope-file docs/dependency_map.md `
  --scope-file docs/experiments/python_markdown_structured_project_memory_r0_spec.md `
  --scope-file docs/experiments/local_project_memory_packet_mcp_sidecar_r0_spec.md `
  --scope-file docs/experiments/local_project_memory_packet_mcp_sidecar_r0_trial.md `
  --output $packet
```

Expected:

```text
The packet is written under %TEMP% and bound to Arm B's exact buggy snapshot.
```

- [ ] **Step 5: Smoke-check the Arm B packet before agent launch**

Run:

```powershell
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q
$env:MNEMOS_PROJECT_PACKET = $packet
$env:MNEMOS_PROJECT_ROOT = $armB
$env:MNEMOS_PROJECT_REPO_ID = 'mnemos'
python mcp_servers/mnemos_project/server.py
```

Expected:

```text
The failing test still fails in Arm B before any fix.
The MCP process starts on stdio when launched with its required environment.
Stop it after verifying startup.
```

### Task 4: Dispatch the Two Experiment Arms

**Files:**
- Modify: none in `G:\\MNEMOS`
- Uses: Arm A and Arm B temporary worktrees
- Test: both arms run the same failing test command independently

**Interfaces:**
- Consumes: Arm A worktree, Arm B worktree, Arm B packet path, shared failing test command
- Produces: one diagnosis report from each arm before any code mutation

- [ ] **Step 1: Launch Arm A as an isolated sub-agent with no project-memory MCP**

Arm A task payload:

```text
Work only in the Arm A worktree under %TEMP%.
Scoped project-memory retrieval is behaving incorrectly. Use the failing test
command below to diagnose the cause and fix it. Keep the change minimal, verify
the result, and summarize the root cause and fix. Do not use internet resources.

Failing test command:
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q

Stop after diagnosis and before applying any code change. Report:
1. failing symptom,
2. suspected root cause,
3. file you would edit,
4. exact test command(s) you ran.
```

Expected:

```text
Arm A returns a diagnosis only and waits for approval before editing.
```

- [ ] **Step 2: Launch Arm B as an isolated sub-agent with the read-only project-memory packet**

Arm B task payload:

```text
Work only in the Arm B worktree under %TEMP%.
Scoped project-memory retrieval is behaving incorrectly. Use the failing test
command below to diagnose the cause and fix it. Keep the change minimal, verify
the result, and summarize the root cause and fix. Do not use internet resources.

Failing test command:
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q

You may use the configured read-only mnemos_project MCP for source-backed
evidence. After any code mutation, do not treat the packet as fresh unless it
has been explicitly rebuilt.

Stop after diagnosis and before applying any code change. Report:
1. failing symptom,
2. suspected root cause,
3. file you would edit,
4. exact test command(s) you ran,
5. any project-memory evidence used, including file path or source span.
```

Expected:

```text
Arm B returns a diagnosis only, cites any packet-backed evidence it used, and waits for approval before editing.
```

- [ ] **Step 3: Compare diagnoses before approving code changes**

Review rubric:

```text
Check whether both arms identified the actual root cause in prototype/local_project_memory_r0/retrieval.py.
Check whether Arm B used source-backed evidence rather than vague claims.
Check whether either arm wandered outside the intended scope.
```

Expected:

```text
No code changes are applied until human approval is granted separately for both arms.
```

### Task 5: Approve Fixes, Verify Results, and Capture the A/B Memo

**Files:**
- Modify in Arm A or Arm B worktrees only: likely `prototype/local_project_memory_r0/retrieval.py`
- Create: `%TEMP%\\mnemos-local-project-memory-ab-results-2026-07-16.md`
- Modify: none in `G:\\MNEMOS`
- Test: targeted pytest for the seeded regression and nearby focused retrieval checks

**Interfaces:**
- Consumes: approved diagnoses from Task 4
- Produces: two verified fixes and one comparison memo

- [ ] **Step 1: Approve each arm to apply its minimal fix**

Expected fix in either arm:

```python
        for artifact in self.packet.artifacts:
            if prefix and not (
                artifact.file_path == prefix
                or artifact.file_path.startswith(prefix + "/")
            ):
                continue
```

Expected:

```text
Each arm applies only the minimal retrieval gate correction inside its own worktree.
```

- [ ] **Step 2: Run the seeded regression test after each fix**

Run in each arm worktree:

```powershell
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q
```

Expected:

```text
PASS in Arm A and Arm B after the fix.
```

- [ ] **Step 3: Run nearby focused verification in each arm**

Run in each arm worktree:

```powershell
python -m pytest tests/test_local_project_memory_retrieval.py -q
python -m pytest tests/test_local_project_memory_boundaries.py -q
```

Expected:

```text
The focused local-project-memory retrieval and boundary tests pass in both arms.
```

- [ ] **Step 4: Check Arm B stale-memory discipline after mutation**

Review:

```text
Arm B must explicitly acknowledge that the packet is stale after its edit.
Arm B must not present any post-mutation packet lookup as fresh current truth unless a new packet was rebuilt with approval.
```

Expected:

```text
Boundary discipline is captured as a scored outcome, even if no packet rebuild is performed.
```

- [ ] **Step 5: Write the compact A/B results memo under `%TEMP%`**

Memo outline:

```markdown
# Local Project Memory A/B Results

- Date:
- Bug seed commit:
- Arm A worktree:
- Arm B worktree:
- Arm B packet:
- Shared failing test:

## Outcome
- Arm A:
- Arm B:

## Diagnosis Quality
- Arm A:
- Arm B:

## Efficiency
- Arm A elapsed:
- Arm B elapsed:
- Arm A search/test iterations:
- Arm B search/test iterations:

## Evidence Discipline
- Arm A:
- Arm B:

## Stale Memory Handling
- Arm B:

## Conclusion
- Did read-only project memory improve debugging or repair quality?
```

Expected:

```text
The memo is concise, sourceable from the experiment receipts, and sufficient for later SDLC evaluation.
```

- [ ] **Step 6: Pause before cleanup**

Report:

```text
Experiment complete.
Main checkout untouched.
Temporary worktrees and packet remain available until explicit cleanup approval.
```
