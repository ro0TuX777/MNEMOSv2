# Local Project Memory A/B Test Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a cleaner second A/B experiment that reuses the same seeded `path_prefix` bug while removing experiment-doc contamination and capturing time and token measurements for both arms.

**Architecture:** Create a fresh temporary seed worktree from `main`, build sanitized arm-visible repo copies that exclude the experiment design and plan artifacts, seed the same one-line retrieval bug, and dispatch two isolated arms from the same buggy snapshot. Arm A debugs without MNEMOS project memory, while Arm B receives a read-only packet built from the same sanitized buggy snapshot plus exact-or-estimated token accounting and wall-clock receipts.

**Tech Stack:** Git worktrees, PowerShell, Python, pytest, `tools/build_local_project_memory_packet.py`, `mcp_servers/mnemos_project/server.py`, `%TEMP%` staging paths, Codex isolated sub-agents.

## Global Constraints

- no mutation of the main checkout
- no merge back to `main`
- no runtime route changes
- no default MNEMOS collection changes
- no Research Intake dependency for code ingestion
- no code mutation outside temporary experiment worktrees
- no whole-repo default scope
- no VS Code extension work
- only Python and Markdown memory extraction
- no internet resources for either arm
- require human approval before seeding the bug in the sanitized seed worktree
- require human approval after each arm's diagnosis and before either arm applies its fix
- require human approval before any optional rebuild of Arm B's packet after mutation
- require human approval before cleanup of temporary worktrees, packet files, or receipts

---

### Task 1: Prepare Fresh Seed Worktree and Sanitized Experiment Source

**Files:**
- Create: `%TEMP%\\mnemos-ab2-seed\\` worktree
- Create: `%TEMP%\\mnemos-ab2-sanitized-seed\\` directory tree
- Exclude from sanitized source: `docs/superpowers/specs/2026-07-16-local-project-memory-ab-test-design.md`
- Exclude from sanitized source: `docs/superpowers/specs/2026-07-16-local-project-memory-ab-test-round2-design.md`
- Exclude from sanitized source: `docs/superpowers/plans/2026-07-16-local-project-memory-ab-test.md`
- Exclude from sanitized source: `docs/superpowers/plans/2026-07-16-local-project-memory-ab-test-round2.md`
- Test: `git worktree list`, filesystem inspection of excluded docs

**Interfaces:**
- Consumes: local `main` checkout at `G:\\MNEMOS`
- Produces: one fresh seed worktree and one sanitized experiment source tree with experiment docs absent

- [ ] **Step 1: Confirm the base checkout is safe to branch from**

Run:

```powershell
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

Expected:

```text
Current branch is main.
Only known unrelated untracked paths may appear in the base checkout.
HEAD resolves successfully.
```

- [ ] **Step 2: Inspect existing worktrees and avoid removing unknown ones**

Run:

```powershell
$seed = Join-Path $env:TEMP 'mnemos-ab2-seed'
$sanitized = Join-Path $env:TEMP 'mnemos-ab2-sanitized-seed'
git worktree list
```

Expected:

```text
Existing worktrees are visible before setup.
No unknown worktree is removed without explicit controller approval.
```

- [ ] **Step 3: Create the fresh seed worktree from local `main`**

Run:

```powershell
$seed = Join-Path $env:TEMP 'mnemos-ab2-seed'
git worktree add --detach $seed main
git -C $seed switch -c exp/local-project-memory-ab2-seed
```

Expected:

```text
The fresh seed worktree exists at %TEMP%\mnemos-ab2-seed on branch exp/local-project-memory-ab2-seed.
```

- [ ] **Step 4: Build the sanitized experiment source tree from the seed worktree**

Run:

```powershell
$seed = Join-Path $env:TEMP 'mnemos-ab2-seed'
$sanitized = Join-Path $env:TEMP 'mnemos-ab2-sanitized-seed'
if (Test-Path $sanitized) { Remove-Item -LiteralPath $sanitized -Recurse -Force }
New-Item -ItemType Directory -Path $sanitized | Out-Null
robocopy $seed $sanitized /MIR /XD .git .pytest_cache __pycache__ .mypy_cache .ruff_cache .venv venv logs benchmarks
```

Expected:

```text
The sanitized source tree is a filesystem copy of the seed worktree content, excluding transient caches and non-source scratch paths.
```

- [ ] **Step 5: Remove the experiment design and plan docs from the sanitized source tree**

Run:

```powershell
$sanitized = Join-Path $env:TEMP 'mnemos-ab2-sanitized-seed'
$remove = @(
  'docs/superpowers/specs/2026-07-16-local-project-memory-ab-test-design.md',
  'docs/superpowers/specs/2026-07-16-local-project-memory-ab-test-round2-design.md',
  'docs/superpowers/plans/2026-07-16-local-project-memory-ab-test.md',
  'docs/superpowers/plans/2026-07-16-local-project-memory-ab-test-round2.md'
)
foreach ($relative in $remove) {
  $target = Join-Path $sanitized $relative
  if (Test-Path $target) { Remove-Item -LiteralPath $target -Force }
}
```

Expected:

```text
The four experiment files are absent from the sanitized source tree.
```

- [ ] **Step 6: Verify the sanitized source no longer exposes the experiment docs**

Run:

```powershell
$sanitized = Join-Path $env:TEMP 'mnemos-ab2-sanitized-seed'
Get-ChildItem -Recurse $sanitized\\docs\\superpowers\\specs
Get-ChildItem -Recurse $sanitized\\docs\\superpowers\\plans
```

Expected:

```text
The round-one and round-two experiment spec/plan files are absent from the sanitized source tree.
```

- [ ] **Step 7: Pause for approval before seeding the bug**

Report:

```text
Fresh seed worktree and sanitized experiment source are ready.
Experiment design/plan docs are absent from the arm-visible source.
Await approval before seeding the bug.
```

### Task 2: Seed the Same `path_prefix` Bug in the Sanitized Seed

**Files:**
- Modify in sanitized source: `prototype/local_project_memory_r0/retrieval.py`
- Confirm existing test in sanitized source: `tests/test_local_project_memory_retrieval.py`
- Test: `tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates`

**Interfaces:**
- Consumes: sanitized experiment source from Task 1
- Produces: one sanitized buggy snapshot and one deterministic failing test receipt

- [ ] **Step 1: Confirm the scoped retrieval test exists in the sanitized source**

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
The sanitized source still contains the same target regression test as round one.
```

- [ ] **Step 2: Introduce the same one-line semantic bug in the sanitized source**

Change this code in `prototype/local_project_memory_r0/retrieval.py`:

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
The same narrow semantic fault from round one is reproduced in the sanitized source tree.
```

- [ ] **Step 3: Run the targeted test from the sanitized source to verify it fails**

Run:

```powershell
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q
```

Expected:

```text
FAIL at assert hits because the inverted path_prefix gate excludes docs-scoped matches.
```

- [ ] **Step 4: Capture the sanitized buggy snapshot as the shared experiment source**

Run:

```powershell
Get-FileHash prototype/local_project_memory_r0/retrieval.py -Algorithm SHA256
Get-FileHash tests/test_local_project_memory_retrieval.py -Algorithm SHA256
```

Expected:

```text
The controller has content receipts for the sanitized buggy source before creating arm copies.
```

### Task 3: Create Sanitized Arm A and Arm B Worktrees

**Files:**
- Create: `%TEMP%\\mnemos-ab2-arm-a\\` directory tree
- Create: `%TEMP%\\mnemos-ab2-arm-b\\` directory tree
- Test: source hashes and targeted failing test

**Interfaces:**
- Consumes: sanitized buggy source from Task 2
- Produces: two identical sanitized arm-visible repo trees for the experiment

- [ ] **Step 1: Create Arm A by copying the sanitized buggy source**

Run:

```powershell
$sanitized = Join-Path $env:TEMP 'mnemos-ab2-sanitized-seed'
$armA = Join-Path $env:TEMP 'mnemos-ab2-arm-a'
if (Test-Path $armA) { Remove-Item -LiteralPath $armA -Recurse -Force }
robocopy $sanitized $armA /MIR
```

Expected:

```text
Arm A contains the sanitized buggy repo copy with no experiment design/plan docs present.
```

- [ ] **Step 2: Create Arm B by copying the sanitized buggy source**

Run:

```powershell
$sanitized = Join-Path $env:TEMP 'mnemos-ab2-sanitized-seed'
$armB = Join-Path $env:TEMP 'mnemos-ab2-arm-b'
if (Test-Path $armB) { Remove-Item -LiteralPath $armB -Recurse -Force }
robocopy $sanitized $armB /MIR
```

Expected:

```text
Arm B contains the same sanitized buggy repo copy as Arm A.
```

- [ ] **Step 3: Verify both arms still exclude the experiment docs**

Run:

```powershell
Get-ChildItem -Recurse $armA\\docs\\superpowers\\specs
Get-ChildItem -Recurse $armA\\docs\\superpowers\\plans
Get-ChildItem -Recurse $armB\\docs\\superpowers\\specs
Get-ChildItem -Recurse $armB\\docs\\superpowers\\plans
```

Expected:

```text
Neither arm-visible repo contains the round-one or round-two experiment spec/plan files.
```

- [ ] **Step 4: Verify both arms reproduce the same failing test**

Run in each arm:

```powershell
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q
```

Expected:

```text
Both arms fail in the same way before any fix is attempted.
```

### Task 4: Build Arm B Packet and Prepare Measurement Receipts

**Files:**
- Create: `%TEMP%\\mnemos-project-memory-ab2-<timestamp>.md`
- Create: `%TEMP%\\mnemos-ab2-arm-a-receipt.json`
- Create: `%TEMP%\\mnemos-ab2-arm-b-receipt.json`
- Test: Arm B packet build, receipt file initialization

**Interfaces:**
- Consumes: sanitized Arm B source tree
- Produces: Arm B packet and per-arm measurement receipts

- [ ] **Step 1: Build Arm B's packet from the sanitized Arm B source with explicit scope**

Run:

```powershell
$armB = Join-Path $env:TEMP 'mnemos-ab2-arm-b'
$packet = Join-Path $env:TEMP ("mnemos-project-memory-ab2-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".md")
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
Arm B gets a packet bound to the sanitized buggy snapshot, and the experiment spec/plan docs are absent from both the source tree and packet scope.
```

- [ ] **Step 2: Initialize lightweight measurement receipts for Arm A and Arm B**

Template content for each receipt:

```json
{
  "arm": "A_OR_B",
  "start_time_utc": null,
  "diagnosis_time_utc": null,
  "fix_approval_time_utc": null,
  "fix_complete_time_utc": null,
  "commands": [],
  "test_commands": [],
  "exact_tokens": null,
  "estimated_tokens": null,
  "notes": []
}
```

Expected:

```text
Both arms have a place for timing, command, and token metrics before dispatch.
```

- [ ] **Step 3: Record the exact prompt text that will be dispatched to both arms**

Prompt text:

```text
Scoped project-memory retrieval is behaving incorrectly. Use the failing test
command below to diagnose the cause and fix it. Keep the change minimal, verify
the result, and summarize the root cause and fix. Do not use internet resources.
```

Expected:

```text
The controller preserves the exact prompt text for later token estimation if exact usage is unavailable.
```

### Task 5: Dispatch Both Arms, Capture Diagnosis, and Measure the Run

**Files:**
- Modify: `%TEMP%\\mnemos-ab2-arm-a-receipt.json`
- Modify: `%TEMP%\\mnemos-ab2-arm-b-receipt.json`
- Test: same targeted failing command in both arms

**Interfaces:**
- Consumes: sanitized Arm A source, sanitized Arm B source, Arm B packet, measurement receipt files
- Produces: diagnosis receipts, approved fixes, and focused verification outcomes for both arms

- [ ] **Step 1: Dispatch Arm A with no MNEMOS project memory**

Arm A task payload:

```text
Work only in the Arm A repo copy under %TEMP%.
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
Arm A returns a diagnosis-only report and waits for approval before editing.
```

- [ ] **Step 2: Dispatch Arm B with the sanitized read-only packet**

Arm B task payload:

```text
Work only in the Arm B repo copy under %TEMP%.
Scoped project-memory retrieval is behaving incorrectly. Use the failing test
command below to diagnose the cause and fix it. Keep the change minimal, verify
the result, and summarize the root cause and fix. Do not use internet resources.

Failing test command:
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q

You may use the configured read-only mnemos_project packet/MCP for source-backed
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
Arm B returns a diagnosis-only report, notes any packet-backed evidence it used, and waits for approval before editing.
```

- [ ] **Step 3: Record diagnosis timestamps, command lists, and any exact token usage exposed by the harness**

Expected:

```text
Each arm receipt contains start and diagnosis timestamps, executed commands, and exact token totals if exposed by the runtime.
```

- [ ] **Step 4: Approve both arms to apply the minimal fix and run focused verification**

Run in each arm after approval:

```powershell
python -m pytest tests/test_local_project_memory_retrieval.py::test_filters_are_eligibility_gates -q
python -m pytest tests/test_local_project_memory_retrieval.py -q
python -m pytest tests/test_local_project_memory_boundaries.py -q
```

Expected:

```text
Both arms apply the one-line fix and pass the same focused verification set.
```

- [ ] **Step 5: Record fix-complete timestamps and stale-packet handling**

Expected:

```text
Arm B explicitly acknowledges packet staleness after mutation.
Both receipts capture final timing and verification data.
```

### Task 6: Compute Token Results and Write Round-Two Memo

**Files:**
- Create: `%TEMP%\\mnemos-local-project-memory-ab-round2-results-2026-07-16.md`
- Create if needed: `%TEMP%\\mnemos-local-project-memory-ab-round2-token-estimate.json`
- Modify: `%TEMP%\\mnemos-ab2-arm-a-receipt.json`
- Modify: `%TEMP%\\mnemos-ab2-arm-b-receipt.json`
- Test: consistency check between receipts and memo

**Interfaces:**
- Consumes: per-arm receipts, exact token totals if available, otherwise prompt/report text
- Produces: final round-two comparison memo and token accounting artifact

- [ ] **Step 1: Use exact per-arm token totals if the harness exposes them**

Expected:

```text
If exact usage is available, record prompt tokens, completion tokens, and total tokens for Arm A and Arm B directly into the receipts and memo.
```

- [ ] **Step 2: Otherwise compute the fallback token estimate consistently for both arms**

Fallback estimate inputs:

```text
- exact controller-dispatched prompt text
- exact controller follow-up text
- returned diagnosis and fix reports
- one consistent tokenizer/model assumption across both arms
```

Expected:

```text
Estimated token totals are computed the same way for both arms and clearly labeled as estimates.
```

- [ ] **Step 3: Write the round-two memo with round-one comparison**

Memo outline:

```markdown
# Local Project Memory A/B Round 2 Results

- Date:
- Shared bug:
- Arm A path:
- Arm B path:
- Arm B packet:
- Shared failing test:

## Outcome
- Arm A:
- Arm B:

## Time
- Arm A diagnosis time:
- Arm B diagnosis time:
- Arm A total time:
- Arm B total time:

## Tokens
- Exact or estimated method:
- Arm A:
- Arm B:
- Delta:

## Command Effort
- Arm A command count:
- Arm B command count:
- Arm A test count:
- Arm B test count:

## Evidence Discipline
- Arm A:
- Arm B:

## Contamination Check
- Were experiment docs absent from both arm-visible repos?
- Did Arm B materially use packet-backed evidence?

## Conclusion
- Did MNEMOS reduce time and/or tokens to a correct fix?
- How does round two compare to round one?
```

Expected:

```text
The memo explicitly answers whether MNEMOS saved time or tokens, and how much of the result rests on exact versus estimated token accounting.
```

- [ ] **Step 4: Pause before cleanup**

Report:

```text
Round-two experiment complete.
Main checkout untouched.
Temporary worktrees, packets, and measurement receipts remain available until explicit cleanup approval.
```
