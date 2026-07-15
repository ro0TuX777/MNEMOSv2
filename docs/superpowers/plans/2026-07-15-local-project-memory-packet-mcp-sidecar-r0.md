# Local Project Memory Packet + MCP Sidecar R0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only, path-scoped Python/Markdown memory packet for a local Git project and expose snapshot-verified source evidence to a VS Code frontier agent through a separate stdio MCP sidecar.

**Architecture:** A standalone builder validates an explicit project/scope, captures a deterministic Git working-tree snapshot, structurally extracts Python and Markdown into one signed Markdown/JSON packet, and never contacts MNEMOS runtime storage. A separate read-only MCP server loads that packet, revalidates live file hashes before retrieval, and provides deterministic structured lexical search; a separate stack verifier diagnoses the installed Compose services and ports without starting or changing containers.

**Tech Stack:** Python 3.10+ standard library (`argparse`, `ast`, `dataclasses`, `hashlib`, `json`, `pathlib`, `subprocess`, `tokenize`, `urllib`), existing MCP bridge environment (`mcp==1.28.1`), pytest, Git CLI, Docker Compose CLI.

## Global Constraints

- First target project: the MNEMOS Git repository at an operator-supplied `--project-root`.
- Require explicit `--project-root`, `--repo-id`, `--output`, and at least one `--scope-root` or `--scope-file`.
- Never infer current-directory, whole-repository, or default scope.
- Admit only Git-tracked `.py` and `.md` files in the explicit scope.
- Exclude untracked, ignored, secret, binary, cache, build, dependency, and outside-root paths before hashing.
- Do not import, execute, compile, lint, format, or mutate target-project code.
- Write only the explicit packet path, which must be outside the target project and absent before execution.
- Do not use Research Intake, `mnemos_sdk`, the MNEMOS REST API, Qdrant, PostgreSQL, or any configured/default collection for packet construction or sidecar retrieval.
- Do not modify runtime routes, `service/app.py`, `mnemos/config.py`, `mnemos/retrieval/`, `docker-compose.yml`, default collection settings, or existing MCP write tools.
- Do not add a VS Code extension or frontier API integration.
- Preserve exact source bytes, one-based inclusive spans, file/content hashes, source URIs, snapshot ID, exclusions, and authority boundaries.
- MCP search must fail closed on packet-integrity, repo-ID, parse-completeness, or live-snapshot mismatch.
- MCP sidecar exposes no indexing, observation, decision, lint, shell, file-write, or mutation tool.
- Stack verification is read-only: no container start, stop, restart, recreate, pull, build, or configuration writes.
- Human approval is required before lint execution and separately before code mutation; after mutation the packet is stale.
- Preserve unrelated user changes and untracked `benchmarks/reports/` and `logs/` content.

---

## File Structure

### New isolated package

- `prototype/local_project_memory_r0/__init__.py`: exported R0 contracts only.
- `prototype/local_project_memory_r0/errors.py`: stable fail-closed reason codes and exception.
- `prototype/local_project_memory_r0/models.py`: immutable scope, snapshot, artifact, packet, search, and verification models.
- `prototype/local_project_memory_r0/canonical.py`: canonical JSON, hashing, source URI, artifact ID, and path normalization.
- `prototype/local_project_memory_r0/snapshot.py`: explicit scope validation, read-only Git discovery, exclusions, dirty state, hashing, and live verification.
- `prototype/local_project_memory_r0/python_extractor.py`: source-only AST/token Python extraction.
- `prototype/local_project_memory_r0/markdown_extractor.py`: source-line Markdown extraction.
- `prototype/local_project_memory_r0/packet.py`: packet assembly, integrity validation, sentinel serialization, and loading.
- `prototype/local_project_memory_r0/retrieval.py`: deterministic structured lexical indexing, filtering, ranking, and evidence envelopes.

### New commands and MCP sidecar

- `tools/build_local_project_memory_packet.py`: packet-builder CLI.
- `tools/verify_mnemos_local_stack.py`: read-only Docker/endpoint preflight and JSON receipt writer.
- `mcp_servers/mnemos_project/__init__.py`: package marker.
- `mcp_servers/mnemos_project/server.py`: read-only stdio MCP facade over one packet.
- `mcp_servers/mnemos_project/README.md`: VS Code/Claude-style MCP configuration and authority guidance.

### New tests and trial artifacts

- `tests/test_local_project_memory_models.py`
- `tests/test_local_project_memory_snapshot.py`
- `tests/test_local_project_memory_extractors.py`
- `tests/test_local_project_memory_packet_cli.py`
- `tests/test_local_project_memory_retrieval.py`
- `tests/test_mnemos_project_mcp.py`
- `tests/test_verify_mnemos_local_stack.py`
- `tests/test_local_project_memory_boundaries.py`
- `docs/experiments/local_project_memory_packet_mcp_sidecar_r0_trial.md`

The implementation must not create a checked-in packet containing the full
MNEMOS source tree. Trial packets and stack receipts go under `$env:TEMP`.

---

### Task 1: Freeze Common Contracts and Canonical Identity

**Files:**
- Create: `prototype/local_project_memory_r0/__init__.py`
- Create: `prototype/local_project_memory_r0/errors.py`
- Create: `prototype/local_project_memory_r0/models.py`
- Create: `prototype/local_project_memory_r0/canonical.py`
- Test: `tests/test_local_project_memory_models.py`

**Interfaces:**
- Produces: `ErrorCode`, `ProjectMemoryError`, `ScopeSpec`, `SourceSpan`, `SnapshotFile`, `SnapshotManifest`, `ProjectArtifact`, `ProjectPacket`, `SearchHit`, `SnapshotVerification`.
- Produces: `canonical_json_bytes(value)`, `sha256_bytes(value)`, `normalize_relative_path(value)`, `source_uri(repo_id, path, snapshot_id)`, `artifact_id(...)`.
- Consumes: Python standard library only.

- [ ] **Step 1: Write failing contract and canonicalization tests**

```python
from dataclasses import replace

import pytest

from prototype.local_project_memory_r0.canonical import (
    artifact_id,
    canonical_json_bytes,
    normalize_relative_path,
    sha256_bytes,
    source_uri,
)
from prototype.local_project_memory_r0.errors import ErrorCode, ProjectMemoryError
from prototype.local_project_memory_r0.models import ScopeSpec, SourceSpan


def test_canonical_json_and_hash_are_stable():
    left = canonical_json_bytes({"b": 2, "a": [1]})
    right = canonical_json_bytes({"a": [1], "b": 2})
    assert left == right == b'{"a":[1],"b":2}'
    assert sha256_bytes(left).startswith("sha256:")


@pytest.mark.parametrize("value", ["../secret.py", "/tmp/a.py", "C:/a.py", "a/../../b.py"])
def test_relative_path_escape_fails_closed(value):
    with pytest.raises(ProjectMemoryError) as exc:
        normalize_relative_path(value)
    assert exc.value.code is ErrorCode.SCOPE_OUTSIDE_PROJECT


def test_artifact_identity_binds_snapshot_span_and_content():
    snapshot = "sha256:" + "a" * 64
    span = SourceSpan(start_line=10, end_line=12)
    content_hash = "sha256:" + "b" * 64
    first = artifact_id("mnemos", snapshot, "mnemos/config.py", "python_symbol", "Settings", span, content_hash)
    second = artifact_id("mnemos", snapshot, "mnemos/config.py", "python_symbol", "Settings", replace(span, end_line=13), content_hash)
    assert first != second
    assert source_uri("mnemos", "mnemos/config.py", snapshot) == f"repo://mnemos/mnemos/config.py?snapshot={snapshot}"


def test_empty_scope_contract_is_rejected():
    with pytest.raises(ProjectMemoryError) as exc:
        ScopeSpec(roots=(), files=(), excludes=())
    assert exc.value.code is ErrorCode.SCOPE_REQUIRED
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_local_project_memory_models.py -q
```

Expected: collection fails with `ModuleNotFoundError: prototype.local_project_memory_r0`.

- [ ] **Step 3: Implement immutable models and stable errors**

Define `ErrorCode` with these exact values:

```python
class ErrorCode(str, Enum):
    PROJECT_ROOT_REQUIRED = "PROJECT_ROOT_REQUIRED"
    NOT_GIT_WORKTREE = "NOT_GIT_WORKTREE"
    REPO_ID_REQUIRED = "REPO_ID_REQUIRED"
    SCOPE_REQUIRED = "SCOPE_REQUIRED"
    SCOPE_OUTSIDE_PROJECT = "SCOPE_OUTSIDE_PROJECT"
    SCOPE_FILE_NOT_ADMITTED = "SCOPE_FILE_NOT_ADMITTED"
    EMPTY_ADMITTED_SCOPE = "EMPTY_ADMITTED_SCOPE"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    SECRET_PATH_REJECTED = "SECRET_PATH_REJECTED"
    DIRTY_STATE_UNRESOLVED = "DIRTY_STATE_UNRESOLVED"
    STRUCTURED_PARSE_INCOMPLETE = "STRUCTURED_PARSE_INCOMPLETE"
    OUTPUT_INSIDE_PROJECT = "OUTPUT_INSIDE_PROJECT"
    OUTPUT_ALREADY_EXISTS = "OUTPUT_ALREADY_EXISTS"
    PACKET_INTEGRITY_INVALID = "PACKET_INTEGRITY_INVALID"
    REPO_ID_MISMATCH = "REPO_ID_MISMATCH"
    SNAPSHOT_MISMATCH = "SNAPSHOT_MISMATCH"
    CROSS_SCOPE_EVIDENCE = "CROSS_SCOPE_EVIDENCE"
```

Use frozen dataclasses. `ScopeSpec.__post_init__` rejects empty scope and
non-positive ceilings. `SourceSpan.__post_init__` requires `start_line >= 1`
and `end_line >= start_line`. Store hashes as lowercase `sha256:<64 hex>`.

- [ ] **Step 4: Implement canonical JSON and identity helpers**

Use UTF-8, NFC normalization, sorted JSON keys, compact separators, and
`ensure_ascii=False`. `normalize_relative_path()` converts `\` to `/`, rejects
absolute/drive-qualified/traversing paths, and returns a POSIX relative path.
Build artifact identities from a length-prefixed list of repo, snapshot, path,
type, qualified name, span, and content hash; never concatenate ambiguous raw
strings.

- [ ] **Step 5: Run tests and verify GREEN**

```powershell
python -m pytest tests/test_local_project_memory_models.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the contract slice**

```powershell
git add prototype/local_project_memory_r0 tests/test_local_project_memory_models.py
git commit -m "feat: add local project memory contracts"
```

---

### Task 2: Build the Explicit-Scope Read-Only Snapshot

**Files:**
- Create: `prototype/local_project_memory_r0/snapshot.py`
- Test: `tests/test_local_project_memory_snapshot.py`

**Interfaces:**
- Consumes: `ScopeSpec`, `SnapshotFile`, `SnapshotManifest`, canonical helpers, and `ProjectMemoryError`.
- Produces: `GitReader`, `build_snapshot(project_root: Path, repo_id: str, scope: ScopeSpec) -> SnapshotManifest`, `verify_snapshot(project_root: Path, manifest: SnapshotManifest) -> SnapshotVerification`.

- [ ] **Step 1: Write failing scope, exclusion, dirty-state, and mutation tests**

```python
def test_scope_admits_only_tracked_python_and_markdown(git_repo):
    git_repo.commit({
        "mnemos/a.py": "VALUE = 1\n",
        "docs/a.md": "# A\n",
        "mnemos/data.json": "{}\n",
        "outside.py": "VALUE = 2\n",
    })
    scope = ScopeSpec(roots=("mnemos",), files=("docs/a.md",), excludes=())
    manifest = build_snapshot(git_repo.path, "fixture", scope)
    assert [item.path for item in manifest.files] == ["docs/a.md", "mnemos/a.py"]


def test_explicit_untracked_file_fails(git_repo):
    git_repo.write("mnemos/untracked.py", "TOKEN = 'not admitted'\n")
    scope = ScopeSpec(roots=(), files=("mnemos/untracked.py",), excludes=())
    with pytest.raises(ProjectMemoryError) as exc:
        build_snapshot(git_repo.path, "fixture", scope)
    assert exc.value.code is ErrorCode.SCOPE_FILE_NOT_ADMITTED


def test_empty_filtered_scope_fails(git_repo):
    git_repo.commit({"selected/data.json": "{}\n"})
    with pytest.raises(ProjectMemoryError) as exc:
        build_snapshot(git_repo.path, "fixture", ScopeSpec(roots=("selected",), files=(), excludes=()))
    assert exc.value.code is ErrorCode.EMPTY_ADMITTED_SCOPE


def test_snapshot_reads_without_mutating_project(git_repo):
    git_repo.commit({"selected/a.py": "VALUE = 1\n"})
    before = tree_hashes(git_repo.path)
    build_snapshot(git_repo.path, "fixture", ScopeSpec(roots=("selected",), files=(), excludes=()))
    assert tree_hashes(git_repo.path) == before


def test_dirty_bytes_change_snapshot_not_base_commit(git_repo):
    git_repo.commit({"selected/a.py": "VALUE = 1\n"})
    clean = build_snapshot(git_repo.path, "fixture", scope_for("selected"))
    git_repo.write("selected/a.py", "VALUE = 2\n")
    dirty = build_snapshot(git_repo.path, "fixture", scope_for("selected"))
    assert dirty.commit_hash == clean.commit_hash
    assert dirty.snapshot_id != clean.snapshot_id
    assert dirty.working_tree_state == "dirty"
```

Also cover outside-root symlinks, secret filenames, ignored/untracked counts,
operator exclusions, deleted tracked files, detached HEAD, max files, max file
bytes, total bytes, and deterministic ordering.

- [ ] **Step 2: Run the snapshot tests and verify RED**

```powershell
python -m pytest tests/test_local_project_memory_snapshot.py -q
```

Expected: import failure for `snapshot.py`.

- [ ] **Step 3: Implement a read-only `GitReader`**

All Git calls use argument arrays, `cwd=project_root`, `shell=False`, captured
UTF-8 output, and only these commands:

```text
git rev-parse --show-toplevel
git rev-parse HEAD
git symbolic-ref --quiet --short HEAD
git ls-files --cached -z -- <explicit paths>
git status --porcelain=v1 -z --untracked-files=all
git check-ignore -z --stdin
```

Reject a resolved `--project-root` that differs from `git rev-parse
--show-toplevel`; R0 binds one repository root, not a nested implicit root.

- [ ] **Step 4: Implement admission and snapshot hashing**

Use a versioned hard-exclusion policy. Resolve candidates before reading,
exclude unsupported descendants with reason codes, fail explicit-file
violations, read bytes once, enforce ceilings, and compute exact hashes.
Construct `snapshot_id` from the canonical stable manifest preimage without
creation time or absolute paths. Include dirty state for admitted tracked paths
and their working-tree byte hashes. Report excluded/untracked counts, but do not
put repository activity outside the admitted scope into the snapshot identity
or freshness comparison; changing excluded `logs/` must not stale the packet.

- [ ] **Step 5: Implement live verification**

`verify_snapshot()` rechecks repository identity, admitted path presence and
hashes, scope membership, and dirty-state identity. Return all mismatches in a
`SnapshotVerification`; never silently choose a newer packet or snapshot.

- [ ] **Step 6: Run tests and verify GREEN**

```powershell
python -m pytest tests/test_local_project_memory_models.py tests/test_local_project_memory_snapshot.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit the snapshot boundary**

```powershell
git add prototype/local_project_memory_r0/snapshot.py tests/test_local_project_memory_snapshot.py
git commit -m "feat: add explicit scoped project snapshots"
```

---

### Task 3: Extract Source-Backed Python and Markdown Artifacts

**Files:**
- Create: `prototype/local_project_memory_r0/python_extractor.py`
- Create: `prototype/local_project_memory_r0/markdown_extractor.py`
- Test: `tests/test_local_project_memory_extractors.py`

**Interfaces:**
- Consumes: `project_root`, `SnapshotManifest`, and one `SnapshotFile` whose bytes already match the manifest.
- Produces: `extract_python(...) -> tuple[ProjectArtifact, ...]`, `extract_markdown(...) -> tuple[ProjectArtifact, ...]`.

- [ ] **Step 1: Write failing exact-span extractor tests**

```python
def test_python_symbols_preserve_decorators_and_exact_spans(snapshot_fixture):
    source = "@router.get('/health')\ndef health():\n    return {'status': 'ok'}\n"
    fixture = snapshot_fixture({"service/app.py": source})
    artifacts = extract_python(fixture.root, fixture.manifest, fixture.file("service/app.py"))
    handler = next(item for item in artifacts if item.qualified_name == "service.app.health")
    assert handler.span == SourceSpan(1, 3)
    assert handler.content == source
    assert handler.metadata["route_path"] == "/health"
    assert handler.metadata["route_detection"] == "heuristic"


def test_python_does_not_import_or_execute_source(snapshot_fixture, monkeypatch):
    fixture = snapshot_fixture({"selected/a.py": "raise RuntimeError('must not execute')\n"})
    monkeypatch.setattr("builtins.__import__", guarded_import)
    result = extract_python(fixture.root, fixture.manifest, fixture.file("selected/a.py"))
    assert result


def test_markdown_heading_sections_have_exact_lines(snapshot_fixture):
    source = "# Decision\n\nStatus: Accepted\n\n## Boundary\nRead only.\n"
    fixture = snapshot_fixture({"docs/decision.md": source})
    artifacts = extract_markdown(fixture.root, fixture.manifest, fixture.file("docs/decision.md"))
    boundary = next(item for item in artifacts if item.metadata.get("heading_path") == ["Decision", "Boundary"])
    assert boundary.span == SourceSpan(5, 6)
    assert boundary.content == "## Boundary\nRead only.\n"


def test_python_syntax_error_is_structurally_incomplete(snapshot_fixture):
    fixture = snapshot_fixture({"selected/broken.py": "def broken(:\n"})
    with pytest.raises(ProjectMemoryError) as exc:
        extract_python(fixture.root, fixture.manifest, fixture.file("selected/broken.py"))
    assert exc.value.code is ErrorCode.STRUCTURED_PARSE_INCOMPLETE
```

- [ ] **Step 2: Run extractor tests and verify RED**

```powershell
python -m pytest tests/test_local_project_memory_extractors.py -q
```

Expected: extractor modules are missing.

- [ ] **Step 3: Implement Python extraction**

Decode with strict UTF-8, parse with `ast.parse`, and use `lineno`,
`end_lineno`, decorator locations, `ast.get_source_segment`, and `tokenize`
only. Emit module, class, function, method, import, uppercase literal constant,
decorator, syntactic test, and heuristic route artifacts. Literal-evaluate only
AST literal nodes; never call `eval`, `exec`, `compile`, or import target code.

- [ ] **Step 4: Implement Markdown extraction**

Recognize ATX and Setext headings, YAML-front-matter delimiters as text, and
explicit labelled status/date/supersession fields. Emit document and heading
sections with exact original newlines and classify ADR/decision/handoff/
evaluation/agent-instruction only from explicit names, headings, or fields.

- [ ] **Step 5: Validate all artifact hashes and source slices**

For every artifact, recompute the exact content hash, artifact ID, source URI,
file hash, and one-based inclusive slice. Raise `PACKET_INTEGRITY_INVALID` on
any inconsistency rather than repairing metadata.

- [ ] **Step 6: Run extractor and snapshot tests**

```powershell
python -m pytest tests/test_local_project_memory_extractors.py tests/test_local_project_memory_snapshot.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit extraction**

```powershell
git add prototype/local_project_memory_r0/python_extractor.py prototype/local_project_memory_r0/markdown_extractor.py tests/test_local_project_memory_extractors.py
git commit -m "feat: extract source-backed project artifacts"
```

---

### Task 4: Assemble and Write the Single Packet CLI

**Files:**
- Create: `prototype/local_project_memory_r0/packet.py`
- Create: `tools/build_local_project_memory_packet.py`
- Test: `tests/test_local_project_memory_packet_cli.py`

**Interfaces:**
- Consumes: explicit CLI arguments, `build_snapshot()`, and both extractors.
- Produces: `build_packet(...) -> ProjectPacket`, `write_packet(path, packet)`, `load_packet(path) -> ProjectPacket`, and CLI exit codes `0` complete / `2` validation failure / `3` incomplete parse.

- [ ] **Step 1: Write failing CLI and packet round-trip tests**

```python
def test_cli_requires_explicit_scope_and_never_defaults_to_repo(tmp_path, git_repo):
    output = tmp_path / "packet.md"
    result = run_cli("--project-root", git_repo.path, "--repo-id", "fixture", "--output", output)
    assert result.returncode == 2
    assert "SCOPE_REQUIRED" in result.stderr
    assert not output.exists()


def test_output_inside_project_is_rejected(git_repo):
    result = run_cli(
        "--project-root", git_repo.path,
        "--repo-id", "fixture",
        "--scope-root", "selected",
        "--output", git_repo.path / "packet.md",
    )
    assert result.returncode == 2
    assert "OUTPUT_INSIDE_PROJECT" in result.stderr


def test_packet_round_trip_preserves_integrity(tmp_path, complete_packet):
    path = tmp_path / "packet.md"
    write_packet(path, complete_packet)
    loaded = load_packet(path)
    assert loaded.snapshot.snapshot_id == complete_packet.snapshot.snapshot_id
    assert loaded.packet_sha256 == complete_packet.packet_sha256
    assert "MNEMOS_PROJECT_PACKET_JSON_BEGIN:v1" in path.read_text(encoding="utf-8")


def test_existing_output_is_not_overwritten(tmp_path, complete_packet):
    path = tmp_path / "packet.md"
    path.write_text("operator data", encoding="utf-8")
    with pytest.raises(ProjectMemoryError) as exc:
        write_packet(path, complete_packet)
    assert exc.value.code is ErrorCode.OUTPUT_ALREADY_EXISTS
    assert path.read_text(encoding="utf-8") == "operator data"
```

Also assert target-tree hashes before/after, no absolute project root in the
payload, exclusions and approval checkpoints present, deterministic snapshot
ID, changing creation time leaves snapshot ID unchanged, and packet tampering
is rejected.

- [ ] **Step 2: Run packet tests and verify RED**

```powershell
python -m pytest tests/test_local_project_memory_packet_cli.py -q
```

Expected: packet/CLI modules are missing.

- [ ] **Step 3: Implement packet assembly and integrity**

Dispatch only manifest files to the matching extractor. Sort files and
artifacts deterministically. Set `usable=false` when any parse failure exists.
Compute `packet_sha256` from canonical payload data with the hash field omitted.
Validate every artifact against its manifest file before serialization.

- [ ] **Step 4: Implement the versioned Markdown container**

Write exactly one file using exclusive creation mode and these sentinels:

```text
<!-- MNEMOS_PROJECT_PACKET_JSON_BEGIN:v1 -->
```json
{canonical pretty-printed payload}
```
<!-- MNEMOS_PROJECT_PACKET_JSON_END:v1 -->
```

Follow with a human report containing identity, boundaries, admitted files,
exclusions, parse status, and the four approval statements. Do not repeat full
source content in the report.

- [ ] **Step 5: Implement the CLI parser and safe exit behavior**

Make all four required input groups explicit. Repeat scope/exclude flags with
`action="append"`. Print a concise receipt containing output, repo ID, snapshot,
file/artifact counts, and usability. On validation failure print the stable
code to stderr without a traceback or sensitive path values.

- [ ] **Step 6: Run packet tests and CLI help smoke test**

```powershell
python -m pytest tests/test_local_project_memory_packet_cli.py tests/test_local_project_memory_extractors.py -q
python tools/build_local_project_memory_packet.py --help
```

Expected: tests pass; help lists all required scope and ceiling arguments and
contains no collection, MNEMOS URL, lint, or mutation option.

- [ ] **Step 7: Commit the builder**

```powershell
git add prototype/local_project_memory_r0/packet.py tools/build_local_project_memory_packet.py tests/test_local_project_memory_packet_cli.py
git commit -m "feat: build immutable local project memory packets"
```

---

### Task 5: Add Deterministic Structured Retrieval

**Files:**
- Create: `prototype/local_project_memory_r0/retrieval.py`
- Test: `tests/test_local_project_memory_retrieval.py`

**Interfaces:**
- Consumes: one integrity-validated, usable `ProjectPacket`.
- Produces: `ProjectMemoryIndex(packet)`, `search(query, top_k=8, path_prefix=None, artifact_types=()) -> tuple[SearchHit, ...]`, `get(artifact_id) -> ProjectArtifact`.

- [ ] **Step 1: Write failing retrieval and scope-isolation tests**

```python
def test_exact_symbol_and_path_rank_first(project_index):
    hits = project_index.search("RetrievalRouter.search", top_k=5)
    assert hits[0].artifact.qualified_name.endswith("RetrievalRouter.search")
    assert "exact_qualified_name" in hits[0].match_reasons


def test_concept_terms_return_source_backed_logic(project_index):
    hits = project_index.search("reject stale snapshot source hashes", top_k=8)
    assert hits
    assert all(hit.artifact.content for hit in hits)
    assert all(hit.artifact.span.start_line >= 1 for hit in hits)


def test_filters_are_eligibility_gates(project_index):
    hits = project_index.search("health", path_prefix="docs/", artifact_types=("markdown_section",))
    assert hits
    assert all(hit.artifact.file_path.startswith("docs/") for hit in hits)
    assert all(hit.artifact.artifact_type == "markdown_section" for hit in hits)


def test_cross_scope_artifact_is_never_indexed(complete_packet):
    injected = add_artifact(complete_packet, file_path="outside/secret.py", content="TOKEN='x'")
    with pytest.raises(ProjectMemoryError) as exc:
        ProjectMemoryIndex(injected)
    assert exc.value.code is ErrorCode.CROSS_SCOPE_EVIDENCE
```

- [ ] **Step 2: Run retrieval tests and verify RED**

```powershell
python -m pytest tests/test_local_project_memory_retrieval.py -q
```

Expected: retrieval module is missing.

- [ ] **Step 3: Implement structured tokenization and ranking**

Normalize prose tokens case-insensitively while separately preserving exact
snake-case, dotted qualified-name, route, uppercase constant, heading, and path
tokens. Rank with deterministic integer components:

```text
100 exact qualified-name
90 exact relative path
80 exact route/config/test literal
60 exact heading
20 per distinct query token in qualified name/path
10 per distinct query token in content
5 artifact-type structural prior
```

Sort by total descending, exact-component tuple descending, then artifact ID
ascending. Return component scores and match reasons; do not label R0 results
semantic.

- [ ] **Step 4: Validate eligibility and evidence envelopes**

On index construction, confirm every artifact maps to one admitted manifest
file with matching repo, snapshot, path, and file hash. Apply filters before
ranking. Each hit includes repo/snapshot, source URI, exact span, file/content
hashes, content, score components, and authority boundary.

- [ ] **Step 5: Run retrieval tests**

```powershell
python -m pytest tests/test_local_project_memory_retrieval.py tests/test_local_project_memory_packet_cli.py -q
```

Expected: all tests pass with deterministic ordering across repeated runs.

- [ ] **Step 6: Commit retrieval**

```powershell
git add prototype/local_project_memory_r0/retrieval.py tests/test_local_project_memory_retrieval.py
git commit -m "feat: retrieve scoped source-backed project evidence"
```

---

### Task 6: Expose the Packet Through a Read-Only MCP Sidecar

**Files:**
- Create: `mcp_servers/mnemos_project/__init__.py`
- Create: `mcp_servers/mnemos_project/server.py`
- Create: `mcp_servers/mnemos_project/README.md`
- Test: `tests/test_mnemos_project_mcp.py`

**Interfaces:**
- Consumes: `MNEMOS_PROJECT_PACKET`, `MNEMOS_PROJECT_ROOT`, `MNEMOS_PROJECT_REPO_ID`, `load_packet()`, `verify_snapshot()`, and `ProjectMemoryIndex`.
- Produces MCP tools: `project_memory_health`, `get_project_identity`, `search_project_memory`, `get_project_artifact`, `verify_project_snapshot`.

- [ ] **Step 1: Write failing MCP tool and stale-abstention tests**

```python
def test_server_exposes_only_read_only_tools(mcp_module):
    names = set(mcp_module.registered_tool_names())
    assert names == {
        "project_memory_health",
        "get_project_identity",
        "search_project_memory",
        "get_project_artifact",
        "verify_project_snapshot",
    }
    assert not names & {"write_observation", "record_decision", "index", "lint", "mutate"}


def test_search_returns_hashes_spans_and_boundary(configured_server):
    response = configured_server.search_project_memory("snapshot verification", top_k=3)
    assert response["status"] == "ok"
    assert response["snapshot_id"].startswith("sha256:")
    assert response["results"]
    assert response["results"][0]["provenance_span"]["start_line"] >= 1
    assert response["results"][0]["file_hash"].startswith("sha256:")
    assert "human approval" in response["authority_boundary"].lower()


def test_changed_file_forces_abstention(configured_server):
    configured_server.project_file.write_text("changed\n", encoding="utf-8")
    response = configured_server.search_project_memory("anything", top_k=3)
    assert response["status"] == "abstained"
    assert response["reason_code"] == "SNAPSHOT_MISMATCH"
    assert response["results"] == []
```

Also cover missing env vars, repo-ID mismatch, tampered packet, unusable packet,
missing artifact, invalid filters JSON, top-k bounds `1..20`, and stdio MCP
initialization using the existing `mcp_servers/mnemos/.venv`.

- [ ] **Step 2: Run MCP tests and verify RED**

```powershell
python -m pytest tests/test_mnemos_project_mcp.py -q
```

Expected: project MCP package is missing.

- [ ] **Step 3: Implement lazy, fail-closed server state**

At first tool call, resolve all three required environment values, load and
validate the packet, compare repo ID, build the index, and verify the live
snapshot. Cache the parsed packet/index only; run live snapshot verification
before every evidence-returning call. Return structured abstention envelopes
instead of tracebacks.

- [ ] **Step 4: Implement exactly five MCP tools**

Use `FastMCP("mnemos-project-memory-r0")`. `project_memory_health` reports
configuration, packet integrity, and snapshot freshness. Identity returns
scope/exclusions without source. Search and get return evidence only when
fresh. Verification returns mismatch paths and hashes without reading excluded
files. Every response includes the non-authority and approval boundary.

- [ ] **Step 5: Document VS Code MCP configuration**

Provide a Windows JSON example using:

```json
{
  "mcpServers": {
    "mnemos-project": {
      "command": "G:\\MNEMOS\\mcp_servers\\mnemos\\.venv\\Scripts\\python.exe",
      "args": ["G:\\MNEMOS\\mcp_servers\\mnemos_project\\server.py"],
      "env": {
        "MNEMOS_PROJECT_PACKET": "C:\\Users\\operator\\AppData\\Local\\Temp\\mnemos-project-memory.md",
        "MNEMOS_PROJECT_ROOT": "G:\\MNEMOS",
        "MNEMOS_PROJECT_REPO_ID": "mnemos"
      }
    }
  }
}
```

Also include the recommended agent instruction: search MNEMOS project memory
before making logic claims, verify returned spans against the live file when
editing, present lint commands for approval, present proposed edits for separate
approval, and rebuild memory after edits.

- [ ] **Step 6: Run MCP unit and stdio smoke tests**

```powershell
python -m pytest tests/test_mnemos_project_mcp.py tests/test_local_project_memory_retrieval.py -q
mcp_servers/mnemos/.venv/Scripts/python.exe mcp_servers/mnemos_project/server.py --help
```

Expected: tests pass; server help succeeds without contacting MNEMOS REST.

- [ ] **Step 7: Commit the sidecar**

```powershell
git add mcp_servers/mnemos_project tests/test_mnemos_project_mcp.py
git commit -m "feat: expose read-only project memory over MCP"
```

---

### Task 7: Add Read-Only MNEMOS Stack and Port Verification

**Files:**
- Create: `tools/verify_mnemos_local_stack.py`
- Test: `tests/test_verify_mnemos_local_stack.py`

**Interfaces:**
- Produces: `run_preflight(config: StackConfig, runner: CommandRunner, http: HttpProbe) -> StackReceipt`, CLI exit `0` all required healthy / `1` required failure / `2` configuration error.
- Consumes: Docker Compose CLI and configured local HTTP URLs; no MNEMOS Python runtime imports.

- [ ] **Step 1: Write failing discovery and diagnostic tests**

```python
def test_compose_roles_are_discovered_without_fixed_container_names(fake_runner, fake_http):
    fake_runner.compose_config(services={
        "qdrant": {"image": "qdrant/qdrant:v1.17.1"},
        "postgres": {"image": "postgres:16-alpine"},
        "mnemos": {"image": "local/mnemos:test"},
    })
    fake_runner.running_container(service="mnemos", name="random-project-mnemos-1", image="local/mnemos:test")
    receipt = run_preflight(default_config(), fake_runner, fake_http)
    assert receipt.services["mnemos"].container_name == "random-project-mnemos-1"


def test_required_connection_failure_is_actionable_and_nonzero(fake_runner, fake_http):
    fake_http.fail("http://127.0.0.1:8700/health", "connection refused")
    receipt = run_preflight(default_config(), fake_runner, fake_http)
    assert receipt.ok is False
    assert receipt.services["mnemos"].reason_code == "HTTP_CONNECTION_FAILED"
    assert "8700" in receipt.services["mnemos"].remediation


def test_optional_absent_differs_from_configured_unhealthy(fake_runner, fake_http):
    receipt = run_preflight(core_only_config(), fake_runner, fake_http)
    assert receipt.services["research-ui"].status == "not_configured"
    assert receipt.services["openwebui-proxy"].status == "not_configured"


def test_verifier_never_runs_mutating_docker_commands(fake_runner, fake_http):
    run_preflight(default_config(), fake_runner, fake_http)
    commands = [" ".join(call) for call in fake_runner.calls]
    assert not any(word in command for command in commands for word in (" up ", " down ", " restart ", " rm ", " pull ", " build "))
```

Also test Qdrant image mismatch, PostgreSQL `pg_isready` failure, degraded
MNEMOS health, invalid capabilities JSON, port collision/ownership mismatch,
custom Compose file, overridden URLs, timeouts, JSON receipt redaction, and
Research UI/proxy endpoint failures.

- [ ] **Step 2: Run verifier tests and verify RED**

```powershell
python -m pytest tests/test_verify_mnemos_local_stack.py -q
```

Expected: verifier module is missing.

- [ ] **Step 3: Implement Compose discovery**

Use only:

```text
docker compose -f <file> config --format json
docker compose -f <file> ps --format json
docker inspect <container-id>
docker compose -f <file> exec -T postgres pg_isready -U <configured user> -d <configured database>
```

Parse service identity from Compose service names and
`com.docker.compose.service` labels. Record observed container name/image but
do not use them as primary identity. Redact environment values whose names
contain `PASSWORD`, `TOKEN`, `SECRET`, `KEY`, or `DSN`.

- [ ] **Step 4: Implement endpoint and port probes**

Use `urllib.request` with explicit timeouts. Defaults:

```text
qdrant_url=http://127.0.0.1:6333
mnemos_url=http://127.0.0.1:8700
research_ui_url=http://127.0.0.1:8788
proxy_url=http://127.0.0.1:8790
```

Probe Qdrant `/healthz`, MNEMOS `/health` and
`/v1/mnemos/capabilities`, Research UI `/`, and proxy `/health`. Inspect
published-port bindings and distinguish endpoint reachability from ownership.

- [ ] **Step 5: Implement CLI and JSON receipt**

Support `--compose-file`, four URL overrides, `--qdrant-service`,
`--postgres-service`, `--mnemos-service`, `--research-ui-service`,
`--openwebui-proxy-service`, `--timeout-s`, `--require-research-ui`,
`--require-openwebui-proxy`, and `--output-json`. Service-name defaults match
`docker-compose.yml`, while explicit overrides support differently named
installer Compose files. Do not include database passwords or full DSNs in
console/JSON output. Print a compact table with role, service, observed
container/image, endpoint, status, latency, reason, and one actionable
remediation.

- [ ] **Step 6: Run unit tests and live read-only preflight**

```powershell
python -m pytest tests/test_verify_mnemos_local_stack.py -q
$receipt = Join-Path $env:TEMP 'mnemos-local-stack-receipt.json'
python tools/verify_mnemos_local_stack.py --compose-file docker-compose.yml --require-research-ui --require-openwebui-proxy --output-json $receipt
```

Expected on the current workstation:

```text
qdrant: healthy, qdrant/qdrant:v1.17.1, host port 6333
postgres: healthy, postgres:16-alpine, host port 5432
mnemos: healthy, Compose service mnemos, observed container mnemos-service, image mnemos-mnemos, host port 8700
research-ui: healthy, observed container mnemos-research-ui, host port 8788
openwebui-proxy: healthy, observed container mnemos-openwebui-proxy, host port 8790
```

- [ ] **Step 7: Commit the verifier**

```powershell
git add tools/verify_mnemos_local_stack.py tests/test_verify_mnemos_local_stack.py
git commit -m "feat: verify local MNEMOS stack connectivity"
```

---

### Task 8: Enforce Boundaries and Run the MNEMOS Self-Memory Trial

**Files:**
- Create: `tests/test_local_project_memory_boundaries.py`
- Create: `docs/experiments/local_project_memory_packet_mcp_sidecar_r0_trial.md`
- Modify: `mcp_servers/mnemos_project/README.md`

**Interfaces:**
- Consumes: complete builder, packet loader, retrieval index, MCP sidecar, stack verifier, and the live MNEMOS checkout.
- Produces: a reproducible operator trial record with no checked-in source packet.

- [ ] **Step 1: Write boundary tests before the live trial**

```python
def test_project_memory_lane_has_no_runtime_or_storage_coupling():
    files = project_memory_python_files()
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    forbidden = (
        "mnemos_sdk",
        "MNEMOS_BASE_URL",
        "MNEMOS_QDRANT_COLLECTION",
        "qdrant_client",
        "psycopg",
        "/v1/mnemos/index",
        "subprocess.Popen",
    )
    assert not any(value in text for value in forbidden)


def test_builder_and_sidecar_expose_no_mutation_flags_or_tools():
    builder_help = run_builder("--help").stdout
    assert all(flag not in builder_help for flag in ("--lint", "--fix", "--write", "--collection"))
    assert set(project_mcp_tool_names()) == READ_ONLY_PROJECT_TOOL_NAMES


def test_target_tree_unchanged_by_build_search_and_verify(mnemos_trial_scope, tmp_path):
    before = tracked_and_untracked_hashes(mnemos_trial_scope.root)
    packet = tmp_path / "mnemos-project-memory.md"
    build_trial_packet(mnemos_trial_scope, packet)
    search_packet(packet, "where is collection configuration defined")
    verify_packet(packet, mnemos_trial_scope.root)
    assert tracked_and_untracked_hashes(mnemos_trial_scope.root) == before
```

Also inspect that `service/app.py`, `mnemos/config.py`, `mnemos/retrieval/`,
`docker-compose.yml`, and `mcp_servers/mnemos/server.py` have no trial diff.

- [ ] **Step 2: Run the focused R0 suite**

```powershell
python -m pytest tests/test_local_project_memory_models.py tests/test_local_project_memory_snapshot.py tests/test_local_project_memory_extractors.py tests/test_local_project_memory_packet_cli.py tests/test_local_project_memory_retrieval.py tests/test_mnemos_project_mcp.py tests/test_verify_mnemos_local_stack.py tests/test_local_project_memory_boundaries.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Build the first MNEMOS packet outside the repository**

```powershell
$packet = Join-Path $env:TEMP 'mnemos-project-memory-r0.md'
if (Test-Path $packet) { Remove-Item -LiteralPath $packet }
python tools/build_local_project_memory_packet.py `
  --project-root G:\MNEMOS `
  --repo-id mnemos `
  --scope-root mnemos `
  --scope-root service `
  --scope-root mnemos_sdk `
  --scope-root mcp_servers/mnemos `
  --scope-root tests `
  --scope-file README.md `
  --scope-file docs/architecture.md `
  --scope-file docs/dependency_map.md `
  --scope-file docs/experiments/python_markdown_structured_project_memory_r0_spec.md `
  --output $packet
```

Expected: complete packet, approximately 242 admitted tracked Python/Markdown
files, no target-tree diff, and a printed snapshot ID.

- [ ] **Step 4: Run frozen retrieval checks against the packet**

Use an in-process harness or MCP stdio client to assert these query families:

```text
"where is the default qdrant collection configured" -> mnemos/config.py or docker configuration evidence in admitted scope
"how does the REST health endpoint decide degraded status" -> service/app.py health source span
"how does the MCP bridge search memory" -> mcp_servers/mnemos/server.py search_memory source span
"what tests cover qdrant collection dimension compatibility" -> relevant tests/ source span
"why are default collections excluded from project-memory R0" -> structured project-memory specification section
```

Every result must use `repo_id=mnemos`, the active snapshot, an admitted path,
matching hashes, and exact one-based spans. No Research Intake or outside-scope
candidate may appear.

- [ ] **Step 5: Verify stale behavior without retaining project mutation**

Copy the admitted fixture subset to an OS temporary Git repository, build its
packet, modify one admitted file in that temporary repository, and confirm MCP
search returns `SNAPSHOT_MISMATCH` with no results. Do not modify `G:\MNEMOS`
for this test.

- [ ] **Step 6: Run the live stack preflight and record non-sensitive evidence**

```powershell
$receipt = Join-Path $env:TEMP 'mnemos-local-stack-r0.json'
python tools/verify_mnemos_local_stack.py `
  --compose-file docker-compose.yml `
  --require-research-ui `
  --require-openwebui-proxy `
  --output-json $receipt
```

Copy only summarized statuses, versions, ports, latencies, and reason codes into
the trial document. Link to the temporary receipt path for the local operator;
do not commit host-specific IDs, absolute user paths, tokens, passwords, or
DSNs.

- [ ] **Step 7: Complete the operator trial document**

Record command, scope, snapshot ID, admitted/excluded counts, packet hash,
retrieval observations, stale-abstention evidence, stack-preflight summary,
known limitations, and these explicit conclusions:

```text
PROJECT_MEMORY_PACKET_R0_COMPLETE or PROJECT_MEMORY_PACKET_R0_FAILED
READ_ONLY_MCP_RETRIEVAL_VERIFIED or READ_ONLY_MCP_RETRIEVAL_FAILED
TARGET_PROJECT_MUTATION_COUNT=0
DEFAULT_COLLECTION_MUTATION_COUNT=0
RUNTIME_ROUTE_CHANGE_COUNT=0
LINT_EXECUTION_COUNT=0
CODE_MUTATION_REQUIRES_SEPARATE_HUMAN_APPROVAL
```

- [ ] **Step 8: Run regression and diff checks**

```powershell
python -m pytest tests -q
git status --short
git diff -- service/app.py mnemos/config.py mnemos/retrieval docker-compose.yml mcp_servers/mnemos/server.py
```

Expected: the repository-local test suite passes subject to already-documented
environment exclusions; protected runtime files have no diff; only planned R0
implementation, tests, and docs are changed.

- [ ] **Step 9: Commit the trial evidence**

```powershell
git add tests/test_local_project_memory_boundaries.py docs/experiments/local_project_memory_packet_mcp_sidecar_r0_trial.md mcp_servers/mnemos_project/README.md
git commit -m "test: verify local project memory R0 trial"
```

---

## Final Acceptance Checklist

- [ ] Builder requires explicit project root, repo ID, output, and non-empty scope.
- [ ] Only tracked Python/Markdown files inside explicit scope are admitted.
- [ ] Empty, escaped, ambiguous, secret, unsupported explicit, or excessive scope fails closed.
- [ ] Output is a single integrity-checked packet outside the target project.
- [ ] File hashes, content hashes, source URIs, and one-based spans verify exactly.
- [ ] Packet records exclusions, resource boundaries, dirty state, and approval checkpoints.
- [ ] Python extraction never imports or executes target code.
- [ ] MCP sidecar revalidates live snapshot before every evidence return.
- [ ] MCP sidecar exposes only the five approved read-only tools.
- [ ] Structured retrieval returns active, source-backed MNEMOS Python/Markdown evidence.
- [ ] Cross-scope, tampered, incomplete, or stale evidence causes abstention.
- [ ] Research Intake, runtime REST, Qdrant, PostgreSQL, and the default collection are absent from the packet/retrieval path.
- [ ] Stack verifier detects Compose services by role/labels rather than fixed container names.
- [ ] Stack verifier validates Qdrant, PostgreSQL, MNEMOS, Research UI, proxy, and published ports without mutation.
- [ ] Current default MNEMOS collection, runtime routes, and target-project code remain unchanged.
- [ ] Packet explicitly requires human approval before lint and separate approval before mutation.
- [ ] Any mutation invalidates the packet and requires rebuild before trusted retrieval.
