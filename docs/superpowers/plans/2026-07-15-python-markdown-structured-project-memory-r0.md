# Python + Markdown Structured Project Memory R0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, offline R0 prototype that converts a frozen Session Context Assembler subsystem snapshot into versioned Python and Markdown Engrams, retrieves them in four evaluated modes, and rejects stale or wrong-corpus evidence.

**Architecture:** Add a new `prototype/python_markdown_project_memory_r0/` package with no import path from `service/`, `mnemos/`, or the existing Session Context Assembler. An explicit subsystem scope produces immutable active and historical manifests; AST and Markdown extractors emit validated source-backed artifacts; a file-backed snapshot collection and injected local embedder support lexical, semantic, hybrid, and one-hop expanded evaluation without touching the configured MNEMOS runtime collection.

**Tech Stack:** Python 3.10+ standard library (`ast`, `tokenize`, `subprocess`, `hashlib`, `json`, `pathlib`), existing MNEMOS `Engram`, NumPy, existing `sentence-transformers` dependency in local-files-only mode, pytest, Git CLI with `shell=False`.

## Global Constraints

- First selected subsystem: **Session Context Assembler + consumer-neutral shadow adapter**.
- Active fixture origin is pinned to repository commit `79ae3342eaa01c2e9dd7c1ab0be289f046a1baeb`.
- Historical fixture is explicitly `fixture_local_synthetic_history`; it must never be described as a real MNEMOS Git commit.
- Process only an explicit subsystem allowlist. An absent or empty scope is an error; never infer whole-repository scope.
- No runtime route changes.
- No default retrieval changes.
- No code execution, repository-module import, dynamic introspection, `eval`, `exec`, or `compile` of repository content.
- Snapshot-builder Git subprocess calls use argument arrays, `shell=False`, read-only Git commands, and the selected repository root. The fixture freezer may use `git init/add/commit` only inside a newly created OS-temporary fixture directory; it must reject the MNEMOS working tree as its Git mutation target.
- No code mutation. The prototype reads fixture/repository content and writes only explicit fixture, temporary collection, or benchmark-result paths.
- No automatic durable-memory promotion, governance mutation, contradiction mutation, Resolution Engram mutation, or generated-summary authority.
- No frontier API integration.
- No VS Code extension.
- No graph database or GraphRAG.
- No network listener, SDK surface, deployment configuration, or consumer connection.
- No fallback to the configured/default MNEMOS collection.
- Semantic evaluation uses an explicitly supplied local embedding model with `local_files_only=True`; unavailable model state fails closed.
- Preserve artifact-local lineage, exact source slices, one-based inclusive line spans, file/content hashes, and immutable snapshot identity.
- Do not modify `service/app.py`, `mnemos/retrieval/`, `mnemos/config.py`, `requirements.txt`, existing Session Context Assembler code, or existing Session Context Assembler tests.
- Do not stage or modify the unrelated untracked `logs/` directory.

---

## Selected Subsystem and Fixture Boundary

The Session Context Assembler + shadow adapter is selected because the current
repository contains all R0 evidence classes in a compact boundary:

- Python implementation: `prototype/session_context_assembler/` and
  `prototype/session_context_assembler/shadow_adapter/`;
- tests: `tests/test_session_context_assembler_selector_s1.py` and
  `tests/test_session_context_assembler_shadow_adapter.py`;
- decisions: ADR 0007 and ADR 0008;
- evaluation/closeout evidence:
  `docs/session_context_assembler_phase_4r_notes.md` and
  `docs/session_context_assembler_shadow_adapter_implementation_notes.md`;
- fixture-local agent instructions and handoff records; and
- a controlled historical version of `LocalShadowAdapter.process` and the
  ADR/handoff status fields for stale-source evaluation.

Baseline before planning: the two selected existing test modules pass `61/61`.

The active fixture copies selected files byte-for-byte from the pinned commit.
The historical fixture is a controlled benchmark tree with the same repository
paths but an older fixture-local `LocalShadowAdapter.process` implementation,
ADR 0008 status `Proposed`, and a blocked handoff. Its manifest records
`history_kind = fixture_local_synthetic_history` and a deterministic fixture
Git commit. No report may cite it as historical product evidence.

The required **frozen active snapshot fixture** and **frozen historical snapshot fixture**
are therefore both repository-shaped Git snapshots, while
only the active tree claims byte provenance from the pinned MNEMOS commit.

---

## File Structure

### New prototype package

- Create `prototype/python_markdown_project_memory_r0/__init__.py`: export only the isolated R0 public contracts.
- Create `prototype/python_markdown_project_memory_r0/errors.py`: structured fail-closed error codes and exception.
- Create `prototype/python_markdown_project_memory_r0/canonical.py`: canonical JSON, SHA-256, source URI, collection name, and deterministic artifact identity.
- Create `prototype/python_markdown_project_memory_r0/models.py`: immutable common artifact, snapshot, extraction, candidate, and evaluation dataclasses.
- Create `prototype/python_markdown_project_memory_r0/schema.py`: common field, hash, span, enum, lineage, and Engram-conversion validation.
- Create `prototype/python_markdown_project_memory_r0/snapshot.py`: explicit scope validation, read-only Git identity, file admission, dirty-state capture, and manifest building.
- Create `prototype/python_markdown_project_memory_r0/python_extractor.py`: AST/token-based Python source segmentation without importing code.
- Create `prototype/python_markdown_project_memory_r0/markdown_extractor.py`: source-line Markdown document/section and explicit metadata extraction.
- Create `prototype/python_markdown_project_memory_r0/ingest.py`: manifest-driven extractor dispatch and validated artifact bundle creation.
- Create `prototype/python_markdown_project_memory_r0/collection.py`: immutable file-backed collection per repo snapshot and injected embedding provider.
- Create `prototype/python_markdown_project_memory_r0/retrieval.py`: scoped lexical, semantic, hybrid, and H+E retrieval.
- Create `prototype/python_markdown_project_memory_r0/staleness.py`: active-manifest, historical, dirty-state, and summary-parent checks.
- Create `prototype/python_markdown_project_memory_r0/evaluation.py`: truth-set execution, metrics, hard gates, and report payload.

### New tools and artifacts

- Create `tools/freeze_python_markdown_project_memory_r0_fixtures.py`: one-purpose fixture freezer with a hardcoded selected-path allowlist and deterministic fixture Git history.
- Create `tools/run_python_markdown_project_memory_r0.py`: offline evaluation entry point and JSON/Markdown report writer.
- Create `benchmarks/fixtures/python_markdown_project_memory_r0/session_context_assembler/active/`: frozen active fixture tree.
- Create `benchmarks/fixtures/python_markdown_project_memory_r0/session_context_assembler/historical/`: frozen fixture-local historical tree.
- Create `benchmarks/fixtures/python_markdown_project_memory_r0/session_context_assembler/fixture_origin.json`: pinned origin and controlled historical deltas.
- Create `benchmarks/fixtures/python_markdown_project_memory_r0/session_context_assembler/active_snapshot.manifest.json`.
- Create `benchmarks/fixtures/python_markdown_project_memory_r0/session_context_assembler/historical_snapshot.manifest.json`.
- Create `benchmarks/truthsets/python_markdown_project_memory_r0.json`: fourteen frozen queries across seven families.
- Create `benchmarks/truthsets/python_markdown_project_memory_r0.manifest.json`: truth-set hash and exact oracle IDs.
- Create `benchmarks/results/python_markdown_project_memory_r0.json`.
- Create `benchmarks/results/python_markdown_project_memory_r0.md`.

### New tests

- Create `tests/test_python_markdown_project_memory_r0_schema.py`.
- Create `tests/test_python_markdown_project_memory_r0_snapshot.py`.
- Create `tests/test_python_markdown_project_memory_r0_fixtures.py`.
- Create `tests/test_python_markdown_project_memory_r0_python.py`.
- Create `tests/test_python_markdown_project_memory_r0_markdown.py`.
- Create `tests/test_python_markdown_project_memory_r0_ingest.py`.
- Create `tests/test_python_markdown_project_memory_r0_collection.py`.
- Create `tests/test_python_markdown_project_memory_r0_retrieval.py`.
- Create `tests/test_python_markdown_project_memory_r0_staleness.py`.
- Create `tests/test_python_markdown_project_memory_r0_evaluation.py`.
- Create `tests/test_python_markdown_project_memory_r0_boundaries.py`.

---

## Stage A — Contracts, Snapshot Builder, and Frozen Evidence

### Task 1: Common Artifact Schema and Canonical Identity

**Files:**
- Create: `prototype/python_markdown_project_memory_r0/__init__.py`
- Create: `prototype/python_markdown_project_memory_r0/errors.py`
- Create: `prototype/python_markdown_project_memory_r0/canonical.py`
- Create: `prototype/python_markdown_project_memory_r0/models.py`
- Create: `prototype/python_markdown_project_memory_r0/schema.py`
- Test: `tests/test_python_markdown_project_memory_r0_schema.py`

**Interfaces:**
- Produces: `ProjectMemoryError`, `ErrorCode`, `SourceSpan`, `ProjectArtifact`, `SnapshotFile`, `SnapshotManifest`, `ExtractionFailure`, `ExtractionResult`, `ArtifactBundle`, `RetrievalCandidate`, `EvaluationRecord`.
- Produces: `canonical_json_bytes(value)`, `sha256_bytes(value)`, `project_source_uri(repo_id, file_path, snapshot_id)`, `project_collection_name(repo_id, snapshot_id)`, `artifact_identity(repo_id, snapshot_id, file_path, artifact_type, qualified_name, span, content_hash)`, `validate_artifact(artifact)`, `artifact_to_engram(artifact)`.
- Consumes: existing `mnemos.engram.model.Engram` only as an output representation; no retrieval or runtime import.

- [ ] **Step 1: Write failing canonical and schema tests**

```python
from dataclasses import replace

import pytest

from prototype.python_markdown_project_memory_r0.canonical import artifact_identity, sha256_bytes
from prototype.python_markdown_project_memory_r0.errors import ProjectMemoryError
from prototype.python_markdown_project_memory_r0.models import ProjectArtifact, SourceSpan
from prototype.python_markdown_project_memory_r0.schema import artifact_to_engram, validate_artifact


def _artifact() -> ProjectArtifact:
    content = "def process(self):\n    return True\n"
    snapshot_id = "sha256:" + "a" * 64
    span = SourceSpan(10, 11)
    content_hash = sha256_bytes(content.encode("utf-8"))
    return ProjectArtifact(
        artifact_id=artifact_identity(
            "mnemos", snapshot_id, "prototype/session_context_assembler/adapter.py",
            "python_symbol", "adapter.LocalShadowAdapter.process", span, content_hash,
        ),
        repo_id="mnemos",
        snapshot_id=snapshot_id,
        branch="main",
        commit_hash="b" * 40,
        file_path="prototype/session_context_assembler/adapter.py",
        file_hash="sha256:" + "c" * 64,
        language="python",
        artifact_type="python_symbol",
        qualified_name="adapter.LocalShadowAdapter.process",
        span=span,
        content=content,
        content_hash=content_hash,
        metadata={"symbol_name": "process", "symbol_kind": "method"},
    )


def test_valid_artifact_converts_to_lineage_engram():
    artifact = _artifact()
    validate_artifact(artifact)
    engram = artifact_to_engram(artifact)
    assert engram.id == artifact.artifact_id
    assert engram.metadata["artifact_version"] == artifact.snapshot_id
    assert engram.metadata["provenance_span"] == {"start_line": 10, "end_line": 11}
    assert engram.metadata["source_linked"] is True


@pytest.mark.parametrize("changed", [
    {"content_hash": "sha256:" + "0" * 64},
    {"span": SourceSpan(0, 1)},
    {"language": "javascript"},
    {"artifact_type": "unknown"},
])
def test_invalid_artifact_fails_closed(changed):
    with pytest.raises(ProjectMemoryError):
        validate_artifact(replace(_artifact(), **changed))
```

- [ ] **Step 2: Run schema tests and verify RED**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_schema.py -q
```

Expected: `ModuleNotFoundError` for the new package.

- [ ] **Step 3: Implement immutable contracts and canonical helpers**

Use frozen dataclasses. `ErrorCode` must include all ten spec abstention codes
plus `COLLECTION_UNAVAILABLE` and `COLLECTION_MANIFEST_INVALID` for the harness.
`validate_artifact()` recomputes `content_hash` and `artifact_id`, validates
required fields, controlled types, exact hash formats, relative normalized
paths, line spans, source URI, and language-specific metadata. It never fills
missing lineage silently.

`artifact_to_engram()` copies every common field into Engram metadata and maps
`qualified_name` to `qualified_symbol_name` for Python or to the canonical
document/heading identifier for Markdown. It also writes `artifact_id`,
`artifact_version`, `chunk_id`, `source_uri`, `provenance_span`,
`ingestion_schema_version`, `source_linked=True`, and `is_superseded`.

```python
@dataclass(frozen=True)
class SourceSpan:
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ProjectArtifact:
    artifact_id: str
    repo_id: str
    snapshot_id: str
    branch: str
    commit_hash: str
    file_path: str
    file_hash: str
    language: Literal["python", "markdown"]
    artifact_type: str
    qualified_name: str
    span: SourceSpan
    content: str
    content_hash: str
    metadata: Mapping[str, object]
    is_superseded: bool = False

    @property
    def source_uri(self) -> str:
        return project_source_uri(self.repo_id, self.file_path, self.snapshot_id)

    @property
    def artifact_version(self) -> str:
        return self.snapshot_id

    @property
    def chunk_id(self) -> str:
        return self.artifact_id

    @property
    def provenance_span(self) -> Mapping[str, int]:
        return {"start_line": self.span.start_line, "end_line": self.span.end_line}


@dataclass(frozen=True)
class SnapshotFile:
    file_path: str
    file_hash: str
    language: Literal["python", "markdown"]
    byte_size: int


@dataclass(frozen=True)
class SnapshotManifest:
    repo_id: str
    snapshot_id: str
    branch: str
    commit_hash: str
    working_tree_state: Literal["clean", "dirty"]
    files: tuple[SnapshotFile, ...]
    scope_roots: tuple[str, ...]
    scope_files: tuple[str, ...]
    modified_paths: tuple[str, ...]
    deleted_paths: tuple[str, ...]
    staged_paths: tuple[str, ...]
    admitted_untracked_paths: tuple[str, ...]
    excluded_untracked_count: int
    history_kind: Literal["active", "fixture_local_synthetic_history"]
    ingestion_schema_version: str
    collection_name: str


@dataclass(frozen=True)
class ExtractionFailure:
    code: str
    file_path: str
    safe_message: str
    line: int | None = None


@dataclass(frozen=True)
class ExtractionResult:
    artifacts: tuple[ProjectArtifact, ...]
    failures: tuple[ExtractionFailure, ...]
    parse_status: Literal["parsed", "partial", "failed"]


@dataclass(frozen=True)
class ArtifactBundle:
    manifest: SnapshotManifest
    artifacts: tuple[ProjectArtifact, ...]
    failures: tuple[ExtractionFailure, ...]
```

Define `RetrievalCandidate` with artifact, fused score, candidate origin,
component scores/ranks, applied filters, relationship type, expansion origin,
and hop count. Define `EvaluationRecord` with query/mode identity, expected and
returned IDs, rankings, filters, exclusions, abstention, latency, versions,
collection, snapshot, and model ID. These exact fields are serialized with
canonical JSON in Tasks 7–10.

- [ ] **Step 4: Run schema tests and verify GREEN**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_schema.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the contract slice**

```powershell
git add prototype/python_markdown_project_memory_r0 tests/test_python_markdown_project_memory_r0_schema.py
git commit -m "feat: add project memory R0 artifact contracts"
```

### Task 2: Explicit-Scope Snapshot Manifest Builder

**Files:**
- Create: `prototype/python_markdown_project_memory_r0/snapshot.py`
- Test: `tests/test_python_markdown_project_memory_r0_snapshot.py`

**Interfaces:**
- Consumes: snapshot dataclasses, canonical helpers, `ProjectMemoryError`.
- Produces: `SubsystemScope`, `GitState`, `GitReader`, `build_snapshot_manifest(repo_root, repo_id, scope, git_reader)`, `verify_live_snapshot(repo_root, manifest, git_reader)`.

- [ ] **Step 1: Write failing scope, dirty-tree, and immutability tests**

```python
def test_empty_scope_never_defaults_to_whole_repository(tmp_path):
    repo = make_git_repo(tmp_path, {"allowed/a.py": "VALUE = 1\n", "secret.py": "TOKEN='x'\n"})
    with pytest.raises(ProjectMemoryError, match="REPO_SCOPE_REQUIRED"):
        build_snapshot_manifest(repo, "fixture", SubsystemScope(), GitReader())


def test_allowlist_admits_only_selected_python_and_markdown(tmp_path):
    repo = make_git_repo(tmp_path, {
        "prototype/session_context_assembler/a.py": "VALUE = 1\n",
        "docs/adr/0007.md": "# ADR 0007\n",
        "outside.py": "VALUE = 2\n",
    })
    scope = SubsystemScope(
        roots=("prototype/session_context_assembler",),
        files=("docs/adr/0007.md",),
        max_files=20,
        max_total_bytes=100_000,
    )
    manifest = build_snapshot_manifest(repo, "fixture", scope, GitReader())
    assert [item.file_path for item in manifest.files] == [
        "docs/adr/0007.md", "prototype/session_context_assembler/a.py",
    ]


def test_dirty_tree_changes_snapshot_not_base_commit(tmp_path):
    repo = make_git_repo(tmp_path, {"selected/a.py": "VALUE = 1\n"})
    clean = build_snapshot_manifest(repo, "fixture", scope_for("selected"), GitReader())
    (repo / "selected/a.py").write_text("VALUE = 2\n", encoding="utf-8")
    dirty = build_snapshot_manifest(repo, "fixture", scope_for("selected"), GitReader())
    assert dirty.commit_hash == clean.commit_hash
    assert dirty.snapshot_id != clean.snapshot_id
    assert dirty.working_tree_state == "dirty"
    assert dirty.modified_paths == ("selected/a.py",)
```

Also test symlink escape rejection, ignored/untracked exclusion, admitted
untracked opt-in, file/byte ceilings, branch mismatch, detached HEAD, unchanged
file hashes before/after building, and `SNAPSHOT_MISMATCH` after a live change.

- [ ] **Step 2: Run snapshot tests and verify RED**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_snapshot.py -q
```

Expected: import failure for `snapshot.py`.

- [ ] **Step 3: Implement read-only Git and manifest construction**

```python
@dataclass(frozen=True)
class SubsystemScope:
    roots: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    include_untracked: bool = False
    max_files: int = 200
    max_total_bytes: int = 2_000_000


class GitReader:
    ALLOWED_VERBS = frozenset({"rev-parse", "branch", "ls-files", "status", "check-ignore"})

    def run(self, repo_root: Path, *args: str) -> str:
        if not args or args[0] not in self.ALLOWED_VERBS:
            raise ProjectMemoryError(ErrorCode.DISCLOSURE_DENIED, "read-only Git verb required")
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args], check=True,
            capture_output=True, text=True, encoding="utf-8", shell=False,
        )
        return completed.stdout
```

Resolve and confine paths below the repository root. Admit only `.py` and
`.md`. Hash exact bytes, sort arrays, disclose dirty state, and derive
`snapshot_id` from canonical manifest content excluding creation time and the
snapshot ID itself. Treat `collection_name` as a deterministic derived field
and exclude it from the snapshot-ID preimage; otherwise deriving the collection
name from `snapshot_id` creates a circular hash dependency. The serialized
manifest receives a separate SHA-256 integrity hash that covers the derived
collection name.

- [ ] **Step 4: Run snapshot and selected existing tests**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_snapshot.py tests/test_session_context_assembler_selector_s1.py tests/test_session_context_assembler_shadow_adapter.py -q
```

Expected: all new tests pass and the existing subtotal remains `61 passed`.

- [ ] **Step 5: Commit the snapshot slice**

```powershell
git add prototype/python_markdown_project_memory_r0/snapshot.py tests/test_python_markdown_project_memory_r0_snapshot.py
git commit -m "feat: add scoped project snapshot manifests"
```

### Task 3: Freeze Active/Historical Fixtures and Truth Set Before Extractor Tuning

**Files:**
- Create: `tools/freeze_python_markdown_project_memory_r0_fixtures.py`
- Create: `tests/test_python_markdown_project_memory_r0_fixtures.py`
- Create: `benchmarks/fixtures/python_markdown_project_memory_r0/session_context_assembler/**`
- Create: `benchmarks/truthsets/python_markdown_project_memory_r0.json`
- Create: `benchmarks/truthsets/python_markdown_project_memory_r0.manifest.json`

**Interfaces:**
- Consumes: snapshot builder and canonical identity helpers.
- Produces: immutable fixture trees, exact snapshot IDs, fourteen query records, exact oracle IDs, and frozen hashes.

- [ ] **Step 1: Write the failing fixture-integrity test**

```python
EXPECTED_FAMILIES = {
    "exact_symbol_lookup": 2,
    "conceptual_behavior_lookup": 2,
    "test_association_lookup": 2,
    "adr_decision_lookup": 2,
    "stale_version_detection": 2,
    "cross_file_relationship_lookup": 2,
    "wrong_corpus_rejection": 2,
}


def test_fixture_origin_and_truthset_are_frozen():
    origin = load_json(FIXTURE_ROOT / "fixture_origin.json")
    truthset = load_json(TRUTHSET)
    manifest = load_json(TRUTHSET_MANIFEST)
    assert origin["active_origin_commit"] == "79ae3342eaa01c2e9dd7c1ab0be289f046a1baeb"
    assert origin["historical_history_kind"] == "fixture_local_synthetic_history"
    assert Counter(row["family"] for row in truthset["queries"]) == EXPECTED_FAMILIES
    assert all(row["expected_artifact_ids"] for row in truthset["queries"][:-2])
    assert verify_manifest_hashes(origin, manifest)
```

- [ ] **Step 2: Run fixture test and verify RED**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_fixtures.py -q
```

Expected: fixture files do not exist.

- [ ] **Step 3: Implement the freezer with an exact allowlist**

```python
ACTIVE_SOURCE_PATHS = (
    "prototype/session_context_assembler/selector_s1.py",
    "prototype/session_context_assembler/shadow_adapter/adapter.py",
    "prototype/session_context_assembler/shadow_adapter/kill_switch.py",
    "prototype/session_context_assembler/shadow_adapter/response_builder_and_digest.py",
    "tests/test_session_context_assembler_selector_s1.py",
    "tests/test_session_context_assembler_shadow_adapter.py",
    "docs/adr/0007-session-context-assembler-shadow-only.md",
    "docs/adr/0008-consumer-neutral-read-only-shadow-adapter-implementation.md",
    "docs/session_context_assembler_phase_4r_notes.md",
    "docs/session_context_assembler_shadow_adapter_implementation_notes.md",
)
```

Copy exact active bytes with `git show <pinned-commit>:<path>`, argument arrays,
and `shell=False`. Add fixture-authored `AGENTS.md` and
`docs/session_context_assembler_r0_handoff.md`. The active handoff is
`Status: Complete`; historical is `Status: Blocked` with explicit
`Superseded by`.

Create historical from active, then apply only three recorded deltas: replace
`adapter.py` with a fixture-local older implementation that still defines a
changed `LocalShadowAdapter.process`; change ADR 0008 status to `Proposed`;
and use the blocked handoff. Create deterministic temporary Git commits, record
their real fixture commit hashes, remove `.git`, and freeze only admitted
files/manifests. Refuse overwrite unless `--replace` is explicit.

The controlled historical adapter source is exactly:

```python
"""Fixture-local historical shadow adapter; not MNEMOS product history."""

from __future__ import annotations


class LocalShadowAdapter:
    """Earlier fixture contract without replay-policy pinning."""

    def process(
        self,
        request: dict,
        inputs: dict,
        policy: dict,
        transport: dict,
        now: object | None = None,
    ) -> dict:
        return {
            "ok": True,
            "shadow_only": True,
            "historical_fixture": True,
            "request_id": request.get("request_id"),
        }
```

- [ ] **Step 4: Freeze fourteen oracle queries**

Use two queries per family and these fixed targets:

```text
exact: LocalShadowAdapter.process; validate_response_contract
concept: kill-switch assembly blocking; artifact-local digest validation
test association: replay conflict tests; budget-abstention tests
ADR/decision: shadow-only reason in ADR 0007; adapter authorization in ADR 0008
stale: historical LocalShadowAdapter.process; historical Proposed ADR status
cross-file: adapter -> validate_response_contract; test -> LocalShadowAdapter
wrong corpus: R²-Mem Evaluator Learner; GDPR Article 141
```

Oracle records contain explicit path, qualified/heading name, manually
enumerated one-based span, source-slice hash, and exact artifact ID. The freezer
verifies each oracle slice occurs exactly once and must not call either future
extractor.

- [ ] **Step 5: Freeze and verify**

```powershell
python tools/freeze_python_markdown_project_memory_r0_fixtures.py
python -m pytest tests/test_python_markdown_project_memory_r0_fixtures.py -q
```

Expected: freeze succeeds and integrity tests pass.

- [ ] **Step 6: Commit the frozen evidence**

```powershell
git add tools/freeze_python_markdown_project_memory_r0_fixtures.py tests/test_python_markdown_project_memory_r0_fixtures.py benchmarks/fixtures/python_markdown_project_memory_r0 benchmarks/truthsets/python_markdown_project_memory_r0.json benchmarks/truthsets/python_markdown_project_memory_r0.manifest.json
git commit -m "test: freeze structured project memory R0 corpus"
```

After this commit, extractor/retrieval tuning must not edit R0 fixtures or
truth sets. Corrections create an R1 corpus.

---

## Stage B — Structured Extractors and Admission

### Task 4: Python AST Extractor

**Files:**
- Create: `prototype/python_markdown_project_memory_r0/python_extractor.py`
- Test: `tests/test_python_markdown_project_memory_r0_python.py`

**Interfaces:**
- Consumes: `SnapshotManifest`, `SnapshotFile`, `ProjectArtifact`, `ExtractionResult`, canonical helpers.
- Produces: `extract_python_file(repo_root, manifest, snapshot_file) -> ExtractionResult`.
- Guarantees: no import/execution of target code; exact source slices; controlled artifact/symbol kinds.

- [ ] **Step 1: Write failing AST extraction tests**

```python
PYTHON_SOURCE = '''\
import os
from pathlib import Path as RepoPath

MAX_TOKEN_BUDGET: int = 200

def audit(func):
    return func

class LocalShadowAdapter:
    @audit
    def process(self, request: dict) -> dict:
        return {"ok": True}

@app.post("/v1/context")
async def context_route() -> dict:
    return {"ok": True}

def test_process_rejects_stale_source():
    assert True
'''


def test_python_extractor_emits_structural_artifacts(repo_fixture):
    result = extract_selected_python(repo_fixture, PYTHON_SOURCE)
    kinds = {(a.artifact_type, a.metadata["symbol_kind"]) for a in result.artifacts}
    assert ("python_module", "module") in kinds
    assert ("python_symbol", "class") in kinds
    assert ("python_symbol", "method") in kinds
    assert ("python_symbol", "route_handler") in kinds
    assert ("python_symbol", "test_function") in kinds
    assert ("python_import_block", "import_block") in kinds
    assert ("python_config_constant", "config_constant") in kinds
    assert ("python_decorator_application", "decorator_application") in kinds
    assert all(a.content == source_lines(PYTHON_SOURCE, a.span) for a in result.artifacts)
    required = {"symbol_name", "symbol_kind", "parent_symbol", "imports", "test_marker"}
    assert all(required <= set(a.metadata) for a in result.artifacts)
```

Also test nested qualified names, parent symbols, imports/aliases, async flags,
test signals, literal-only constants, route heuristic labels, decorators, large
class `embedding_eligible=False`, syntax-error `partial`, unsupported encoding,
and absence of repository imports or dynamic execution.

- [ ] **Step 2: Run Python extractor tests and verify RED**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_python.py -q
```

Expected: import failure for `python_extractor.py`.

- [ ] **Step 3: Implement AST/token segmentation**

```python
def extract_python_file(
    repo_root: Path,
    manifest: SnapshotManifest,
    snapshot_file: SnapshotFile,
) -> ExtractionResult:
    source_bytes = (repo_root / snapshot_file.file_path).read_bytes()
    if sha256_bytes(source_bytes) != snapshot_file.file_hash:
        raise ProjectMemoryError(ErrorCode.STALE_SOURCE_DETECTED, snapshot_file.file_path)
    source = source_bytes.decode("utf-8")
    tree = ast.parse(source, filename=snapshot_file.file_path, type_comments=True)
    return PythonArtifactVisitor(manifest, snapshot_file, source).extract(tree)
```

Use AST `lineno`/`end_lineno`, decorator line numbers, and exact
`splitlines(keepends=True)` slices. Implement module, class, function, method,
test, decorator application, contiguous import block, literal uppercase
constant, and allowlisted route-handler rules from the spec. Record
`detection_basis`; never infer decorator definitions or runtime route status.

- [ ] **Step 4: Run extractor and fixture tests**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_python.py tests/test_python_markdown_project_memory_r0_fixtures.py -q
```

Expected: all tests pass and fixture hashes remain unchanged.

- [ ] **Step 5: Commit the Python extractor**

```powershell
git add prototype/python_markdown_project_memory_r0/python_extractor.py tests/test_python_markdown_project_memory_r0_python.py
git commit -m "feat: extract source-backed Python artifacts"
```

### Task 5: Markdown Structural Extractor

**Files:**
- Create: `prototype/python_markdown_project_memory_r0/markdown_extractor.py`
- Test: `tests/test_python_markdown_project_memory_r0_markdown.py`

**Interfaces:**
- Consumes: common snapshot/artifact contracts.
- Produces: `extract_markdown_file(repo_root, manifest, snapshot_file) -> ExtractionResult`.
- Guarantees: exact document/section slices and explicit-only status/date/supersession.

- [ ] **Step 1: Write failing Markdown tests**

```python
ADR_SOURCE = '''\
# ADR 0008: Consumer-Neutral Adapter

Date: 2026-06-22
Status: Accepted
Supersedes: ADR 0006

## Context

The adapter is read-only.

## Decision

Keep the adapter shadow-only.
'''


def test_markdown_preserves_heading_status_and_supersession(repo_fixture):
    result = extract_selected_markdown(repo_fixture, "docs/adr/0008-example.md", ADR_SOURCE)
    document = next(a for a in result.artifacts if a.artifact_type == "markdown_adr")
    decision = next(
        a for a in result.artifacts
        if a.metadata["heading_path"] == ["ADR 0008: Consumer-Neutral Adapter", "Decision"]
    )
    assert document.metadata["heading_path"] == []
    assert document.metadata["status"] == {"raw": "Accepted", "normalized": "accepted"}
    assert document.metadata["decision_date"] == "2026-06-22"
    assert document.metadata["supersedes"] == ["ADR 0006"]
    assert decision.content == "## Decision\n\nKeep the adapter shadow-only.\n"
    required = {"heading_path", "status", "decision_date", "supersedes", "superseded_by"}
    assert all(required <= set(a.metadata) for a in result.artifacts)
```

Also test ATX/Setext headings, nested heading paths, YAML front matter,
unrecognized explicit status -> `unknown_explicit`, no prose status inference,
explicit `superseded_by`, unresolved literal references, `AGENTS.md`, handoff,
evaluation closeout, preamble, and inert embedded HTML.

- [ ] **Step 2: Run Markdown tests and verify RED**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_markdown.py -q
```

Expected: import failure for `markdown_extractor.py`.

- [ ] **Step 3: Implement source-line Markdown extraction**

Implement these exact typed interfaces:

```python
def extract_markdown_file(
    repo_root: Path,
    manifest: SnapshotManifest,
    snapshot_file: SnapshotFile,
) -> ExtractionResult:
    source_bytes = (repo_root / snapshot_file.file_path).read_bytes()
    if sha256_bytes(source_bytes) != snapshot_file.file_hash:
        raise ProjectMemoryError(ErrorCode.STALE_SOURCE_DETECTED, snapshot_file.file_path)
    source = source_bytes.decode("utf-8")
    blocks = parse_heading_blocks(source)
    explicit = explicit_document_metadata(source)
    return MarkdownArtifactBuilder(manifest, snapshot_file, source).build(blocks, explicit)
```

The private helpers are `parse_heading_blocks(source) -> Sequence[HeadingBlock]`,
`classify_markdown_document(file_path, source) -> tuple[str, Sequence[str]]`,
and `explicit_document_metadata(source) -> ExplicitMarkdownMetadata`.

Classification uses filename/path, front matter, title, heading, and labelled
fields only. Preserve raw status and map only approved normalized values. Do
not infer supersession by date/similarity. Each section ends before the next
heading of equal or higher level and includes its heading line.

- [ ] **Step 4: Run Markdown, schema, and fixture tests**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_markdown.py tests/test_python_markdown_project_memory_r0_schema.py tests/test_python_markdown_project_memory_r0_fixtures.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the Markdown extractor**

```powershell
git add prototype/python_markdown_project_memory_r0/markdown_extractor.py tests/test_python_markdown_project_memory_r0_markdown.py
git commit -m "feat: extract structured Markdown evidence"
```

### Task 6: Manifest-Driven Ingestion and Common Validation Gate

**Files:**
- Create: `prototype/python_markdown_project_memory_r0/ingest.py`
- Test: `tests/test_python_markdown_project_memory_r0_ingest.py`

**Interfaces:**
- Consumes: frozen manifest, Python/Markdown extractors, `validate_artifact()`.
- Produces: `ingest_snapshot(repo_root, manifest) -> ArtifactBundle`.
- Guarantees: manifest files only, every artifact validated, parse failures retained, no discovery/fallback.

- [ ] **Step 1: Write failing admission tests**

```python
def test_ingestion_reads_only_manifest_files(active_fixture):
    unexpected = active_fixture.root / "outside_scope.py"
    unexpected.write_text("SECRET = 'not admitted'\n", encoding="utf-8")
    bundle = ingest_snapshot(active_fixture.root, active_fixture.manifest)
    assert "outside_scope.py" not in {a.file_path for a in bundle.artifacts}


def test_parse_failure_is_not_flat_chunked(broken_fixture):
    bundle = ingest_snapshot(broken_fixture.root, broken_fixture.manifest)
    assert bundle.failures[0].code == "STRUCTURED_PARSE_INCOMPLETE"
    assert not any(a.file_path == "selected/broken.py" for a in bundle.artifacts)
```

Also test deterministic order, duplicate ID rejection, metadata completeness,
file/content hashes, source spans, and no fixture mutation.

- [ ] **Step 2: Run ingestion tests and verify RED**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_ingest.py -q
```

Expected: import failure for `ingest.py`.

- [ ] **Step 3: Implement strict manifest dispatch**

```python
EXTRACTORS = {"python": extract_python_file, "markdown": extract_markdown_file}


def ingest_snapshot(repo_root: Path, manifest: SnapshotManifest) -> ArtifactBundle:
    artifacts: list[ProjectArtifact] = []
    failures: list[ExtractionFailure] = []
    for snapshot_file in manifest.files:
        result = EXTRACTORS[snapshot_file.language](repo_root, manifest, snapshot_file)
        for artifact in result.artifacts:
            validate_artifact(artifact)
            artifacts.append(artifact)
        failures.extend(result.failures)
    ensure_unique_artifact_ids(artifacts)
    return ArtifactBundle(tuple(sorted(artifacts, key=artifact_sort_key)), tuple(failures))
```

Do not call `rglob()`, `os.walk()`, or Git discovery in ingestion.

- [ ] **Step 4: Run Stage B tests**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_schema.py tests/test_python_markdown_project_memory_r0_python.py tests/test_python_markdown_project_memory_r0_markdown.py tests/test_python_markdown_project_memory_r0_ingest.py -q
```

Expected: all Stage B tests pass.

- [ ] **Step 5: Commit the ingestion gate**

```powershell
git add prototype/python_markdown_project_memory_r0/ingest.py tests/test_python_markdown_project_memory_r0_ingest.py
git commit -m "feat: validate scoped project artifact ingestion"
```

---

## Stage C — Isolated Collections and Four Retrieval Modes

### Task 7: Immutable Snapshot Collection Isolation Harness

**Files:**
- Create: `prototype/python_markdown_project_memory_r0/collection.py`
- Test: `tests/test_python_markdown_project_memory_r0_collection.py`

**Interfaces:**
- Consumes: `ArtifactBundle`, `SnapshotManifest`, `ProjectArtifact`, NumPy.
- Produces: `EmbeddingProvider`, `FrozenTestEmbedder`, `LocalSentenceTransformerEmbedder`, `SnapshotCollectionHarness`.
- Persists per collection: `manifest.json`, `artifacts.jsonl`, `vectors.npy`, `SEALED`.

- [ ] **Step 1: Write failing physical-isolation tests**

```python
def test_one_physical_collection_per_snapshot(tmp_path, active_bundle, historical_bundle):
    harness = SnapshotCollectionHarness(tmp_path, FrozenTestEmbedder())
    active = harness.create(active_bundle.manifest, active_bundle.artifacts)
    historical = harness.create(historical_bundle.manifest, historical_bundle.artifacts)
    assert active.path != historical.path
    assert active.collection_name != historical.collection_name
    assert active.path.joinpath("SEALED").exists()
    assert historical.path.joinpath("SEALED").exists()


def test_missing_collection_never_falls_back(tmp_path, active_manifest):
    harness = SnapshotCollectionHarness(tmp_path, FrozenTestEmbedder())
    with pytest.raises(ProjectMemoryError, match="COLLECTION_UNAVAILABLE"):
        harness.open(active_manifest)


def test_manifest_tamper_quarantines_collection(tmp_path, active_bundle):
    harness = SnapshotCollectionHarness(tmp_path, FrozenTestEmbedder())
    handle = harness.create(active_bundle.manifest, active_bundle.artifacts)
    handle.path.joinpath("manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ProjectMemoryError, match="COLLECTION_MANIFEST_INVALID"):
        harness.open(active_bundle.manifest)
```

Also seed a fake research-paper collection and prove an active handle cannot
enumerate/search it; test sealed immutability and repo/snapshot checks on every
artifact.

- [ ] **Step 2: Run collection tests and verify RED**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_collection.py -q
```

Expected: import failure for `collection.py`.

- [ ] **Step 3: Implement file-backed collections**

The `EmbeddingProvider` protocol exposes read-only `model_id: str`,
`dimension: int`, `embed_documents(texts: Sequence[str]) -> np.ndarray`, and
`embed_query(text: str) -> np.ndarray`. `SnapshotCollectionHarness` exposes
`__init__(root, embedder)`, `create(manifest, artifacts) -> CollectionHandle`,
and `open(manifest) -> CollectionHandle`.

Derive the name
only from repo/snapshot. Write a temporary sibling, verify, atomically rename,
then seal. Reuse a sealed collection only when byte-identical. `open()` accepts
an expected manifest, never a free-form/default name. The local sentence
transformer uses CPU and `local_files_only=True` without online retry.

- [ ] **Step 4: Run collection tests**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_collection.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the collection harness**

```powershell
git add prototype/python_markdown_project_memory_r0/collection.py tests/test_python_markdown_project_memory_r0_collection.py
git commit -m "feat: isolate immutable project snapshot collections"
```

### Task 8: Lexical, Semantic, and Hybrid Retrieval

**Files:**
- Create: `prototype/python_markdown_project_memory_r0/retrieval.py`
- Test: `tests/test_python_markdown_project_memory_r0_retrieval.py`

**Interfaces:**
- Consumes: exact collection handle, rows/vectors, metadata filters.
- Produces: `RetrievalMode`, `ProjectQuery`, `search_project(handle, query) -> Sequence[RetrievalCandidate]`.
- Candidate origins: `lexical`, `semantic`, `hybrid`; H+E is completed in Task 9.

- [ ] **Step 1: Write failing mode/filter tests**

```python
def test_exact_identifier_ranks_first(active_handle):
    query = ProjectQuery(
        text="LocalShadowAdapter.process",
        repo_id=active_handle.manifest.repo_id,
        snapshot_id=active_handle.manifest.snapshot_id,
        mode=RetrievalMode.LEXICAL,
        top_k=5,
    )
    results = search_project(active_handle, query)
    assert results[0].artifact.metadata["qualified_symbol_name"].endswith("LocalShadowAdapter.process")
    assert results[0].candidate_origin == "lexical"


def test_metadata_filters_are_eligibility_gates(active_handle):
    results = search_project(
        active_handle,
        project_query("shadow-only", mode="hybrid", language="markdown", artifact_type="markdown_adr"),
    )
    assert results
    assert all(row.artifact.language == "markdown" for row in results)
    assert all(row.artifact.artifact_type == "markdown_adr" for row in results)
```

Also test paths, routes, errors, env vars, constants, snake/dotted tokens,
conceptual semantic retrieval, deterministic RRF/component scores, stable
tie-breaks, wrong scope, and zero collection fan-out.

- [ ] **Step 2: Run retrieval tests and verify RED**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_retrieval.py -q
```

Expected: import failure for `retrieval.py`.

- [ ] **Step 3: Implement scoped L/S/H retrieval**

```python
class RetrievalMode(str, Enum):
    LEXICAL = "L"
    SEMANTIC = "S"
    HYBRID = "H"
    HYBRID_EXPANDED = "H+E"


@dataclass(frozen=True)
class ProjectQuery:
    text: str
    repo_id: str
    snapshot_id: str
    mode: RetrievalMode
    top_k: int = 5
    branch: str | None = None
    commit_hash: str | None = None
    language: str | None = None
    artifact_type: str | None = None
    symbol_kind: str | None = None
    file_path: str | None = None
    historical: bool = False
```

Lexical scoring indexes content plus identifier/path metadata and boosts exact
full-field matches. Semantic uses cosine similarity. Hybrid uses deterministic
RRF with `k=60`, equal weights, and artifact ID final tie-break. Record lane
scores/ranks and applied filters.

- [ ] **Step 4: Run L/S/H tests**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_retrieval.py tests/test_python_markdown_project_memory_r0_collection.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the three retrieval modes**

```powershell
git add prototype/python_markdown_project_memory_r0/retrieval.py tests/test_python_markdown_project_memory_r0_retrieval.py
git commit -m "feat: add scoped project retrieval modes"
```

### Task 9: Staleness Guard and One-Hop Structural Expansion

**Files:**
- Create: `prototype/python_markdown_project_memory_r0/staleness.py`
- Modify: `prototype/python_markdown_project_memory_r0/retrieval.py`
- Create: `tests/test_python_markdown_project_memory_r0_staleness.py`
- Modify: `tests/test_python_markdown_project_memory_r0_retrieval.py`

**Interfaces:**
- Produces: `validate_active_candidate(manifest, artifact)`, `validate_summary_parents(manifest, summary, artifacts_by_id)`, `explicit_relationships(bundle)`, and H+E candidates.
- Allowed relationships: `contains`, `decorated_by`, `imports_name`, `exact_test_reference`, `document_contains`, `explicit_markdown_reference`.

- [ ] **Step 1: Write failing stale/H+E tests**

```python
def test_historical_same_name_symbol_is_stale(active_manifest, historical_process):
    assert historical_process.qualified_name.endswith("LocalShadowAdapter.process")
    with pytest.raises(ProjectMemoryError, match="STALE_SOURCE_DETECTED"):
        validate_active_candidate(active_manifest, historical_process)


def test_h_plus_e_adds_only_explicit_one_hop_relationships(active_handle):
    results = search_project(active_handle, project_query("LocalShadowAdapter", mode="H+E"))
    expanded = [row for row in results if row.candidate_origin == "structural_expansion"]
    assert expanded
    assert all(row.expansion_hops == 1 for row in expanded)
    assert {row.relationship_type for row in expanded} <= ALLOWED_RELATIONSHIP_TYPES
    assert not any(row.relationship_type in {"calls", "called_by"} for row in expanded)
```

Also test changed file/span/content, deleted path, explicitly historical query,
stale summary parents, dirty/branch/commit mismatch, parse failure, cross-scope
hard failure, one-hop cap, and relationship provenance.

- [ ] **Step 2: Run stale/H+E tests and verify RED**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_staleness.py tests/test_python_markdown_project_memory_r0_retrieval.py -q
```

Expected: failures for missing staleness and expansion behavior.

- [ ] **Step 3: Implement active validation and explicit relationships**

Build active file and qualified-symbol maps. Active candidates must match repo,
snapshot, file hash, source span, content hash, and artifact identity.
Historical mode is explicit and labels snapshot/commit. Relationships derive
only from extracted containment, decorators, syntactic imports, and exact
source references. Never emit runtime caller/callee edges. Add at most one hop
and five expanded artifacts, validating and deduplicating every result.

- [ ] **Step 4: Run all Stage C tests**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_collection.py tests/test_python_markdown_project_memory_r0_retrieval.py tests/test_python_markdown_project_memory_r0_staleness.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit staleness and H+E**

```powershell
git add prototype/python_markdown_project_memory_r0/staleness.py prototype/python_markdown_project_memory_r0/retrieval.py tests/test_python_markdown_project_memory_r0_staleness.py tests/test_python_markdown_project_memory_r0_retrieval.py
git commit -m "feat: reject stale project evidence and expand explicit context"
```

---

## Stage D — Evaluation, Boundary Audit, and R0 Report

### Task 10: Truth-Set Evaluation Engine and Report Renderer

**Files:**
- Create: `prototype/python_markdown_project_memory_r0/evaluation.py`
- Create: `tools/run_python_markdown_project_memory_r0.py`
- Test: `tests/test_python_markdown_project_memory_r0_evaluation.py`

**Interfaces:**
- Consumes: frozen fixtures/truth set, four retrieval modes, injected embedder.
- Produces: `run_evaluation(active_fixture, historical_fixture, truthset, embedder, collection_root) -> EvaluationReport`, deterministic JSON, and Markdown.
- Hard gates match Section 12.2 of the approved specification.

- [ ] **Step 1: Write failing metric/report tests**

```python
def test_evaluation_runs_all_queries_in_all_modes(frozen_test_embedder):
    report = run_frozen_evaluation(frozen_test_embedder)
    assert report.query_count == 14
    assert report.modes == ("L", "S", "H", "H+E")
    assert len(report.records) == 56
    assert report.hard_gates["wrong_repository_leakage"] == 0
    assert report.hard_gates["wrong_snapshot_active_leakage"] == 0
    assert report.hard_gates["research_paper_collection_leakage"] == 0
    assert report.hard_gates["unlabelled_stale_artifact_count"] == 0


def test_cross_scope_candidate_forces_fail(report_factory):
    report = report_factory(cross_scope_candidates=1)
    assert report.recommendation != "PASS"
    assert "CROSS_SCOPE_CANDIDATE_DETECTED" in report.failure_codes
```

Also test metadata/hash/span rates, exact-symbol top-1, family recall floors,
inferred-call-graph count, execution/mutation counts, model/manifest disclosure,
determinism, and JSON/Markdown agreement.

- [ ] **Step 2: Run evaluation tests and verify RED**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_evaluation.py -q
```

Expected: import failure for `evaluation.py`.

- [ ] **Step 3: Implement evaluation and gates**

```python
REQUIRED_GATES = {
    "required_metadata_completeness": 1.0,
    "file_hash_verification_rate": 1.0,
    "content_hash_verification_rate": 1.0,
    "source_span_fidelity": 1.0,
    "exact_symbol_top1_accuracy": 1.0,
    "required_artifact_recall_overall_min": 0.90,
    "required_artifact_recall_family_min": 0.80,
    "wrong_repository_leakage": 0,
    "wrong_snapshot_active_leakage": 0,
    "research_paper_collection_leakage": 0,
    "unlabelled_stale_artifact_count": 0,
    "inferred_call_graph_claims": 0,
    "unauthorized_code_execution": 0,
    "unauthorized_memory_or_governance_mutation": 0,
}
```

Each query/mode record includes expected/returned IDs, ranks, lane scores,
candidate origin, filters, expansion relationship, stale/exclusion/abstention,
latency, candidate count, parser/ingestion versions, collection, snapshot, and
model ID. Zero-tolerance failures override aggregate recall.

The CLI requires explicit active/historical fixture and manifest paths,
truth-set paths, collection root, embedding model, and result paths.
`--local-files-only` is mandatory. It must not accept a MNEMOS base URL or a
default collection name.

- [ ] **Step 4: Run evaluation harness tests with injected deterministic embeddings**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_evaluation.py -q
```

Expected: all harness tests pass; this is not the final semantic quality claim.

- [ ] **Step 5: Commit evaluation code before the final corpus run**

```powershell
git add prototype/python_markdown_project_memory_r0/evaluation.py tools/run_python_markdown_project_memory_r0.py tests/test_python_markdown_project_memory_r0_evaluation.py
git commit -m "feat: evaluate structured project memory R0"
```

### Task 11: Boundary Gate, Full Verification, and Final R0 Report

**Files:**
- Create: `tests/test_python_markdown_project_memory_r0_boundaries.py`
- Create: `benchmarks/results/python_markdown_project_memory_r0.json`
- Create: `benchmarks/results/python_markdown_project_memory_r0.md`

**Interfaces:**
- Consumes: complete isolated package, frozen manifests, locally cached sentence transformer.
- Produces: static boundary evidence and final R0 result artifacts.

- [ ] **Step 1: Write failing architecture-boundary tests**

```python
FORBIDDEN_IMPORT_PREFIXES = (
    "service", "fastapi", "flask", "requests", "socket", "importlib", "runpy",
)
FORBIDDEN_CALL_NAMES = {"eval", "exec", "compile", "__import__"}


def test_r0_package_has_no_runtime_network_or_execution_imports():
    violations = scan_python_boundary(
        PACKAGE_ROOT, FORBIDDEN_IMPORT_PREFIXES, FORBIDDEN_CALL_NAMES,
    )
    assert violations == []


def test_snapshot_git_commands_are_read_only_and_shell_is_false():
    source = Path("prototype/python_markdown_project_memory_r0/snapshot.py").read_text(encoding="utf-8")
    assert "shell=False" in source
    assert "git add" not in source
    assert "git commit" not in source
    assert "git checkout" not in source


def test_no_runtime_module_imports_r0_package():
    for root in (Path("service"), Path("mnemos")):
        assert "python_markdown_project_memory_r0" not in read_all_python(root)
```

Also assert the runner has no MNEMOS URL/default collection option, scope is
non-empty/capped, no graph dependency exists, source trees are never write
targets, and selected SCA files retain their pre-plan Git hashes.

- [ ] **Step 2: Run boundary tests and handle RED honestly**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_boundaries.py -q
```

Expected: report the actual result. If RED, fix only the new package/tool
violation and rerun; never weaken the boundary assertion.

- [ ] **Step 3: Run complete R0 and selected regression suites**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_schema.py tests/test_python_markdown_project_memory_r0_snapshot.py tests/test_python_markdown_project_memory_r0_fixtures.py tests/test_python_markdown_project_memory_r0_python.py tests/test_python_markdown_project_memory_r0_markdown.py tests/test_python_markdown_project_memory_r0_ingest.py tests/test_python_markdown_project_memory_r0_collection.py tests/test_python_markdown_project_memory_r0_retrieval.py tests/test_python_markdown_project_memory_r0_staleness.py tests/test_python_markdown_project_memory_r0_evaluation.py tests/test_python_markdown_project_memory_r0_boundaries.py tests/test_session_context_assembler_selector_s1.py tests/test_session_context_assembler_shadow_adapter.py -q
```

Expected: all tests pass; existing SCA subtotal remains `61 passed`.

- [ ] **Step 4: Preflight the local semantic model without network fallback**

```powershell
$env:MNEMOS_PROJECT_MEMORY_R0_MODEL='sentence-transformers/all-MiniLM-L6-v2'
@'
import os
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(
    os.environ["MNEMOS_PROJECT_MEMORY_R0_MODEL"],
    device="cpu",
    local_files_only=True,
)
print(model.get_sentence_embedding_dimension())
'@ | python -
```

Expected: a positive dimension with no download. If unavailable, stop with
`MODEL_UNAVAILABLE`; do not download online or claim final R0 completion.

- [ ] **Step 5: Run the final frozen R0 evaluation**

```powershell
$collectionRoot = Join-Path $env:TEMP 'mnemos-project-memory-r0-collections'
python tools/run_python_markdown_project_memory_r0.py `
  --active-fixture benchmarks/fixtures/python_markdown_project_memory_r0/session_context_assembler/active `
  --active-manifest benchmarks/fixtures/python_markdown_project_memory_r0/session_context_assembler/active_snapshot.manifest.json `
  --historical-fixture benchmarks/fixtures/python_markdown_project_memory_r0/session_context_assembler/historical `
  --historical-manifest benchmarks/fixtures/python_markdown_project_memory_r0/session_context_assembler/historical_snapshot.manifest.json `
  --truthset benchmarks/truthsets/python_markdown_project_memory_r0.json `
  --truthset-manifest benchmarks/truthsets/python_markdown_project_memory_r0.manifest.json `
  --collection-root $collectionRoot `
  --embedding-model $env:MNEMOS_PROJECT_MEMORY_R0_MODEL `
  --local-files-only `
  --output-json benchmarks/results/python_markdown_project_memory_r0.json `
  --output-markdown benchmarks/results/python_markdown_project_memory_r0.md
```

Expected: exit `0` only when every zero-tolerance and numeric gate passes.
Exit `1` is an evaluated failure; exit `2` is preflight/harness failure. Retain
the outcome honestly and do not turn a failure into a promotion claim.

- [ ] **Step 6: Verify reports and out-of-scope modifications**

```powershell
python -m pytest tests/test_python_markdown_project_memory_r0_evaluation.py tests/test_python_markdown_project_memory_r0_boundaries.py -q
git diff --check
git status --short
```

Expected: tests pass, no whitespace errors, and only planned R0 paths are
modified. `logs/` may remain untracked but must not be staged.

- [ ] **Step 7: Commit final evidence**

```powershell
git add tests/test_python_markdown_project_memory_r0_boundaries.py benchmarks/results/python_markdown_project_memory_r0.json benchmarks/results/python_markdown_project_memory_r0.md
git commit -m "test: record structured project memory R0 evaluation"
```

---

## Final Acceptance Checklist

- [ ] Selected subsystem is Session Context Assembler + shadow adapter.
- [ ] Active fixture is pinned to `79ae3342eaa01c2e9dd7c1ab0be289f046a1baeb`.
- [ ] Historical fixture is labelled fixture-local synthetic history everywhere.
- [ ] Active and historical snapshot manifests verify byte-for-byte.
- [ ] Truth set contains fourteen frozen queries, two per required family.
- [ ] Every non-wrong-corpus oracle has exact expected artifact IDs.
- [ ] Every artifact validates common fields, file/content hashes, source URI, and line span.
- [ ] Python extraction covers module, class, function, method, test, decorator application, import block, literal config constant, and syntactic route handler.
- [ ] Markdown extraction covers document, section, ADR, decision, handoff, closeout, and agent instruction.
- [ ] Empty scope and accidental whole-repo discovery fail closed.
- [ ] Collection identity is one repository plus one snapshot; no default fallback exists.
- [ ] L, S, H, and H+E execute and report separately.
- [ ] Structural expansion is explicit, one hop, and never claims callers/callees.
- [ ] Stale file, symbol, summary, branch, commit, and dirty-tree mismatch rules are tested.
- [ ] Wrong-repo, wrong-snapshot, and research-paper leakage are zero.
- [ ] No repository code is imported or executed.
- [ ] No runtime route, retrieval default, governance, promotion, frontier, VS Code, graph, or GraphRAG path changes.
- [ ] Existing selected SCA tests remain `61/61` passing.
- [ ] Final JSON and Markdown reports agree and disclose fixture/model/claim boundaries.

## Completion Boundary

Completing this plan yields isolated R0 evidence only. Even a passing report
does not authorize a runtime route, default retrieval change, context-packet
command, VS Code integration, automatic memory write, or production claim. A
failed gate is retained as evidence and routes work back to a narrowed design
or R1 corpus; it is not tuned away by editing the frozen R0 truth set.
