# Engram R0 Local CLI Design

Date: 2026-07-17
Status: Proposed

## Goal

Define the R0 architecture for Engram as a standalone, local-first coding
context application built around one core library and one thin CLI. R0 should
optimize for repository scanning, code/doc/test extraction, hybrid retrieval,
and budgeted context bundle output while preserving clean seams for a later
containerized sidecar or service wrapper.

## Product Shape

Engram R0 is one codebase with two deliverables:

- a core library that owns scanning, extraction, indexing, retrieval, and
  bundle assembly; and
- a thin CLI that turns explicit user input into library calls and renders
  human-readable or machine-readable output.

R0 is not a broad service surface, hosted API, IDE extension, or platform.
Service-readiness matters at the architecture boundary, but the service layer
itself is out of scope for this release.

## Why This Shape

Engram is intended to be developer working memory for coding agents, not a
direct clone of MNEMOS's evidence-heavy workflow. The R0 design should preserve
the architectural strengths worth reusing from MNEMOS, such as typed artifacts,
retrieval tiering, and bounded context assembly, without carrying over the full
packet ceremony and audit weight into normal coding workflows.

The guiding R0 principle is:

```text
local repository -> typed artifact extraction -> local hybrid retrieval ->
budgeted task context bundle
```

Everything in R0 should reinforce fast, local, inspectable coding context.

## Architecture Overview

R0 should be organized as a single standalone application with the following
library modules:

- `scan`: repository discovery, scope policy, exclusion policy, snapshot
  identity, and file manifests
- `extract`: typed artifact and explicit relationship extraction from admitted
  files
- `index`: persistence of snapshot metadata, artifacts, lexical structures,
  and vector state
- `retrieve`: lexical search, semantic search, fusion, ranking, and bounded
  structural expansion
- `bundle`: task-scoped context assembly under explicit size budgets
- `api`: stable programmatic entrypoints used by the CLI today and by a later
  sidecar or service wrapper

The CLI remains intentionally thin. It should parse arguments, call stable
library entrypoints, and print structured results. It should not contain
separate retrieval logic, hidden indexing behavior, or service-only contracts.

## Persistence Model

R0 should default to an embedded local store rather than a network service.
The recommended default is:

- SQLite for snapshot manifests, admitted files, artifact metadata,
  relationships, lexical retrieval structures, and retrieval receipts
- local vector index files stored beside the SQLite database under the same
  configured Engram index root

This posture keeps Engram fully local-first, single-process, and containerable
without forcing a background daemon or a remote dependency. It also gives R0
strong support for deterministic snapshots, metadata filtering, incremental
rescans, and budget-aware retrieval.

The storage boundary should remain narrow. Higher-level modules consume typed
records and stable interfaces rather than directly depending on SQLite tables or
vector engine internals. That keeps the core reusable if a later sidecar or
containerized service needs to reuse the same logic.

## Container-Ready Constraints

R0 should be container-ready from day one without introducing a service surface.
The architecture must therefore avoid host-only assumptions and support mounted
repository execution later.

Required constraints:

- no hidden dependence on editor state or ephemeral UI context
- no canonical identity based on absolute host paths
- explicit configuration for index/cache/output directories
- commands that work in a one-shot process model
- outputs whose identities remain valid whether Engram runs on the host or in a
  container with the repository mounted

Container-readiness in R0 means stable boundaries and configuration, not an
active HTTP or MCP process.

## Snapshot Model

The library should treat every scan as the creation or refresh of a scoped
`RepoSnapshot`. A snapshot represents the admitted repository state for one repo
root plus one explicit scope policy.

Each `RepoSnapshot` should include at minimum:

- `repo_id`
- repository root reference
- branch or detached-head state
- base commit identity when available
- dirty-working-tree receipt
- normalized scope and exclusion policy
- admitted file manifest
- deterministic `snapshot_id`
- scan and extraction schema versions

Snapshot identity must be deterministic for the same admitted repository state.
Excluded files outside scope must not churn identity. Dirty-state capture should
be explicit because coding-agent usefulness depends on freshness, not just on
the last clean commit.

## Core Data Model

The core library should revolve around four primary record types.

### `RepoSnapshot`

The scoped repo state for one scan, including repo identity, scope policy,
freshness state, and the admitted file manifest.

### `Artifact`

One extracted unit of source-backed coding context. Examples include a Python
function, class, method, test, import block, config constant, Markdown section,
ADR section, handoff section, or document-level fallback. Every artifact should
carry:

- stable artifact identity
- repository-relative path
- source span
- source content or excerpt basis
- content hash
- artifact kind
- lightweight typed metadata

### `Relation`

An explicit source-derived edge between artifacts. R0 relationships should stay
bounded and inspectable, such as:

- `contains`
- `imports`
- `tests`
- `documents`
- `references`

R0 should not infer deep runtime call graphs or speculative semantic edges and
present them as structural fact.

### `ContextBundle`

The final task-oriented output object. It should include:

- bundle identity and generation metadata
- repo and snapshot identity
- selected artifacts
- compact excerpts and file/line anchors
- inclusion rationale
- budget accounting
- omissions caused by trimming
- freshness or completeness warnings

## Initial Artifact Coverage

R0 should start with a narrow, high-value extraction set that supports coding
tasks well without overcommitting to broad language support.

Python artifacts:

- module
- class
- function
- method
- test function or test method
- import block
- config constant

Markdown artifacts:

- document
- section
- ADR classification
- decision classification
- handoff classification
- instruction classification

If a file cannot be structurally parsed but is still intentionally admitted, R0
may use an explicitly labeled fallback artifact. Parse failures must remain
visible and must not silently masquerade as high-confidence structure.

## Retrieval Flow

R0 retrieval should be hybrid by default, but tightly scoped and predictable.
The baseline flow is:

1. Resolve the active `RepoSnapshot` for the requested repo and scope.
2. Run lexical retrieval over paths, symbol names, test names, headings, config
   keys, and exact query terms.
3. Run embedding retrieval over artifact text within that same scoped snapshot.
4. Fuse lexical and semantic candidates with simple, inspectable weighting.
5. Apply one-hop expansion over explicit relations when that improves task
   usefulness.
6. Pass the ranked candidates into bundle assembly.

Retrieval must never widen scope. Semantic similarity helps rank candidates
inside the admitted snapshot only. It must not pull from neighboring repos,
older snapshots, or unrelated indexes.

## Hybrid Retrieval Principles

Lexical retrieval is critical for coding tasks because identifiers, file paths,
test names, error strings, headings, and config keys often matter more than
paraphrased meaning. Semantic retrieval is still useful for conceptual task
phrases such as "where stale bundles are rejected" or "tests around budget
trimming." The fusion layer should preserve the strengths of both.

R0 fusion should therefore be:

- inspectable rather than opaque
- versioned with explicit defaults
- easy to tune later without reshaping the data model

Bounded expansion should remain source-derived and one-hop only in R0. Good
examples are a function plus its nearby test, a Markdown decision section that
documents a code area, or a class plus one retrieved method. Expansion should
improve task usefulness, not flood the bundle with adjacency.

## Bundle Assembly

Engram becomes useful when it assembles compact, task-oriented context rather
than returning a long ranked list. The bundle assembler should transform the
final candidate set into a `ContextBundle` that is sized for direct use by a
coding agent.

Every bundle should aim to include:

- repo and snapshot identity
- selected artifacts with exact file and line anchors
- compact excerpts rather than full uncontrolled blobs
- nearby tests or docs when they are retrieved directly or linked explicitly
- overlap suppression and deduplication
- explicit budget accounting
- omission notes when relevant artifacts were cut by budget
- freshness warnings when the working tree changed after scan

R0 should expose three bundle modes:

- `brief`: smallest practical context for fast prompts
- `standard`: balanced default for normal coding tasks
- `max`: fills the configured upper budget for harder tasks

The bundle contract is the heart of Engram. It should prioritize prompt
compression and task usefulness over evidence-packet ceremony.

## Library API Boundary

The core library should expose stable entrypoints that later wrappers can
reuse without reworking the core. R0 does not need a public network API, but it
does need an internal API shaped for future reuse.

Suggested entrypoints:

- `scan_repo(...) -> RepoSnapshot`
- `refresh_snapshot(...) -> RepoSnapshot`
- `query_snapshot(...) -> RetrievalResultSet`
- `build_context_bundle(...) -> ContextBundle`
- `get_artifact(...) -> Artifact`
- `verify_snapshot_freshness(...) -> FreshnessReport`

The signatures can evolve during implementation, but the boundary should remain
clear: all core behavior lives behind reusable library functions, not behind
CLI-only side effects.

## CLI Contract

The initial CLI should remain narrow and explicit around three commands:

- `engram scan`
- `engram query`
- `engram bundle`

### `engram scan`

Creates or refreshes a scoped snapshot and updates local indexes.

Expected responsibilities:

- resolve repo and scope
- apply exclusions
- extract artifacts and relations
- persist metadata and vector state
- print snapshot identity, admitted counts, warnings, and output location

### `engram query`

Runs retrieval without full bundle assembly so operators can inspect candidates
and ranking behavior.

Expected responsibilities:

- resolve target snapshot
- run lexical and semantic retrieval
- print ranked artifacts with rationale, scores, and anchors
- support JSON output for evaluation and wrappers

### `engram bundle`

Builds the compact task context output for direct coding-agent use.

Expected responsibilities:

- resolve target snapshot
- run retrieval and bounded expansion
- assemble a budgeted bundle
- print either human-readable output or machine-readable JSON

Important CLI flags for R0 should include:

- `--repo`
- `--scope`
- `--task`
- `--budget`
- `--format`
- `--index-dir`

The CLI should treat machine-readable JSON as a first-class output mode because
that makes later MCP or service wrapping straightforward.

## Failure and Downgrade Behavior

R0 should fail closed where correctness matters and degrade explicitly where
partial utility is still safe.

Required behavior:

- unsupported or unreadable files are skipped with structured warnings
- parse failures are recorded, not silently flattened into misleading structure
- snapshot mismatch marks retrieval or bundles stale
- missing vector state may downgrade retrieval to lexical-only, but the
  downgrade must be explicit in outputs
- budget overflow trims the lowest-value material and records omissions
- index corruption or schema mismatch produces a visible rebuild requirement

The user should always know whether the bundle is fresh, partial, downgraded,
or complete enough for trust in normal coding use.

## Verification Strategy

R0 verification should focus on the promises that matter most for local coding
context.

Scanner and snapshot tests:

- explicit scope control
- exclusion handling
- deterministic snapshot identity
- dirty-tree capture
- no accidental admission of excluded paths

Extractor tests:

- stable Python source spans
- correct class/function/method/test extraction
- correct Markdown section and classification extraction
- visible parse-failure handling

Retrieval tests:

- exact symbol and path lookup
- hybrid retrieval on conceptual task phrasing
- no cross-snapshot leakage
- bounded one-hop expansion only

Bundle tests:

- deduplication and overlap suppression
- deterministic ordering under equal settings
- budget enforcement
- omission reporting
- stale-snapshot warnings

CLI tests:

- `scan` JSON output
- `query` JSON output
- `bundle` JSON output
- non-zero exit behavior for unrecoverable errors

## R0 Non-Goals

R0 should not include:

- a hosted API or long-running service layer
- an MCP server
- an IDE extension
- multi-repo retrieval
- broad multi-language extraction
- speculative call-graph authority
- audit-heavy response framing by default
- full evidence packet ceremony in standard coding workflows
- export into MNEMOS memory or governance flows by default

Those may become later layers, but they must not distort the core library and
thin CLI shape of R0.

## Recommended Implementation Direction

The implementation should begin with the local-first scanner and snapshot
machinery, then add typed Python/Markdown extraction, then hybrid retrieval,
then bundle assembly, and only after that consider any wrapper surfaces.

That sequence preserves the product goal:

```text
one Engram codebase
one core library
one thin CLI
local-first scanning and retrieval
budgeted context bundles
clean seams for a later sidecar or service wrapper
```

## Acceptance Criteria

This design is successful if the resulting R0 implementation can:

- scan a scoped local repository without needing a service
- persist deterministic local snapshot and artifact state
- extract code, docs, and tests into typed records
- perform hybrid retrieval inside one admitted snapshot only
- build compact task-scoped context bundles under explicit budgets
- expose the functionality through a thin CLI with JSON output support
- remain container-ready without already becoming a containerized service

## Recommendation

`DESIGN_READY_FOR_IMPLEMENTATION_PLANNING`
