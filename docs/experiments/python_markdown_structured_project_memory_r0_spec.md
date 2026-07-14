# Python + Markdown Structured Project Memory R0 Specification

Date: 2026-07-15

Status: **SPEC ONLY / NO IMPLEMENTATION AUTHORIZED**

## 1. Purpose

Define the minimum structured ingestion and retrieval model needed for MNEMOS
to support future task-specific coding context packets from one VS Code or Git
repository subsystem.

This specification is narrower than a full `Repo Snapshot -> Task Context
Packet R0`. It covers only the source-grounded representation, isolation,
version checks, and evaluation contract for Python and Markdown artifacts. It
does not define a public command, editor integration, code execution path, or
production retrieval change.

## 2. Core Principle

Code is versioned, executable, structured evidence. It must not be ingested as
undifferentiated PDF-like text.

MNEMOS stores source-backed records about code; it does not execute code or
make code authoritative merely because it was retrieved. Parser-derived
structure is descriptive metadata. The checked-out repository remains the
source of truth.

The R0 flow is:

```text
one repository + one selected subsystem
-> immutable repository snapshot manifest
-> Python AST and Markdown structural segmentation
-> source-backed Engrams with artifact-local lineage
-> snapshot-scoped lexical and semantic retrieval
-> optional bounded structural expansion
-> evaluated candidate set for a future task context packet
```

## 3. Relationship to Existing MNEMOS Boundaries

R0 extends the existing Engram content, metadata, and lineage model. It does
not create a second durable-memory authority or a parallel project-memory
database.

Existing boundaries remain binding:

- retrieved material is evidence, not a decision, authorization, or action;
- source URI, artifact identity, version, chunk identity, and provenance span
  remain artifact-local;
- any future context package is read-only, non-authoritative, and
  non-promotable;
- no project artifact may mutate governance, contradiction, promotion,
  retrieval ranking, or Resolution Engram state; and
- default MNEMOS collections and retrieval behavior remain unchanged.

The current focused repository seeder reads selected files as whole text
documents. R0 replaces that behavior only inside a future isolated project
ingestion harness. It does not alter that seeder or the served runtime.

## 4. R0 Scope

R0 is limited to:

- one repository;
- one explicitly selected subsystem root or allowlisted file set;
- tracked Python (`.py`) and Markdown (`.md`) files within that scope;
- one immutable repository snapshot per evaluation run;
- Python parsing with the standard-library AST plus source-token inspection
  where needed;
- Markdown heading and explicit metadata parsing without rendered-document
  interpretation;
- one dedicated physical collection per snapshot;
- offline or isolated retrieval evaluation; and
- enough output structure to inform a later context-packet design.

Files outside the selected subsystem are excluded unless explicitly included
as required documentation, agent instructions, or tests. Symlinks resolving
outside the repository root, generated environments, dependency caches,
secrets, and ignored files are excluded.

## 5. Common Identity and Lineage Contract

Every Python and Markdown record MUST be a source-backed Engram with exact
content copied from a repository artifact. Parser-generated descriptions MAY
be indexed as separately labelled synthetic retrieval aids later, but they are
outside R0 and MUST NOT replace the source-backed record.

### 5.1 Common required fields

| Field | Requirement |
| --- | --- |
| `repo_id` | Stable, non-secret repository identifier. Prefer an explicitly configured slug; do not expose credentials embedded in a remote URL. |
| `snapshot_id` | Immutable digest identifying the complete R0 snapshot manifest. |
| `branch` | Checked-out branch name, or the literal `DETACHED_HEAD`. |
| `commit_hash` | Full Git `HEAD` commit hash. It identifies the base commit even when the working tree is dirty. |
| `file_path` | POSIX-style repository-relative path. Absolute local paths MUST NOT be stored as the canonical path. |
| `file_hash` | SHA-256 of the exact file bytes captured by the snapshot manifest. |
| `language` | `python` or `markdown`. |
| `artifact_type` | Controlled artifact type defined in this specification. |
| `start_line` | One-based inclusive source line. |
| `end_line` | One-based inclusive source line; MUST be greater than or equal to `start_line`. |
| `content_hash` | SHA-256 of the exact UTF-8 source slice stored in `Engram.content`. |
| `source_uri` | Repository-scoped URI in the form `repo://<repo_id>/<file_path>?snapshot=<snapshot_id>`. |
| `artifact_id` | Deterministic ID derived from repository, snapshot, file, artifact identity, span, and content hash. |
| `artifact_version` | Equal to `snapshot_id` for R0. |
| `chunk_id` | Equal to the deterministic `artifact_id` unless later segmentation creates a subordinate chunk. |
| `provenance_span` | Structured `{start_line, end_line}` matching the stored source slice. |
| `ingestion_schema_version` | Fixed value `python_markdown_project_memory_r0`. |
| `source_linked` | MUST be `true`. |
| `is_superseded` | `false` for active snapshot artifacts; historical records are not queried as active records. |

`artifact_id` MUST be reproducible from:

```text
sha256(
  repo_id || snapshot_id || file_path || artifact_type ||
  qualified_artifact_name || start_line || end_line || content_hash
)
```

All string components use UTF-8, NFC normalization, and an unambiguous
length-prefixed canonical encoding. Hashes are lowercase hexadecimal prefixed
with `sha256:`.

### 5.2 Snapshot manifest

The immutable snapshot manifest MUST record:

- `repo_id`, `snapshot_id`, `branch`, and `commit_hash`;
- selected subsystem roots and explicit inclusions/exclusions;
- Git tracked-file set admitted to R0;
- per-file `file_path`, `file_hash`, byte size, and language;
- dirty-working-tree state described in Section 9;
- parser and ingestion schema versions;
- collection identity; and
- manifest creation time.

`snapshot_id` is the SHA-256 digest of canonical JSON containing all manifest
fields except `snapshot_id` and creation time. Creation time is excluded so
the same repository state produces the same snapshot identity.

## 6. Python Artifact Model

Python structure MUST be extracted without importing or executing repository
code. The standard-library AST is the normative R0 parser. Token inspection
MAY recover comments, decorator spans, and exact source boundaries.

### 6.1 Python artifact types

| Represented artifact | `artifact_type` | `symbol_kind` | Source-backed content rule |
| --- | --- | --- | --- |
| Module | `python_module` | `module` | Entire file only when within the configured chunk limit; otherwise module docstring/header material with child artifacts carrying definitions. |
| Class | `python_symbol` | `class` | Exact class source span. Large classes MAY be retained for lineage but excluded from semantic embedding when child methods provide retrieval coverage. |
| Function | `python_symbol` | `function` | Exact decorated function source span. |
| Method | `python_symbol` | `method` | Exact decorated method source span with owning class in `parent_symbol`. |
| Test function/method | `python_symbol` | `test_function` or `test_method` | Exact decorated function or method source span. |
| Decorator application | `python_decorator_application` | `decorator_application` | Exact decorator expression lines attached to the decorated parent symbol. |
| Import block | `python_import_block` | `import_block` | One contiguous run of `import` and `from ... import ...` statements. |
| Configuration constant | `python_config_constant` | `config_constant` | Exact top-level assignment span accepted by the R0 constant rule. |
| Route/API handler | `python_symbol` | `route_handler` | Exact function or method span when a configured, syntactically present route decorator is detected. |

Overlapping source spans are permitted because class/module lineage and
method-level retrieval serve different purposes. Packet assembly must later
deduplicate overlapping content; R0 retrieval evaluation reports overlap but
does not invent lossy summaries to avoid it.

### 6.2 Python required metadata

Every Python artifact MUST include all common fields plus:

- `symbol_name`: local name, or the module name for a module artifact;
- `qualified_symbol_name`: module-qualified name, including class ownership;
- `symbol_kind`: controlled value from Section 6.1;
- `parent_symbol`: qualified parent name for methods and nested definitions,
  otherwise `null`;
- `imports`: cheaply extractable syntactic imports relevant to the artifact;
- `test_marker`: structured object described below, or `null`;
- `decorators`: exact syntactic decorator names where available;
- `parser`: `python_ast`;
- `parser_version`: running Python major/minor version; and
- `parse_status`: `parsed`, `partial`, or `failed`.

Additional metadata MAY include `async`, parameter names, return annotation
text, route information, or framework detection, provided it is syntactically
derived and labelled with `detection_basis`.

A decorator definition remains an ordinary function or class because Python
does not provide a reliable static declaration that a callable is a decorator.
R0 represents each syntactically visible decorator application as its own
source-backed artifact and records the decorator expression on its owning
symbol. It MUST NOT reclassify a callable as a decorator definition based only
on naming or semantic similarity.

### 6.3 Imports

Imports are syntactic evidence only. R0 MAY extract:

- module names from `import x`;
- module and imported names from `from x import y`;
- aliases; and
- relative-import level.

R0 MUST NOT claim that an import resolves, loads successfully, or identifies a
runtime dependency. Wildcard imports remain literal and unresolved.

Module-level contiguous imports produce `python_import_block` artifacts.
Symbol-level `imports` MAY include imports lexically nested within that symbol.

### 6.4 Test detection

`test_marker` is non-null when one or more cheap, explicit signals exist:

- file is under an allowlisted test directory;
- file name matches `test_*.py` or `*_test.py`;
- symbol name begins with `test_`;
- owning class name begins with `Test`; or
- a configured test decorator or base class is syntactically present.

It MUST record the matched signals. Test detection is descriptive and does not
claim the test is collected, runnable, or passing.

### 6.5 Configuration constants

R0 recognizes a configuration constant only when it is a top-level `Assign`
or `AnnAssign` whose target is a simple uppercase identifier and whose value is
an AST literal or a container composed only of literals. Values MUST NOT be
evaluated. Assignments involving calls, attribute access, comprehensions, or
arbitrary expressions remain ordinary module content.

### 6.6 Route/API handler detection

Route detection is allowlist-driven and syntactic. R0 MAY recognize configured
decorator terminal names such as `route`, `get`, `post`, `put`, `patch`, or
`delete`. It MAY extract a route path and method only from literal decorator
arguments.

Every detected handler MUST include:

- `route_detection = heuristic`;
- `detection_basis` naming the exact decorator expression;
- literal `route_path` when present, otherwise `null`; and
- literal `http_methods` when present, otherwise an empty list.

The record MUST NOT claim that the route is registered or reachable at
runtime.

### 6.7 Python parse failures

A syntax error or unsupported encoding MUST NOT silently fall back to
undifferentiated semantic ingestion. The file receives a manifest-level parse
failure with location and non-sensitive error category. Exact lexical lookup
of the file MAY remain available from a clearly labelled whole-file fallback,
but structured retrieval MUST treat the file as incomplete. A task requiring
that file may trigger packet abstention.

## 7. Markdown Artifact Model

Markdown parsing is source-line based. R0 recognizes ATX and Setext headings,
YAML front matter when present, and explicit status/date/reference fields. It
does not execute embedded HTML or directives.

### 7.1 Markdown artifact types

| Represented artifact | `artifact_type` | Detection rule |
| --- | --- | --- |
| Document | `markdown_document` | Every admitted Markdown file. |
| Heading section | `markdown_section` | Each heading and its body up to the next heading of equal or higher level. |
| ADR | `markdown_adr` | Explicit ADR path/name/front matter/title convention. |
| Decision note | `markdown_decision` | Explicit decision-labelled path, front matter, title, or heading. |
| Handoff | `markdown_handoff` | Explicit handoff-labelled path, front matter, title, or heading. |
| Test/evaluation closeout | `markdown_evaluation_closeout` | Explicit test, evaluation, benchmark, result, evidence, or closeout labelling. |
| Agent instruction file | `markdown_agent_instruction` | Allowlisted names such as `AGENTS.md`, `CLAUDE.md`, or explicitly configured equivalents. |

A document may have one document-level type more specific than
`markdown_document`, while its child sections use `markdown_section` plus
`document_artifact_type` metadata. R0 MUST NOT duplicate identical section
content solely to express multiple classifications.

### 7.2 Markdown required metadata

Every Markdown artifact MUST include all common fields plus:

- `heading_path`: ordered heading titles from document root to the section;
- `heading_level`: integer for a section, or `0` for the document record;
- `status`: explicit normalized status plus raw value, or `null`;
- `decision_date`: ISO date parsed from an explicit decision/date field, or
  `null`;
- `supersedes`: explicit referenced artifact/path identifiers, otherwise an
  empty list;
- `superseded_by`: explicit referenced artifact/path identifiers, otherwise an
  empty list;
- `detection_basis`: matched filename, front-matter key, heading, or explicit
  field used for special classification;
- `parser`: `markdown_structural_r0`;
- `parser_version`; and
- `parse_status`: `parsed`, `partial`, or `failed`.

For a document-level artifact, `heading_path` is an empty list. For a section,
it includes the section heading as the final element.

Document-level records MAY contain the entire document only when it is within
the configured chunk limit. For longer documents, the document record contains
the exact pre-heading preamble, if any, while heading sections carry the body.

### 7.3 Status, dates, and supersession

Status and decision dates MUST be taken only from explicit front matter,
labelled fields, or conventional ADR headers. Sentiment or prose interpretation
MUST NOT infer status.

When an explicit status is present, metadata retains its exact `raw` value and
maps `normalized` to one of `proposed`, `accepted`, `deprecated`,
`superseded`, `complete`, `incomplete`, `blocked`, `pass`, `fail`, or
`unknown_explicit`. An unrecognized explicit value is never discarded or
silently mapped to a stronger status.

`supersedes` and `superseded_by` MUST be recorded only when an explicit
relationship exists. A newer date or similar title does not establish
supersession. Unresolved references remain literal and are reported during
evaluation; they MUST NOT be silently linked to the nearest semantic match.

## 8. Retrieval Contract

R0 project retrieval is an isolated evaluation pipeline. It may compose
existing retrieval primitives, but it does not change MNEMOS's default
retrieval mode, fusion policy, ranking, or served API.

### 8.1 Mandatory scope filters

Every project query MUST bind:

- physical snapshot collection;
- `repo_id`;
- `snapshot_id`; and
- active/historical intent.

It MAY additionally filter by branch, commit, language, file path,
artifact type, symbol kind, or test marker. Repository and snapshot filtering
are eligibility gates, not ranking features.

### 8.2 Exact lexical retrieval

The lexical lane targets:

- identifiers and qualified symbol names;
- repository-relative paths;
- route literals;
- error codes and exception names;
- environment-variable names;
- configuration constants; and
- literal Markdown headings or ADR identifiers.

Exact matches MUST be retained as candidates even when semantic similarity is
weak. Identifier tokenization MUST preserve snake case, dotted names, path
segments, leading route slashes, and uppercase environment-variable forms.

### 8.3 Semantic retrieval

The semantic lane targets conceptual intent such as "where stale memory is
rejected" or "tests governing context budget abstention." It searches only
the already scoped snapshot collection. Semantic similarity MUST NOT broaden
repository, branch, snapshot, or artifact eligibility.

### 8.4 Candidate combination

R0 compares at least:

```text
L: lexical only
S: semantic only
H: scoped lexical + semantic hybrid
H+E: hybrid plus bounded structural expansion
```

The evaluation harness MUST record component scores and candidate origin.
R0 does not prescribe a new production fusion policy. Any R0 weighting is
fixture-local and versioned in the evaluation configuration.

### 8.5 Optional structural expansion

Expansion is bounded to explicit, source-derived relationships in the same
snapshot:

- module to contained class/function/method artifacts;
- class to contained methods;
- symbol to applied decorator artifacts;
- symbol to syntactically imported module names;
- test to an exact symbol/path explicitly named in its source;
- Markdown section to its containing document;
- ADR/decision/handoff to explicit referenced paths or symbols; and
- source artifact to an explicitly referenced test, document, or decision.

Expansion MUST NOT infer runtime callers, callees, route registration,
dependency resolution, or semantic links and present them as structural fact.
Each expanded candidate records the originating artifact and relationship
type. Expansion is capped by configurable candidate and hop limits; R0 uses a
maximum of one hop.

## 9. Staleness and Version Rules

### 9.1 Current and stale files

For an active-snapshot query, a file record is current only when its
`file_hash` equals the hash for the same `file_path` in the active snapshot
manifest. A differing hash, missing path, or record from another snapshot is
stale for that query.

Historical retrieval is allowed only when explicitly requested and MUST label
all returned records with their historical snapshot and commit.

### 9.2 Current and stale symbols

A symbol chunk is current only when:

- its file is current;
- its deterministic artifact identity exists in the active manifest-derived
  index; and
- its `content_hash`, source span, and qualified name match the active parse.

If the same qualified name exists with a different content hash or span, the
older record is stale. A rename is not inferred; the old symbol is absent and
the new symbol is distinct unless an explicit rename record exists.

### 9.3 Stale summaries

R0 does not generate summaries. If a pre-existing summary is admitted as an
explicit evaluation input, it MUST list parent artifact IDs, parent content
hashes, and snapshot ID. It is stale when any parent is missing or changed, or
when its snapshot differs from the active snapshot. A stale summary MUST NOT
substitute for current source and is excluded from active packet evidence.

### 9.4 Dirty working tree

The snapshot manifest MUST disclose:

- tracked modified files;
- tracked deleted files;
- staged changes;
- admitted untracked files, if explicitly enabled; and
- excluded untracked or ignored files by count only, without exposing secret
  paths unnecessarily.

When dirty, `commit_hash` remains the base `HEAD`; `snapshot_id` incorporates
the captured working-tree file hashes and dirty-state manifest. The snapshot
MUST be labelled `working_tree_state = dirty` and MUST NOT be described as an
exact representation of the commit.

Untracked files are excluded by default. If enabled for a controlled test,
they must pass the same subsystem allowlist, secret/path exclusions, hashing,
and disclosure rules as tracked files.

### 9.5 Branch and commit mismatch

Before active retrieval or packet preparation, the consumer-provided repository
state is compared with the snapshot manifest. A branch, commit, or working-tree
digest mismatch fails closed as `SNAPSHOT_MISMATCH`; it does not silently use
the nearest or newest collection.

### 9.6 Required abstention conditions

A future packet builder MUST abstain, fully or explicitly partially, when:

- repository or snapshot scope is absent or ambiguous;
- the live repository does not match the requested snapshot;
- required source artifacts have stale or missing hashes;
- a relevant Python file failed structured parsing;
- required provenance spans are missing or inconsistent;
- retrieval returns cross-repository or cross-snapshot candidates;
- mandatory evidence cannot fit the packet budget;
- dirty-tree state cannot be captured accurately; or
- source access or disclosure policy blocks required evidence.

An abstention records a non-sensitive reason code and no apparently complete
packet. Minimum R0 reason codes are:

```text
REPO_SCOPE_REQUIRED
SNAPSHOT_SCOPE_REQUIRED
SNAPSHOT_MISMATCH
STALE_SOURCE_DETECTED
STRUCTURED_PARSE_INCOMPLETE
LINEAGE_INCOMPLETE
CROSS_SCOPE_CANDIDATE_DETECTED
BUDGET_INSUFFICIENT
DIRTY_STATE_UNRESOLVED
DISCLOSURE_DENIED
```

## 10. Dedicated Collection Rules

R0 uses one immutable physical collection per repository snapshot. A suggested
non-authoritative naming convention is:

```text
mnemos_project_r0__<repo_id_digest_12>__<snapshot_id_digest_12>
```

The full identities live in the collection manifest; truncated names are not
security or lineage boundaries.

Required rules:

1. A collection contains artifacts from exactly one `repo_id` and one
   `snapshot_id`.
2. A collection is immutable after its manifest is frozen. Changed files
   produce a new snapshot and collection.
3. Branches do not share an active collection, even when their current file
   trees happen to match. Branch remains explicit provenance.
4. Retrieval binds directly to the requested physical collection and also
   applies repo/snapshot metadata filters as defense in depth.
5. General research-paper, uploaded-document, chat-memory, and other repository
   collections are never included in the R0 search fan-out.
6. There is no implicit fallback to the default MNEMOS collection when the
   requested project collection is unavailable.
7. A collection whose manifest digest fails verification is quarantined from
   retrieval.
8. Historical collections are retained or deleted only under an explicit
   operator policy; they are never treated as active merely because they are
   newer by creation time.

These rules require a future isolated collection-binding mechanism. They do
not authorize changing the current service's configured collection or adding
request-selectable collection routing.

## 11. Future Context Packet Implications

These records are intended to support a later proposal for:

```text
mnemos project-context --repo <path> --task "<task>"
```

This specification does not implement or authorize that command.

A future command could use R0 records to produce:

- verified repository, branch, commit, snapshot, and dirty-tree identity;
- exact files and symbols relevant to the task;
- nearby source-backed tests;
- applicable ADRs, decisions, handoffs, and agent instructions;
- explicit import or containment relationships;
- stale-source and parse-completeness warnings;
- inclusion and exclusion rationale;
- source paths and line spans;
- bounded content with overlap suppression; and
- an evidence receipt and package digest.

The future packet builder must revalidate the live repository state immediately
before issuing a packet. Retrieval success alone does not prove snapshot
freshness. Packet content remains context, not authority, and cannot be written
back as durable memory without a separate source-grounded admission process.

## 12. R0 Evaluation Design

The truth set and repository snapshot MUST be frozen before implementation
tuning. The selected subsystem should contain at least:

- two Python modules;
- one class with methods;
- one route or configured handler pattern, if the subsystem has one;
- one test module;
- one configuration constant;
- one ADR or explicit decision note;
- one handoff or evaluation closeout; and
- one deliberately retained historical snapshot containing a changed symbol.

### 12.1 Query families

| Family | Example intent | Required observation |
| --- | --- | --- |
| Exact symbol lookup | `RetrievalRouter.search` or another frozen qualified name | Correct active symbol ranks first; exact identifier is preserved. |
| Conceptual behavior lookup | "Where are stale project records rejected?" | Expected source-backed implementation artifact appears in top-k. |
| Test association lookup | "Which tests cover context budget abstention?" | Correct test artifact and target symbol/path evidence are returned without inferred execution claims. |
| ADR/decision lookup | "Why is the adapter shadow-only?" | Current explicit ADR/decision section is returned with status and source span. |
| Stale version detection | Query a symbol present in both active and historical snapshots | Active query excludes or labels the historical symbol and reports no stale leakage. |
| Cross-file relationship lookup | Query a symbol and an explicitly importing or referencing test/doc | One-hop expansion returns only explicit relationships with origin metadata. |
| Wrong-corpus rejection | Use terminology strongly represented in the research-paper collection but absent from the project snapshot | No research-paper or cross-repo candidate is returned. |

At least two frozen queries per family are required, including one paraphrase
for every non-exact family. Exact expected artifact IDs and acceptable top-k
sets are stored in a versioned truth-set manifest.

### 12.2 Required gates

| Gate | R0 threshold |
| --- | ---: |
| Required metadata completeness | `1.0` |
| File hash verification rate | `1.0` |
| Content hash verification rate | `1.0` |
| Source-span fidelity | `1.0` |
| Exact-symbol top-1 accuracy | `1.0` |
| Required artifact recall at frozen top-k | `>= 0.90` overall and no query family below `0.80` |
| Wrong-repository leakage | `0` |
| Wrong-snapshot active leakage | `0` |
| Research-paper collection leakage | `0` |
| Unlabelled stale artifact count | `0` |
| Inferred call-graph claims | `0` |
| Unauthorized code execution | `0` |
| Unauthorized memory/governance mutation | `0` |

Any non-zero cross-scope leakage, unlabelled stale evidence, code execution, or
authority mutation is a hard failure regardless of retrieval quality.

### 12.3 Conditions and reporting

The evaluation MUST report lexical-only, semantic-only, hybrid, and
hybrid-plus-expansion conditions separately. It MUST retain:

- query and expected artifact IDs;
- collection and snapshot manifest digests;
- returned artifact IDs, ranks, scores, and candidate origins;
- filters applied;
- expansion relationships;
- stale/exclusion/abstention decisions;
- latency and candidate counts; and
- exact parser and ingestion versions.

The result is local R0 evidence for the frozen repository and subsystem only.
It does not establish general code understanding, production readiness, or
performance across languages or repositories.

## 13. Security and Disclosure Constraints

Before hashing or ingestion, R0 MUST exclude:

- credential files and configured secret patterns;
- `.git` internals other than safe command-derived identity;
- virtual environments, caches, build outputs, and vendored dependencies;
- files outside the repository root after path resolution;
- binary files; and
- files denied by the subsystem allowlist or operator policy.

Hashing a secret does not make it safe to ingest. Error messages and manifests
must avoid exposing excluded secret values or unnecessary absolute user paths.
R0 is local-only and performs no hidden upload.

## 14. Non-Goals

R0 does not authorize or provide:

- code execution, importing repository modules, or dynamic introspection;
- call-graph authority unless a relationship is explicitly derived and
  labelled by a later approved design;
- automatic code mutation;
- automatic durable-memory promotion;
- automatic summary generation or summary authority;
- frontier API integration;
- a VS Code extension;
- a public `mnemos project-context` command;
- a graph database;
- GraphRAG;
- runtime dependency resolution;
- multi-language support;
- multi-repository or monorepo-wide ingestion;
- automatic collection switching in the served MNEMOS API;
- a change to default MNEMOS retrieval policy, ranking, governance, promotion,
  disclosure, or authority behavior; or
- a production-readiness claim.

## 15. Advancement and Rollback Boundary

This specification is ready to inform a separate isolated implementation plan.
It does not itself authorize implementation.

Any implementation proposal must remain outside the served runtime, use a
frozen evaluation corpus and dedicated collections, and include a kill switch
or equivalent removal path that leaves existing MNEMOS collections and routes
unchanged.

If an implementation executes repository code, searches the default research
collection, admits cross-snapshot results as active, loses artifact-local
lineage, writes generated context as durable memory, or changes default
retrieval behavior, the R0 lane must stop and return to design review.

## 16. Recommendation

`SPEC_READY_FOR_IMPLEMENTATION`

Meaning: the structured evidence, isolation, staleness, retrieval, and
evaluation contracts are sufficiently bounded for a separate implementation
plan and explicit authorization decision. This label is not implementation
authorization and is not a runtime promotion decision.

```text
PYTHON_MARKDOWN_PROJECT_MEMORY_R0_SPEC_COMPLETE
STRUCTURED_CODE_EVIDENCE_MODEL
VERSIONED_PROJECT_CONTEXT_REQUIRED
NO_IMPLEMENTATION_AUTHORIZED
NO_CODE_EXECUTION
NO_CODE_MUTATION
NO_AUTOMATIC_MEMORY_PROMOTION
NO_FRONTIER_API_INTEGRATION
NO_RUNTIME_BEHAVIOR_CHANGE
```
