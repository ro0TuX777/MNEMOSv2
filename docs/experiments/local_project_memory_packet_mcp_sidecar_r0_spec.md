# Local Project Memory Packet + MCP Sidecar R0 Specification

Date: 2026-07-15

Status: **IMPLEMENTED / R0 TRIAL VERIFIED**

## 1. Goal

Give a frontier agent running in VS Code read-only, structured memory about one
explicitly scoped local project while leaving all code execution and editing
authority with the frontier agent and operator.

The first trial project is the MNEMOS repository itself. The operator builds a
snapshot-bound packet, configures a separate read-only MCP sidecar to serve that
packet, and instructs the VS Code agent to use the sidecar for source-backed
Python and Markdown evidence.

## 2. Authority Boundary

```text
local Git project (source of truth)
  -> read-only packet builder
  -> immutable packet outside the target project
  -> read-only project-memory MCP sidecar
  -> VS Code frontier agent
  -> operator-approved lint or code mutation using the agent's own tools
```

MNEMOS supplies evidence. It does not authorize a conclusion, run project code,
run lint, or mutate the project. The MCP sidecar exposes no write tools. The
frontier agent's native editor and terminal tools remain outside MNEMOS's
control, so lint and mutation approval is an explicit procedural checkpoint
included in every packet and MCP response.

Any admitted-file change makes the packet stale. A stale sidecar abstains from
returning source evidence until the packet is rebuilt.

## 3. Command

The builder is:

```text
tools/build_local_project_memory_packet.py
```

Required arguments:

- `--project-root PATH`
- `--repo-id ID`
- at least one repeatable `--scope-root RELATIVE_PATH` or `--scope-file RELATIVE_PATH`
- `--output PATH`

Optional arguments:

- repeatable `--exclude RELATIVE_PATH`
- `--max-files` (default `500`)
- `--max-total-bytes` (default `10485760`)
- `--max-file-bytes` (default `1048576`)

There is no current-directory project default, inferred scope, whole-repository
fallback, MNEMOS URL, collection option, model option, lint option, or mutation
option. The output must resolve outside the target project and must not already
exist.

## 4. Admission

The admitted set is:

```text
Git-tracked files
INTERSECT explicit scope roots/files
INTERSECT {.py, .md}
MINUS hard exclusions
MINUS operator exclusions
```

All scope and exclusion values are repository-relative POSIX paths after
normalization. Absolute values, traversal, missing explicit files, unsupported
explicit files, untracked explicit files, and paths or symlinks resolving
outside the project root fail closed. Unsupported descendants of a scope root
are exclusions, not errors. Empty admitted scope fails closed.

Hard exclusions apply before hashing and cover `.git`, virtual environments,
caches, build outputs, vendored dependencies, binaries, credential filenames,
and configured secret-path patterns. Untracked and ignored files are excluded
in R0. Excluded secret values and unnecessary absolute paths never enter the
packet.

Only Python and Markdown are supported. Python uses standard-library `ast` and
`tokenize` without importing or executing project modules. Markdown uses
source-line heading and explicit-metadata parsing without rendering HTML or
directives.

## 5. Packet

The output is one Markdown file with an authoritative canonical JSON payload
between versioned sentinel comments plus a human-readable report. Source text
appears only in the JSON payload to avoid duplicate evidence.

The payload contains:

- schema and extractor versions;
- `repo_id`, branch, commit, and dirty-state receipt;
- explicit scope, exclusions, and resource ceilings;
- `snapshot_id` and `packet_sha256`;
- admitted file paths, byte sizes, languages, and exact byte hashes;
- exclusions grouped by stable reason code;
- source-backed Python and Markdown artifacts;
- one-based inclusive source spans and content hashes;
- parse failures and packet completeness;
- authority boundaries and human approval checkpoints.

Canonical source URIs use:

```text
repo://<repo_id>/<relative-path>?snapshot=<snapshot_id>
```

Absolute project paths are not canonical evidence fields. The MCP process
receives its local project root separately through configuration.

`snapshot_id` is the SHA-256 of canonical JSON containing the stable schema and
extractor versions, repository identity, branch and base commit, dirty state of
admitted tracked paths, normalized scope and exclusion policy, resource
ceilings, and admitted file metadata. Creation time, absolute local paths, and
repository activity outside the admitted scope are excluded. Counts of ignored,
untracked, and otherwise excluded files remain report fields but are not
freshness gates; changing `logs/` or another excluded path cannot stale an
otherwise identical scoped packet. The packet hash covers the complete
canonical payload except its own hash field.

## 6. Extraction

Python artifacts include module, class, function, method, import, configuration
constant, decorator, and syntactically detected test/route records. Markdown
artifacts include document and heading-section records plus explicit status,
date, supersession, ADR, decision, handoff, evaluation, and agent-instruction
classification.

Every artifact carries exact source text, relative file path, file hash,
content hash, source URI, qualified name or heading path, parser status, and
one-based inclusive span. Parser-derived structure is descriptive evidence,
not a runtime claim.

A Python parse failure is recorded as `STRUCTURED_PARSE_INCOMPLETE`. The builder
may write an explicitly incomplete report for diagnosis, but it must not mark
the packet usable by the MCP sidecar.

## 7. Read-Only MCP Sidecar

The separate stdio MCP server is configured with:

- `MNEMOS_PROJECT_PACKET`: absolute packet path;
- `MNEMOS_PROJECT_ROOT`: absolute live project root;
- `MNEMOS_PROJECT_REPO_ID`: expected packet repository ID.

It provides only:

- `project_memory_health()`;
- `get_project_identity()`;
- `search_project_memory(query, top_k, path_prefix, artifact_types)`;
- `get_project_artifact(artifact_id)`;
- `verify_project_snapshot()`.

R0 retrieval is deterministic structured lexical retrieval with exact boosts
for qualified symbols, paths, route/config literals, Markdown headings, and
test names. It does not claim embedding-based semantic retrieval. Semantic
retrieval can be evaluated later without changing the R0 packet contract.

Before returning evidence, the sidecar verifies repo identity and current
admitted-file hashes. `SNAPSHOT_MISMATCH`, missing files, packet-integrity
failure, incomplete parsing, or cross-scope evidence causes abstention. All
responses include snapshot identity, source spans, hashes, and the authority
boundary.

The sidecar does not call the MNEMOS REST service, Qdrant, PostgreSQL, Research
Intake, or the default collection. It has no write, decision, observation,
indexing, lint, shell, or mutation tool.

## 8. Stack Verification

A separate read-only preflight tool verifies an installation without being a
dependency of packet construction or sidecar retrieval:

```text
tools/verify_mnemos_local_stack.py
```

It discovers Compose services through `docker compose config` and container
labels rather than assuming fixed container names. It verifies:

- Qdrant image/version and HTTP `/healthz` on the configured host URL;
- PostgreSQL image/version, container health, and `pg_isready` through
  `docker compose exec -T` without printing credentials;
- MNEMOS `/health` and `/v1/mnemos/capabilities`;
- Research UI root endpoint when that optional service is configured;
- OpenWebUI proxy `/health` when that optional service is configured;
- published-port ownership and collisions;
- configured service URL, observed container/image, status, latency, and
  non-sensitive failure reason.

Defaults match the current development stack: Qdrant `6333`, PostgreSQL `5432`,
MNEMOS `8700`, Research UI `8788`, and proxy `8790`. CLI arguments and
environment variables can override URLs, Compose file, and Compose role/service
names for other installations. Required core-service failures return non-zero.
Optional-service absence is reported separately from an unhealthy configured
service.

The verifier writes an optional JSON receipt and does not start, stop, restart,
or reconfigure containers.

## 9. First MNEMOS Trial Scope

The first frozen command explicitly includes:

```text
mnemos/
service/
mnemos_sdk/
mcp_servers/mnemos/
tests/
README.md
docs/architecture.md
docs/dependency_map.md
docs/experiments/python_markdown_structured_project_memory_r0_spec.md
```

This is approximately 242 tracked Python/Markdown files at planning time and
fits the default 500-file boundary. `tools/` is not included wholesale; later
tasks may add individual tool files explicitly when needed.

## 10. Approval Checkpoints

Every complete packet and MCP response states:

1. The operator must verify repo ID, snapshot ID, scopes, and exclusions.
2. Before lint, the agent must display the exact command and confirm it has no
   autofix, formatting, write, or generated-file behavior.
3. Before code mutation, the agent must summarize proposed edits and receive
   separate human approval.
4. After any mutation, the prior packet is stale and must be rebuilt before
   further source-backed retrieval is trusted.

## 11. Non-Goals

R0 does not add a VS Code extension, runtime route, default collection change,
Research Intake dependency, code execution, lint execution, code mutation,
automatic memory promotion, frontier API integration, GraphRAG, or production
semantic-retrieval claim.

## 12. Acceptance

R0 passes only when:

- empty or ambiguous scope fails closed;
- target-project bytes never change;
- packet output is outside the target project;
- all admitted file and artifact hashes/spans verify exactly;
- excluded/cross-scope content cannot be retrieved;
- stale snapshots cause MCP abstention;
- the MCP server exposes no mutation-capable tools;
- default MNEMOS collections and runtime routes are unchanged;
- the MNEMOS repository trial retrieves expected active Python and Markdown
  evidence through the VS Code-callable stdio MCP server; and
- stack verification gives actionable, non-sensitive diagnostics on both the
  current Compose topology and overridden installation ports.
