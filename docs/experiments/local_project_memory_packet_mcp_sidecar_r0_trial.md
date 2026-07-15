# Local Project Memory Packet + MCP Sidecar R0 Trial

Date: 2026-07-16

Status: **PASS**

## Purpose

This trial verifies that an operator can select an explicit local Git project
and Python/Markdown scope, build one immutable source-backed packet outside the
project, and expose it to a VS Code frontier agent through a separate read-only
MCP server. The frontier agent retains its own editing tools. MNEMOS project
memory supplies evidence only and grants no lint, shell, or mutation authority.

The first trial target was the MNEMOS repository itself.

## Operator command

The packet path was a new file under `$env:TEMP`; it was not checked in.

```powershell
python tools/build_local_project_memory_packet.py `
  --project-root <MNEMOS-project-root> `
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
  --output (Join-Path $env:TEMP 'mnemos-project-memory-r0-<timestamp>.md')
```

No scope was inferred. Only Git-tracked `.py` and `.md` files in these explicit
roots/files were eligible.

## Packet evidence

- Repository ID: `mnemos`
- Base branch and commit: `main`, `0e1de735ff97a977b4718fe6e155ba654b45c51d`
- Snapshot ID: `sha256:8720f473c52e44ed4995ccad88aab2e3804c13ad69b45551b5bab742cf503443`
- Packet hash: `sha256:e355a02a263b49e2407f8b18a18cba3581f538978648c2ead6d6c2c130a37844`
- Admitted files: 243
- Source-backed artifacts: 5,313
- Packet size: 11,562,744 bytes
- Working-tree state bound into the snapshot: clean
- Explicit extraction failures: 0
- Unsupported tracked files excluded: 16
- Untracked files excluded before hashing: 154
- Target status before/after construction: unchanged
- Packet usable: true

The snapshot records each admitted file hash, language, size, scope boundary,
exclusion reason/count, Git identity, and dirty state. Each artifact records an
exact one-based inclusive source span, file hash, content hash, parser identity,
repository ID, snapshot ID, and `repo://` source URI.

## Retrieval observations

The deterministic structured lexical index was queried with `top_k=8`. Every
returned item was checked to use `repo_id=mnemos`, the active snapshot, an
admitted path, exact spans, and matching hashes.

| Query family | Expected evidence observed | Best relevant rank |
|---|---|---:|
| default Qdrant collection configuration | `mnemos/config.py` | 5 |
| REST health degraded behavior | `service/app.py:2178-2192` (`service.app.health`) | 3 |
| MCP bridge search behavior | `mcp_servers/mnemos/server.py:109-127` (`search_memory`) | 1 |
| Qdrant dimension compatibility tests | `tests/test_qdrant_embedding_dim_compat.py` | 2 |
| exclusion of default collections from R0 | structured project-memory specification, dedicated collection rules | 1 |

No Research Intake artifact or outside-scope candidate appeared.

## Read-only MCP evidence

An actual newline-delimited JSON-RPC stdio session performed `initialize`,
`tools/list`, and `tools/call`. It exited successfully and exposed exactly:

- `project_memory_health`
- `get_project_identity`
- `search_project_memory`
- `get_project_artifact`
- `verify_project_snapshot`

The search call returned `mcp_servers/mnemos/server.py:109-127`, its file and
content hashes, repository and snapshot IDs, source URI, scoring explanation,
and the authority boundary. No indexing, observation, decision, lint, shell,
write, or mutation tool was present.

The standard Python MCP client helper successfully delivered requests but its
Windows task-group teardown raised `BrokenResourceError` in this environment.
The protocol-level subprocess test is therefore the frozen transport regression
test; it verifies both response delivery and clean process exit without relying
on that helper's teardown behavior.

## Stale abstention

`test_changed_file_forces_abstention` builds a packet from an isolated temporary
Git repository, changes one admitted file, and verifies that MCP search returns
`status=abstained`, `reason_code=SNAPSHOT_MISMATCH`, and no results. The real
MNEMOS target was not changed for this test. A changed admitted file requires a
new packet before evidence can be trusted again.

## Local stack preflight

The separate verifier was run against the installed checkout's Compose file
with both optional services required. The receipt remained under
`$env:TEMP\mnemos-local-stack-receipt.json` and contains no passwords, tokens,
keys, secrets, or DSNs.

| Role | Observed image/version | Host port | Status | Reason | HTTP latency |
|---|---|---:|---|---|---:|
| Qdrant | `qdrant/qdrant:v1.17.1` | 6333 | healthy | `OK` | 167.6 ms |
| PostgreSQL | `postgres:16-alpine` | 5432 | healthy | `OK` | `pg_isready` |
| MNEMOS | `mnemos-mnemos` | 8700 | healthy | `OK` | 15.3 ms |
| Research UI | local MNEMOS build | 8788 | healthy | `OK` | 15.4 ms |
| OpenWebUI proxy | local MNEMOS build | 8790 | healthy | `OK` | 15.6 ms |

Compose service identity, not a fixed container name, is authoritative. The
first probe against the temporary worktree's Compose copy correctly found no
containers because Compose project identity derives from the selected Compose
file location. Installers should pass the Compose file that owns the running
stack, or use explicit service-name and URL overrides.

The verifier ran only Compose `config`, Compose `ps`, and PostgreSQL
`pg_isready`, plus HTTP GET probes. It did not start, stop, restart, recreate,
pull, build, or configure a container.

## Approval checkpoints and limitations

Before any lint execution, the frontier agent must display the exact read-only
lint command and obtain human approval. Before any code mutation, it must obtain
a separate human approval. These approvals are outside MNEMOS and are not MCP
tools. After mutation, the current packet is stale and must be rebuilt.

R0 supports only Git-tracked UTF-8 Python and Markdown. Retrieval is local,
deterministic lexical ranking rather than embeddings. There is one active packet
per sidecar process. The packet duplicates source excerpts for transparent
offline consumption, so its byte size can exceed the admitted source-byte cap.

## Verification

- Focused R0 suite: 74 passed.
- Full repository suite: 1,469 passed, 8 explicitly skipped.
- Protected-file diff (`service/app.py`, `mnemos/config.py`,
  `mnemos/retrieval/`, `docker-compose.yml`, and the existing MNEMOS MCP
  server): empty.
- Packet construction, retrieval, snapshot verification, and live stack
  preflight made no change to the selected target checkout.

## Conclusions

```text
PROJECT_MEMORY_PACKET_R0_COMPLETE
READ_ONLY_MCP_RETRIEVAL_VERIFIED
TARGET_PROJECT_MUTATION_COUNT=0
DEFAULT_COLLECTION_MUTATION_COUNT=0
RUNTIME_ROUTE_CHANGE_COUNT=0
LINT_EXECUTION_COUNT=0
CODE_MUTATION_REQUIRES_SEPARATE_HUMAN_APPROVAL
```
