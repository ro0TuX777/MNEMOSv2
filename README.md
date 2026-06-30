<div align="center">

# MNEMOS

**Governed, source-grounded memory and retrieval for AI systems.**

Retrieve context with lineage, boundaries, and evidence for why it should influence the next action.

[![MNEMOS CI Gates](https://github.com/ro0TuX777/MNEMOSv2/actions/workflows/mnemos-gates.yml/badge.svg)](https://github.com/ro0TuX777/MNEMOSv2/actions/workflows/mnemos-gates.yml)

[Quickstart](#quickstart) |
[Architecture](docs/architecture.md) |
[Deployment](docs/deployment_profiles.md) |
[Benchmarks](docs/benchmark.md) |
[Documentation](docs/README.md) |
[Research Status](#capability-status)

</div>

## What MNEMOS Is

MNEMOS is a containerised, source-grounded memory and retrieval service for AI-native applications.

It helps systems retrieve durable context with provenance, bounded candidate selection, audit trails, and explicit controls around what retrieved or derived information may influence.

## Why MNEMOS

Many retrieval systems can return something related. MNEMOS is designed for systems that also need to know:

- What source supports this result?
- Is it current, superseded, contradictory, or derived?
- What route selected it?
- What is it allowed to influence?
- Can the result be inspected, replayed, or rolled back?

## Core Capabilities

| Capability | What it provides | Status |
| --- | --- | --- |
| Source-grounded Engrams | Content, metadata, provenance, and lineage-ready records | Core |
| Semantic and hybrid retrieval | Qdrant/pgvector profile options with bounded retrieval controls | Core / configurable |
| Forensic ledger | Auditable index, search, and mutation events | Core |
| Governance controls | Candidate evaluation, contradiction handling, lifecycle controls | Core / configurable |
| Deployment profiles | Docker Compose deployment postures for different operating needs | Core |
| Experimental research lanes | Shadow, read-only, or opt-in evaluation tracks | Experimental / research |

## Architecture Overview

```text
Sources and project artifacts
            |
            v
   Engrams + source lineage
            |
            v
 Semantic / hybrid retrieval
            |
            v
 Governance and evidence checks
            |
            v
 Bounded context or retrieval result
            |
            v
 AI application, operator, or agent
```

MNEMOS keeps source evidence, retrieval, governance, and consumer context assembly as separate concerns. Research lanes do not alter default retrieval or authority surfaces unless explicitly enabled and independently evaluated.

See the concise [architecture guide](docs/architecture.md), the detailed [whitepaper](docs/whitepaper.md), and the [ADR index](docs/adr/README.md).

## Quickstart

Prerequisites:

- Docker and Docker Compose
- NVIDIA container runtime for the checked-in GPU-oriented service image
- Python 3.11+ for local tools and SDK examples

Clone and enter the repository:

```bash
git clone https://github.com/ro0TuX777/MNEMOSv2.git
cd MNEMOSv2
```

Prepare the local Python environment:

```bash
python -m pip install -r requirements.txt
```

Start the default local Compose stack:

```bash
docker compose up -d --build
```

Check service health:

```bash
curl http://localhost:8700/health
```

Minimal index/search example:

```bash
curl -X POST http://localhost:8700/v1/mnemos/index \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"content":"MNEMOS quickstart verification record","source":"docs:quickstart","neuro_tags":["docs","quickstart"]}]}'

curl -X POST http://localhost:8700/v1/mnemos/search \
  -H "Content-Type: application/json" \
  -d '{"query":"quickstart verification record","top_k":1}'
```

For profile selection, generated Compose files, and operational promotion rules, see [deployment profiles](docs/deployment_profiles.md) and the [operator playbook](docs/mnemos_operator_playbook.md).

## Capability Status

| Status | Meaning |
| --- | --- |
| Core | Supported in the primary runtime and covered by maintained tests |
| Experimental | Available only through explicit opt-in or controlled operator evaluation |
| Research / shadow | Implemented for evidence gathering; does not change default runtime delivery or authority |
| Planned | Documented direction; not implemented or not authorized for runtime use |

### Core

- Engram storage and retrieval
- Supported deployment profiles
- Forensic ledger
- Semantic retrieval and configurable hybrid retrieval
- Configurable governance behavior
- Source-grounded evidence summaries in search responses

### Experimental

- `graph_hybrid_experimental`, read-only and outside the public default retrieval surface
- Hybrid retrieval as a targeted evaluation/configuration mode, not a broad default
- Derived-facts evaluation lane when explicitly enabled and bounded by its documented gates
- Associative candidate expansion only where explicitly requested and gated

### Research / Shadow

- EBIR refinement
- Session Context Assembler local shadow adapter
- GateMem reference baseline
- Associative Routing E0/E1 shadow work
- Derived-facts work unless and until a specific deployment artifact authorizes a narrower posture
- TimesFM predictive pulse and related advisory predictive lanes

### Planned Or Blocked

- Context Atlas P0 and Associative Retrieval A1 beyond their current specs
- ColBERT/reranker production promotion
- Any production authorization-security or broad performance claim not backed by a current evidence artifact

The [support matrix](docs/support_matrix.md) is the public boundary for status claims.

## Evidence And Benchmarks

MNEMOS maintains benchmark and evaluation artifacts alongside implementation work. Results are scoped to their recorded corpus, configuration, and date.

See:

- [Benchmark methodology and current results](docs/benchmark.md)
- [Support matrix](docs/support_matrix.md)
- [Deployment profiles](docs/deployment_profiles.md)
- [Architecture guide](docs/architecture.md) and [ADR index](docs/adr/README.md)
- [Operator playbook](docs/mnemos_operator_playbook.md)

Research and local evaluation results are not general performance, security, or production-readiness claims unless explicitly stated by their evidence artifact.

## Documentation Index

Start with [docs/README.md](docs/README.md) for grouped links to getting started, architecture, deployment, evidence, governance boundaries, research lanes, and ADRs.

## Contributing And Project Boundaries

Contributions are welcome, particularly around reproducible benchmarks, deployment reliability, documentation, and source-grounded retrieval evaluation.

Changes that affect retrieval ranking, governance, authority, disclosure, promotion, or deletion must include explicit tests and evidence artifacts.

This repository currently declares a proprietary license posture.
