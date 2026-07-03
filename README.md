<div align="center">

# MNEMOS

**Governed context operations for AI systems.**

Source-grounded memory, bounded retrieval, and evidence for what AI may know,
what it may influence, and why.

[Quickstart](#quickstart) ·
[Architecture](docs/architecture.md) ·
[Deployment](docs/deployment_profiles.md) ·
[Benchmarks](docs/benchmark.md) ·
[Documentation](docs/README.md) ·
[Capability Status](#capability-status)

</div>

## What MNEMOS Is

MNEMOS is a containerised, source-grounded context service for AI-native applications.

It helps AI systems retrieve and assemble durable working context with provenance, bounded candidate selection, audit trails, and explicit controls around what retrieved or derived information may influence.

MNEMOS is designed for the operational layer around AI: maintaining useful context, preserving source lineage, constraining reach, and producing evidence that supports review, replay, and correction.

## Why MNEMOS

Model capability alone is not enough to make AI reliable in real work.

Many retrieval systems can return something related. MNEMOS is designed for systems that also need to know whether context is current, supported, permitted, bounded, and suitable to influence the next step.

<img width="1672" height="941" alt="MNEMOS" src="https://github.com/user-attachments/assets/dcd76974-0525-4ac9-8c18-22b016a9b780" />

- What source supports this result, and what is its lineage?
- Is it current, superseded, contradictory, provisional, or derived?
- Why was it selected, and what route or evaluation influenced that selection?
- What may this context influence, and what remains outside its scope?
- Can the result, evidence, and decision path be inspected, replayed, corrected, or rolled back?

The goal is not simply more memory. It is maintained, inspectable context that can be used safely inside a bounded AI workflow.

## Core Capabilities

| Capability | What it provides | Status |
| --- | --- | --- |
| Source-grounded Engrams | Content, metadata, provenance, and lineage-ready context records | Core |
| Semantic and hybrid retrieval | Qdrant/pgvector profile options with bounded candidate selection | Core |
| Forensic ledger | Auditable index, search, retrieval, and mutation events | Core |
| Context governance controls | Candidate evaluation, contradiction handling, lifecycle controls, and explicit influence boundaries | Core / configurable |
| Context assembly boundary | Separation between source evidence, retrieval, governance, and downstream consumer context | Core |
| Deployment profiles | Docker Compose deployment postures for different operating needs | Core |
| Research and shadow lanes | Read-only, evaluation, or observational tracks that do not alter default delivery or authority surfaces | Research / experimental |

## Architecture Overview

```text
Sources, records, and project artifacts
                  ↓
      Engrams + source lineage
                  ↓
 Semantic / hybrid candidate retrieval
                  ↓
 Governance, lifecycle, and evidence checks
                  ↓
 Bounded, inspectable working context
                  ↓
 AI application, operator, or agent workflow
```

MNEMOS separates source evidence, retrieval, governance, and consumer context assembly.

This separation is intentional. A retrieved item is not automatically authoritative, current, permitted, or suitable to influence an action. MNEMOS preserves the evidence and boundaries needed for an application, operator, or workflow to make that decision.

Research lanes do not alter default retrieval, delivery, or authority surfaces unless explicitly enabled and independently evaluated.

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

The installer is platform-aware. On macOS, including Apple Silicon, it
automatically generates a CPU-safe compose file without `runtime: nvidia` and
sets `MNEMOS_GPU_DEVICE=cpu` in `.env.mnemos`. Linux hosts with an NVIDIA GPU
and NVIDIA Container Toolkit use CUDA by default. You can override detection
with `--compute-mode cpu` or `--compute-mode cuda`.

CPU mode is suitable for local evaluation and smaller corpora, but first-query
latency may be higher while embedding models warm up.

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

For local model workflows, see the [Ollama MFS adapter guide](docs/integrations/ollama_mnemos_mfs.md). It retrieves MNEMOS evidence through the SDK/API boundary, sends bounded context to Ollama, and returns citations without changing retrieval or enforcement policy.

## Capability Status

MNEMOS distinguishes supported runtime capabilities from experimental and research work.

A documented concept, ADR, prototype, benchmark baseline, or shadow adapter is not automatically a supported runtime capability. Status reflects the current repository implementation, explicit enablement path, tests, and operational evidence.

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
- Associative Routing E2: opt-in candidate expansion behind an explicit request flag and a global kill switch (disabled by default). Normal retrieval remains primary.
  - candidate-addition only
  - normal retrieval remains primary
  - expansion candidates are governed independently, source-linked, origin-labeled, bounded, and appended without suppressing or re-ranking normal results

Associative Routing E2 is an opt-in experimental candidate-expansion path.

It can append a small number of governed, source-linked candidates after normal retrieval and governance have completed. It does not alter default retrieval, suppress normal results, inject authority fields, or make durable writes.

Its current evidence supports limited usefulness for selected query classes, including supersession and evidence-completeness questions. It does not support a general retrieval quality, production-readiness, or broad superiority claim.

### Research / Shadow

- EBIR refinement
- Session Context Assembler local shadow adapter
- GateMem reference baseline
- Associative Routing E0 prototype and E1 opt-in shadow behavior; E1 is observational and does not change delivered results
- Derived-facts work unless and until a specific deployment artifact authorizes a narrower posture
- TimesFM predictive pulse and related advisory predictive lanes

### Planned Or Blocked

- Context Atlas P0 and Associative Retrieval A1 beyond their current specs
- ColBERT/reranker production promotion
- Any production authorization-security or broad performance claim not backed by a current evidence artifact

This boundary is deliberate: research may inform future context and harness operations, but it does not silently change default retrieval behavior, authority, or delivery.

The [support matrix](docs/support_matrix.md) is the public boundary for status claims.

## Evidence And Benchmarks

MNEMOS maintains benchmark, evaluation, and operational evidence artifacts alongside implementation work.

Results are scoped to their recorded corpus, configuration, deployment profile, and date. They are intended to make behavior inspectable and to support repeatable evaluation as context sources, retrieval paths, governance rules, and research lanes evolve.

See:

- [Benchmark methodology and current results](docs/benchmark.md)
- [Support matrix](docs/support_matrix.md)
- [Deployment profiles](docs/deployment_profiles.md)
- [Architecture guide](docs/architecture.md) and [ADR index](docs/adr/README.md)
- [Operator playbook](docs/mnemos_operator_playbook.md)

Research, local evaluation, and shadow-mode results are not general performance, security, reliability, or production-readiness claims unless explicitly stated and supported by their corresponding evidence artifact.

MNEMOS treats evidence as part of the operating surface: context should be traceable not only when it is retrieved, but when it is reviewed, challenged, refreshed, or retired.

### Associative Routing E2

Associative Routing E2 is retained as an opt-in experimental candidate-expansion mechanism.

In its recorded 22-query comparison, expansion triggered on two queries and added two source-linked candidates classified as correct and needed: one supersession answer that normal semantic retrieval missed, and one missing evidence item that completed an otherwise incomplete result set.

The comparison used current local code against the live Qdrant and PostgreSQL backends. It did not exercise the deployed service container's HTTP path because the container image was confirmed to predate the E1 and E2 implementation.

E2 remains disabled by default. The available evidence supports evaluation for selected query classes only; it is not a broad claim of retrieval superiority, production quality, or authorization safety.

See:

- [E2 closeout record](docs/associative_routing_e2_closeout.md)
- [E2 design note and limitations](docs/associative_routing_e2_design_note.md)
- [E2 commit manifest](docs/associative_routing_e2_commit_manifest.md)
- [Recorded E2 comparison artifact](benchmarks/results/associative_routing_e2_live_comparison_run_001.json)
- [Benchmark methodology](docs/benchmark.md)
- [Support matrix](docs/support_matrix.md)

"Live backend" evaluation and "deployed service" evaluation are not interchangeable.

Where an evidence artifact used local in-process execution against live data stores rather than the deployed HTTP service, MNEMOS documentation must describe that distinction plainly.

## Documentation Index

Start with [docs/README.md](docs/README.md) for grouped links to getting started, architecture, deployment, evidence, governance boundaries, research lanes, and ADRs.

## Contributing And Project Boundaries

Contributions are welcome, particularly around reproducible benchmarks, deployment reliability, documentation, source-grounded retrieval evaluation, provenance, lifecycle management, and evidence-backed context operations.

Changes that affect retrieval ranking, candidate selection, governance, authority, disclosure, promotion, deletion, or downstream influence boundaries must include explicit tests and evidence artifacts.

MNEMOS is intended to support bounded AI workflows. Contributions should preserve the separation between context retrieval, governance evaluation, consumer decision-making, and action execution.

This repository currently declares a proprietary license posture.
