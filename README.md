# MNEMOS

**Multi-tier Neuro-tagged Engram Memory with Optimal Near-lossless Index Compression**

MNEMOS is a containerized, contract-governed memory service for AI-native applications. It gives agents and RAG systems a reusable memory layer that can ingest knowledge, retrieve it under latency budgets, explain why results won or lost, and reconcile contradictions without deleting the original evidence.

## Why MNEMOS Exists

Every AI application that persists and retrieves knowledge ends up rebuilding the same infrastructure: embedding pipelines, vector databases, rerankers, metadata filters, compression, audit trails, and governance logic. MNEMOS packages those concerns as an application-agnostic memory appliance with a versioned REST contract and a Python boundary SDK.

The current MNEMOSv2 workstream moves the system beyond static retrieval. It now supports adaptive routing, hierarchy-aware synthesis, additive consensus behavior, and a TimesFM-backed predictive layer so memory can be routed, governed, reconciled, audited, and pre-warmed before demand arrives.

## What You Get

| Capability | What it provides |
|---|---|
| **Deployment profiles** | Core Memory Appliance (Qdrant), Governance Native (pgvector), or Custom Manual profiles with generated Compose/env/manifest files |
| **Tiered vector search** | Nomic Matryoshka embeddings with 64-dimensional coarse prefetch and 768-dimensional rescore |
| **TurboQuant compression** | 4-bit embedding compression with an 8x storage reduction target while preserving full-vector rescore fidelity |
| **Adaptive routing** | Embedded query-complexity classification routes simple, multi-hop, and global synthesis queries to different retrieval postures |
| **Budget-aware retrieval** | EWMA cost model and degradation ladder for latency-budgeted responses, with degraded components surfaced in the API contract |
| **Hierarchical summaries** | RAPTOR-lite summary engrams for global/theme queries, isolated from default factoid search by reserved server-side sentinels |
| **Consensus governance** | Resolution Engrams structure factual collisions, preserve parent lineage, and receive governed read-path priority |
| **Predictive pulse** | TimesFM-backed operational forecasts for query volume, p95 latency, cache pressure, and degradation risk |
| **Predictive hygiene** | Semantic volatility signals accelerate decay and reconciliation for high-change engram families |
| **Pre-cognitive retrieval** | Intent trajectories can populate pre-cognitive cache entries through low-priority shadow search |
| **Citation-ready evidence contract** | Search responses expose per-result provenance and grouped source summaries for downstream chat, RAG, and agent UIs |
| **EBIR shadow refinement** | Offline evidence-based refinement benchmarks reconcile conflicting parent evidence without promoting derived claims into live memory |
| **EBIR-R2 reviewer harness** | Frozen truthsets, blinded Markdown packets, deterministic assignment, compiler validation, and gold-label scoring for human evaluation |
| **Context Atlas / associative retrieval specs** | Deferred design lanes for context exploration and projection-based associative retrieval, kept separate from live runtime behavior |
| **Reflect precision** | Lexical or NLI-backed usage detection for safer post-answer reinforcement |
| **Forensic auditability** | PostgreSQL-backed audit trail, immutable lineage edges, and explainable governance modifiers |
| **Boundary SDK** | Python client with readiness polling, retry, timeout, and graceful degradation helpers |

## Current Phase Highlights

MNEMOSv2 Phase 7-10 advances are documented in [docs/whitepaperupdates.md](docs/whitepaperupdates.md) and incorporated into the main [whitepaper](docs/whitepaper.md).

| Phase | Result |
|---|---|
| **Phase 7: Matryoshka migration** | Nomic MRL runtime promoted with coarse prefetch/full-vector rescore and warmup readiness |
| **Phase 8: Adaptive routing** | Embedded complexity classifier reached `1.0` hold-out accuracy |
| **Phase 9: Hierarchical retrieval** | Summary isolation validated with `0.7342` mean hierarchy similarity |
| **Phase 10: Knowledge reconciliation** | Consensus gate passed `5/5`; Resolution Engrams ranked first while preserving conflicting parents for audit |
| **Phase 11: Pulse** | Observed and forecasted operational telemetry exposed for self-awareness |
| **Phase 12: Autonomous pre-warm** | High-confidence forecast spikes can trigger governed warmup with cooldown and audit logs |
| **Phase 13: Predictive hygiene** | Volatility forecasts bias freshness decay and proactive reconciliation |
| **Phase 14: Shadow search** | Forecasted intent trajectories populate pre-cognitive cache entries |
| **Phase 15: Cognitive-cycle transparency** | CoALA-aligned cycle records expose bounded, redacted, adapter-compatible evidence paths |
| **EBIR-R1/R2: Evidence refinement evaluation** | Shadow-only refinement and reviewer-trial harness validate contradiction handling without changing live retrieval or authority |
| **Evidence contract** | `/search` results carry normalized provenance packets and `meta.evidence_summary` for citation-aware integrations |

Benchmark evidence lives in:

- [benchmarks/results/phase_8_complexity_accuracy.json](benchmarks/results/phase_8_complexity_accuracy.json)
- [benchmarks/results/phase_9_hierarchy_sim.json](benchmarks/results/phase_9_hierarchy_sim.json)
- [benchmarks/results/phase_10_consensus_gate.json](benchmarks/results/phase_10_consensus_gate.json)
- [benchmarks/results/coala_baseline_v3.2.json](benchmarks/results/coala_baseline_v3.2.json)
- [benchmarks/results/latency_slo_burn_in.json](benchmarks/results/latency_slo_burn_in.json) — supplemental illustrative trend series (not harness-measured; per-phase measured evidence is in the three gate artifacts above)

## Deployment Profiles

| Profile | Stack | Best for |
|---|---|---|
| **Core Memory Appliance** | Qdrant + PostgreSQL + MNEMOS | Semantic memory, agent systems, general-purpose RAG |
| **Governance Native** | PostgreSQL/pgvector + MNEMOS | Provenance-heavy, metadata-filtered, compliance-aware retrieval |
| **Custom Manual** | Operator-defined | Advanced multi-tier setups and experiments |

## Quick Start

```bash
# Install with guided profile selection
python -m installer

# Start the generated stack
docker compose -f docker-compose.generated.yml up -d --build

# Validate health, capabilities, and contract fields
python tools/mnemos_health_audit.py
```

```python
from mnemos_sdk import MnemosClient, MnemosConfig

client = MnemosClient(MnemosConfig.from_env())
client.wait_until_ready(warmup=True)

client.index([
    {
        "content": "Gravity waves detected by LIGO in 2015",
        "source": "arxiv:1602.03837",
        "neuro_tags": ["physics", "gravitational-waves"],
    }
])

for hit in client.search("gravitational wave detection", top_k=5):
    print(f"[{hit.score:.3f}] {hit.engram['content'][:80]}")
```

## API Surface

Every response includes contract metadata such as `contract_version`, `status`, `profile`, `tiers`, and `degraded_components`. Adaptive retrieval responses can also include complexity classification and routing posture metadata.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Container health check |
| `/v1/mnemos/capabilities` | GET | Active profile, backends, degraded components, compression config |
| `/v1/mnemos/index` | POST | Ingest documents into engrams |
| `/v1/mnemos/search` | POST | Query active backends with optional metadata filters and governance explanation |
| `/v1/mnemos/engrams/{id}` | GET | Retrieve an engram by ID |
| `/v1/mnemos/engrams/{id}` | DELETE | Remove an engram from all backends |
| `/v1/mnemos/audit` | GET | Query the forensic ledger |
| `/v1/mnemos/stats` | GET | Backend sizes, profile info, and compression stats |
| `/v1/mnemos/warmup` | POST | Preload models and reduce first-query latency |
| `/v1/mnemos/pulse` | GET | Return observed pulse telemetry and predictive trend signals when enabled |

## Operational Tooling

| Tool | Command | Purpose |
|---|---|---|
| Installer | `python -m installer` | Guided profile selection and config generation |
| Health audit | `python tools/mnemos_health_audit.py` | Validate health, contract fields, and version drift |
| Reconciliation dry run | `python tools/run_phase10_reconciliation_dry_run.py --mode dry-run --fail-on-conflict` | Detect unresolved contradiction clusters |
| Reconciliation apply | `python tools/run_phase10_reconciliation_dry_run.py --apply` | Persist Resolution Engrams when an indexer is available |
| Resolution gate | `python tools/validate_phase10_resolution_gate.py` | Verify Resolution Engram priority and parent suppression |
| Contract diff | `python tools/contract_diff.py --old v1.json --new v2.json` | Backward/forward compatibility checks |
| CI gates | `python tools/mnemos_ci_gates.py --run-health-audit` | Pipeline gate runner |
| EBIR-R1 refinement benchmark | `python tools/run_ebir_refinement_benchmark.py --truthset benchmarks/truthsets/ebir_r1_adversarial.json --fail-on-gate` | Shadow-only refinement validation over adversarial contradiction fixtures |
| EBIR-R2 preflight | `python tools/run_ebir_r2_preflight.py --truthset benchmarks/truthsets/ebir_r2_reviewer_tasks.json --reviewers configs/ebir_r2_reviewers.json --blind --output-dir eval_results/ebir_r2 --fail-on-gate` | Validate blinded reviewer packets before human distribution |
| EBIR-R2 compiler | `python tools/compile_ebir_r2_pilot_report.py --manifest eval_results/ebir_r2/assignment_manifest.json --responses-dir eval_results/ebir_r2/pilot_responses --output eval_results/ebir_r2/ebir_r2_pilot_report.md --fail-on-gate` | Compile completed Markdown responses without exposing gold labels to reviewers |
| EBIR-R2 gold scoring | `python tools/score_ebir_r2_gold_report.py --truthset benchmarks/truthsets/ebir_r2_full_reviewer_tasks.json --compiled-report eval_results/ebir_r2_full/ebir_r2_full_report.md --output eval_results/ebir_r2_full/ebir_r2_gold_report.json --fail-on-gate` | Restricted post-freeze unblinding and condition-level scoring |

## Repository Layout

```text
mnemos/              Core library: retrieval, governance, engrams, compression, audit
mnemos_sdk/          Boundary SDK for consumer apps
service/             Flask REST API and MFS contract surface
installer/           Guided installer, probes, profiles, renderer
tools/               Operational and validation tooling
benchmarks/          Truthsets, simulations, and result artifacts
tests/               Unit and regression tests
docs/                Whitepaper, operator playbook, and phase reports
```

## Documentation

- [Whitepaper](docs/whitepaper.md): architecture, governance, benchmarks, deployment model
- [Phase 7-10 supplement](docs/whitepaperupdates.md): adaptive routing, hierarchy, and consensus governance update
- [Operator playbook](docs/mnemos_operator_playbook.md): diagnostics, rollout, rollback, and incident procedures
- [Installation guide](INSTALL.md): installer usage, deployment profiles, and manual setup
- [Chat integration evidence contract](docs/chat_integration_evidence_contract.md): normalized provenance fields for citation-aware consumers
- [EBIR-R1 acceptance](docs/ebir_r1_acceptance.md): shadow-only refinement acceptance gates
- [EBIR-R2 trial protocol](docs/ebir_r2_trial_protocol.md): blinded reviewer-trial protocol and scoring workflow
- [Context Atlas P0 spec](docs/context_atlas_spec.md): deferred exploration API design
- [Associative retrieval A1 spec](docs/associative_retrieval_a1_spec.md): deferred benchmark-first graph projection design

## Status Notes

- `graph_hybrid_experimental` remains experimental and read-only; it is not exposed on the public retrieval-mode surface by default.
- Summary and Resolution Engrams are synthetic governed artifacts with lineage back to raw parents.
- EBIR refinement and EBIR-R2 scoring are offline, shadow-only evaluation paths; they do not write memory, alter ranking, or promote authority.
- Context Atlas P0 and A1 associative retrieval remain specifications until their prerequisite gates are complete.
- Reserved sentinels such as `__exclude_derived__` and `__exclude_summaries__` are server-managed and rejected when supplied directly by clients.
- Predictive features are controlled by `MNEMOS_TIMESFM_ENABLED` and `MNEMOS_PULSE_ACTIONS`; advisory mode is the default production posture.

## License

Proprietary.
