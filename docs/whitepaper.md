# MNEMOS — White Paper

**A containerised, contract-governed memory and retrieval service for AI-native applications.**

*Version 3.4 · July 2026*

> [!NOTE]
> **As of June 11, 2026:** Benchmark conclusions in dated sections remain scoped to their cited artifact timestamps. For full methodology, raw artifacts, and latest measured runs, see `docs/benchmark.md`.
> For release, promotion, rollback, and incident execution runbooks, see `docs/mnemos_operator_playbook.md`.
> **Governance layer (MemArchitect Waves 1–4) is implemented and CI-gated** — per-candidate policy pipeline, entity-slot contradiction detection, reflect-path reinforcement, and background hygiene control loops (dry-run + enforceable `apply` mode) are in place.
> **Memory Over Maps Phases 1–5 are implemented and benchmark-gated** — source-grounded lineage, bounded candidate envelope, on-demand derived views, cache + invalidation, and bounded semantic reflect evolution passed phase gates on March 30, 2026; gates remain enforced in CI.
> **Derived Facts lane (PIT-0→PIT-10) is production-adjacent and pilot-ready** — isolated shadow evaluation via `/api/v1/evaluate_derived_shadow`; default retrieval remains derived-fact-free. See §4.8.
> **Adaptive Routing, Hierarchical Retrieval, and Consensus Governance (Phases 8-10) are operationally enforced** — embedded query-complexity classification reached 1.0 hold-out accuracy, the Phase 9b summary-isolation sentinel is live in the service container, and Phase 10 Resolution Engrams passed the live consensus gate. See `benchmarks/results/phase_8_complexity_accuracy.json`, `benchmarks/results/phase_9_hierarchy_sim.json`, and `benchmarks/results/phase_10_consensus_gate.json`.
> **MNEMOS-Thinking predictive stack (Phases 11-14) is implemented** — TimesFM-backed pulse forecasting, autonomous pre-warm, semantic volatility hygiene, and pre-cognitive shadow search have passed synthetic phase gates. See §2.1 and §4.9.
> **CoALA Cognitive Cycle (v3.2) is implemented** — MNEMOS cognitive behaviours are now explicit, auditable, and interoperable via `mnemos/cognitive/`. Add `cognitive_cycle: true` to any `/search` request to receive a `CognitiveCycleRecord` in the response. See §4.10.
> **PatternEngramCandidate Extraction Harness (v3.3) is implemented** — Phases 16–21 complete. `CycleEvaluator`, `PatternLearner`, `PatternConsolidator`, `PatternCandidateStore`, and `PatternEngram` ship as advisory-only, governance-gated pattern abstraction. 182 new tests; Phase 21 gate harness (`tools/run_pattern_phase_gate.py`) passes 8/8 scenarios and 5/5 cross-cutting gates. See §4.11.
> **EBIR-R1 shadow refinement lane is technically accepted and CI-gated** — RepFusion-inspired Evidence-Bounded Iterative Reconciliation evaluates multi-pass evidence challenge/revision for contradiction clusters in shadow only. Authoritative Resolution Engram promotion remains blocked. See §4.12 and `docs/ebir_r1_acceptance.md`.
> **Session Context Assembler local shadow milestone is technically accepted** — a consumer-neutral, read-only adapter can assemble bounded, provenance-labeled context packages in an isolated local harness. It has no listener, route, SDK, external consumer connection, deployment, or effect on MNEMOS authority surfaces. See §4.13 and ADR 0008.
> **GateMem governance reference baseline is frozen** — MNEMOS completed a governed authorization/disclosure research lane from clean-input benchmark isolation through a deterministic offline reference implementation. G4 matched 36/36 expected synthetic outcomes and passed 33/33 reference gates. This is regression-only research evidence, not production authorization security or held-out benchmark performance. See §4.14 and ADRs 0009–0013.
> **Graph Tier (`graph_hybrid_experimental`) is experimental and read-only** — offline/live resolver validation complete (MG-Test-1→10); not exposed on the public HTTP retrieval-mode surface. See §4.2 and `docs/graph_tier/operator_guide.md`.
> **Open WebUI evidence lane (v3.4) is operational as a local consumer-boundary workflow** — a research intake UI, an Ollama/OpenAI-compatible evidence proxy, and per-answer evidence receipts let a generic chat front end answer from MNEMOS evidence with token streaming, multi-turn query condensation, and deterministic post-hoc verification annotations (citation check, truncation honesty, score spread, tamper-evidence hash). Context-only adapter: it does not alter MNEMOS retrieval, write memory, or enforce admission policy. Local single-user workflow evidence only. See §4.16.
> **Deployment model:** MNEMOS runtime services are deployed as a Docker Compose stack; all serving components run in containers.
> **Developer model:** tooling, benchmarks, and tests are typically run from host Python unless explicitly containerized.

### Changes since v3.0 (March 2026)

| Date | Change |
|---|---|
| 2026-07-10/11 | **Open WebUI evidence lane hardening (v3.4)** - Real token streaming through the evidence proxy (Ollama NDJSON passthrough re-emitted as OpenAI SSE / Ollama chunks); multi-turn support with footer-stripped history forwarding and standalone retrieval-query condensation recorded in receipt metadata; receipt verification annotations (`citation_check`, `generation` truncation honesty, `score_stats`, `content_hash`); receipt overflow archived instead of deleted; real token usage passthrough; waitress serving. Deterministic, passive annotations only — no admission enforcement. Local single-user workflow evidence. |
| 2026-07-05→08 | **Open WebUI evidence lane established** - Research intake UI (upload → extract → chunk → index, with Docling OCR fallback and page lineage), Ollama/OpenAI-compatible evidence proxy, per-answer evidence receipts with a receipt browser, and containerized deployment of both services in the compose stack. |
| 2026-06-26 | **AI developer MCP memory trial** - MNEMOS was exposed through an MFS-compatible MCP bridge and tested in local paired app-building trials against no-memory controls. The first pilot exposed infrastructure-readiness and measurement gaps; the refreshed E1 paired run used a dedicated seeded collection and structured telemetry. Both conditions completed and passed acceptance. MNEMOS retrieved useful task context with provenance and no observed quality degradation, but added memory/tool-call overhead and did not establish a speed or token-efficiency claim. Local development evidence only; no general memory-performance claim. |
| 2026-03-30 | Enhancement roadmap closed: CI phase gates, Wave 4 hygiene gate, reflect precision guards, tenant policy profiles, explainability traces, economics counters, SLO reliability gate, operator playbook |
| 2026-04–05 | Qdrant v1.17 server-side RRF (`qdrant_rrf`), relevance feedback adapter, Cross-Encoder rerank hardening |
| 2026-05–06 | Graph Tier evaluation track (MG-Test-1→10); read-only `QdrantEngramResolver` with batched prefetch |
| 2026-06 | Derived Facts production-adjacent lane (PIT-0→PIT-10) approved for limited controlled operator pilot; DFE human-value trials (dfe_12→dfe_21); ops certification closeout (ops_0→ops_4) |
| 2026-06-10 | Test suite: 564 tests collected; all CI gates passing (MoM phases, governance evidence, hygiene, SLO canary_25) |
| 2026-06-11 | Phase 8 embedded-reflex adaptive routing, Phase 9b live hierarchy isolation, and Phase 10 consensus Resolution Engrams activated; live consensus gate passed 5/5 collisions |
| 2026-06-12 | MNEMOS-Thinking activation — TimesFM sidecar integration, Pulse telemetry normalization, predictive pre-warming, volatility-driven hygiene, and intent-trajectory shadow search are live in the standalone framework |
| 2026-06-14 | **CoALA Cognitive Cycle (v3.2)** — New `mnemos/cognitive/` overlay module. `CognitiveCycleRecord`, `WorkingMemorySnapshot`, `AttentionContract`, `ForecastOutcomeRecord`, and `CycleAssembler` ship as a zero-cost opt-in. New endpoint: `GET /v1/mnemos/cognitive/cycles`. 77 new cycle tests; Phase 15 operational validation adds 8 representative cases and 40 passing focused tests. |
| 2026-06-15 | **PatternEngramCandidate Extraction Harness (v3.3)** — Phases 16–21 complete. `CycleEvaluator` (R²-Mem rubric scorer), `PatternLearner` (ExpeL-style IF-THEN extractor), `PatternConsolidator` (A-MEM deduplication), `PatternCandidateStore` (advisory accumulation + governed promotion), and `PatternEngram` (authoritative promoted pattern). Advisory recall integrated into cognitive cycle (`advisory_patterns` field). 182 new tests; Phase 21 phase gate (`tools/run_pattern_phase_gate.py`) passes 8 scenarios + 5 cross-cutting gates. Gate evidence: `benchmarks/results/pattern_phase_gate.json`. |
| 2026-06-18 | **EBIR-R1 shadow refinement lane** — RepFusion-inspired Evidence-Bounded Iterative Reconciliation scaffold added around `ReconciliationRunner`; adversarial technical acceptance pack covers 10 conflict classes and is CI-gated via `tools/run_ebir_refinement_benchmark.py`. EBIR is shadow-only; authoritative promotion remains blocked pending R2 human-review value trials. |
| 2026-06-22 | **Session Context Assembler research milestone** — governed selector and consumer-neutral, read-only local shadow adapter accepted through ADR 0008. Technical gates cover lineage, digest verification, binding budgets, content-free telemetry, determinism, isolation, kill-switch behavior, and mutation sensitivity. No external integration or production surface is authorized. |
| 2026-06-24 | **GateMem governance reference baseline** — external benchmark isolation, frozen G2/G2A characterization, principal-bound authorization/disclosure semantics, and a deterministic offline G4 reference implementation completed. The 36-case synthetic development corpus matched 36/36 expected outcomes and 33/33 G4 gates passed. Runtime integration, deletion, production authorization, and fresh benchmark claims remain blocked. |

---

## 1. Problem Statement

Every AI application that persists and retrieves knowledge must solve the same set of problems:

- **Storage**: Where do embeddings live, and how do they scale?
- **Retrieval**: How do you find the right information across thousands of documents with high precision?
- **Compression**: How do you keep memory footprint manageable as the index grows?
- **Enrichment**: How do you go beyond raw vectors to attach semantic meaning, provenance, and relationships?
- **Auditability**: How do you trace what was stored, retrieved, modified, and when?
- **Integration**: How do you wire memory into an application without tight coupling or fragile glue code?

Today, each project re-implements these capabilities from scratch — writing custom embedding pipelines, bolting on vector databases, and building ad-hoc search logic. The result is fragile, inconsistent, and impossible to reuse across projects.

## 2. Solution: MNEMOS

**MNEMOS** (Multi-tier Neuro-tagged Engram Memory with Optimal Near-lossless Index Compression) is a GPU-accelerated, production-grade memory service for AI-native applications. It deploys via **named deployment profiles** — each profile defines a retrieval backend, container topology, and operational posture:

- **Core Memory Appliance** — Qdrant + PostgreSQL + MNEMOS (3 containers). Semantic ANN with payload filtering.
- **Governance Native** — PostgreSQL/pgvector + MNEMOS (2 containers). ANN + SQL metadata filtering in one query.
- **Custom Manual** — Operator-defined configuration for advanced multi-backend setups.
- **Hybrid Retrieval Mode (Gate C)** - optional lexical + semantic fusion mode inside existing profiles (not a separate profile).

A guided Python installer (`python -m installer`) probes the host, resolves platform-safe compute mode, asks deployment questions, recommends a profile, and generates all deployment files. The service exposes a versioned REST API governed by an MFS contract.

MNEMOS is **application-agnostic** — it knows nothing about the domain of the consuming application. It stores, enriches, compresses, retrieves, and audits knowledge. That’s it.

**What's new in v3:**
- **Deployment profiles** replace the flat tier model — named profiles with distinct retrieval architectures
- **pgvector tier** — PostgreSQL-native ANN with SQL WHERE metadata filtering (Governance Native profile)
- **Guided installer** — Q/A + host probes → profile recommendation → compose + env + manifest generation
- **Profile benchmarks** — per-profile retrieval latency, recall, and throughput data
- **Deployment manifest** — `mnemos_profile.yaml` as durable deployment artifact
- **Governance layer (MemArchitect Waves 1–4)** — per-candidate policy pipeline, contradiction detection, reflect-path reinforcement, tenant policy profiles, explainability traces, and background hygiene control loops; advisory and enforced read path modes; default is `off`
- **Memory Over Maps lane (Phases 1–5)** — source-grounded artifact lineage, deterministic candidate narrowing, on-demand derived views, deterministic cache + invalidation with dry-run parity, and bounded semantic reflect benchmark pack
- **Adaptive complexity routing (Phase 8)** — embedded linear classifier over the active query embedding space routes CLASS_A factoid, CLASS_B multi-hop, and CLASS_C global synthesis queries; hold-out accuracy is 1.0 with sub-millisecond classifier overhead after embedding reuse
- **Hierarchical retrieval (Phase 9b)** — RAPTOR-lite summary engrams support global CLASS_C retrieval while the `__exclude_summaries__` sentinel prevents summary nodes from leaking into default factoid searches
- **Consensus governance (Phase 10)** — contradiction clusters can synthesize additive Resolution Engrams that preserve parent lineage, take Tier-1 read-path priority with a 1.25 contradiction modifier, and suppress conflicting parents without deleting them
- **EBIR shadow refinement lane (R1)** — RepFusion-inspired, evidence-bounded iterative reconciliation for complex contradiction clusters; CI-gated as shadow research only, with no retrieval, ranking, governance-score, parent-mutation, or promotion-path effects
- **Session Context Assembler (research milestone)** — governed, budget-aware session selection and a consumer-neutral, read-only local shadow adapter; provenance, abstention, replay, digest, disclosure, telemetry, and kill-switch behavior are test-gated without changing runtime, retrieval, governance, or write paths
- **GateMem governance reference baseline (research milestone)** — evaluator-safe clean-input projection, honest weak-governance characterization, principal-bound authorization/disclosure semantics, and a frozen deterministic offline reference implementation; regression-only, with no runtime, production-security, deletion, or fresh benchmark claim
- **Anticipatory cognition (Phases 11-14)** — integration with Google TimesFM enables MNEMOS to forecast its operational pulse, predict factual obsolescence, and execute shadow searches from user intent trajectories
- **Server-side hybrid RRF + relevance feedback** — `qdrant_rrf` fusion policy and governance-driven `discover_points()` exemplar biasing (opt-in)
- **Derived Facts lane (production-adjacent pilot)** — isolated shadow evaluation packets with authority matrices and source traceability; default retrieval invariant: zero derived facts
- **Graph Tier (experimental)** — read-only graph-neighbor expansion via `graph_hybrid_experimental`; double opt-in, no write path
- **SLO-driven promotion gates** — automated canary-stage reliability checks with rollback discipline (`tools/run_slo_reliability_gate.py`)
- **Operator playbook** — single operational runbook for deploy/promote/rollback/incident execution (`docs/mnemos_operator_playbook.md`)
- **Open WebUI evidence lane (v3.4)** — local research intake UI, an Ollama/OpenAI-compatible evidence proxy for generic chat front ends, and per-answer evidence receipts with streaming, multi-turn condensation, and deterministic verification annotations; context-only, no MNEMOS authority-surface change (§4.16)

MNEMOS also ships with a **Boundary SDK** (Python client library) and a suite of **operational tools** (health audit, contract evolution, onboarding, CI gates, and staged cutover) — making it a complete platform that can be deployed with a single `python -m installer`.

Operationally, the current architecture posture is: fast retrieval substrate + adaptive routing + enforced summary isolation + governed consensus controls + source-grounded, bounded, on-demand synthesis + predictive self-optimization.

### 2.1 MNEMOS-Thinking Milestone

MNEMOS-Thinking is the TimesFM-backed predictive layer that moves MNEMOS from
reactive retrieval toward anticipatory memory. It treats operational load,
semantic volatility, and user intent as time-series signals.

| Phase | Capability | Function | Status |
|---|---|---|---|
| 11 | Self-awareness | Pulse forecasts query volume, p95 latency, cache pressure, and degradation risk. | Implemented |
| 12 | Anticipatory warmup | High-confidence forecasts pre-warm retrieval and summary layers before predicted demand. | Implemented |
| 13 | Predictive hygiene | Volatility forecasts shorten semantic half-life and trigger proactive reconciliation. | Implemented |
| 14 | Pre-cognitive retrieval | Intent trajectories run shadow searches and populate pre-cognitive cache entries. | Implemented |

Operational bounds:
- `MNEMOS_TIMESFM_ENABLED=false` returns MNEMOS to the reactive baseline.
- `MNEMOS_PULSE_ACTIONS=advisory` is the default production posture.
- Autonomous pre-warm is reserved for high-confidence triggers (`confidence_score > 0.85`) with cooldown enforcement.
- Forecast-driven actions are auditable through the forensic ledger.
- TimesFM runs in an isolated `mnemos-timesfm` sidecar, not the main request process.

Synthetic phase traces validated pulse spike detection, autonomous pre-warm,
staleness anticipation, and a pre-cognitive cache hit for a predicted future
query.

---

## 3. Architecture

MNEMOS is organised as a layered stack with a pluggable retrieval tier selected by **deployment profile**:

```text
+------------------------------------------------------------+
|                     REST API (:8700)                       |
| /index /search /reflect /pulse /warmup /stats /audit       |
+------------------------------------------------------------+
|            Anticipatory Brain (TimesFM Pulse)              |
| intent trajectory | volatility forecast | SLO prediction   |
+------------------------------------------------------------+
|         Retrieval, Governance, Cache, and Ledger Stack      |
+------------------------------------------------------------+
```

```
┌────────────────────────────────────────────────────────────┐
│                       REST API (:8700)                      │
│   /index   /search   /reflect   /stats   /audit            │
├────────────────────────────────────────────────────────────┤
│                  Engram Enrichment Layer                    │
│   neuro-tags · provenance · confidence · relationship      │
│   edges · extensible metadata                              │
├────────────────────────────────────────────────────────────┤
│            Retrieval (profile-selected)                     │
│                                                             │
│  ┌──────────────────────────┐  ┌────────────────────────┐  │
│  │  Core Memory Appliance   │  │   Governance Native    │  │
│  │  Qdrant (HNSW, CUDA)     │  │   pgvector (Postgres)  │  │
│  └──────────────────────────┘  └────────────────────────┘  │
├────────────────────────────────────────────────────────────┤
│             TurboQuant Compression Layer                    │
│   4-bit quantised storage · 8× raw compression             │
│   Near-optimal distortion (arXiv:2504.19874)               │
├────────────────────────────────────────────────────────────┤
│     Embedding Engine (GPU-accelerated, swappable)          │
│   BGE Base (Default) │ Nomic (Long-Context)                │
│   + optional Cross-Encoder Rerank Lane                     │
├────────────────────────────────────────────────────────────┤
│          Forensic Ledger (PostgreSQL audit trail)           │
│   Immutable · every operation logged · replayable          │
│   tsvector FTS · connection pooling · SQLite fallback      │
└────────────────────────────────────────────────────────────┘
```

The architecture is layered by concern: the API layer handles routing and auth; the engram layer enriches raw content; the retrieval layer is determined by the selected deployment profile; an optional governance layer evaluates, scores, and suppresses retrieval candidates post-retrieval before the response is assembled; the compression layer reduces storage footprint; and the audit layer logs every mutation to PostgreSQL.

---

## 4. Core Components

### 4.1 Engram: The Universal Memory Unit

An **Engram** is the atomic unit of knowledge in MNEMOS. It wraps a raw document chunk with machine-generated enrichments that make retrieval smarter and operations auditable.

| Field | Type | Purpose |
|---|---|---|
| `id` | `string` | Unique identifier (UUID) |
| `content` | `string` | Raw text content |
| `embedding` | `vector` | Dense vector (quantised on disk) |
| `neuro_tags` | `string[]` | Auto-generated semantic labels |
| `source` | `string` | Provenance URI (where this data came from) |
| `confidence` | `float` | Quality signal (0.0–1.0) used for ranking |
| `created_at` | `datetime` | Ingestion timestamp |
| `metadata` | `object` | Extensible application-specific data |
| `edges` | `string[]` | IDs of related engrams (knowledge graph) |
| `_governance` | `GovernanceMeta` | Optional governance metadata (lifecycle state, trust/utility/freshness scores, conflict state, lineage); `null` for legacy engrams |

The Engram is **domain-agnostic** — the consuming application decides what `neuro_tags` mean, what `source` URIs look like, and what goes in `metadata`. MNEMOS provides the schema, storage, indexing, and retrieval.

### 4.2 Retrieval Backends

MNEMOS supports multiple retrieval backends, selected by deployment profile. All embedding inference is **GPU-accelerated** via CUDA.

| Backend | Profile | Embedding Model | Strength |
|---|---|---|---|
| **Qdrant** | Core Memory Appliance | BAAI/bge-base-en-v1.5 or Nomic v1.5 MRL (CUDA) | Fast semantic ANN, HNSW index, named-vector prefetch/rescore, payload filtering, horizontal scaling |
| **pgvector** | Governance Native | BAAI/bge-base-en-v1.5 (768-dim, CUDA) | ANN + SQL metadata filtering in one query, single-database deployment |
| **PostgreSQL FTS** | Hybrid mode (Core/Governance) | n/a (lexical lane) | Exact-term/title/acronym retrieval via full-text lexical matching |
| **Cross-Encoder** | Precision Lane (optional) | BAAI/bge-reranker-base | Dense reranking for long-context and technical text via stateless cross-encoder |

*(Note: ColBERT late-interaction has been moved to an experimental/research appendix and is no longer a standard deployment profile.)*

**Why Qdrant** (Core Memory Appliance): Standalone service with its own HNSW index, snapshotting, replication, and sharding. Supports concurrent reads and writes without single-process bottlenecks, payload-based filtering without post-filtering, and survives independently of the MNEMOS process.

**Why pgvector** (Governance Native): Vectors live inside the same PostgreSQL instance as the forensic ledger. ANN retrieval can be combined with SQL `WHERE` clauses on tenant, provenance, department, security markings, or any relational metadata — in a single query. This eliminates the need for a separate vector service in governance-heavy deployments.

**Matryoshka tiered search (Phase 7):** The Qdrant tier now supports Nomic `nomic-embed-text-v1.5` Matryoshka Representation Learning (MRL) with named vectors. The hot path performs a 64-dimensional coarse prefetch (`dense_64`) to reduce the candidate pool, then rescoring uses the full 768-dimensional vector (`dense_768`) for final fidelity. This keeps the retrieval economy visible to the budget router: coarse prefetch is cheap, full rescore is precise, and both stages are exposed to latency-budget degradation. Measured posture (June 11, 2026): the complexity classifier adds `0.0502ms` p95 after embedding reuse (`benchmarks/results/phase_8_complexity_accuracy.json`); at the current 2.1K-point evaluation corpus the summary-layer route runs at latency parity with flat search (p95 `30.4ms` vs `32.5ms`) while cutting the candidate pool by `99.5%` — the wall-clock latency gate is scale-aware and deferred to production-scale (≥100K point) benchmark runs (`tools/run_phase9_hierarchy_gate.py`). The Phase 7 migration replay flagged budget p95 `53.5ms` vs a `50.3ms` target on the small replay sample (REVIEW; see `docs/reports/mnemos_phase7_burn_in_report.md`).

**Hybrid fusion**: In Gate C hybrid mode, MNEMOS merges lexical (PostgreSQL FTS) and semantic candidates with deterministic normalization and weighted fusion. Four fusion policies are available:

| Policy | Lexical | Semantic | Engine |
|---|---|---|---|
| `semantic_dominant` | 0.25 | 0.75 | Python-side rank fusion |
| `balanced` | 0.50 | 0.50 | Python-side rank fusion |
| `lexical_dominant` | 0.75 | 0.25 | Python-side rank fusion |
| `qdrant_rrf` | — | — | Qdrant server-side Reciprocal Rank Fusion via `prefetch` (v1.17+) |

The `qdrant_rrf` policy delegates fusion to Qdrant's built-in RRF engine, combining a dense vector prefetch with a full-text payload prefetch in a single `query_points()` call — eliminating the second network round-trip. A full-text index on the `content` payload field is created automatically during tier initialization. If unavailable, the router falls back to Python-side fusion transparently. Optional explain output returns component scores and source attribution per hit.

**Relevance feedback**: MNEMOS can feed governance `reflect_path` labels (Used / Ignored) back into retrieval via Qdrant's `discover_points()` API. When enabled, previously-used engrams become positive exemplars and previously-ignored engrams become negative exemplars, biasing future queries toward results that have demonstrated utility. The adapter maintains a TTL-bounded exemplar cache (1.6M writes/sec, 100% cache hit rate after warmup). This feature is opt-in (`relevance_feedback.enabled: true` in `rerank_policy.yaml`).

**Graph Tier (experimental, read-only):**

The Graph Tier augments semantic retrieval with curated knowledge-graph neighbors sourced from engram `edges` and Qdrant payload metadata. Evaluation track MG-Test-1 through MG-Test-10 validated shadow telemetry, hub-penalty filtering, candidate-envelope isolation, double opt-in controls, operator shadow trials, and live batched `QdrantEngramResolver` retrieval (p95 ≈ 6.8 ms under live load, zero governance leaks, zero write-path mutations).

| Property | Value |
|---|---|
| Mode name | `graph_hybrid_experimental` (locked; not `graph_hybrid`) |
| HTTP API exposure | **Not** in public `retrieval_mode` enum (`semantic`, `hybrid` only); router-internal / service-config path |
| Global opt-in | `enable_experimental_graph_hybrid=True` on `RetrievalRouter` |
| Request opt-in | `retrieval_mode="graph_hybrid_experimental"` at router level |
| Posture | Experimental, isolated, read-only — no edge generation, no persistence mutations |
| Operator guide | `docs/graph_tier/operator_guide.md` |
| Closeout evidence | `docs/mg_test_10_experimental_closeout.md` |

Rollback is flag-based: disable the global flag or omit the experimental mode; the router falls back to `semantic` or `hybrid` with no data cleanup.

**Adaptive routing (Phase 8):**

MNEMOS no longer treats every query as the same retrieval problem. The `embedded-linear-softmax` complexity classifier runs over the active query embedding space and classifies each query into one of three route postures:

| Class | Query shape | Route posture |
|---|---|---|
| `CLASS_A` | Factoid, direct lookup, acronym, single-policy question | Flat semantic retrieval with aggressive budget posture |
| `CLASS_B` | Relationship, conflict, lineage, multi-hop question | Graph-capable or balanced route with forced reranking posture |
| `CLASS_C` | Global synthesis, thematic summary, corpus-level comparison | Hierarchical summary layer route with fallback |

The Phase 8 hold-out gate reached `1.0000` accuracy across 15 held-out queries (`5/5` per class), with p95 classifier latency of `0.0502ms` after embedding reuse. Evidence is recorded in `benchmarks/results/phase_8_complexity_accuracy.json`.

**Budget-aware retrieval (Phase 7):** Callers may pass `latency_budget_ms` on `POST /v1/mnemos/search`. The `BudgetAwareRouter` maintains an EWMA stage-cost model (embed, prefetch, rescore, rerank) and sheds stages down a fixed degradation ladder when the predicted cost exceeds the budget: drop rerank → reduce MRL oversample (3.0 → 1.5) → reduce HNSW `ef` (128 → 64) → drop rescore. Retrieval never degrades below coarse prefetch; responses that cannot meet the budget are flagged `budget_infeasible` so the consumer sees the precision-for-latency trade explicitly. Phase 8 complexity classes select the starting posture (CLASS_A aggressive shedding, CLASS_B forced rerank).

**Reserved retrieval sentinels:** Server-injected filter keys (`__exclude_derived__`, `__exclude_summaries__`, `__mrl_oversample__`, `__hnsw_ef__`, `__prefetch_only__`) are consumed inside the vector tiers and rejected with HTTP 400 if a client supplies them — isolation and budget controls cannot be spoofed or disabled from outside the service.

### 4.3 TurboQuant Compression

All stored embeddings are compressed via **TurboQuant** (arXiv:2504.19874), a near-optimal vector quantisation algorithm.

> [!NOTE]
> **Dimensionality Disclaimer**: The compression tables and examples below anchor on 128-dim and 384-dim arrays (legacy standard). While BGE operates at 768 dimensions, the fundamental 8×/4-bit compression ratios and fidelity bounds remain empirically valid across architectures.

**How it works:**
1. Input vectors are randomly rotated so each coordinate follows a Beta distribution
2. Optimal Lloyd-Max scalar quantisers compress each coordinate to 1–4 bits
3. For inner-product operations, a 1-bit QJL residual preserves unbiased estimation

**Performance at 4-bit (default):**

| Metric | Value |
|---|---|
| Storage compression | 8× raw bytes, 8.3–8.4× file (.npz) |
| Recall@10 | 84.2% (128-dim), 84.8% (384-dim) |
| MSE | 7.3×10⁻⁵ (128-dim), 2.5×10⁻⁵ (384-dim) — well below 0.009 bound |
| Cosine fidelity | 0.995 avg (practically indistinguishable from float32) |
| Encode throughput | 67K–175K docs/sec (CPU, NumPy) |

> *All values measured empirically on a 10,000-document synthetic corpus. Benchmark source: `benchmarks/run_benchmarks.py`. Results: `benchmarks/results.json`.*

#### Compression & Fidelity Across Bit-Widths

| Bits | MSE (128d) | MSE (384d) | Cosine Sim | Raw Ratio | File Ratio |
|---|---|---|---|---|---|
| 1-bit | 0.0028 | 0.0009 | 0.799 | 32× | 31× |
| 2-bit | 0.0009 | 0.0003 | 0.940 | 16× | 16× |
| 3-bit | 0.0003 | 0.00009 | 0.983 | 8× | 11× |
| **4-bit** | **0.00007** | **0.00003** | **0.995** | **8×** | **8.3×** |

#### Recall@10 (Nearest-Neighbour Fidelity)

Measured on 10K corpus / 100 queries — fraction of true float32 top-10 neighbours preserved after quantisation:

| Bits | Recall@10 (128d) | Recall@10 (384d) |
|---|---|---|
| 1-bit | 22.8% | 23.5% |
| 2-bit | 50.3% | 53.0% |
| 3-bit | 72.8% | 73.0% |
| **4-bit** | **84.2%** | **84.8%** |

#### Encoding Throughput (4-bit, CPU)

| Batch Size | 128-dim | 384-dim |
|---|---|---|
| 100 docs | 3,842 docs/s (0.26 ms/doc) | 1,909 docs/s (0.52 ms/doc) |
| 1,000 docs | 33,589 docs/s (0.03 ms/doc) | 15,810 docs/s (0.06 ms/doc) |
| 10,000 docs | 175,685 docs/s (0.006 ms/doc) | 67,195 docs/s (0.015 ms/doc) |

#### Storage at Scale (4-bit, 128-dim)

| Corpus Size | float32 | TurboQuant 4-bit | Ratio |
|---|---|---|---|
| 10K documents | 5.1 MB | 0.7 MB | 7.5× |
| 100K documents | 51.2 MB | 6.8 MB | 7.5× |
| 1M documents | 512.0 MB | 68.0 MB | 7.5× |
| 10M documents | 5,120 MB | 680 MB | 7.5× |

**Why it matters**: Without compression, a 1M-document index at 128 dimensions consumes ~512 MB in float32. With 4-bit TurboQuant, that drops to ~68 MB — enabling deployment on memory-constrained edge devices, smaller cloud instances, and faster cold starts.

#### Cross-domain Validation

The TurboQuant algorithm has been independently validated for **LLM KV cache compression** by the llama.cpp community ([turboquant_plus](https://github.com/TheTom/turboquant_plus), 6.7K+ stars, 30+ testers across Apple Silicon, NVIDIA, and AMD hardware). Key findings that reinforce MNEMOS's algorithm choice:

1. **Rotation Gaussianization confirmed on real model tensors** — Walsh-Hadamard rotation reduces raw KV tensor kurtosis from 900 to 2.9 (Gaussian = 3.0), validating the theoretical foundation used by MNEMOS for embedding rotation.
2. **Asymmetric sensitivity** — Value tensors (weighted sums) are safely compressible to 2-bit with zero quality loss, while Key tensors (softmax routing) require higher precision. This finding is specific to attention KV pairs and does not apply to MNEMOS's embedding use case, where all dimensions are treated uniformly.
3. **Cross-architecture stability** — 4-bit TurboQuant achieves 3.8× KV cache compression with near-q8\_0 quality, validated end-to-end from 1.5B to 104B parameter models at up to 128K context length.

> [!NOTE]
> MNEMOS uses TurboQuant exclusively for **embedding vector compression** (application-side, before storage). The KV cache application operates at a different level of the inference stack and is implemented in C/Metal/CUDA kernels within llama.cpp. The two applications share the same mathematical foundation but have independent implementations.


### 4.4 Profile Retrieval Benchmarks

This section summarizes current measured profile behavior from the reproducible benchmark suite.
Detailed run history, commands, and raw artifacts are maintained in `docs/benchmark.md`.

#### Retrieval (Real Corpus, 79 PDFs / 5,967 engrams)

Reference run: `20260329_123129_profile_benchmarks.json`

| Backend | Ingest Throughput | Search p50 (semantic) | Search p50 (light filter) | Search p50 (heavy filter) |
|---|---:|---:|---:|---:|
| Core Memory Appliance (Qdrant) | 399.6 docs/s | 31.1 ms | 31.1 ms | 31.1 ms |
| Governance Native (pgvector) | 20.1 docs/s | 55.7 ms | 56.5 ms | 56.5 ms |

Observed on this workload:
- Core (Qdrant) is materially faster on ingest and query latency.
- Relevance metrics between Core and Governance are near-parity in multiple filtered regimes.
- Current evidence supports Core as default for performance-sensitive workloads.

#### Governance-Focused Constraint Testing

Reference run: `20260329_120756_profile_benchmarks.json` (adversarial filter pressure)

| Regime | Qdrant Compliance@10 | pgvector Compliance@10 |
|---|---:|---:|
| light_filter | 0.1190 | 0.1190 |
| heavy_filter | 0.1649 | 0.1646 |

Observed on this workload:
- Constraint-correctness metrics are active and measurable.
- No meaningful backend separation has been demonstrated yet in compliance/violation behavior.
- Governance Native remains best framed as governance/operational posture (single-system SQL deployment), not a proven relevance-quality winner under current benchmark design.

#### Cross-Encoder Rerank (Track 2)

*Status: Implemented as a Conditional Reranking Policy. Replaces legacy ColBERT late-interaction path.*

Reference rerun: `20260422_124854_profile_benchmarks.json` (BAAI/bge-reranker-base run)

Observed on this synthetic workload:
- **Baseline Qdrant:** MRR=0.5134, nDCG=0.2105, p50=30.5ms
- **Cross-Encoder limits @50:** MRR=0.3566 (Δ-0.15), nDCG=0.2114 (Δ+0.00), p50=45.5ms (Δ+15ms)

**Production Posture:**
- While the Cross-Encoder pipeline is significantly more stable operationally than previous late-interaction policies, synthetic zero-shot reranking still shows baseline semantic dominance (negative MRR uplift).
- The system is currently in a **shadow-ready conditional state**: Dense-only remains the safe default path.
- **Conditional Rerank Policy Enforcement:** Code evaluates hard and soft skip reasons before invoking the reranker. Reranking is explicitly gated behind heuristics (scaffolded hybrid zero-shot classifiers returning a low confidence "unknown" to guarantee baseline performance).
- **Safety Gates & Telemetry:** Real-time `.health()` probes attached to the reranker, timeout circuit-breakers, and durable operational JSONL telemetry sinks (`logs/retrieval_telemetry.jsonl`) collect trigger rates and skip-reason distributions safely before turning reranking fully on.

#### Hybrid Retrieval (Gate C, Real Corpus)

Reference decision run: `20260329_225832_profile_benchmarks.json`  
Decision report: `20260329_225907_gate_c_decision.md`

Observed on this workload:
- Track execution complete: `True`
- Quality class win found: `False`
- Latency threshold satisfied: `True`
- Sprint exit pass: `False`

Interpretation:
- Hybrid retrieval is implemented, benchmarkable, and operationally viable.
- Hybrid did not demonstrate a differentiated quality-class win on this real-corpus benchmark.
- Semantic-only remains the production default at this time.
- Hybrid remains available as an evaluation mode for targeted enterprise query classes.

**Phase 2 update (v2):** The addition of `qdrant_rrf` provides a second hybrid evaluation path. Server-side RRF eliminates the Python-side fusion overhead and the second network round-trip, making hybrid search viable in latency-sensitive deployments where the original Gate C latency budget was marginal. The relevance feedback adapter (`discover_points()`) adds a further signal dimension not present in the original Gate C evaluation. Both features are subject to future Gate C re-evaluation on updated workloads.

### 4.5 Forensic Ledger (PostgreSQL)

Every operation that touches stored memory is immutably logged to **PostgreSQL** via a connection-pooled `psycopg3` driver:

| Field | Purpose |
|---|---|
| `timestamp` | When the operation occurred (TIMESTAMPTZ, server-side) |
| `component` | Which service component performed it |
| `action` | What happened (index, search, delete, update) |
| `raw_data` | Human-readable description |
| `status` | `success`, `failure`, `warning` |
| `latency` | Operation duration in seconds |
| `metadata` | JSONB structured details (IDs affected, query text, result count) |
| `search_vector` | Auto-generated tsvector for full-text search (GIN-indexed) |

**Why PostgreSQL**: SQLite's single-writer lock becomes a bottleneck under concurrent agent workloads. PostgreSQL provides connection pooling, ACID transactions, concurrent writes, and native full-text search via `tsvector` + `GIN` indexes — replacing FTS5 with a language-aware, ranked search engine. A SQLite fallback remains available for local development and testing.

**Use cases:**
- **Compliance** — demonstrate when data was ingested, accessed, or deleted
- **Debugging** — replay the sequence of operations that led to a retrieval failure
- **Analytics** — track query patterns, ingestion rates, and error trends

### 4.6 Governance Layer (MemArchitect)

The **Governance Layer** is an in-process post-retrieval pipeline that evaluates, scores, and optionally suppresses candidates before they are returned to the caller. It operates on the `GovernanceMeta` attached to each Engram and produces a `GovernanceDecision` per candidate alongside optional `ContradictionRecord` objects.

**Three governance modes:**

| Mode | Behavior |
|---|---|
| `off` | No governance; results returned unchanged (default) |
| `advisory` | All candidates evaluated; none suppressed; results re-ranked by `governed_score`; decisions included in response for inspection |
| `enforced` | Suppressed candidates removed; survivors re-ranked by `governed_score` and trimmed to `top_k` |

**Score formula:**

```
governed_score = retrieval_score
              × trust_modifier
              × utility_modifier
              × freshness_modifier
              × contradiction_modifier
              × veto_modifier
```

**Policy pipeline (per-candidate):**

| Policy | What it does |
|---|---|
| `RelevanceVetoPolicy` | Hard veto for score-below-threshold, deleted (`soft_deleted`/`tombstone`), or `toxic`-flagged candidates; exponential freshness decay on `freshness_modifier` |
| `UtilityPolicy` | Maps `trust_score` and `utility_score` from `GovernanceMeta` to `[0.75, 1.25]`-range modifiers |

**Cross-candidate contradiction detection (Wave 2):**

Candidates that carry `entity_key`, `attribute_key`, and `normalized_value` in their `GovernanceMeta` are grouped by `(entity_key, attribute_key)`. Groups with two or more distinct normalized values are contradiction clusters. A winner is selected deterministically by:

1. `trust_score` (higher wins)
2. `created_at` (newer wins)
3. `utility_score` (higher wins)
4. `source_authority` (higher wins)
5. `engram.id` (lexicographically lower — always resolves ties)

The winner receives `contradiction_modifier = 1.0`; losers receive `0.25`. In enforced mode, losers are removed from the result set.

**Consensus Resolution Engrams (Phase 10):** Offline contradiction clusters can now be passed to `ReconciliationRunner`, which synthesizes an additive Resolution Engram rather than mutating or deleting parent memories. Resolution Engrams carry `metadata.is_resolution_engram = true`, `source = derived://reconciliation/<entity_key>`, and `edges` pointing to every conflicting parent. When a Resolution Engram appears in the same entity-slot cluster as its parents, `ContradictionPolicy` gives it Tier-1 priority with `contradiction_modifier = 1.25` and suppresses the parent candidates through the existing contradiction path. The live Phase 10 gate validated 5/5 collisions: Resolution Engram rank #1, parent suppression, and visible 1.25 modifier audit (`benchmarks/results/phase_10_consensus_gate.json`). Supporting this, the Qdrant tier now persists `GovernanceMeta` as `gov_`-prefixed payload fields and rehydrates it on retrieval, so entity/attribute slot keys survive the index round-trip and the read path can group resolution engrams with their parents. Resolution engrams are intentionally not caught by `__exclude_derived__` (they set `is_resolution_engram`, not `is_derived_fact`): they must be co-retrievable with their parents to take read-path priority.

**Configuration (environment variables):**

| Variable | Default | Description |
|---|---|---|
| `MNEMOS_GOVERNANCE_MODE` | `off` | Default mode for all search requests |
| `MNEMOS_GOVERNANCE_MIN_SCORE` | `0.0` | Veto threshold (0.0 disables score-floor veto) |
| `MNEMOS_GOVERNANCE_FRESHNESS_HALF_LIFE` | `180.0` | Freshness decay half-life in days |

The governance mode can also be overridden per-request via the `governance` parameter on `POST /v1/mnemos/search`. Per-tenant tuning uses `governance_profile` (loaded from `MNEMOS_GOVERNANCE_POLICY_PROFILES_JSON`).

**Explainability (read path):**

| Flag | Effect |
|---|---|
| `explain: true` | Hybrid component scores, fusion policy, and retrieval sources per hit |
| `explain_governance: true` | Full `governance` modifier breakdown per result, plus `governance_trace` (outcome, reason, rank shift, top score factors) and `meta.governance_explain.suppressed_candidates` summary |

**Counterfactual explainability (Phase 6):** governance traces include deterministic, arithmetic counterfactuals for the top results — the exact score distance to rank 1 and which modifier would close it (e.g. the `trust_score` delta required to tie, or the freshness modifier inverted into a human-readable age limit). No ML in the loop: counterfactuals are derived purely from the modifier product, so they are reproducible and auditable.

**Reflect path (Wave 3):**

After retrieval and governance evaluation, the reflect path closes the feedback loop. When the calling application sends back the generated answer alongside the candidate set (`POST /v1/mnemos/governance/reflect`), the `UsageDetector` assigns each memory a usage label:

| Label | Signal |
|---|---|
| `USED` | Present in `cited_ids`, or semantically entailed by the generated answer under the NLI critic; lexical overlap remains a guarded fallback |
| `IGNORED` | No overlap signal and not cited |
| `CONTRADICTED` | Was a contradiction loser in the read-path decision |
| `VETOED` | Failed a policy veto in the read path |

**Reflect precision guards** (default profile; tunable per tenant): memories with fewer than `min_memory_tokens_for_overlap` tokens (default 3) skip overlap classification; overlap classification additionally requires at least `min_overlap_tokens` shared tokens (default 2) before the recall-oriented threshold is evaluated.

The Phase 7 reflect path adds a DeBERTa-v3 NLI critic for semantic usage detection. This resolves the prior lexical false-positive boundary by requiring entailment rather than word overlap when the critic is available; adversarial validation improved USED precision from `0.57` to `1.00`. Selection is per-tenant via the `reflect_precision_mode` profile field (`lexical` | `nli`); NLI model-load failure falls back to the lexical detector with the failure cached, so a broken critic cannot stall the reflect path.

Reinforcement is then applied in-place to each memory's `GovernanceMeta`:

| Label | `utility_score` | `trust_score` | `stability` |
|---|---|---|---|
| `USED` | +0.05 | +0.02 | +0.02 |
| `IGNORED` | −0.01 | — | — |
| `CONTRADICTED` | −0.03 | −0.02 | — |
| `VETOED` / `UNKNOWN` | — | — | — |

All deltas are clamped to [0.0, 1.0]. The response includes `utility_deltas` and `trust_deltas` per memory for caller inspection. Persistence of governance score updates is caller-owned; the reflect endpoint is stateless with respect to backend score mutation. This is separate from derived-view caching in Memory Over Maps, which stores reproducible read artifacts and invalidates them deterministically without persisting governance mutations.

**Validation Evidence:**

The governance behavioral claims are backed by the maintained governance evidence suite (`tests/test_governance*.py`, `tests/test_hygiene_*.py`, and benchmark result artifacts). The suite consists of named scenarios, each mapping a specific failure mode to a deterministic, in-process test. The following guarantees are proven, not asserted:

| Guarantee | Failure mode addressed |
|---|---|
| Reinforcement converges — used memories strengthen toward ceiling, not forever | Runaway score accumulation without a floor |
| Ignored memories weaken over repeated cycles | Stale utility retention |
| Contradiction winners and losers separate in utility over time | Contradiction resolution without reinforcement divergence |
| Stale memories decay via ignore penalties before any backend hygiene run | Obsolete memory retaining pre-decay scores indefinitely |
| Sub-3-character tokens produce no overlap signal — no false positives from zero-content memories | Short generic content matching every answer |
| Contradiction state outranks lexical overlap — a loser stays `CONTRADICTED` even when it shares words with the winning answer | Contradiction loser accruing positive reinforcement through phrasing coincidence |
| Overlap threshold is a documented precision/recall dial, not a hidden heuristic | Unknown precision behavior at deployment time |

**Known precision boundaries (documented, not hidden):**

- Two-token generic memories achieve 100% word overlap with any answer containing both tokens. At the default 0.15 threshold this is a classification false positive. Mitigations: raise the threshold above 0.50, or enforce a minimum content token count at write time.
- The overlap detector is purely lexical. Proper-noun or entity-name overlap fires regardless of topical relevance. Semantic re-ranking is the long-term mitigation path.

**Wave 4 — Hygiene path (background memory health):**

Three runners, chained by `HygienePipeline`, handle long-horizon memory health between query cycles:

| Runner | What it does |
|---|---|
| `DecayRunner` | Linear utility decay past inactivity horizon (default 60 days). Sets `lifecycle_state = "stale"` when `utility_score < 0.20`. `last_used_at` takes priority over `created_at`. Floor at 0.0. |
| `PrunePromoter` | Composite score floor: `utility × trust × contradiction_factor < 0.05` → `lifecycle_state = "prune_candidate"`. Stale memories always promoted. |
| `ContradictionSweepRunner` | Offline entity-slot contradiction detection over the full corpus. Catches contradictions between memories never co-retrieved in the same query context. Reuses `ContradictionPolicy` resolution logic. |
| `ReconciliationRunner` | Synthesizes Resolution Engrams from contradiction sweep output. In dry-run mode it previews consensus artifacts; in action mode it indexes additive resolution memories with parent lineage preserved. |

All runners support `dry_run=True` (compute report, mutate nothing) and `apply` mode. Lifecycle hygiene mutates governance state in memory with artifact emission; reconciliation can additionally persist additive Resolution Engrams through an explicit indexer. No physical deletion. No irreversible consolidation. `Governor.run_hygiene()` is the single entry point; hygiene counters are reported via `GET /v1/mnemos/governance/stats`. CI enforces a dry-run hygiene gate via `tools/run_wave4_hygiene.py --fail-on-gate`.

**Per-tenant policy profiles:**

`GovernancePolicyProfile` allows per-tenant tuning of read-path thresholds, reflect-path precision, and all reinforcement deltas without restarting the service. Profiles are loaded from `MNEMOS_GOVERNANCE_POLICY_PROFILES_JSON`. The `"default"` profile always exists; additional profiles are selected per-request via `governance_profile` on `POST /v1/mnemos/search` and `POST /v1/mnemos/governance/reflect`.

**Validated gaps (tested in Wave 4):**

Long-horizon calendar-based decay, offline contradiction sweep coverage, and stale-state lifecycle transitions were documented as untested in Validation Pack v1 and are now covered by 61 Wave 4 tests. Remaining open gaps: enforced-mode drift divergence, trust recovery after contradiction penalty, and concurrent reflect cycle safety — scheduled for Phase 2 (persistence) and beyond.

---

### 4.7 Memory Over Maps (Phase-Gated)

Memory Over Maps is now an implemented architecture lane in MNEMOS and has been advanced only through benchmark-gated phase exits.

| Phase | Capability | Gate Status (March 30, 2026) |
|---|---|---|
| 1 | Source-grounded lineage contract + audit hooks | PASS |
| 2 | Deterministic bounded candidate envelope | PASS |
| 3 | On-demand derived views (evidence, contradiction, preference, timeline) | PASS |
| 4 | Derived-view cache + invalidation + dry-run traces | PASS |
| 5 | Bounded semantic reflect evolution scenarios | PASS |

Phase 8-10 extend this lane from on-demand views into active routing, hierarchy, and consensus:

| Phase | Capability | Gate Status (June 11, 2026) |
|---|---|---|
| 8 | Embedded-reflex complexity classifier for CLASS_A/B/C adaptive routing | PASS: 1.0 hold-out accuracy |
| 9b | RAPTOR-lite summary hierarchy with summary isolation sentinel | PASS: mean similarity 0.7342 and zero live factoid summary leaks |
| 10 | Additive Resolution Engrams for contradiction reconciliation | PASS: 5/5 live consensus gate |

Current artifact family:
- `benchmarks/outputs/raw/memory_over_maps_<timestamp>_raw.json`
- `benchmarks/outputs/summaries/memory_over_maps_<timestamp>_report.md`
- `benchmarks/outputs/summaries/memory_over_maps_<timestamp>_decision.md`

Latest phase artifacts (March 30, 2026):
- Phase 1: `benchmarks/outputs/raw/memory_over_maps_20260330_135417_raw.json`, `benchmarks/outputs/summaries/memory_over_maps_20260330_135417_report.md`, `benchmarks/outputs/summaries/memory_over_maps_20260330_135417_decision.md`
- Phase 2: `benchmarks/outputs/raw/memory_over_maps_20260330_150121_raw.json`, `benchmarks/outputs/summaries/memory_over_maps_20260330_150121_report.md`, `benchmarks/outputs/summaries/memory_over_maps_20260330_150121_decision.md`
- Phase 3: `benchmarks/outputs/raw/memory_over_maps_20260330_150942_raw.json`, `benchmarks/outputs/summaries/memory_over_maps_20260330_150942_report.md`, `benchmarks/outputs/summaries/memory_over_maps_20260330_150942_decision.md`
- Phase 4: `benchmarks/outputs/raw/memory_over_maps_20260330_151515_raw.json`, `benchmarks/outputs/summaries/memory_over_maps_20260330_151515_report.md`, `benchmarks/outputs/summaries/memory_over_maps_20260330_151515_decision.md`
- Phase 5: `benchmarks/outputs/raw/memory_over_maps_20260330_151822_raw.json`, `benchmarks/outputs/summaries/memory_over_maps_20260330_151822_report.md`, `benchmarks/outputs/summaries/memory_over_maps_20260330_151822_decision.md`

Interpretation:
- Source truth is explicit and traceable.
- Expensive reasoning is bounded before governance/synthesis work.
- Derived views are reproducible and input-declared.
- Cache correctness is validated with explicit invalidation evidence.
- Summary and Resolution Engrams are synthetic, but never anonymous: every synthetic node carries metadata and parent `edges` for forensic review.
- The Phase 9b `__exclude_summaries__` sentinel makes hierarchy operationally safe by excluding summary engrams from default factoid retrieval while keeping them reachable through explicit CLASS_C summary-layer routes.

Phase gates remain enforced in CI (`tests/test_memory_over_maps_benchmark_runner.py`). Post–March-30 operator and pilot evidence (PIT, DFE) is cited in §4.8 and companion reports under `docs/reports/`.

### 4.8 Derived Facts Lane (Production-Adjacent Pilot)

The Derived Facts lane is a **separate, isolated evaluation track** (PIT-0 through PIT-10) for operator-facing shadow packets. It is **not** part of default retrieval and must not leak into production prompts.

**Status (June 2026):** `PIT_10_PRODUCTION_ADJACENT_EVALUATION_LANE_READY_FOR_LIMITED_PILOT` — see `docs/reports/pit_10_closeout_and_pilot_readiness.md`.

**What is proven:**

- Default retrieval (`POST /v1/mnemos/search`, `POST /api/v1/query`) returns zero derived facts; a runtime `SEV-STOP` guard fires on any leak.
- Shadow endpoint `/api/v1/evaluate_derived_shadow` requires kill-switch enablement, client whitelist (`X-Client-Id`), and double opt-in JSON flags (`evaluation_mode=true`, `include_derived_facts=true`).
- Derived facts render with `[MNEMOS-DERIVED]` authority labels, source traceability, and bounded shadow packet limits (`PIT3_MAX_DERIVED_FACTS_PER_SHADOW_PACKET`, `PIT3_MAX_DERIVED_FACT_TOKENS`).
- Controlled operator trials reported high usefulness for shadow packets; p50/p95 shadow generation stabilizes at ~2–4 ms after cross-encoder warm-up.

**Required configuration:**

| Variable | Default | Purpose |
|---|---|---|
| `MNEMOS_DERIVED_ENABLED` | `false` | Global kill-switch; `false` → HTTP 503 on shadow lane |
| `MNEMOS_DERIVED_WHITELIST` | `["eval_dashboard", "governance_auditor"]` | Allowed `X-Client-Id` values |

**Telemetry (tracked in `GET /v1/mnemos/stats` → `derived_lane`):**

`query.default_retrieval.derived_fact_count`, `derived_lane.execution_count`, `derived_lane.denied_count`, `derived_lane.kill_switch_count`, `evaluate_derived_shadow.request_count`, `evaluate_derived_shadow.rendered_derived_fact_count`

**Prohibited (SEV-STOP protected):**

- Derived facts in default retrieval or production EchoFrame prompts outside explicit evaluation mode
- Candidate-envelope mixing with raw engram fusion on the default path
- Automatic promotion to governance ledgers or schema/fact extraction (remains blocked — see README project status)

Human operator value trials (DFE-12 through DFE-21) and selection/rescue tuning evidence live in `docs/reports/dfe_*` and `eval_results/dfe_*`.

### 4.9 Predictive Cognition (MNEMOS-Thinking)

MNEMOS leverages Google TimesFM as a time-series foundation model to move from reactive retrieval toward a self-aware cognitive engine. The integration uses a sidecar provider model: the main MNEMOS service records telemetry and makes governed decisions, while the `mnemos-timesfm` sidecar performs forecast work so predictive overhead does not compromise retrieval SLOs.

**Pulse layer (`GET /v1/mnemos/pulse`)**

The service maintains an in-memory circular buffer of 1,440 one-minute `PulsePatch` records. Each patch normalizes operational telemetry into forecastable sequences: query volume, p95 latency, cache hit rate, degradation count, and candidate-envelope pressure. Production deployments pin local TimesFM artifacts and can disable forecasting with `MNEMOS_TIMESFM_ENABLED=false`.

**Track 1: Predictive pre-warming**

The Pulse Engine watches query-class and latency trends. If a high-confidence forecast predicts a material volume spike or p95 rise inside the 15-minute horizon, MNEMOS can trigger autonomous pre-warm under `MNEMOS_PULSE_ACTIONS=autonomous`. Advisory mode remains the default. Autonomous warmup is cooldown-protected and records `forecast_reason` plus `confidence_score` in the forensic ledger.

**Track 2: Predictive hygiene and semantic volatility**

The Volatility Harvester tracks memory event density by engram family: index updates, contradiction events, and usage frequency. When volatility bias is enabled, the relevance policy can shorten the effective semantic half-life for high-velocity families. Forecasted contradiction spikes can also trigger proactive reconciliation so Resolution Engrams are produced before users encounter stale conflicts.

**Track 4: Intent trajectory and shadow search**

The Intent Engine maps per-session query sequences to semantic cluster IDs. When the trajectory becomes predictable, the ShadowSearchRunner performs a low-priority search for the forecasted cluster centroid and stores the result in the derived-view cache with `pre_cognitive: true`. If the user later asks the predicted query, `fuzzy_pre_cognitive_get()` can return the cached result immediately.

**Operational safety**

| Control | Behavior |
|---|---|
| `MNEMOS_TIMESFM_ENABLED=false` | Reverts forecast providers to the reactive baseline/fallback path |
| `MNEMOS_PULSE_ACTIONS=advisory` | Logs suggested actions without mutating routing or triggering warmup |
| `MNEMOS_PULSE_ACTIONS=off` | Disables pulse-driven action paths |
| Confidence threshold | Autonomous warmup requires high-confidence forecasts and cooldown enforcement |
| Forensic ledger | Forecast-driven actions include reason, confidence, target, and result metadata |

### 4.10 CoALA Cognitive Cycle (v3.2)

MNEMOS v3.2 introduces a **CoALA-aligned cognitive cycle layer** (`mnemos/cognitive/`) that makes MNEMOS' existing cognitive behaviours explicit, auditable, and interoperable. Prior to v3.2, routing decisions, attention gates, forecast advisory states, and governance outcomes were distributed across internal call sites with no unified schema for consuming systems to read. The cognitive cycle overlay captures all of this in a single per-request record without changing any retrieval or governance behaviour.

**Conceptual mapping (CoALA → MNEMOS)**

| CoALA concept | MNEMOS implementation |
|---|---|
| Working memory | `WorkingMemorySnapshot` — transient per-cycle state (query, candidate counts, active modes, gate flags) |
| Attention | `List[AttentionDecision]` — 11 named dimensions: retrieval mode, candidate envelope, summary inclusion, graph expansion, derived facts, governance gate, forecast advisory, shadow search, intent trajectory, query classification, cross-encoder rerank |
| Retrieval actions | `List[ActionRecord]` — semantic_search, python_hybrid_rrf, qdrant_rrf, candidate_envelope, cross_encoder_rerank, pre_cognitive_cache |
| Reasoning actions | `List[ActionRecord]` — contradiction_detection, derived views, shadow_search_intent |
| Learning writes | `List[LearningWrite]` — reflect_path reinforcement, volatility hygiene, episodic writes |
| Governance | `List[GovernanceEvalSummary]` — aggregate veto/suppression/contradiction counts |
| Forecasting | `List[ActionRecord]` + `ForecastOutcomeRecord` — pulse advisory, intent trajectory, volatility forecast |
| Audit | `List[str] forensic_ledger_refs` — cycle record cross-referenced to forensic ledger transactions |

**OperationType taxonomy**

Every action in a `CognitiveCycleRecord` carries one or more `OperationType` labels from the CoALA action space: `RETRIEVAL`, `REASONING`, `LEARNING`, `GROUNDING`, `GOVERNANCE`, `FORECASTING`, `AUDIT`. Multi-typed actions (e.g. `contradiction_detection` is both `REASONING` and `GOVERNANCE`) are expressed as lists.

**How to use**

Add `"cognitive_cycle": true` to any `/search` request body. The response will include a top-level `"cognitive_cycle"` key containing the full `CognitiveCycleRecord` dict. Recent cycle records are also available at `GET /v1/mnemos/cognitive/cycles?limit=N`.

```json
POST /v1/mnemos/search
{
  "query": "...",
  "cognitive_cycle": true
}
```

Response includes:
```json
{
  "results": [...],
  "meta": {...},
  "cognitive_cycle": {
    "cycle_id": "uuid",
    "trigger_type": "search",
    "working_memory_snapshot": { ... },
    "attention_decisions": [ { "dimension": "retrieval_mode", "decision": "hybrid:balanced", "reason": "..." }, ... ],
    "retrieval_actions": [ { "name": "python_hybrid_rrf", "operation_types": ["retrieval"], ... } ],
    "governance_evaluations": [ { "mode": "advisory", "vetoed": 2, "net_candidates_returned": 8, ... } ],
    "forecast_actions": [...],
    "forensic_ledger_refs": ["txn-..."],
    "selected_route": "return",
    "final_status": "completed",
    "cycle_latency_ms": 14.2
  }
}
```

**ForecastOutcomeRecord**

The `ForecastOutcomeRecord` class (`mnemos/cognitive/forecast_outcome.py`) adds the missing second half of MNEMOS forecasting: recording whether forecasts were accurate, useful, or harmful. It is created at prediction time via factory methods (`from_pulse_advisory`, `from_autonomous_warmup`, `from_proactive_reconciliation`, `from_intent_trajectory`) and resolved when the predicted condition is confirmed, refuted, or expires. The `future_policy_candidate` field is advisory only — procedural memory is never mutated automatically.

**Phase 15 operational validation**

The CoALA operational validation report (`docs/reports/coala_cycle_operational_validation.md`) seals the v3.2 transparency baseline. The deterministic harness (`tools/run_coala_cycle_validation.py`) validates 8 representative cognitive paths: CLASS_A direct lookup, CLASS_B multi-hop, CLASS_C global synthesis, contradiction/reconciliation, high-volatility governance, forecast-triggered pulse, pre-cognitive shadow search, and derived-view evidence bundle.

Validation gates passed:
- `attention_faithfulness`: attention decisions are evidence-derived from runtime, router, forecast, governance, policy, or config metadata.
- `bounded_record`: cycle telemetry remains bounded and caps `query_or_event` at 240 characters.
- `redaction`: cycle and forecast records reject secret, token, raw prompt, private reasoning, and raw engram content leakage.
- `adapter_compatibility`: records expose stable consumer-facing keys and operation-type labels.
- `forecast_resolution`: forecast-triggered cycles link to resolved `ForecastOutcomeRecord` lifecycle state.
- `learning_boundary`: learning writes declare explicit write classes, and PatternEngram candidates remain advisory.

Release evidence: `python -m pytest tests/test_coala_cycle_validation.py tests/test_learning_boundary.py tests/test_cognitive_cycle.py` passed 40 tests. The archived synthetic trace baseline is `benchmarks/results/coala_baseline_v3.2.json`.

**Safety invariants**

All existing safety boundaries are preserved:
- Default retrieval lane remains derived-fact-free (unchanged)
- Graph hybrid remains blocked on the HTTP surface (explicitly surfaced in attention contract as `graph_expansion: blocked`)
- Governance decisions are never mutated by the `CycleAssembler`
- Procedural memory is never written automatically (future_policy_candidate is advisory dict only)
- PatternEngram candidates can be `promotion_recommended`, but automatic promotion to authoritative `PatternEngram` is blocked until explicit governance approval (Phase 20 promotion gate — see §4.11)
- Learning writes distinguish `semantic_candidate_write`, `procedural_change_candidate`, `blocked_procedural_mutation`, and append-only `audit_write`
- All governed actions remain traceable through the forensic ledger
- Attention decisions must be evidence-derived, not explanation-generated
- The `CycleAssembler` is zero-cost when `cognitive_cycle` is not set in the request

**Module structure**

```
mnemos/cognitive/
  __init__.py          — public exports
  cycle.py             — CognitiveCycleRecord, WorkingMemorySnapshot, AttentionDecision,
                         ActionRecord, GovernanceEvalSummary, LearningWrite, OperationType,
                         OPERATION_TYPE_MAP
  attention.py         — build_attention_decisions() — pure function, 12 dimensions
                         (dimension 12: pattern_advisory — Phase 19)
  assembler.py         — CycleAssembler — incremental builder, zero cost when not used
  forecast_outcome.py  — ForecastOutcomeRecord — full forecast lifecycle
  learning_boundary.py — semantic/procedural write-class boundary
  pattern_engram.py    — advisory PatternEngramCandidate schema (Phase 17)
  cycle_evaluator.py   — CycleEvaluator, CycleEvaluationRecord (Phase 16)
  pattern_learner.py   — PatternLearner, PatternConsolidator, SituationAbstractor (Phase 17)
  pattern_store.py     — PatternCandidateStore — advisory accumulation + promotion (Phase 18/20)
  promoted_pattern.py  — PatternEngram — authoritative promoted pattern (Phase 20)
```

Test coverage: 77 cycle-layer tests across `test_cognitive_cycle.py` (29), `test_attention_contract.py` (31), and `test_forecast_outcome.py` (17), plus 40 focused Phase 15 validation tests covering CoALA gates, learning boundaries, and regression behavior. An additional 182 tests cover Phase 16–21 (pattern extraction harness) in `test_cycle_evaluator.py`, `test_pattern_learner.py`, `test_pattern_store.py`, `test_pattern_recall.py`, `test_promoted_pattern.py`, and `test_pattern_endpoints.py`.

### 4.11 PatternEngramCandidate Extraction Harness (v3.3)

MNEMOS v3.3 extends the CoALA cognitive cycle overlay with **experience abstraction** — the third stage in the "From Storage to Experience" taxonomy. Phases 16–21 implement an ExpeL + R²-Mem-inspired pipeline that harvests `CognitiveCycleRecord` history into governed, advisory `PatternEngramCandidate` objects and, after explicit governance approval, into authoritative `PatternEngram` objects.

**Paper basis**

| Priority | Paper | Role in MNEMOS |
|---|---|---|
| 1 | **ExpeL** — LLM Agents Are Experiential Learners | Cross-cycle IF-THEN insight extraction via `PatternLearner` |
| 2 | **R²-Mem** — Reflective Experience for Memory Search | Six-dimension rubric scorer (`CycleEvaluator`); quality thresholds KLOW=7, KHIGH=13 |
| 3 | **A-MEM** — Agentic Memory for LLM Agents | Jaccard-based situation deduplication and contradiction linking (`PatternConsolidator`) |
| 4 | **Governing Evolving Memory** | Safety counterweight: advisory-only boundary, blocked *_mutation types, no autonomous promotion |

**Pipeline**

```
CognitiveCycleRecords (per-request history)
        ↓
CycleEvaluator (Phase 16)
  R²-Mem rubric: 6 dimensions × 0–3 → aggregate 0–18
  quality_label: good (≥13) | neutral (8–12) | bad (≤7)
        ↓  (good/bad only — neutral filtered out)
SituationAbstractor (Phase 17)
  entity-free SituationSummary from structured cycle fields — no LLM calls
        ↓
PatternLearner (Phase 17)
  IF-THEN pattern_summary  →  PatternEngramCandidate
  types: descriptive | operational_recommendation (never *_mutation)
        ↓
PatternConsolidator (Phase 17)
  Jaccard similarity ≥ 0.85 → merge (same quality) or contradict (opposite quality)
        ↓
PatternCandidateStore (Phase 18)
  advisory-only accumulation; offline runner: tools/run_pattern_accumulation.py
        ↓
Advisory recall (Phase 19)
  cognitive_cycle=true → advisory_patterns field in CognitiveCycleRecord
  12th AttentionContract dimension: pattern_advisory
        ↓  (never automatic)
Promotion governance gate (Phase 20)
  POST /v1/mnemos/pattern-candidates/{id}/recommend  (requires gate_id)
  POST /v1/mnemos/pattern-candidates/{id}/approve    (requires governance_review_id)
        ↓
PatternEngram (Phase 20)
  authoritative advisory fact — SEMANTIC_WRITE only, never inserted into retrieval index
```

**CycleEvaluator rubric dimensions**

| Dimension | Max | What it measures |
|---|---|---|
| `routing_precision` | 3 | Query-class confidence and cycle completion |
| `candidate_efficiency` | 3 | Pre/post governance candidate pool ratio |
| `governance_appropriate` | 3 | Governance mode matched to query risk |
| `forecast_utilisation` | 3 | Forecast signals acted upon |
| `attention_coverage` | 3 | Attention dimensions populated (≥9→3, ≥7→2, ≥4→1) |
| `write_integrity` | 3 | Learning writes carry valid `write_class` |

**PatternCandidateStore**

The store (`mnemos/cognitive/pattern_store.py`) holds candidates separately from the main engram index. It is configured via `MNEMOS_PATTERN_STORE_PATH` and wired into `MnemosRuntime` at startup. Lifecycle:

- `add(candidate)` → status `candidate`
- `recommend(id, gate_id=…)` → status `promotion_recommended`
- `promote(id, governance_review_id=…)` → status `approved`, produces `PatternEngram`
- `reject(id)` → status `rejected`

**Advisory recall**

When `cognitive_cycle: true` and a `PatternCandidateStore` is configured, the runtime calls `store.find_relevant(situation_text, top_k=3)` and injects results into the `CognitiveCycleRecord` as `advisory_patterns`. These are informational only — they never alter retrieval candidates or ranking.

**HTTP endpoints (Phase 20)**

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/mnemos/pattern-candidates` | List candidates (optional `?status=` filter) |
| `GET` | `/v1/mnemos/pattern-candidates/{id}` | Get single candidate |
| `POST` | `/v1/mnemos/pattern-candidates/{id}/recommend` | Mark as promotion_recommended (requires `gate_id`) |
| `POST` | `/v1/mnemos/pattern-candidates/{id}/approve` | Promote to PatternEngram (requires `governance_review_id`) |
| `POST` | `/v1/mnemos/pattern-candidates/{id}/reject` | Mark as rejected |
| `GET` | `/v1/mnemos/pattern-candidates/promoted` | List all promoted PatternEngrams |

**Safety invariants (all preserved from v3.2)**

- `PatternEngramCandidate.authoritative_for_retrieval` — always `False`; enforced at class level
- `PatternEngramCandidate.affects_ranking` — always `False`
- `PatternEngramCandidate.mutates_policy` — always `False`
- `PatternLearner` never produces `policy_mutation`, `routing_mutation`, or `template_mutation` types
- `PatternEngram.from_approved_candidate()` raises `PermissionError` if promotion status is not `PROMOTION_APPROVED`
- `PatternEngram.write_class` — always `SEMANTIC_WRITE`; approved patterns are semantic facts, not procedural mutations
- No LLM calls anywhere in the extraction pipeline

**Phase 21 gate evidence**

The deterministic harness (`tools/run_pattern_phase_gate.py`) validates 8 scenarios and 5 cross-cutting gate assertions:

Scenarios: `eval_good_class_a`, `eval_bad_class_b`, `learner_descriptive`, `learner_operational`, `consolidator_merge`, `consolidator_contradict`, `recall_advisory`, `promotion_gate`

Gate assertions: `evaluator_determinism`, `safety_invariant`, `promotion_boundary`, `blocked_types_never_promoted`, `ledger_traceability`

Release evidence: `python -m pytest tests/test_cycle_evaluator.py tests/test_pattern_learner.py tests/test_pattern_store.py tests/test_pattern_recall.py tests/test_promoted_pattern.py tests/test_pattern_endpoints.py` — 182 tests pass. Gate artifact: `benchmarks/results/pattern_phase_gate.json`.

### 4.12 EBIR-R1 Shadow Refinement Lane

MNEMOS now includes **Evidence-Bounded Iterative Reconciliation (EBIR)**, a RepFusion-inspired shadow research lane for testing whether controlled multi-pass evidence reconciliation improves synthetic Resolution Engram review quality. This is not a literal RepFusion integration: there is no diffusion, RAE, image generation, or representation-space denoising. The borrowed pattern is narrower: repeatedly challenge an evolving candidate against a bounded evidence packet instead of treating the first synthetic resolution as final.

EBIR wraps `ReconciliationRunner` in shadow mode through `RepFusionRefiner` (`mnemos/governance/hygiene/repfusion_refiner.py`). For each contradiction cluster it:

1. Builds a reconciliation packet from parent engrams: entity slot, normalized conflicting values, source authority, timestamps, evidence spans, lineage IDs, and governance metadata.
2. Generates a constrained candidate Resolution Engram representation.
3. Runs a structured evidence challenge for unsupported claims, missing parent coverage, temporal ambiguity, authority-policy conflict, and overconfident language.
4. Produces a controlled revision whose deltas are tied to the structured challenge.

**Acceptance status (EBIR-R1):** technically accepted for shadow-only burn-in. The R1 adversarial pack contains 10 fixture classes: authority inversion, temporal trap, insufficient evidence, unsupported synthesis trap, scope overreach, hidden contradiction, evidence omission, revision regression, packet immutability, and operator-review clarity. Latest artifact: `benchmarks/results/ebir_refinement_benchmark.json`.

**Safety boundaries:**

- Shadow-only; no default retrieval changes.
- No governance scoring, ranking, or policy changes.
- No Phase 10 consensus behavior changes.
- No parent engram mutation.
- No automatic or authoritative Resolution Engram promotion.
- Packet hashes are recorded across passes; packet drift, parent mutation, side effects, or promotion-path leakage fail the R1 gate.
- Forensic ledger records structured evidence, critique categories, revision deltas, latency, token cost, and confidence only; hidden chain-of-thought is not persisted.

**CI gate:**

```bash
python tools/run_ebir_refinement_benchmark.py
```

The gate requires zero EBIR regressions, zero safety violations, packet-hash equality across passes, complete parent-support maps, no unsupported claims outside fixture labels, correct abstention where required, authority/temporal alignment, cross-run stability, bounded pass count, bounded latency/token cost, and `promotion_status = blocked_from_authoritative_resolution_promotion`.

**R2 boundary:** EBIR-R2 is the next authorized horizon and remains a human-review value trial only. It must compare raw evidence review, one-pass reconciliation, and EBIR output on difficult conflict packets, measuring correct resolution, correct abstention/escalation, evidence-supported decision quality, reviewer confidence calibration, review time, unsupported-claim detection, latency, token cost, and trigger selectivity. No product promotion is approved until EBIR improves real human review outcomes, not merely benchmarked reconciliation quality.

### 4.13 Session Context Assembler: Consumer-Neutral Local Shadow Milestone

The Session Context Assembler is a governed MNEMOS research capability for constructing bounded context packages from eligible session artifacts. Its S1 selector reserves budget in policy order for prior decisions, unresolved or mixed contradictions, and their required source-linked evidence before allocating remaining space to task-relevant supporting episodes and optional semantic fill. If mandatory eligible context cannot fit, the package explicitly abstains instead of silently reporting success.

Selected artifacts retain artifact-local provenance and safety labels: `synthetic_context`, `non_authoritative`, `non_promotable`, `parent_engram_ids`, `parent_source_ids`, lineage completeness, and artifact type. The consumer receives context, not memory authority: package content is not source truth and cannot alter Engrams, contradiction state, governance, promotion, retrieval ranking, or durable write paths.

ADR 0008 authorizes and closes out only an **isolated, consumer-neutral, read-only local shadow adapter**. The local implementation demonstrates:

- authenticated request validation through a transport-neutral local abstraction;
- tenant/scope, entitlement, classification, disclosure, and redaction enforcement;
- policy- and version-pinned replay behavior with fail-closed drift handling;
- canonical package digests and artifact-local lineage verification;
- bounded package assembly with visible omission and abstention state;
- content-free shadow telemetry; and
- an atomic kill switch that blocks new assembly, cache writes, sink events, and delivery attempts without disturbing MNEMOS state.

The isolated gate passes frozen-corpus integrity, assembly, digest, lineage, budget, telemetry, determinism, and runtime/network-isolation checks. It also detects eight deliberate mutations covering digest tampering, lineage removal, telemetry escape, kill-switch bypass, replay-policy bypass, authorization bypass, redaction bypass, and abstention suppression. The focused evidence is **175 relevant tests**, not a whole-repository release certification; unrelated failures and optional dependency availability are outside this feature-lane claim.

**Research and release boundary:** this milestone contains no network listener, API route, SDK, external consumer connection, live routing, staging or production deployment, durable memory write, retrieval change, governance mutation, or consumer active-path effect. A passing gate authorizes review of a separately governed consumer-neutral shadow-evaluation proposal only. Model-assisted answer-fidelity testing remains surrogate evidence, and prepared owner review remains non-independent and non-generalizable; neither is a broad human-value claim.

For a future separately authorized consumer evaluation, a downstream application could inspect a source-linked, artifact-lineage-preserving, `synthetic_context`-labeled, non-authoritative, non-promotable, policy-scoped, budget-bounded package with explicit omissions or abstention. This could reduce irrelevant context while preserving earlier decisions, contradictions, and source evidence, without turning the consumer into an alternate memory authority.

Evidence: `docs/adr/0008-consumer-neutral-read-only-shadow-adapter-implementation.md`, `docs/session_context_assembler_spec.md`, `docs/session_context_assembler_shadow_adapter_implementation_notes.md`, and `benchmarks/results/session_context_assembler_shadow_adapter_gate.md`.

---

### 4.14 GateMem Governance Reference Baseline

MNEMOS completed a governed research program for memory authorization and disclosure. The work does not establish that MNEMOS has solved production authorization governance; it establishes a reproducible research boundary, an honestly measured weak baseline, a normative decision model, and a deterministic reference implementation for regression testing.

The program progressed through five bounded milestones:

1. **G0 — environment and gap assessment.** GateMem was pinned as an external research dependency, with its runtime, evaluator, licensing, and deletion-governance gaps recorded without modifying upstream.
2. **G1 — clean-input projection.** Visible turns, requester identity/role, and permitted metadata were projected through a strict allowlist. Evaluator-only annotations remained unavailable to the decision path.
3. **G2/G2A — frozen baseline characterization.** The unchanged offline adapter was evaluated across all four released GateMem domains. Provenance integrity remained 1.0, while utility, leakage, and over-refusal results showed that visible-text heuristics and candidate filtering were not adequate disclosure controls. These domains are historical characterization data, not a future held-out set.
4. **G3 — principal-bound semantics.** The design binds disclosure to authenticated identity, identity-derived tenant/session scope, scoped role assignment, entitlement, purpose, artifact/source classification, time validity, redaction obligations, replay controls, and content-free audit correlation. Caller filters and query wording are never authority; unknown policy or failed redaction denies.
5. **G4 — offline reference conformance.** A local deterministic implementation exercised those contracts against 36 MNEMOS-owned, inspectable synthetic development cases. It matched 36/36 expected outcomes and passed 33/33 focused gates, including forged identity, scope widening, entitlement drift, lineage change, redaction residue, replay drift, audit leakage, evaluator-field injection, and harness-owned HMAC-key isolation.

The frozen G4 implementation/corpus composite is `ed3b5c7672e591b039183eaa2d8c7c7630a655575bfc7866558d53f2eb874c52`. A read-only verifier checks pinned source hashes, corpus fingerprint, evidence counts, gate counts, and claim classification before regression tests run. Changes to the implementation or corpus create a new development iteration rather than revising the frozen result.

**Focused verification:** 59 focused tests passed; 8/8 frozen-reference checks passed; 36/36 synthetic outcomes matched; 33/33 G4 reference gates passed; and the external GateMem checkout remained clean.

**Research and release boundary:** G4 is isolated from MNEMOS runtime modules, network services, hosted models, GateMem imports, durable memory, shared caches, and deletion paths. This milestone does not claim production authorization security, held-out benchmark performance, legal compliance, active forgetting, or deletion capability. It is focused research-lane evidence, not a full-repository certification. No hosted judge, runtime integration, public benchmark submission, or leaderboard submission occurred.

GateMem policy work is paused pending an independent sealed-evaluation custodian and a newly sealed or independent evaluation corpus. Until then, G4 is retained for regression testing only.

Evidence: `docs/reports/gatemem_governance_reference_baseline.md`, `docs/benchmarks/gatemem_program_status.md`, `docs/benchmarks/gatemem_g4_offline_reference_implementation.md`, `benchmarks/results/gatemem_g4_frozen_reference_manifest.json`, and ADRs 0009–0013.

---

### 4.15 AI Developer MCP Memory Trial

MNEMOS was tested as an agentic memory substrate for an AI developer workflow
through an MFS-compatible MCP bridge. The question was narrow: when a coding
agent builds the same small local application, does MNEMOS-assisted project
memory reduce logged orientation and rework signals compared with a no-memory
control?

The bridge exposes MNEMOS through MCP tools for agent use:

```text
health_check
get_capabilities
search_memory
write_observation
record_decision
find_related_context
detect_contradictions
summarize_session_handoff
explain_memory_provenance
```

The evaluation progressed in two steps. The first pilot proved that the MCP
bridge could be used from an AI developer session, but also exposed
infrastructure-readiness and measurement gaps: the backing service had to be
started correctly, early calls could miss or route to the wrong corpus, and the
original logs did not contain enough retrieval-integrity fields to support a
retrieval-conditioned comparison.

The refreshed E1 trial fixed those gaps before rerunning the paired task. The
same starter repository was reset into both workspaces. The MNEMOS condition
used a dedicated Qdrant collection (`mnemos_ai_dev_e1_task_01`) seeded with the
task documents and identified by seed snapshot `0a578569ef136afa`. Both
conditions were required to write structured trial artifacts under
`trial_results/`, including route logs, repo activity, test runs, wrong turns,
user interventions, token estimates, and a final report. The MNEMOS-enabled
condition additionally had to log per-call retrieval telemetry: execution path,
retrieval fingerprint, returned source labels, provenance, usefulness,
abstention behavior, and whether retrieved material influenced the next action.
A verifier rejected incomplete or inconsistent folders before comparison.

The refreshed task was a local issue tracker completion task: saved views,
deterministic sorting, schema migration/default handling, repair of a seeded
`priority_desc` sorting defect, responsive/focus-safe UI polish, and a
Windows-compatible acceptance-test script without modifying the frozen
acceptance suite.

| Metric | MNEMOS enabled | No memory |
|---|---:|---:|
| Task completion | 1.0 | 1.0 |
| Acceptance-test pass rate | 1.0 | 1.0 |
| Build result | Pass | Pass |
| Total estimated tokens | 57,000 | 51,000 |
| Logged tool calls | 14 | 5 |
| Memory calls | 7 | 0 |
| Route log rows | 7 | 7 |
| Repo activity rows | 16 | 29 |
| Raw failed-test count | 5 | 0 |
| Failed-test metric status | not comparable due to harness failures | not comparable due to harness failures |
| Harness/environment failures | 1 | 1 |
| Expected RED acceptance failures | 4 | 0 |
| Agent-caused test failures | 0 | 0 |
| Wrong-turn rows | 2 | 2 |
| Files changed | 7 | 8 |
| User interventions | 0 | 0 |
| Seed snapshot recorded | `0a578569ef136afa` | n/a |
| Seed snapshot layer | task seed manifest hash | n/a |
| Collection snapshot from executed route | `mnemos_ai_dev_e1_task_01:2437be792647c500` | n/a |
| Collection snapshot layer | retrieval index snapshot | n/a |
| Executed-route fingerprint recorded | Yes | n/a |
| Retrieved-context usefulness | 0.857 | n/a |
| Provenance retained | 1.0 | n/a |
| Irrelevant-context rate | 0.0 | n/a |

Both runs completed and passed the same acceptance/build checks. The
MNEMOS-enabled run used the required memory tools (`health_check`,
`get_capabilities`, `find_related_context`, `search_memory`, `record_decision`,
`write_observation`, and `summarize_session_handoff`). Its useful
`search_memory` call returned the seeded task-doc neighborhood, including
`docs/product_scope.md`, `docs/architecture_decisions.md`, and
`docs/data_contract.md`; the agent then verified those results against local
project files before relying on them. Prior-run memory appeared in the result
set but was rejected for direct reliance rather than used as authority.

The no-memory control also completed successfully without memory calls. It
required no user intervention and reached the same final quality bar, but logged
more repo-activity rows. The MNEMOS condition logged fewer repo-activity rows
but more tool calls and a higher estimated token count. In this run, MNEMOS
showed useful retrieval and no observed quality degradation; it did not show a
token or speed advantage. The raw failed-test count is retained for audit, but
is not used as a workflow-quality comparison because both legs encountered the
same Windows glob/script harness issue. The normalized agent-caused test
failure count was `0` in both conditions.

**Interpretation:** MNEMOS can operate as a useful AI-developer orientation and
continuity layer when the collection is correctly seeded and the MCP path is
healthy. The strongest evidence from E1 is retrieval quality and auditability:
correct source neighborhood, source labels retained, retrieval fingerprints
recorded, and stale/prior-run context rejected rather than used. The evidence
does not support a broad claim that MNEMOS makes agents faster or cheaper. The
reported seed snapshot and executed-route collection snapshot are different
lineage layers: `0a578569ef136afa` identifies the task seed manifest, while
`mnemos_ai_dev_e1_task_01:2437be792647c500` identifies the retrieval index
snapshot observed in the executed route.

**Claim boundary:** this is local development evidence for one paired task, one
seed snapshot, one dedicated collection, and the tested MCP configuration. It
does not establish general AI-developer performance improvement, universal
token reduction, broad retrieval quality, production readiness as a universal
agent memory store, or causality for all observed differences. More paired
trials are needed, including warm-start repeats, resume tests, bug-regression
tests, design-constraint retention tests, stale-memory rejection tests, and
multi-task comparisons with exact token/latency instrumentation.

Evidence: `docs/experiments/ai_dev_mnemos_enabled_trial_instructions.md`,
`docs/experiments/ai_dev_no_memory_trial_instructions.md`,
`tools/verify_ai_dev_memory_trial.py`,
`tools/compare_ai_dev_memory_trials.py`,
`benchmarks/results/ai_dev_memory_trial_comparison_001.json`,
`benchmarks/results/ai_dev_memory_trial_comparison_001.md`,
`benchmarks/results/ai_dev_memory_quality_e1_task_01_comparison_004.json`, and
`benchmarks/results/ai_dev_memory_quality_e1_task_01_comparison_004.md`.

### 4.16 Open WebUI Evidence Lane (v3.4)

The Open WebUI evidence lane answers a consumer-boundary question: can a
generic, MNEMOS-unaware chat front end deliver answers grounded in MNEMOS
evidence — with per-answer proof — without widening MNEMOS's authority surface?
The lane adds no new MNEMOS endpoints and no new write paths. Open WebUI never
learns MNEMOS exists; MNEMOS never learns chat exists. The only component that
knows both worlds is a local proxy.

**Topology (local compose stack + host Ollama):**

| Component | Port | Role |
|---|---|---|
| `research-ui` | :8788 | Intake desk: upload PDFs/docs → extract (pypdf, Docling OCR fallback) → chunk → index into MNEMOS with source/page lineage; hosts the `/evidence` receipt browser |
| `openwebui-proxy` | :8790 | Evidence proxy: presents an Ollama/OpenAI-compatible API to chat front ends; retrieves bounded evidence from MNEMOS, sends evidence + question to Ollama, appends an evidence footer, writes a receipt |
| `open-webui` | :8088 | Generic chat window (separate container); speaks the standard model API and must be connected **only** to the proxy |
| Ollama | :7777 (host) | Local model runtime; the only generation engine in the lane |
| Evidence receipts | `./logs/evidence_receipts` | Host-mounted JSON receipts, shared read-only with the receipt browser |

**Chat flow:** the proxy takes the latest user message (plus sanitized prior
turns), asks MNEMOS for bounded evidence, streams Ollama's generation back to
the chat client token-by-token, appends a deterministic evidence footer
(citations, receipt link, claim boundary), and writes a receipt. If MNEMOS
returns no evidence or errors, the answer is withheld and the footer says so —
Ollama is not called.

**Multi-turn handling (July 2026):** prior turns are forwarded to Ollama with
evidence footers stripped, capped by `MNEMOS_PROXY_HISTORY_MAX_TURNS`
(default 8). Follow-up questions are condensed into standalone retrieval
queries via a bounded temperature-0 Ollama call before hitting MNEMOS;
condensation is env-gated (`MNEMOS_PROXY_QUERY_CONDENSE`), falls back to the
raw query on any failure, and is fully recorded in the receipt
(`history_turns`, `query_condensed`, `original_query`, `retrieval_query`).
Factual claims remain bound to the supplied evidence; history is
reference-resolution context only.

**Evidence receipts** are per-answer JSON artifacts recording the query, the
retrieval query actually sent, requested/actual model, the full evidence block
sent to Ollama, citations with scores and engram IDs, retrieval metadata
(mode, fingerprint, fusion policy, latency, low-relevance abstention signal
when reported), the answer, and the claim boundary. Receipts are viewable as
HTML at both the proxy and the research UI.

**Verification annotations (July 2026)** — each receipt additionally carries
deterministic post-hoc annotations. These are recorded, never enforced:

| Field | What it records |
|---|---|
| `citation_check` | Which `[n]` indices the answer cited, indices citing non-existent evidence, evidence never cited, coverage ratio, and a verdict (`all_evidence_cited`, `partial_evidence_cited`, `cites_missing_evidence`, `no_citations_in_answer`, `no_evidence_available`) |
| `generation` | Ollama `done_reason`, a `truncated` flag (token-limit stops also warn in the served footer that citations may be incomplete), and prompt/eval token counts |
| `score_stats` | Min/max/mean/count of retrieval scores, making the relevance spread of the evidence visible instead of presenting top-k chunks as equally admissible |
| `content_hash` | `sha256` over the receipt's factual core (id, created, query, answer, evidence block, citations) — recomputable tamper evidence |

**Receipt lifecycle:** receipts past `MNEMOS_EVIDENCE_RECEIPT_MAX_FILES`
(default 500) are moved to an `archive/` subfolder with a log line — proof
artifacts are never silently deleted. Real token usage
(`prompt_eval_count`/`eval_count`) passes through to the chat client. The
proxy serves via waitress (threaded, unbuffered chunk flushing) with a Flask
dev-server fallback.

**Operational note:** the chat front end must be connected only to the proxy.
A parallel direct-to-Ollama connection exposing the same model IDs causes the
front end to load-balance between them, nondeterministically bypassing
retrieval and producing answers with no footer and no receipt.

**Claim boundary:** the adapter is stamped
`MFS_OLLAMA_ADAPTER_R0_CONTEXT_ONLY` in every receipt: it retrieves MNEMOS
evidence and asks Ollama to answer from that evidence; it does not alter
MNEMOS retrieval, write memory, or enforce R1/R2 admission policy. The
verification annotations are passive shadow-style observability consistent
with the retained R0 posture — Evidence Admission R1 enforcement remains not
retained, and nothing in this lane blocks, rewrites, or re-ranks answers. The
`content_hash` is tamper evidence on a mutable file, not a signed or immutable
ledger entry. Behavior is validated by the unit suites and local end-to-end
runs of the deployed stack (streaming cadence, condensation trail, truncation
capture); this is local single-user workflow evidence, not a benchmark, and no
relevance-quality or performance claim is made.

Evidence: `tools/mnemos_ollama_chat.py`,
`tools/mnemos_ollama_openwebui_proxy.py`, `tools/mnemos_research_ui.py`,
`tools/mnemos_research_intake.py`, `tests/test_mnemos_ollama_chat.py`,
`tests/test_mnemos_ollama_openwebui_proxy.py`,
`tests/test_mnemos_research_ui.py`,
`docs/integrations/openwebui_mnemos_local_chat_readme.md`,
`docs/integrations/ollama_mnemos_mfs.md`, and `topology.md`.

---

## 5. API Contract

MNEMOS follows the **MFS Contract Pattern**: every response includes `contract_version`, `status`, `source`, and `error` fields, ensuring the consuming application can always determine the health and trustworthiness of the data it receives.

### Contract (service/contract.json)

`service/contract.json` defines the **minimum** MFS envelope validated by CI (`contract_version`, `status`, `source`, `generated_at`, `feature`, `supports`, `error`). The live `/v1/mnemos/capabilities` response extends this baseline with deployment and feature-discovery fields (`profile`, `tiers`, `retrieval_modes`, `fusion_policies`, `governance`, `memory_over_maps`, `compression`, `gpu_device`). Contract evolution checks use `tools/contract_diff.py`.

### Core Endpoints

```
GET    /health                              — Container health check
GET    /v1/mnemos/capabilities              — Feature discovery, profile, backends, governance/MoM flags
POST   /v1/mnemos/index                     — Ingest documents → engrams
POST   /v1/mnemos/search                    — Query across active backends (canonical)
POST   /api/v1/query                        — Search alias (same handler; derived-fact leak guard)
POST   /api/v1/evaluate_derived_shadow      — Production-adjacent derived-fact shadow evaluation (pilot only)
POST   /v1/mnemos/governance/reflect        — Post-generation reflect loop (usage labels + reinforcement)
GET    /v1/mnemos/engrams/{id}              — Retrieve a specific engram
DELETE /v1/mnemos/engrams/{id}              — Remove from all backends
GET    /v1/mnemos/audit                     — Query the forensic ledger
GET    /v1/mnemos/stats                     — Backend sizes, economics, derived-lane telemetry, cache stats
GET    /v1/mnemos/governance/stats          — Governance + reflect + hygiene aggregate stats
```

### Predictive Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/mnemos/pulse` | GET | Returns the observed service heartbeat and TimesFM 15-minute forecast |
| `/v1/mnemos/warmup` | POST | Explicitly forces model-load and retrieval-path pre-warming |
| `/v1/mnemos/governance/reflect` | POST | Post-generation reflect loop; NLI-backed usage detection remains available through governance profiles |

### Example: /capabilities Response

```json
{
  "contract_version": "v1",
  "status": "healthy",
  "source": "mnemos-service",
  "generated_at": "2026-06-10T05:00:00Z",
  "feature": "mnemos_memory",
  "profile": "core_memory_appliance",
  "supports": ["index", "search", "engrams", "audit", "stats", "pulse", "warmup"],
  "tiers": ["qdrant"],
  "retrieval_modes": ["semantic", "hybrid"],
  "fusion_policies": ["semantic_dominant", "balanced", "lexical_dominant", "qdrant_rrf"],
  "retrieval_mode_default": "semantic",
  "fusion_policy_default": "balanced",
  "lexical_lane_available": true,
  "explain_support": true,
  "compression": { "enabled": true, "bits": 4 },
  "gpu_device": "cuda",
  "governance": {
    "supported_modes": ["advisory", "enforced", "off"],
    "default_mode": "off",
    "policy_profiles": ["default"]
  },
  "memory_over_maps": {
    "phase1_enabled": false,
    "phase2_enabled": false,
    "phase3_enabled": false,
    "phase4_enabled": false,
    "phase5_enabled": false,
    "supported_derived_views": ["contradiction", "evidence", "preference", "timeline"]
  },
  "pulse": {
    "enabled": true,
    "timesfm_enabled": true,
    "actions_mode": "advisory",
    "horizon_minutes": 15,
    "provider": "timesfm_sidecar_with_linear_fallback"
  },
  "error": null
}
```

A consumer can determine: active profile, backends, retrieval/fusion options, governance modes and policy profiles, Memory Over Maps phase flags, and compression — without inspecting env vars or deployment files.

### Example: Indexing a Document

```json
POST /v1/mnemos/index
{
  "documents": [
    {
      "content": "The quarterly results exceeded expectations...",
      "source": "s3://reports/Q1-2026.pdf",
      "neuro_tags": ["finance", "quarterly"],
      "confidence": 0.95,
      "metadata": { "department": "finance", "fiscal_year": 2026 }
    }
  ],
  "options": {
    "tiers": ["qdrant"]
  }
}
```

### Example: Searching

```json
POST /v1/mnemos/search
{
  "query": "What were the Q1 revenue figures?",
  "top_k": 10,
  "tiers": ["qdrant"],
  "filters": { "metadata.department": "finance" },
  "retrieval_mode": "hybrid",
  "fusion_policy": "qdrant_rrf",
  "governance": "advisory",
  "governance_profile": "default",
  "explain": true,
  "explain_governance": true,
  "bounded_envelope": { "enabled": true, "candidate_pool_limit": 50 },
  "derive_views": ["evidence", "contradiction"]
}
```

**Search request parameters (selected):**

| Parameter | Type | Description |
|---|---|---|
| `retrieval_mode` | string | `semantic` or `hybrid` (public HTTP surface) |
| `fusion_policy` | string | `semantic_dominant`, `balanced`, `lexical_dominant`, `qdrant_rrf` |
| `governance` | string | `off`, `advisory`, `enforced` |
| `governance_profile` | string | Tenant policy profile ID |
| `explain` | boolean | Per-hit hybrid attribution |
| `explain_governance` | boolean | Governance traces and suppressed-candidate summary |
| `bounded_envelope` | object | Phase 2 candidate pool limits |
| `derive_views` | string[] | On-demand derived views when MoM phases enabled |
| `enable_derived_facts` | boolean | **Pilot only** — routes to derived trial path when kill-switch + whitelist pass |

**Search response extensions (`meta`):**

`economics` (envelope compression, cache hits/misses, estimated cost units), `candidate_envelope`, `hybrid_telemetry`, `governance_summary`, `governance_explain` (when `explain_governance=true`)

### Example: Derived Shadow Evaluation (pilot only)

```json
POST /api/v1/evaluate_derived_shadow
X-Client-Id: eval_dashboard

{
  "query": "What is the SIGINT operations reference?",
  "top_k": 10,
  "evaluation_mode": true,
  "include_derived_facts": true
}
```

Requires `MNEMOS_DERIVED_ENABLED=true` and whitelisted client. Returns default search results plus `shadow_evaluation.rendered_block` and stage latency telemetry.

### Example: Governance Reflect

```json
POST /v1/mnemos/governance/reflect
{
  "query": "What were the Q1 revenue figures?",
  "answer": "Revenue exceeded expectations in Q1...",
  "results": [ ... ],
  "decisions": [ ... ],
  "governance_profile": "default"
}
```

### Stats and economics (`GET /v1/mnemos/stats`)

Beyond backend sizes and compression, stats expose:

- `stats.economics` — per-query envelope compression aggregates, cache hit ratio, invalidation event count
- `stats.derived_lane` — derived-fact lane counters (see §4.8)
- `stats.memory_over_maps` — MoM runtime counters and derived-view cache fanout

---

## 6. Boundary SDK

MNEMOS ships with a Python client library (`mnemos_sdk/`) that provides the standard way for consumer applications to call the service. The SDK handles concerns that raw HTTP calls don't:

| Capability | Description |
|---|---|
| **Readiness polling** | `wait_until_ready()` polls `/health` until the service is available or timeout |
| **Retry with backoff** | Configurable retry count and delay for transient failures |
| **Timeout management** | Per-request timeout with graceful fallback |
| **Auto-start** | Optional: start MNEMOS container on first call (for local dev) |
| **Typed responses** | `MnemosResponse`, `SearchHit`, `IndexResult` data classes |
| **Pulse access** | `client.pulse()` returns observed heartbeat and forecast/advisory metadata |
| **Graceful degradation** | Returns structured `degraded`/`unavailable` envelopes instead of exceptions |
| **Env-var config** | `MnemosConfig.from_env()` wires everything from `MNEMOS_*` env vars |

### Usage

```python
from mnemos_sdk import MnemosClient, MnemosConfig

# Configure from environment (MNEMOS_BASE_URL, MNEMOS_TOKEN, etc.)
config = MnemosConfig.from_env()
client = MnemosClient(config)

# Wait for service startup
client.wait_until_ready()

# Store knowledge
client.index([{
    "content": "Gravity waves detected by LIGO in 2015",
    "source": "arxiv:1602.03837",
    "neuro_tags": ["physics", "gravitational-waves"],
}])

# Recall knowledge
hits = client.search("gravitational wave detection", top_k=5)
for hit in hits:
    print(f"  [{hit.score:.3f}] {hit.engram['content'][:80]}")
```

**Rule**: Consumer apps should always use the SDK for index/search flows. Direct HTTP calls bypass readiness, retry, and degradation handling.

**Advanced endpoints** (`/v1/mnemos/governance/reflect`, `/api/v1/evaluate_derived_shadow`, governance explain flags) are available via HTTP today; SDK wrappers for these paths are not yet shipped — integrators call them explicitly when needed.

---

## 7. Operational Tooling

MNEMOS includes a suite of tools adapted from the MFS framework for production operation:

### 7.1 Health & Contract Audit

```bash
python tools/mnemos_health_audit.py
```

Validates: `/health` returns 200, contract endpoint fields match `contract.json` types, status is in allowed values, contract version has not drifted.

### 7.2 Contract Evolution

```bash
python tools/contract_diff.py --old service/contract.json --new contracts/mnemos_v2.json --mode both
```

Checks backward and forward compatibility when evolving the contract: field additions/removals, type changes, enum narrowing, default value transitions, and major version bump advisories.

### 7.3 Consumer Onboarding

```bash
python tools/mnemos_onboard.py --target /path/to/consumer-app
```

Generates in the consumer app: a pre-wired boundary adapter, `.env.mnemos` template, smoke test spec, and integration quickstart doc.

### 7.4 CI/CD Gates

```bash
python tools/mnemos_ci_gates.py \
  --run-health-audit \
  --smoke-spec tools/mnemos_smoke_spec.json \
  --run-memory-over-maps-gates \
  --run-governance-evidence-gates \
  --run-wave4-hygiene-gate \
  --run-slo-reliability-gate
```

| Gate | Enforces |
|---|---|
| Contract validation | `service/contract.json` well-formedness |
| Health & smoke | Live `/health` and smoke spec against running container |
| Memory Over Maps | Phase 1–5 benchmark regression tests |
| Adaptive routing | Phase 8 hold-out complexity accuracy — committed evidence artifact (`benchmarks/results/phase_8_complexity_accuracy.json`); not yet wired into `mnemos_ci_gates.py` |
| Hierarchical retrieval | Phase 9b similarity + live summary-isolation — committed evidence artifact (`benchmarks/results/phase_9_hierarchy_sim.json`); not yet wired into `mnemos_ci_gates.py` |
| Consensus governance | Phase 10 live resolution gate — committed evidence artifact (`benchmarks/results/phase_10_consensus_gate.json`); not yet wired into `mnemos_ci_gates.py` |
| Pulse integrity | One-minute telemetry normalization, circular-buffer hydration, and `/v1/mnemos/pulse` response shape |
| Prediction fidelity | TimesFM 15-minute forecast accuracy within target error and sidecar round-trip under 100ms |
| Mind-reading gate | Intent trajectory shadow search must prove a pre-cognitive cache hit for the predicted query |
| Governance evidence | Governance, contradiction, reflect, drift validation tests |
| Wave 4 hygiene | `tools/run_wave4_hygiene.py --mode dry-run --fail-on-gate` |
| SLO reliability | `tools/run_slo_reliability_gate.py --stage canary_25 --fail-on-breach` |
| EBIR-R1 shadow acceptance | `tools/run_ebir_refinement_benchmark.py`; adversarial fixtures must pass with zero regressions, zero safety violations, packet-hash integrity, and promotion blocked |

GitHub Actions workflow `.github/workflows/mnemos-gates.yml` runs the full gate suite on `main` PRs and pushes. Promotion is blocked on any gate failure; SLO breach triggers rollback guidance per `docs/mnemos_operator_playbook.md`.

Standalone runners:

```bash
python tools/run_wave4_hygiene.py --mode dry-run --fail-on-gate
python tools/run_slo_reliability_gate.py --stage canary_25 --fail-on-breach
python tools/run_ebir_refinement_benchmark.py
```

### 7.5 Cutover Scaffold

```bash
python tools/mnemos_cutover_scaffold.py --app my-app
```

Generates a staged rollout manifest (shadow → canary 5/25/50% → full) for apps migrating from another memory backend to MNEMOS, with health gates and rollback paths.

### 7.6 Companion Evidence Documents

Detailed trial, certification, and experimental evidence is maintained outside this whitepaper:

| Track | Location | Scope |
|---|---|---|
| Enhancement roadmap | `docs/mnemosv2_enhancement_roadmap.md` | 30/60/90 program closeout |
| Operator playbook | `docs/mnemos_operator_playbook.md` | Deploy, promote, rollback, incident |
| Derived Facts (PIT) | `docs/reports/pit_*.md` | Production-adjacent shadow lane |
| Human operator trials (DFE) | `docs/reports/dfe_*.md` | Derived-fact selection and value assessment |
| Graph Tier (MG-Test) | `docs/graph_tier/`, `docs/mg_test_10_experimental_closeout.md` | Experimental graph hybrid |
| EBIR-R1 shadow refinement | `docs/ebir_r1_acceptance.md`, `benchmarks/truthsets/ebir_r1_adversarial.json`, `benchmarks/results/ebir_refinement_benchmark.json` | RepFusion-inspired evidence-bounded reconciliation; shadow-only, promotion blocked |
| AI developer MCP memory trial | `docs/experiments/ai_dev_*_trial_instructions.md`, `benchmarks/results/ai_dev_memory_trial_comparison_001.*`, `benchmarks/results/ai_dev_memory_quality_e1_task_01_comparison_004.*` | Pilot plus refreshed E1 paired local app-building trial; useful retrieval/provenance evidence and no observed quality degradation, but no speed or token-efficiency claim |
| Ops certification | `docs/reports/ops_*.md`, `docs/cert_binder/` | Release governance and red-lines |
| Validation / shadow (VFR) | `docs/reports/vfr_*` (where present) | Sidecar read-only enforcement |
| Schema/fact extraction (SMC) | `tools/smc_*.py`, `docs/reports/` | **Blocked** pending separate review |

---

## 8. Deployment Profiles

MNEMOS ships with named deployment profiles that determine the retrieval backend, container topology, and operational posture. The guided installer (`python -m installer`) recommends a profile based on use case, priorities, host capabilities, and platform-safe compute mode.

### Profile A: Core Memory Appliance *(default)*

**Best for:** Semantic memory, agent systems, general-purpose RAG.

| Component | Service | Container |
|---|---|---|
| Vector store | Qdrant (HNSW, CUDA or CPU embeddings) | `mnemos-qdrant` |
| Audit ledger | PostgreSQL | `mnemos-postgres` |
| Service | MNEMOS (CUDA or CPU mode) | `mnemos-service` |

3 containers. Qdrant provides fast semantic ANN with payload filtering. Recommended when retrieval is primarily semantic and the corpus exceeds 100K documents.

### Profile B: Governance Native

**Best for:** Provenance-heavy, metadata-filtered, compliance-aware retrieval.

| Component | Service | Container |
|---|---|---|
| Vector store | pgvector (inside PostgreSQL) | `mnemos-postgres` (shared) |
| Audit ledger | PostgreSQL | `mnemos-postgres` (shared) |
| Service | MNEMOS (CUDA or CPU mode) | `mnemos-service` |

2 containers. Vectors and audit share one Postgres instance. ANN retrieval can be combined with SQL `WHERE` clauses on tenant, provenance, or security markings — in a single query. Recommended when metadata filtering matters more than raw ANN throughput.

### Profile C: Custom Manual

**Best for:** Advanced operators, multi-tier setups, experimentation.

No compose generation — the operator provides their own configuration. The installer writes `.env.mnemos` only. Supports any combination of backends including Cross-Encoder reranking.

### Hybrid Retrieval Mode (Gate C) *(inside existing profiles)*

Hybrid retrieval is not a separate deployment profile. It is a retrieval mode available within Core and Governance deployments using lexical + semantic fusion. As of the March 29, 2026 real-corpus benchmark decision, hybrid is supported for targeted evaluation but is not the global default.

---

## 9. Deployment

The installer generates a profile-specific `docker-compose.generated.yml` and `.env.mnemos`. It resolves compute mode first: Linux hosts with an NVIDIA GPU and NVIDIA Container Toolkit use CUDA, while macOS and hosts without the NVIDIA runtime generate CPU-safe files with no `runtime: nvidia` stanza and `MNEMOS_GPU_DEVICE=cpu`. Operators can override with `--compute-mode cpu` or `--compute-mode cuda`.

CUDA-mode example stacks:

### Core Memory Appliance

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.17.1
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=mnemos
      - POSTGRES_USER=mnemos
      - POSTGRES_PASSWORD=mnemos
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mnemos:
    build: .
    runtime: nvidia
    ports:
      - "8700:8700"
    environment:
      - MNEMOS_PROFILE=core_memory_appliance
      - MNEMOS_TIERS=qdrant
      - MNEMOS_GPU_DEVICE=cuda
      - MNEMOS_QDRANT_URL=http://qdrant:6333
      - MNEMOS_POSTGRES_DSN=postgresql://mnemos:mnemos@postgres:5432/mnemos
    depends_on: [qdrant, postgres]

volumes:
  qdrant_data:
  postgres_data:
```

### Governance Native

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=mnemos
      - POSTGRES_USER=mnemos
      - POSTGRES_PASSWORD=mnemos
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mnemos:
    build: .
    runtime: nvidia
    ports:
      - "8700:8700"
    environment:
      - MNEMOS_PROFILE=governance_native
      - MNEMOS_TIERS=pgvector
      - MNEMOS_GPU_DEVICE=cuda
      - MNEMOS_POSTGRES_DSN=postgresql://mnemos:mnemos@postgres:5432/mnemos
    depends_on: [postgres]

volumes:
  postgres_data:
```

### Resource Requirements

| Profile | Containers | RAM | Disk | GPU |
|---|---|---|---|---|
| Core Memory Appliance | 3 | ~2 GB | ~200 MB base | CUDA recommended; CPU supported for local/macOS evaluation |
| Core + Cross-Encoder reranking | 3 | ~4 GB | ~400 MB base | CUDA recommended; CPU slower |
| Governance Native | 2 | ~1.5 GB | ~150 MB base | CUDA recommended; CPU supported for local/macOS evaluation |
| Governance + Cross-Encoder | 2 | ~3.5 GB | ~350 MB base | CUDA recommended; CPU slower |

---

## 9. Integration Workflow

A consumer application adopts MNEMOS in five steps:

```
1. Install        →  python -m installer
2. Configure      →  Review generated .env.mnemos and mnemos_profile.yaml
3. Start          →  docker compose -f docker-compose.generated.yml up -d --build
4. Validate       →  python tools/mnemos_health_audit.py
5. Wire           →  Import mnemos_sdk, use client.index() / client.search()
```

For apps migrating from another memory backend (Redis, Elasticsearch, FAISS):

```
6. Scaffold cutover   →  python tools/mnemos_cutover_scaffold.py --app <app>
7. Shadow writes      →  Write to both old + MNEMOS, read from old
8. Canary promotion   →  Route 5% → 25% → 50% → 100% of reads to MNEMOS
9. Decommission       →  Remove old backend
```

---

## 10. Use Cases

The following are the highest-value scenarios where MNEMOS provides immediate benefit as a drop-in memory layer.

### 10.1 AI Agent / Copilot Platforms

**Recommended profile:** Core Memory Appliance

The most natural fit. Any system that has an LLM doing multi-step work needs persistent, searchable memory.

- **What MNEMOS provides**: Engram-enriched conversation history, tool output recall, semantic search over past interactions.
- **Why not just a raw vector DB**: Neuro-tags give semantic labels for retrieval boosting. The forensic ledger tracks what the agent remembered and when — critical for debugging hallucinations.
- **Example**: A coding assistant that remembers past codebases it has worked on, retrieves relevant patterns, and audits what context influenced each generation.

### 10.2 RAG-Powered Knowledge Bases

**Recommended profile:** Governance Native (compliance) or Core Memory Appliance (general)

Enterprise document search where accuracy and audit trails matter — legal, medical, compliance.

- **What MNEMOS provides**: Profile-matched retrieval (pgvector for governance-heavy, Qdrant for speed), optional hybrid lexical+semantic mode for exact-term sensitive workloads, and TurboQuant for scaling to millions of chunks.
- **Why it wins**: The forensic ledger gives compliance-ready logging of every query and retrieval — *"show me exactly what documents were retrieved for this answer and when."*
- **Example**: Internal knowledge base for a law firm — lawyers query it, each retrieval is logged for audit, and pgvector filters by department and security clearance.

### 10.3 IoT / Edge Deployments

**Recommended profile:** Governance Native (single-database, minimal footprint)

Devices with limited memory and storage that still need intelligent retrieval.

- **What MNEMOS provides**: TurboQuant 4-bit compression means a 1M-document index fits in ~68 MB instead of ~512 MB. Governance Native profile runs with just 2 containers.
- **Why it wins**: Most vector DBs assume cloud-scale resources. MNEMOS can run on a single Postgres instance with pgvector.
- **Example**: A smart home hub that remembers user preferences, schedules, and sensor patterns — compressed on-device, searchable locally without cloud dependency.

### 10.4 Multi-Agent Orchestration Systems

**Recommended profile:** Core Memory Appliance

Systems where multiple specialised agents need shared memory without stepping on each other.

- **What MNEMOS provides**: A centralised, contract-governed memory service that any agent can index to and search from via REST. The MFS contract pattern means agents can trust the response schema.
- **Why it wins**: Without shared memory, each agent re-discovers context. With MNEMOS, Agent A's research becomes Agent B's retrieval — and the audit trail shows who stored what.
- **Example**: A research pipeline where a "Scout" agent gathers papers, an "Analyst" agent extracts insights, and a "Writer" agent drafts reports — all sharing one MNEMOS instance.

### 10.5 Content / Creative Platforms

**Recommended profile:** Core Memory Appliance (semantic default), with optional hybrid mode for exact-term-sensitive workloads

Story generators, game engines, or creative tools that need long-term world memory.

- **What MNEMOS provides**: Engram edges create a knowledge graph of relationships (characters → events → locations). Neuro-tags categorise memory by theme. Hybrid lexical+semantic retrieval can improve exact phrase continuity checks.
- **Why it wins**: Creative tools need precise recall (*"what did character X say about Y in chapter 3?"*) — multi-vector retrieval is dramatically better than single-vector for this.
- **Example**: An interactive fiction engine where the story adapts based on retrieving and referencing past plot points from a compressed engram store.

### Common Thread

Any application that stores, enriches, retrieves, and audits knowledge — and needs production-grade infrastructure without building the plumbing from scratch. MNEMOS gives you deployment profiles tailored to your use case, GPU-accelerated retrieval, scalable audit logging, and a complete operational toolkit in one `python -m installer`.

---

## 11. Design Principles

1. **Application-agnostic** — The service has zero knowledge of what domain it serves. It stores vectors, enriches engrams, and answers queries. Period.
2. **Compute-mode aware** — Embedding inference uses CUDA when an NVIDIA GPU and container runtime are available, while CPU mode is generated automatically for macOS and other non-NVIDIA hosts.
3. **Profile-composable** — Named deployment profiles (Core Memory Appliance, Governance Native) determine the retrieval backend and container topology. The installer recommends, the operator confirms.
4. **Contract-governed** — Every API response follows a strict MFS contract schema, enabling reliable integration without tight coupling.
5. **Compression by default** — TurboQuant is on at 4-bit out of the box. Storage scales sublinearly with document count.
6. **Audit by default** — Every mutation is logged immutably to PostgreSQL. Compliance is a feature, not an afterthought.
7. **Graceful degradation** — If a backend goes unhealthy, remaining backends continue serving. Status is always reported honestly via the contract.
8. **Process isolation** — Each infrastructure component (vector store, audit ledger, service) runs in its own container with independent health checks, volumes, and lifecycle.
9. **SDK-first integration** — Consumer apps use the boundary SDK, never raw HTTP. This ensures readiness, retry, and degradation are handled consistently.
10. **Tooling-complete** — Health audit, contract evolution, onboarding, CI gates, and cutover are included — not left as an exercise for the adopter.
11. **Governance by design** — The governance layer is built into the read path, not bolted on. Reinforcement convergence, contradiction adjudication, freshness decay, suppression policies, and background hygiene are evaluated with deterministic, tunable parameters. Per-tenant policy profiles allow threshold and delta tuning without service restarts. Behavioral guarantees are backed by formal validation evidence (Governance Validation Pack v1), not asserted by architecture language alone. Advisory mode before enforced mode; promotion requires benchmark evidence.
12. **Non-destructive hygiene** — Memory health management uses state transitions, not deletions. The hygiene path promotes memories to `stale` or `prune_candidate`; deletion and consolidation are explicit, operator-gated actions. This keeps the governance story auditable and reversible at every stage.

13. **Self-awareness before magic** — Predictive behavior must be grounded in observed operational telemetry before it is applied to cognitive intent. MNEMOS learns its own pulse before it forecasts user trajectories.
14. **Anticipatory instinct** — The service uses idle time to prepare for forecasted load, pre-warm expensive paths, and reconcile predicted conflicts so peak performance is available before it is requested.
---


## 12. Deployment Manifest (mnemos_profile.yaml)

The guided installer generates a `mnemos_profile.yaml` file alongside the compose and env files. This manifest is a **durable deployment artifact** — the single source of truth for what was installed, why, and how.

```yaml
mnemos_profile:
  version: 1.0
  generated_at: 2026-03-28T04:15:22
  install_type: new
  profile:
    name: governance_native
    display_name: Governance Native
    confidence: high
    reasons:
      - Strict metadata/provenance filtering required
      - pgvector enables SQL WHERE + ANN in one query
    warnings: []
    alternatives:
      - core_memory_appliance
  user_answers:
    use_case: compliance_governed
    priority: metadata_governance
    scale: 100k_to_1m
    strict_filters: true
    prefer_manual: false
  host_facts:
    gpu_available: true
    gpu_name: NVIDIA GeForce RTX 4090
    vram_mb: 24576
    ram_gb: 32.0
    docker_available: true
    nvidia_runtime: true
  enabled_services:
    - postgres
    - mnemos
```

**Why this matters:**

| Purpose | How mnemos_profile.yaml enables it |
|---|---|
| **Reproducibility** | Re-run the installer on a new host with the same answers → identical deployment |
| **Supportability** | Attach the manifest to any support request — it captures what was deployed and why |
| **Upgrade input** | Future installer versions can read the manifest to recommend migration paths |
| **CI validation** | CI gates can validate that the running service matches the declared profile |
| **Audit trail** | The manifest records the host facts at install time (GPU, RAM, Docker version) |

The manifest is not consumed by the MNEMOS runtime — it is a static record for operators, support, and tooling. The runtime reads `MNEMOS_PROFILE` from the environment.

---

## 13. Profile Migration

Once profiles are deployed, operators may need to migrate between them. MNEMOS defines migration rules for the two primary profiles.

### Core Memory Appliance → Governance Native

**When:** An operator wants to simplify operations (3 → 2 containers) or needs SQL-level metadata filtering.

| Step | Action |
|---|---|
| 1 | Run `python -m installer --profile governance_native` to generate new compose/env |
| 2 | Enable pgvector extension: `CREATE EXTENSION IF NOT EXISTS vector` in Postgres |
| 3 | Re-index engrams from Qdrant to pgvector using the MNEMOS search/index API |
| 4 | Validate: `python tools/mnemos_health_audit.py` confirms pgvector is healthy |
| 5 | Decommission Qdrant container |
| 6 | Update `mnemos_profile.yaml` with new profile and migration timestamp |

**Metadata assumptions that change:** pgvector stores metadata as JSONB columns — metadata that was previously Qdrant payload becomes SQL-queryable. Review any metadata schemas for SQL compatibility.

### Governance Native → Core Memory Appliance

**When:** Retrieval latency or throughput requirements exceed what pgvector can deliver, or the corpus grows beyond single-Postgres scale.

| Step | Action |
|---|---|
| 1 | Run `python -m installer --profile core_memory_appliance` |
| 2 | Start Qdrant container |
| 3 | Re-index engrams from pgvector to Qdrant using the search/index API |
| 4 | Validate Qdrant health |
| 5 | Optional: keep pgvector table as read-only archive |
| 6 | Update `mnemos_profile.yaml` |

### Rollback

Both migration paths are non-destructive — the source backend is not modified during migration. If the new profile fails health validation:

1. Revert `docker-compose.generated.yml` to the previous version
2. Revert `.env.mnemos` to the previous `MNEMOS_PROFILE`
3. Restart: `docker compose -f docker-compose.generated.yml up -d`
4. The original backend is still intact and serving

---

## 14. Repository Structure

```
MNEMOS/
├── mnemos/                    Core library
│   ├── compression/           TurboQuant (arXiv:2504.19874)
│   ├── engram/                Engram model and enrichment
│   ├── memory_over_maps/      Phase-gated source-first memory lane
│   ├── retrieval/             Multi-backend retrieval + fusion
│   │   ├── qdrant_tier.py     Qdrant backend (Core Memory Appliance)
│   │   ├── pgvector_tier.py   pgvector backend (Governance Native)
│   │   ├── cross_encoder.py   Cross-Encoder reranker (optional)
│   │   ├── fusion.py          Multi-backend fusion engine
│   │   └── base.py            BaseRetriever interface
│   └── audit/                 Forensic ledger
│   └── governance/            Governance layer (MemArchitect)
│       ├── governor.py        Entry point; wraps registry + read path
│       ├── read_path.py       Advisory / enforced read path; 3-tuple return
│       ├── policy_registry.py Per-candidate policy pipeline with short-circuit
│       ├── policies/          RelevanceVetoPolicy, UtilityPolicy, ContradictionPolicy
│       └── models/            GovernanceMeta, GovernanceDecision, ContradictionRecord
├── mnemos_sdk/                Boundary adapter SDK (client library)
│   ├── client.py              MnemosClient with typed methods
│   └── config.py              MnemosConfig.from_env()
├── service/                   Flask REST API + MFS contract
├── installer/                 Guided deployment installer
│   ├── __main__.py            Entry point (python -m installer)
│   ├── questions.py           Deployment Q/A
│   ├── probes.py              Host capability detection
│   ├── profiles.py            Profile definitions
│   ├── recommend.py           Decision tree recommendation
│   ├── render.py              Compose + env + manifest generator
│   └── templates/             Per-profile compose templates
├── tools/                     Operational tooling
│   ├── mnemos_health_audit.py
│   ├── contract_diff.py
│   ├── mnemos_onboard.py
│   ├── mnemos_ci_gates.py
│   ├── mnemos_cutover_scaffold.py
│   ├── run_wave4_hygiene.py
│   └── run_slo_reliability_gate.py
├── benchmarks/                Reproducible benchmark suite
│   ├── run_memory_over_maps_benchmarks.py
│   └── run_mg_test_*.py        Graph Tier offline/live simulations
├── tests/                     Unit + gate tests (564 collected, June 2026)
├── .github/workflows/         CI gate workflow (mnemos-gates.yml)
├── docs/                      Whitepaper, benchmark, operator playbook, reports
│   ├── graph_tier/            Graph hybrid experimental operator guide
│   └── reports/               PIT, DFE, MG-Test, OPS evidence records
├── Dockerfile                 Production container
└── docker-compose.yml         Default stack (Core Memory Appliance)
```

---

## 15. Provenance

MNEMOS was designed from the ground up as a reusable memory service. Its architecture draws on production experience operating multi-tier vector retrieval, near-lossless compression, and forensic audit logging under continuous autonomous workloads.

| Capability | MNEMOS Component |
|---|---|
| Multi-vector retrieval | Multi-Tier Retrieval Engine |
| Matryoshka coarse-to-full retrieval | Qdrant named-vector MRL path (`dense_64` prefetch + `dense_768` rescore) |
| Semantic tagging | Engram Enrichment Layer |
| Immutable operation logging | Audit Trail |
| Near-optimal quantisation (arXiv:2504.19874) | Compression Layer |
| Telemetry & health reporting | Stats & Health API |
| Versioned contract schema | API Contract + Contract Diff |
| Client library with degradation | MNEMOS SDK (mnemos_sdk/) |
| Service validation | Health & Contract Audit |
| Consumer scaffolding | Consumer Onboarding |
| Pipeline integration | CI/CD Gates |
| Staged rollout | Cutover Scaffold |
| Memory lifecycle governance | Governance Layer (mnemos/governance/) |
| Contradiction detection & resolution | ContradictionPolicy (Wave 2) |
| Adaptive query routing | Embedded-reflex complexity classifier (`mnemos/retrieval/complexity.py`) |
| Hierarchical summary retrieval | RAPTOR-lite hierarchy runner (`mnemos/governance/hygiene/clustering_runner.py`) |
| Consensus resolution | Reconciliation runner + Resolution Engrams (`mnemos/governance/hygiene/reconciliation_runner.py`) |
| Evidence-bounded reconciliation refinement | EBIR shadow lane (`mnemos/governance/hygiene/repfusion_refiner.py`) + R1 gate (`tools/run_ebir_refinement_benchmark.py`) |
| Governed session context assembly | S1 selector + consumer-neutral read-only local shadow adapter (`prototype/session_context_assembler/`) + isolated gate (`tools/run_session_context_assembler_shadow_adapter_gate.py`) |
| Source-grounded selective synthesis | Memory Over Maps lane (mnemos/memory_over_maps/) |
| Background memory hygiene | Wave 4 hygiene pipeline (`mnemos/governance/hygiene/`) |
| Tenant governance tuning | `GovernancePolicyProfile` + `MNEMOS_GOVERNANCE_POLICY_PROFILES_JSON` |
| SLO-governed promotion | `tools/run_slo_reliability_gate.py` + operator playbook |
| Derived-fact shadow evaluation | PIT lane (`/api/v1/evaluate_derived_shadow`, `mnemos/evaluation/`) |
| Experimental graph expansion | Graph Tier (`graph_hybrid_experimental`, `docs/graph_tier/`) |

| Operational telemetry normalization | Pulse Engine (`mnemos/retrieval/pulse.py`) |
| Anticipatory model pre-warming | `PulseEngine.evaluate_and_trigger()` + `/v1/mnemos/warmup` |
| Semantic volatility forecasting | `VolatilityHarvester` + `VolatilityEngine` |
| Intent-trajectory shadow search | `IntentEngine` + `ShadowSearchRunner` |
| Predictive SLO orchestration | `BudgetAwareRouter` + router `forecast_advisory` metadata |
| CoALA cognitive cycle transparency | `CognitiveCycleRecord` + `CycleAssembler` (`mnemos/cognitive/`) — opt-in via `cognitive_cycle: true` |
| Auditable forecast lifecycle | `ForecastOutcomeRecord` (`mnemos/cognitive/forecast_outcome.py`) |
| Structured attention contract | `build_attention_decisions()` (`mnemos/cognitive/attention.py`) — 11 named dimensions |
| Cognitive cycle history | `GET /v1/mnemos/cognitive/cycles` |
| Consumer-boundary evidence chat | Open WebUI evidence lane (`tools/mnemos_ollama_openwebui_proxy.py`, `tools/mnemos_research_ui.py`) — streamed answers with receipts + verification annotations (§4.16) |

What remains is a **pure infrastructure service** — a reusable, tooling-complete foundation for any application that needs intelligent, compressed, auditable memory.
