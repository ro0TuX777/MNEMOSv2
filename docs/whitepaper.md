# MNEMOS — White Paper

**A containerised, contract-governed memory and retrieval service for AI-native applications.**

*Version 3.0 · March 2026*

> [!NOTE]
> **As of March 30, 2026:** Benchmark conclusions in this whitepaper are date-scoped to the current measured runs.
> For full methodology, raw artifacts, and latest updates, see `docs/benchmark.md`.
> For release, promotion, rollback, and incident execution runbooks, see `docs/mnemos_operator_playbook.md`.
> **Governance layer (MemArchitect Waves 1–3) is implemented** — per-candidate policy pipeline, entity-slot contradiction detection, and reflect-path reinforcement are in place; Wave 4 hygiene remains the next expansion lane.
> **Memory Over Maps Phases 1–5 are implemented and benchmark-gated** — source-grounded lineage, bounded candidate envelope, on-demand derived views, cache + invalidation, and bounded semantic reflect evolution all passed phase gates on March 30, 2026.
> **Deployment model:** MNEMOS runtime services are deployed as a Docker Compose stack; all serving components run in containers.
> **Developer model:** tooling, benchmarks, and tests are typically run from host Python unless explicitly containerized.

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

A guided Python installer (`python -m installer`) probes the host, asks 5 questions, recommends a profile, and generates all deployment files. The service exposes a versioned REST API governed by an MFS contract.

MNEMOS is **application-agnostic** — it knows nothing about the domain of the consuming application. It stores, enriches, compresses, retrieves, and audits knowledge. That’s it.

**What's new in v3:**
- **Deployment profiles** replace the flat tier model — named profiles with distinct retrieval architectures
- **pgvector tier** — PostgreSQL-native ANN with SQL WHERE metadata filtering (Governance Native profile)
- **Guided installer** — Q/A + host probes → profile recommendation → compose + env + manifest generation
- **Profile benchmarks** — per-profile retrieval latency, recall, and throughput data
- **Deployment manifest** — `mnemos_profile.yaml` as durable deployment artifact
- **Governance layer (MemArchitect Waves 1-3)** — per-candidate policy pipeline (veto, freshness decay, trust/utility modifiers), cross-candidate entity-slot contradiction detection, and reflect-path reinforcement; advisory and enforced read path modes; default is `off` pending advisory benchmarking
- **Memory Over Maps lane (Phases 1–5)** — source-grounded artifact lineage, deterministic candidate narrowing, on-demand derived views, deterministic cache + invalidation with dry-run parity, and bounded semantic reflect benchmark pack
- **Operator playbook** — single operational runbook for deploy/promote/rollback/incident execution (`docs/mnemos_operator_playbook.md`)

MNEMOS also ships with a **Boundary SDK** (Python client library) and a suite of **operational tools** (health audit, contract evolution, onboarding, CI gates, and staged cutover) — making it a complete platform that can be deployed with a single `python -m installer`.

Operationally, the current architecture posture is: fast retrieval substrate + governed memory controls + source-grounded, bounded, on-demand synthesis.

---

## 3. Architecture

MNEMOS is organised as a layered stack with a pluggable retrieval tier selected by **deployment profile**:

```
┌────────────────────────────────────────────────────────────┐
│                       REST API (:8700)                      │
│   /index   /search   /engrams   /audit   /stats            │
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
| **Qdrant** | Core Memory Appliance | BAAI/bge-base-en-v1.5 (768-dim, CUDA) | Fast semantic ANN, HNSW index, payload filtering, horizontal scaling |
| **pgvector** | Governance Native | BAAI/bge-base-en-v1.5 (768-dim, CUDA) | ANN + SQL metadata filtering in one query, single-database deployment |
| **PostgreSQL FTS** | Hybrid mode (Core/Governance) | n/a (lexical lane) | Exact-term/title/acronym retrieval via full-text lexical matching |
| **Cross-Encoder** | Precision Lane (optional) | BAAI/bge-reranker-base | Dense reranking for long-context and technical text via stateless cross-encoder |

*(Note: ColBERT late-interaction has been moved to an experimental/research appendix and is no longer a standard deployment profile.)*

**Why Qdrant** (Core Memory Appliance): Standalone service with its own HNSW index, snapshotting, replication, and sharding. Supports concurrent reads and writes without single-process bottlenecks, payload-based filtering without post-filtering, and survives independently of the MNEMOS process.

**Why pgvector** (Governance Native): Vectors live inside the same PostgreSQL instance as the forensic ledger. ANN retrieval can be combined with SQL `WHERE` clauses on tenant, provenance, department, security markings, or any relational metadata — in a single query. This eliminates the need for a separate vector service in governance-heavy deployments.

**Hybrid fusion**: In Gate C hybrid mode, MNEMOS merges lexical (PostgreSQL FTS) and semantic candidates with deterministic normalization and weighted fusion. Four fusion policies are available:

| Policy | Lexical | Semantic | Engine |
|---|---|---|---|
| `semantic_dominant` | 0.25 | 0.75 | Python-side rank fusion |
| `balanced` | 0.50 | 0.50 | Python-side rank fusion |
| `lexical_dominant` | 0.75 | 0.25 | Python-side rank fusion |
| `qdrant_rrf` | — | — | Qdrant server-side Reciprocal Rank Fusion via `prefetch` (v1.17+) |

The `qdrant_rrf` policy delegates fusion to Qdrant's built-in RRF engine, combining a dense vector prefetch with a full-text payload prefetch in a single `query_points()` call — eliminating the second network round-trip. A full-text index on the `content` payload field is created automatically during tier initialization. If unavailable, the router falls back to Python-side fusion transparently. Optional explain output returns component scores and source attribution per hit.

**Relevance feedback**: MNEMOS can feed governance `reflect_path` labels (Used / Ignored) back into retrieval via Qdrant's `discover_points()` API. When enabled, previously-used engrams become positive exemplars and previously-ignored engrams become negative exemplars, biasing future queries toward results that have demonstrated utility. The adapter maintains a TTL-bounded exemplar cache (1.6M writes/sec, 100% cache hit rate after warmup). This feature is opt-in (`relevance_feedback.enabled: true` in `rerank_policy.yaml`).

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

**Configuration (environment variables):**

| Variable | Default | Description |
|---|---|---|
| `MNEMOS_GOVERNANCE_MODE` | `off` | Default mode for all search requests |
| `MNEMOS_GOVERNANCE_MIN_SCORE` | `0.0` | Veto threshold (0.0 disables score-floor veto) |
| `MNEMOS_GOVERNANCE_FRESHNESS_HALF_LIFE` | `180.0` | Freshness decay half-life in days |

The governance mode can also be overridden per-request via the `governance` parameter on `POST /v1/mnemos/search`. The `explain_governance: true` parameter returns full modifier breakdowns and conflict state per result.

**Reflect path (Wave 3):**

After retrieval and governance evaluation, the reflect path closes the feedback loop. When the calling application sends back the generated answer alongside the candidate set (`POST /v1/mnemos/governance/reflect`), the `UsageDetector` assigns each memory a usage label:

| Label | Signal |
|---|---|
| `USED` | Present in `cited_ids`, or word-overlap with answer ≥ threshold (default 15%) |
| `IGNORED` | No overlap signal and not cited |
| `CONTRADICTED` | Was a contradiction loser in the read-path decision |
| `VETOED` | Failed a policy veto in the read path |

Reinforcement is then applied in-place to each memory's `GovernanceMeta`:

| Label | `utility_score` | `trust_score` | `stability` |
|---|---|---|---|
| `USED` | +0.05 | +0.02 | +0.02 |
| `IGNORED` | −0.01 | — | — |
| `CONTRADICTED` | −0.03 | −0.02 | — |
| `VETOED` / `UNKNOWN` | — | — | — |

All deltas are clamped to [0.0, 1.0]. The response includes `utility_deltas` and `trust_deltas` per memory for caller inspection. Persistence of governance score updates is caller-owned; the reflect endpoint is stateless with respect to backend score mutation. This is separate from derived-view caching in Memory Over Maps, which stores reproducible read artifacts and invalidates them deterministically without persisting governance mutations.

**Validation Evidence:**

The governance behavioral claims are backed by **Governance Validation Pack v1**, a formal proof artifact (`benchmarks/TEMP/Governance_Validation_Pack_v1.md`). The pack consists of 10 named scenarios, each mapping a specific failure mode to a deterministic, in-process test. The following guarantees are proven, not asserted:

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

All runners support `dry_run=True` (compute report, mutate nothing). No physical deletion. No irreversible consolidation. `Governor.run_hygiene()` is the single entry point; hygiene counters are reported via `GET /v1/mnemos/governance/stats`.

**Per-tenant policy profiles:**

`GovernancePolicyProfile` allows per-tenant tuning of read-path thresholds, reflect-path precision, and all reinforcement deltas without restarting the service. Profiles are loaded from `MNEMOS_GOVERNANCE_POLICY_PROFILES_JSON`. The `"default"` profile always exists; additional profiles are selected per-request via `governance_profile` on `POST /v1/mnemos/search` and `POST /v1/mnemos/governance/reflect`.

**Validated gaps (tested in Wave 4):**

Long-horizon calendar-based decay, offline contradiction sweep coverage, and stale-state lifecycle transitions were documented as untested in Validation Pack v1 and are now covered by 61 Wave 4 tests. Remaining open gaps: enforced-mode drift divergence, trust recovery after contradiction penalty, and concurrent reflect cycle safety — scheduled for Phase 2 (persistence) and beyond.

---

### 4.6 Memory Over Maps (Phase-Gated)

Memory Over Maps is now an implemented architecture lane in MNEMOS and has been advanced only through benchmark-gated phase exits.

| Phase | Capability | Gate Status (March 30, 2026) |
|---|---|---|
| 1 | Source-grounded lineage contract + audit hooks | PASS |
| 2 | Deterministic bounded candidate envelope | PASS |
| 3 | On-demand derived views (evidence, contradiction, preference, timeline) | PASS |
| 4 | Derived-view cache + invalidation + dry-run traces | PASS |
| 5 | Bounded semantic reflect evolution scenarios | PASS |

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

## 5. API Contract

MNEMOS follows the **MFS Contract Pattern**: every response includes `contract_version`, `status`, `source`, and `error` fields, ensuring the consuming application can always determine the health and trustworthiness of the data it receives.

### Contract (service/contract.json)

```json
{
  "service_name": "mnemos-service",
  "contract_version": "v1",
  "endpoint": "/v1/mnemos/capabilities",
  "required_fields": {
    "contract_version": "str",
    "status": "str",
    "source": "str",
    "generated_at": "str",
    "feature": "str",
    "profile": "str",
    "supports": "list",
    "tiers": "list",
    "degraded_components": "list",
    "error": "nullable_str"
  },
  "allowed_status": ["healthy", "degraded", "unavailable"]
}
```

The `profile` field reports the active deployment profile. `tiers` lists the currently active retrieval backends. `degraded_components` lists any backends or subsystems that have failed health checks, enabling consumers to understand exactly what is and isn't working.

### Core Endpoints

```
GET    /health                      — Container health check
GET    /v1/mnemos/capabilities      — Feature discovery, active profile, backend status
POST   /v1/mnemos/index             — Ingest documents → engrams
POST   /v1/mnemos/search            — Query across active backends
GET    /v1/mnemos/engrams/{id}      — Retrieve a specific engram
DELETE /v1/mnemos/engrams/{id}      — Remove from all backends
GET    /v1/mnemos/audit             — Query the forensic ledger
GET    /v1/mnemos/stats             — Profile info, backend sizes, compression ratios
GET    /v1/mnemos/governance/stats  — Governance aggregate stats (veto rate, suppression rate, contradiction counts)
```

### Example: /capabilities Response

```json
{
  "contract_version": "v1",
  "status": "healthy",
  "source": "mnemos-service",
  "generated_at": "2026-03-28T04:15:22Z",
  "feature": "mnemos_memory",
  "profile": "governance_native",
  "supports": ["index", "search", "engrams", "audit", "stats"],
  "tiers": ["pgvector"],
  "compression": { "enabled": true, "bits": 4 },
  "gpu_device": "cuda",
  "degraded_components": [],
  "error": null
}
```

A consumer can always determine: which profile is running, which backends are active, whether any components are degraded, and the compression configuration — without inspecting env vars or deployment files.

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
  "filters": { "metadata.department": "finance" }
}
```

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

**Rule**: Consumer apps should always use the SDK. Direct HTTP calls bypass readiness, retry, and degradation handling.

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
python tools/mnemos_ci_gates.py --run-health-audit --run-container-build
```

Runs contract validation, health audit, and container build as CI pipeline steps. Includes a GitHub Actions workflow template (`.github/workflows/mnemos-gates.yml`).

### 7.5 Cutover Scaffold

```bash
python tools/mnemos_cutover_scaffold.py --app my-app
```

Generates a staged rollout manifest (shadow → canary 5/25/50% → full) for apps migrating from another memory backend to MNEMOS, with health gates and rollback paths.

---

## 8. Deployment Profiles

MNEMOS ships with named deployment profiles that determine the retrieval backend, container topology, and operational posture. The guided installer (`python -m installer`) recommends a profile based on use case, priorities, and host capabilities.

### Profile A: Core Memory Appliance *(default)*

**Best for:** Semantic memory, agent systems, general-purpose RAG.

| Component | Service | Container |
|---|---|---|
| Vector store | Qdrant (HNSW, CUDA embeddings) | `mnemos-qdrant` |
| Audit ledger | PostgreSQL | `mnemos-postgres` |
| Service | MNEMOS (nvidia runtime) | `mnemos-service` |

3 containers. Qdrant provides fast semantic ANN with payload filtering. Recommended when retrieval is primarily semantic and the corpus exceeds 100K documents.

### Profile B: Governance Native

**Best for:** Provenance-heavy, metadata-filtered, compliance-aware retrieval.

| Component | Service | Container |
|---|---|---|
| Vector store | pgvector (inside PostgreSQL) | `mnemos-postgres` (shared) |
| Audit ledger | PostgreSQL | `mnemos-postgres` (shared) |
| Service | MNEMOS (nvidia runtime) | `mnemos-service` |

2 containers. Vectors and audit share one Postgres instance. ANN retrieval can be combined with SQL `WHERE` clauses on tenant, provenance, or security markings — in a single query. Recommended when metadata filtering matters more than raw ANN throughput.

### Profile C: Custom Manual

**Best for:** Advanced operators, multi-tier setups, experimentation.

No compose generation — the operator provides their own configuration. The installer writes `.env.mnemos` only. Supports any combination of backends including Cross-Encoder reranking.

### Hybrid Retrieval Mode (Gate C) *(inside existing profiles)*

Hybrid retrieval is not a separate deployment profile. It is a retrieval mode available within Core and Governance deployments using lexical + semantic fusion. As of the March 29, 2026 real-corpus benchmark decision, hybrid is supported for targeted evaluation but is not the global default.

---

## 9. Deployment

The installer generates a profile-specific `docker-compose.generated.yml` and `.env.mnemos`. Example stacks:

### Core Memory Appliance

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
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
| Core Memory Appliance | 3 | ~2 GB | ~200 MB base | Required (CUDA) |
| Core + Cross-Encoder reranking | 3 | ~4 GB | ~400 MB base | Required (CUDA) |
| Governance Native | 2 | ~1.5 GB | ~150 MB base | Required (CUDA) |
| Governance + Cross-Encoder | 2 | ~3.5 GB | ~350 MB base | Required (CUDA, ≥8 GB VRAM) |

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
2. **GPU-native** — Embedding inference runs on CUDA by default. The service is built on `nvidia/cuda` and requires GPU hardware — CPU fallback exists for resilience, not as a primary mode.
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
│   ├── questions.py           5-question Q/A
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
│   └── mnemos_cutover_scaffold.py
├── benchmarks/                Reproducible benchmark suite
│   ├── run_memory_over_maps_benchmarks.py
├── tests/                     Unit tests
├── .github/workflows/         CI gate template
├── docs/                      Whitepaper + AI dev hand-off
├── Dockerfile                 Production container
└── docker-compose.yml         Default stack (Core Memory Appliance)
```

---

## 15. Provenance

MNEMOS was designed from the ground up as a reusable memory service. Its architecture draws on production experience operating multi-tier vector retrieval, near-lossless compression, and forensic audit logging under continuous autonomous workloads.

| Capability | MNEMOS Component |
|---|---|
| Multi-vector retrieval | Multi-Tier Retrieval Engine |
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
| Source-grounded selective synthesis | Memory Over Maps lane (mnemos/memory_over_maps/) |

What remains is a **pure infrastructure service** — a reusable, tooling-complete foundation for any application that needs intelligent, compressed, auditable memory.
