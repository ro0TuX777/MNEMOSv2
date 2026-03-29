# MNEMOS â€” White Paper

**A containerised, contract-governed memory and retrieval service for AI-native applications.**

*Version 3.0 Â· March 2026*

> [!NOTE]
> **As of March 30, 2026:** Benchmark conclusions in this whitepaper are date-scoped to the current measured runs.
> For full methodology, raw artifacts, and latest updates, see `docs/benchmark.md`.
> **Governance layer (MemArchitect Waves 1 & 2) is implemented** — per-candidate policy pipeline, entity-slot contradiction detection, and advisory/enforced read path modes. Default mode is `off` (conservative); advisory benchmarking against real corpus is the next step before enforced-mode promotion.

---

## 1. Problem Statement

Every AI application that persists and retrieves knowledge must solve the same set of problems:

- **Storage**: Where do embeddings live, and how do they scale?
- **Retrieval**: How do you find the right information across thousands of documents with high precision?
- **Compression**: How do you keep memory footprint manageable as the index grows?
- **Enrichment**: How do you go beyond raw vectors to attach semantic meaning, provenance, and relationships?
- **Auditability**: How do you trace what was stored, retrieved, modified, and when?
- **Integration**: How do you wire memory into an application without tight coupling or fragile glue code?

Today, each project re-implements these capabilities from scratch â€” writing custom embedding pipelines, bolting on vector databases, and building ad-hoc search logic. The result is fragile, inconsistent, and impossible to reuse across projects.

## 2. Solution: MNEMOS

**MNEMOS** (Multi-tier Neuro-tagged Engram Memory with Optimal Near-lossless Index Compression) is a GPU-accelerated, production-grade memory service for AI-native applications. It deploys via **named deployment profiles** â€” each profile defines a retrieval backend, container topology, and operational posture:

- **Core Memory Appliance** â€” Qdrant + PostgreSQL + MNEMOS (3 containers). Semantic ANN with payload filtering.
- **Governance Native** â€” PostgreSQL/pgvector + MNEMOS (2 containers). ANN + SQL metadata filtering in one query.
- **Custom Manual** â€” Operator-defined configuration for advanced multi-backend setups.
- **Hybrid Retrieval Mode (Gate C)** - optional lexical + semantic fusion mode inside existing profiles (not a separate profile).

A guided Python installer (`python -m installer`) probes the host, asks 5 questions, recommends a profile, and generates all deployment files. The service exposes a versioned REST API governed by an MFS contract.

MNEMOS is **application-agnostic** â€” it knows nothing about the domain of the consuming application. It stores, enriches, compresses, retrieves, and audits knowledge. Thatâ€™s it.

**What's new in v3:**
- **Deployment profiles** replace the flat tier model â€” named profiles with distinct retrieval architectures
- **pgvector tier** â€” PostgreSQL-native ANN with SQL WHERE metadata filtering (Governance Native profile)
- **Guided installer** â€” Q/A + host probes â†’ profile recommendation â†’ compose + env + manifest generation
- **Profile benchmarks** â€” per-profile retrieval latency, recall, and throughput data
- **Deployment manifest** â€” `mnemos_profile.yaml` as durable deployment artifact
- **Governance layer (MemArchitect Waves 1 & 2)** â€” per-candidate policy pipeline (veto, freshness decay, trust/utility modifiers) and cross-candidate entity-slot contradiction detection; advisory and enforced read path modes; default is `off` pending advisory benchmarking

MNEMOS also ships with a **Boundary SDK** (Python client library) and a suite of **operational tools** (health audit, contract evolution, onboarding, CI gates, and staged cutover) â€” making it a complete platform that can be deployed with a single `python -m installer`.

---

## 3. Architecture

MNEMOS is organised as a layered stack with a pluggable retrieval tier selected by **deployment profile**:

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                       REST API (:8700)                      â”‚
â”‚   /index   /search   /engrams   /audit   /stats            â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                  Engram Enrichment Layer                    â”‚
â”‚   neuro-tags Â· provenance Â· confidence Â· relationship      â”‚
â”‚   edges Â· extensible metadata                              â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚            Retrieval (profile-selected)                     â”‚
â”‚                                                             â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚  Core Memory Appliance   â”‚  â”‚   Governance Native    â”‚  â”‚
â”‚  â”‚  Qdrant (HNSW, CUDA)     â”‚  â”‚   pgvector (Postgres)  â”‚  â”‚
â”‚  â”‚  + optional ColBERT       â”‚  â”‚   + optional ColBERT   â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚             TurboQuant Compression Layer                    â”‚
â”‚   4-bit quantised storage Â· 8Ã— raw compression             â”‚
â”‚   Near-optimal distortion (arXiv:2504.19874)               â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚     Embedding Engine (GPU-accelerated, swappable)          â”‚
â”‚   all-MiniLM-L6-v2 â”‚ ColBERTv2.0 â”‚ custom model           â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚          Forensic Ledger (PostgreSQL audit trail)           â”‚
â”‚   Immutable Â· every operation logged Â· replayable          â”‚
â”‚   tsvector FTS Â· connection pooling Â· SQLite fallback      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
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
| `confidence` | `float` | Quality signal (0.0â€“1.0) used for ranking |
| `created_at` | `datetime` | Ingestion timestamp |
| `metadata` | `object` | Extensible application-specific data |
| `edges` | `string[]` | IDs of related engrams (knowledge graph) |
| `_governance` | `GovernanceMeta` | Optional governance metadata (lifecycle state, trust/utility/freshness scores, conflict state, lineage); `null` for legacy engrams |

The Engram is **domain-agnostic** â€” the consuming application decides what `neuro_tags` mean, what `source` URIs look like, and what goes in `metadata`. MNEMOS provides the schema, storage, indexing, and retrieval.

### 4.2 Retrieval Backends

MNEMOS supports multiple retrieval backends, selected by deployment profile. All embedding inference is **GPU-accelerated** via CUDA.

| Backend | Profile | Embedding Model | Strength |
|---|---|---|---|
| **Qdrant** | Core Memory Appliance | all-MiniLM-L6-v2 (384-dim, CUDA) | Fast semantic ANN, HNSW index, payload filtering, horizontal scaling |
| **pgvector** | Governance Native | all-MiniLM-L6-v2 (384-dim, CUDA) | ANN + SQL metadata filtering in one query, single-database deployment |
| **PostgreSQL FTS** | Hybrid mode (Core/Governance) | n/a (lexical lane) | Exact-term/title/acronym retrieval via full-text lexical matching |
| **ColBERT** | Experimental (optional) | colbert-ir/colbertv2.0 (128-dim, CUDA) | Reranking path under evaluation; not production-default |

**Why Qdrant** (Core Memory Appliance): Standalone service with its own HNSW index, snapshotting, replication, and sharding. Supports concurrent reads and writes without single-process bottlenecks, payload-based filtering without post-filtering, and survives independently of the MNEMOS process.

**Why pgvector** (Governance Native): Vectors live inside the same PostgreSQL instance as the forensic ledger. ANN retrieval can be combined with SQL `WHERE` clauses on tenant, provenance, department, security markings, or any relational metadata â€” in a single query. This eliminates the need for a separate vector service in governance-heavy deployments.

**Hybrid fusion**: In Gate C hybrid mode, MNEMOS merges lexical (PostgreSQL FTS) and semantic candidates with deterministic normalization and weighted fusion (`semantic_dominant`, `balanced`, `lexical_dominant`). Optional explain output returns component scores and source attribution per hit.

### 4.3 TurboQuant Compression

All stored embeddings are compressed via **TurboQuant** (arXiv:2504.19874), a near-optimal vector quantisation algorithm.

**How it works:**
1. Input vectors are randomly rotated so each coordinate follows a Beta distribution
2. Optimal Lloyd-Max scalar quantisers compress each coordinate to 1â€“4 bits
3. For inner-product operations, a 1-bit QJL residual preserves unbiased estimation

**Performance at 4-bit (default):**

| Metric | Value |
|---|---|
| Storage compression | 8Ã— raw bytes, 8.3â€“8.4Ã— file (.npz) |
| Recall@10 | 84.2% (128-dim), 84.8% (384-dim) |
| MSE | 7.3Ã—10â»âµ (128-dim), 2.5Ã—10â»âµ (384-dim) â€” well below 0.009 bound |
| Cosine fidelity | 0.995 avg (practically indistinguishable from float32) |
| Encode throughput | 67Kâ€“175K docs/sec (CPU, NumPy) |

> *All values measured empirically on a 10,000-document synthetic corpus. Benchmark source: `benchmarks/run_benchmarks.py`. Results: `benchmarks/results.json`.*

#### Compression & Fidelity Across Bit-Widths

| Bits | MSE (128d) | MSE (384d) | Cosine Sim | Raw Ratio | File Ratio |
|---|---|---|---|---|---|
| 1-bit | 0.0028 | 0.0009 | 0.799 | 32Ã— | 31Ã— |
| 2-bit | 0.0009 | 0.0003 | 0.940 | 16Ã— | 16Ã— |
| 3-bit | 0.0003 | 0.00009 | 0.983 | 8Ã— | 11Ã— |
| **4-bit** | **0.00007** | **0.00003** | **0.995** | **8Ã—** | **8.3Ã—** |

#### Recall@10 (Nearest-Neighbour Fidelity)

Measured on 10K corpus / 100 queries â€” fraction of true float32 top-10 neighbours preserved after quantisation:

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
| 10K documents | 5.1 MB | 0.7 MB | 7.5Ã— |
| 100K documents | 51.2 MB | 6.8 MB | 7.5Ã— |
| 1M documents | 512.0 MB | 68.0 MB | 7.5Ã— |
| 10M documents | 5,120 MB | 680 MB | 7.5Ã— |

**Why it matters**: Without compression, a 1M-document index at 128 dimensions consumes ~512 MB in float32. With 4-bit TurboQuant, that drops to ~68 MB â€” enabling deployment on memory-constrained edge devices, smaller cloud instances, and faster cold starts.

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

#### ColBERT Rerank (Track 2)

Reference rerun: `20260329_131856_profile_benchmarks.json`

Observed on this workload:
- Reranking reduced MRR and nDCG at depths 20/50/100 for both backends.
- Recommended depth is currently `n/a` (no rerank by default).
- The current implementation path logged a sentence-transformers mean-pooling fallback for `colbert-ir/colbertv2.0`; final ColBERT policy should be revisited after model-path alignment.

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

**Why PostgreSQL**: SQLite's single-writer lock becomes a bottleneck under concurrent agent workloads. PostgreSQL provides connection pooling, ACID transactions, concurrent writes, and native full-text search via `tsvector` + `GIN` indexes â€” replacing FTS5 with a language-aware, ranked search engine. A SQLite fallback remains available for local development and testing.

**Use cases:**
- **Compliance** â€” demonstrate when data was ingested, accessed, or deleted
- **Debugging** â€” replay the sequence of operations that led to a retrieval failure
- **Analytics** â€” track query patterns, ingestion rates, and error trends

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

---

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
GET    /health                      â€” Container health check
GET    /v1/mnemos/capabilities      â€” Feature discovery, active profile, backend status
POST   /v1/mnemos/index             â€” Ingest documents â†’ engrams
POST   /v1/mnemos/search            â€” Query across active backends
GET    /v1/mnemos/engrams/{id}      â€” Retrieve a specific engram
DELETE /v1/mnemos/engrams/{id}      â€” Remove from all backends
GET    /v1/mnemos/audit             â€” Query the forensic ledger
GET    /v1/mnemos/stats             â€” Profile info, backend sizes, compression ratios
GET    /v1/mnemos/governance/stats  â€” Governance aggregate stats (veto rate, suppression rate, contradiction counts)
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

A consumer can always determine: which profile is running, which backends are active, whether any components are degraded, and the compression configuration â€” without inspecting env vars or deployment files.

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
    "tiers": ["qdrant", "colbert"]
  }
}
```

### Example: Searching

```json
POST /v1/mnemos/search
{
  "query": "What were the Q1 revenue figures?",
  "top_k": 10,
  "tiers": ["qdrant", "colbert"],
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

Generates a staged rollout manifest (shadow â†’ canary 5/25/50% â†’ full) for apps migrating from another memory backend to MNEMOS, with health gates and rollback paths.

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

2 containers. Vectors and audit share one Postgres instance. ANN retrieval can be combined with SQL `WHERE` clauses on tenant, provenance, or security markings â€” in a single query. Recommended when metadata filtering matters more than raw ANN throughput.

### Profile C: Custom Manual

**Best for:** Advanced operators, multi-tier setups, experimentation.

No compose generation â€” the operator provides their own configuration. The installer writes `.env.mnemos` only. Supports any combination of backends including ColBERT reranking.

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
| Core + ColBERT reranking | 3 | ~4 GB | ~400 MB base | Required (CUDA) |
| Governance Native | 2 | ~1.5 GB | ~150 MB base | Required (CUDA) |
| Governance + ColBERT | 2 | ~3.5 GB | ~350 MB base | Required (CUDA, â‰¥8 GB VRAM) |

---

## 9. Integration Workflow

A consumer application adopts MNEMOS in five steps:

```
1. Install        â†’  python -m installer
2. Configure      â†’  Review generated .env.mnemos and mnemos_profile.yaml
3. Start          â†’  docker compose -f docker-compose.generated.yml up -d --build
4. Validate       â†’  python tools/mnemos_health_audit.py
5. Wire           â†’  Import mnemos_sdk, use client.index() / client.search()
```

For apps migrating from another memory backend (Redis, Elasticsearch, FAISS):

```
6. Scaffold cutover   â†’  python tools/mnemos_cutover_scaffold.py --app <app>
7. Shadow writes      â†’  Write to both old + MNEMOS, read from old
8. Canary promotion   â†’  Route 5% â†’ 25% â†’ 50% â†’ 100% of reads to MNEMOS
9. Decommission       â†’  Remove old backend
```

---

## 10. Use Cases

The following are the highest-value scenarios where MNEMOS provides immediate benefit as a drop-in memory layer.

### 10.1 AI Agent / Copilot Platforms

**Recommended profile:** Core Memory Appliance

The most natural fit. Any system that has an LLM doing multi-step work needs persistent, searchable memory.

- **What MNEMOS provides**: Engram-enriched conversation history, tool output recall, semantic search over past interactions.
- **Why not just a raw vector DB**: Neuro-tags give semantic labels for retrieval boosting. The forensic ledger tracks what the agent remembered and when â€” critical for debugging hallucinations.
- **Example**: A coding assistant that remembers past codebases it has worked on, retrieves relevant patterns, and audits what context influenced each generation.

### 10.2 RAG-Powered Knowledge Bases

**Recommended profile:** Governance Native (compliance) or Core Memory Appliance (general)

Enterprise document search where accuracy and audit trails matter â€” legal, medical, compliance.

- **What MNEMOS provides**: Profile-matched retrieval (pgvector for governance-heavy, Qdrant for speed), optional hybrid lexical+semantic mode for exact-term sensitive workloads, and TurboQuant for scaling to millions of chunks.
- **Why it wins**: The forensic ledger gives compliance-ready logging of every query and retrieval â€” *"show me exactly what documents were retrieved for this answer and when."*
- **Example**: Internal knowledge base for a law firm â€” lawyers query it, each retrieval is logged for audit, and pgvector filters by department and security clearance.

### 10.3 IoT / Edge Deployments

**Recommended profile:** Governance Native (single-database, minimal footprint)

Devices with limited memory and storage that still need intelligent retrieval.

- **What MNEMOS provides**: TurboQuant 4-bit compression means a 1M-document index fits in ~68 MB instead of ~512 MB. Governance Native profile runs with just 2 containers.
- **Why it wins**: Most vector DBs assume cloud-scale resources. MNEMOS can run on a single Postgres instance with pgvector.
- **Example**: A smart home hub that remembers user preferences, schedules, and sensor patterns â€” compressed on-device, searchable locally without cloud dependency.

### 10.4 Multi-Agent Orchestration Systems

**Recommended profile:** Core Memory Appliance

Systems where multiple specialised agents need shared memory without stepping on each other.

- **What MNEMOS provides**: A centralised, contract-governed memory service that any agent can index to and search from via REST. The MFS contract pattern means agents can trust the response schema.
- **Why it wins**: Without shared memory, each agent re-discovers context. With MNEMOS, Agent A's research becomes Agent B's retrieval â€” and the audit trail shows who stored what.
- **Example**: A research pipeline where a "Scout" agent gathers papers, an "Analyst" agent extracts insights, and a "Writer" agent drafts reports â€” all sharing one MNEMOS instance.

### 10.5 Content / Creative Platforms

**Recommended profile:** Core Memory Appliance (semantic default), with optional hybrid mode for exact-term-sensitive workloads

Story generators, game engines, or creative tools that need long-term world memory.

- **What MNEMOS provides**: Engram edges create a knowledge graph of relationships (characters â†’ events â†’ locations). Neuro-tags categorise memory by theme. Hybrid lexical+semantic retrieval can improve exact phrase continuity checks.
- **Why it wins**: Creative tools need precise recall (*"what did character X say about Y in chapter 3?"*) â€” multi-vector retrieval is dramatically better than single-vector for this.
- **Example**: An interactive fiction engine where the story adapts based on retrieving and referencing past plot points from a compressed engram store.

### Common Thread

Any application that stores, enriches, retrieves, and audits knowledge â€” and needs production-grade infrastructure without building the plumbing from scratch. MNEMOS gives you deployment profiles tailored to your use case, GPU-accelerated retrieval, scalable audit logging, and a complete operational toolkit in one `python -m installer`.

---

## 11. Design Principles

1. **Application-agnostic** â€” The service has zero knowledge of what domain it serves. It stores vectors, enriches engrams, and answers queries. Period.
2. **GPU-native** â€” Embedding inference runs on CUDA by default. The service is built on `nvidia/cuda` and requires GPU hardware â€” CPU fallback exists for resilience, not as a primary mode.
3. **Profile-composable** â€” Named deployment profiles (Core Memory Appliance, Governance Native) determine the retrieval backend and container topology. The installer recommends, the operator confirms.
4. **Contract-governed** â€” Every API response follows a strict MFS contract schema, enabling reliable integration without tight coupling.
5. **Compression by default** â€” TurboQuant is on at 4-bit out of the box. Storage scales sublinearly with document count.
6. **Audit by default** â€” Every mutation is logged immutably to PostgreSQL. Compliance is a feature, not an afterthought.
7. **Graceful degradation** â€” If a backend goes unhealthy, remaining backends continue serving. Status is always reported honestly via the contract.
8. **Process isolation** â€” Each infrastructure component (vector store, audit ledger, service) runs in its own container with independent health checks, volumes, and lifecycle.
9. **SDK-first integration** â€” Consumer apps use the boundary SDK, never raw HTTP. This ensures readiness, retry, and degradation are handled consistently.
10. **Tooling-complete** â€” Health audit, contract evolution, onboarding, CI gates, and cutover are included â€” not left as an exercise for the adopter.
11. **Governance by design** â€” The governance layer is built into the read path, not bolted on. Memory quality scores, freshness decay, contradiction detection, and suppression policies are first-class concerns evaluated at query time. Advisory mode before enforced mode; benchmark evidence drives promotion.

---

## 12. Deployment Manifest (mnemos_profile.yaml)

The guided installer generates a `mnemos_profile.yaml` file alongside the compose and env files. This manifest is a **durable deployment artifact** â€” the single source of truth for what was installed, why, and how.

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
| **Reproducibility** | Re-run the installer on a new host with the same answers â†’ identical deployment |
| **Supportability** | Attach the manifest to any support request â€” it captures what was deployed and why |
| **Upgrade input** | Future installer versions can read the manifest to recommend migration paths |
| **CI validation** | CI gates can validate that the running service matches the declared profile |
| **Audit trail** | The manifest records the host facts at install time (GPU, RAM, Docker version) |

The manifest is not consumed by the MNEMOS runtime â€” it is a static record for operators, support, and tooling. The runtime reads `MNEMOS_PROFILE` from the environment.

---

## 13. Profile Migration

Once profiles are deployed, operators may need to migrate between them. MNEMOS defines migration rules for the two primary profiles.

### Core Memory Appliance â†’ Governance Native

**When:** An operator wants to simplify operations (3 â†’ 2 containers) or needs SQL-level metadata filtering.

| Step | Action |
|---|---|
| 1 | Run `python -m installer --profile governance_native` to generate new compose/env |
| 2 | Enable pgvector extension: `CREATE EXTENSION IF NOT EXISTS vector` in Postgres |
| 3 | Re-index engrams from Qdrant to pgvector using the MNEMOS search/index API |
| 4 | Validate: `python tools/mnemos_health_audit.py` confirms pgvector is healthy |
| 5 | Decommission Qdrant container |
| 6 | Update `mnemos_profile.yaml` with new profile and migration timestamp |

**Metadata assumptions that change:** pgvector stores metadata as JSONB columns â€” metadata that was previously Qdrant payload becomes SQL-queryable. Review any metadata schemas for SQL compatibility.

### Governance Native â†’ Core Memory Appliance

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

Both migration paths are non-destructive â€” the source backend is not modified during migration. If the new profile fails health validation:

1. Revert `docker-compose.generated.yml` to the previous version
2. Revert `.env.mnemos` to the previous `MNEMOS_PROFILE`
3. Restart: `docker compose -f docker-compose.generated.yml up -d`
4. The original backend is still intact and serving

---

## 14. Repository Structure

```
MNEMOS/
â”œâ”€â”€ mnemos/                    Core library
â”‚   â”œâ”€â”€ compression/           TurboQuant (arXiv:2504.19874)
â”‚   â”œâ”€â”€ engram/                Engram model and enrichment
â”‚   â”œâ”€â”€ retrieval/             Multi-backend retrieval + fusion
â”‚   â”‚   â”œâ”€â”€ qdrant_tier.py     Qdrant backend (Core Memory Appliance)
â”‚   â”‚   â”œâ”€â”€ pgvector_tier.py   pgvector backend (Governance Native)
â”‚   â”‚   â”œâ”€â”€ colbert_tier.py    ColBERT reranker (optional)
â”‚   â”‚   â”œâ”€â”€ fusion.py          Multi-backend fusion engine
â”‚   â”‚   â””â”€â”€ base.py            BaseRetriever interface
â”‚   â””â”€â”€ audit/                 Forensic ledger
â”‚   â””â”€â”€ governance/            Governance layer (MemArchitect)
â”‚       â”œâ”€â”€ governor.py        Entry point; wraps registry + read path
â”‚       â”œâ”€â”€ read_path.py       Advisory / enforced read path; 3-tuple return
â”‚       â”œâ”€â”€ policy_registry.py Per-candidate policy pipeline with short-circuit
â”‚       â”œâ”€â”€ policies/          RelevanceVetoPolicy, UtilityPolicy, ContradictionPolicy
â”‚       â””â”€â”€ models/            GovernanceMeta, GovernanceDecision, ContradictionRecord
â”œâ”€â”€ mnemos_sdk/                Boundary adapter SDK (client library)
â”‚   â”œâ”€â”€ client.py              MnemosClient with typed methods
â”‚   â””â”€â”€ config.py              MnemosConfig.from_env()
â”œâ”€â”€ service/                   Flask REST API + MFS contract
â”œâ”€â”€ installer/                 Guided deployment installer
â”‚   â”œâ”€â”€ __main__.py            Entry point (python -m installer)
â”‚   â”œâ”€â”€ questions.py           5-question Q/A
â”‚   â”œâ”€â”€ probes.py              Host capability detection
â”‚   â”œâ”€â”€ profiles.py            Profile definitions
â”‚   â”œâ”€â”€ recommend.py           Decision tree recommendation
â”‚   â”œâ”€â”€ render.py              Compose + env + manifest generator
â”‚   â””â”€â”€ templates/             Per-profile compose templates
â”œâ”€â”€ tools/                     Operational tooling
â”‚   â”œâ”€â”€ mnemos_health_audit.py
â”‚   â”œâ”€â”€ contract_diff.py
â”‚   â”œâ”€â”€ mnemos_onboard.py
â”‚   â”œâ”€â”€ mnemos_ci_gates.py
â”‚   â””â”€â”€ mnemos_cutover_scaffold.py
â”œâ”€â”€ benchmarks/                Reproducible benchmark suite
â”œâ”€â”€ tests/                     Unit tests
â”œâ”€â”€ .github/workflows/         CI gate template
â”œâ”€â”€ docs/                      Whitepaper + AI dev hand-off
â”œâ”€â”€ Dockerfile                 Production container
â””â”€â”€ docker-compose.yml         Default stack (Core Memory Appliance)
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

What remains is a **pure infrastructure service** â€” a reusable, tooling-complete foundation for any application that needs intelligent, compressed, auditable memory.


