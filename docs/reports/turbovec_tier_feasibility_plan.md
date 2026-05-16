# MNEMOS Phase TQ-1: TurbovecTier Feasibility Benchmark

## Objective
Implement an experimental `TurbovecTier` that can ingest and search MNEMOS embeddings using `turbovec` while preserving Engram identity, source provenance, metadata filtering, and benchmark comparability against Qdrant.

## Non-goals
- Do not remove Qdrant.
- Do not change Core Memory Appliance default.
- Do not replace Hybrid RRF globally.
- Do not change governance scoring.
- Do not rely on turbovec for metadata truth.
- Do not promote to production profile in this sprint.

## Architecture

### 1. Dense Index Wrapper
A thin wrapper over `IdMapIndex` from `turbovec`.
- **turbovec requires uint64 IDs**, so MNEMOS needs a durable mapping between UUIDs and integers.

### 2. Metadata Sidecar
Uses SQLite to provide what turbovec lacks: source lookup, metadata filters, and lexical search.

Tables:
```sql
CREATE TABLE engram_metadata (
  engram_uuid TEXT PRIMARY KEY,
  turbovec_id INTEGER UNIQUE NOT NULL,
  source_uri TEXT,
  content TEXT,
  metadata_json TEXT,
  governance_json TEXT,
  content_hash TEXT,
  created_at TEXT,
  updated_at TEXT,
  deleted INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE engram_fts
USING fts5(engram_uuid UNINDEXED, content, source_uri, metadata_text);
```

### 3. Filtering Strategy
Start with post-filtering:
- `dense_top_k = top_k * oversample_factor`
- filter results in SQLite metadata
- return top_k survivors

Default `oversample_factor = 5`, `max_dense_candidates = 500`.

### 4. Hybrid Search
Replicate current hybrid behavior locally:
- Dense candidates from `turbovec`
- Lexical candidates from SQLite `FTS5`
- Normalize scores and fuse with Python-side RRF
- Apply governance/read-path filters

### 5. Persistence
Persist three things together in the profile directory:
1. `index.tvim`
2. `metadata.sqlite`
3. `manifest.json`

## Benchmark Gates
Compare against Qdrant on the same corpus measuring:
- Ingest docs/sec
- Index size on disk
- RAM usage
- Search p50/p95/p99
- Recall@10 versus float baseline or Qdrant
- Source attribution & metadata filter correctness
- Hybrid quality
- Delete/update correctness
- Cold load time & save/load integrity

## Expected Outcome
Targeting `TURBOVEC_DENSE_ONLY_SHADOW` or `TURBOVEC_PORTABLE_PROFILE_CANDIDATE`. This positions MNEMOS to offer a containerized service (Qdrant) or a compact local appliance (Turbovec + SQLite).
