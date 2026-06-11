# Shadow-Validation Replay Spec — Matryoshka Migration Tool

Date: June 10, 2026
Workstreams: W6 (Performance Economics) + W7 (SLO + Reliability) + W1 (Gate Enforcement)
Parent spec: `docs/reports/phase7_turbomatryoshka_spec.md`
Status: Spec — implementation blocked on MemGraphRAG branch landing

---

## 1. Scope

`tools/mnemos_matryoshka_migrate.py` re-embeds the corpus from BGE-base-en-v1.5
(768-dim, single vector) to nomic-embed-text-v1.5 (MRL, named vectors
`dense_64` + `dense_768`) into a **new collection**, then proves via replay
that the new embedding space is safe to promote. The migration is the
highest-semantic-drift operation in the project's history: aggregate gate
metrics (MRR/nDCG) can pass while individual query classes regress.

Non-destructive throughout: the source collection is never modified
(whitepaper §13 migration pattern).

## 2. Requirement 1 — Prefix injection (correctness precondition)

Nomic v1.5 requires task prefixes; omitting them silently degrades quality.

- Ingestion path: every document embedded as `search_document: {content}`.
- Query path: every query embedded as `search_query: {query}`.
- **Enforcement, not convention:** the embedding-engine wrapper for Nomic
  owns prefixing internally — callers pass raw text. The migration tool
  verifies at startup by embedding a sentinel pair with and without prefix
  and asserting the vectors differ; a wrapper that ignores prefixes fails
  fast before any re-embed work.
- The migration manifest records `prefix_scheme: nomic_v1.5_task_prefixes`
  so the audit trail captures how vectors were produced.

## 3. Requirement 2 — Divergence-aware replay (semantic drift analysis)

### Anchor query set

`benchmarks/truthsets/anchor_queries_v1.json` (new, versioned):

- Seeded from the existing real-corpus benchmark queries (gate_b/gate_c
  truthsets) plus per-tenant "golden queries" operators register.
- Each anchor: `{query_id, query, tenant, expected_ids?: [...], class}` where
  `class ∈ {semantic, exact_term, acronym, long_context}` — drift rarely
  distributes evenly across query classes; per-class reporting is mandatory.

### Replay procedure

For each anchor query, run top-10 against:
- **A (baseline):** BGE collection, current production path.
- **B (candidate):** Nomic collection, prefetch64 + rescore768 full plan.

Compute per query:
- `jaccard@10(A, B)` — result-set overlap.
- `rank_displacement` — mean |rank_A − rank_B| over the intersection.
- When `expected_ids` exist: `recall_A`, `recall_B` → `regression = recall_B < recall_A`.

### Promotion matrix

| Jaccard@10 (median, per class) | Regression on labeled anchors | Verdict |
|---|---|---|
| ≥ 0.6 | none | **PROCEED** — drift within expected re-embedding variance |
| < 0.6 | none | **REVIEW** — divergence without measured harm; sample diverged queries for human inspection before promotion |
| any | any labeled regression | **BLOCK** — even if Phase 7 latency criteria pass |

The BLOCK rule is absolute: latency wins never override relevance
regressions (consistent with the Phase 7 relevance-parity criterion).
Verdicts are per-class; one regressing class blocks promotion for all.

Artifacts: `benchmarks/outputs/raw/matryoshka_shadow_<ts>_raw.json` + report
+ decision markdown, in the established phase-gate format, including the
full per-query divergence table for tenant-level inspection.

## 4. Requirement 3 — Derived-view cache invalidation

Precise framing of the risk: cache keys are built from
`view_type + query_fingerprint + artifact_ids (+ governance hash)`. A new
embedding space changes *which artifacts are retrieved*; a changed artifact
set produces a different key (miss → fresh compute), so stale-serve risk is
narrow — but BGE-era entries become unreachable garbage, and the same
query+artifact key is ambiguous across embedding generations within a TTL
window (currently 3600s).

Mechanism, in order of preference:

1. **Key versioning (correct fix):** add `embed_model_version` to
   `build_cache_key(...)` inputs. Cutover bumps the version; old entries
   become unreachable by construction and expire via TTL. Survives restarts
   and partial rollbacks (rolling back the model rolls back the key space).
2. **Purge (v1 fallback):** the migration tool calls a cache flush at
   cutover. Simpler, but a rollback within the TTL window reintroduces the
   ambiguity; acceptable only because the TTL is short.

The tool performs whichever is implemented and records it in the manifest.
Phase 4's stale-cache-survival metric (`≤ 0.0005` SLO) is re-measured in the
first post-cutover benchmark run.

## 5. Requirement 4 — Resumable checkpointing

`migration_state.json` (alongside the generated collection, atomic
write-rename on every batch):

```json
{
  "migration_id": "matryoshka_20260610_...",
  "source_collection": "engrams_bge",
  "target_collection": "engrams_nomic_mrl",
  "embed_model": "nomic-embed-text-v1.5",
  "prefix_scheme": "nomic_v1.5_task_prefixes",
  "batch_size": 256,
  "last_committed_offset": 18432,
  "total_engrams": 103211,
  "content_checksum_mode": "sha256(content)",
  "failed_ids": [],
  "started_at": "...",
  "updated_at": "..."
}
```

- Resume = scroll from `last_committed_offset`; engram IDs are upserted, so
  re-processing a partial batch is idempotent.
- `failed_ids` collects per-engram embed failures for a targeted retry pass
  rather than aborting a 100k-document run.
- Completion writes the migration manifest (counts, checksums, model
  versions, duration, GPU device) — the §13 audit artifact.

## 6. Edge preservation (MemGraphRAG interaction)

Engram `edges` are ID-references, not vectors — re-embedding does not break
them, but the migration tool must copy them (and all payload fields)
verbatim to the target collection. Once the MemGraphRAG branch lands, the
shadow replay adds a graph-path check: anchor queries that exercise
graph-expanded retrieval must replay against both collections so edge
traversal is validated in the new space, not just dense ranking.

## 7. Execution order

1. MemGraphRAG branch lands (router wiring stabilises).
2. Implement cache key versioning (small, independent).
3. Implement `mnemos_matryoshka_migrate.py` against this spec.
4. Shadow replay on the real corpus → promotion matrix verdict.
5. Phase 7 gate (latency + recall retention + relevance parity + budget
   compliance) on the candidate collection.
6. Canary cutover per the existing staged-rollout scaffold.
