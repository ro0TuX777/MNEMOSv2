# MNEMOS Wave 4 — Hygiene Path Implementation Brief

**Status:** Pre-implementation
**Date:** 2026-03-30
**Scope:** Freshness decay runner · Prune candidate promotion · Contradiction sweep
**Explicitly deferred:** Delete cascade · Memory consolidation

---

## Objective

Wave 4 is the background health layer. Waves 1–3 govern individual read and reflect cycles; Wave 4 keeps the corpus healthy between queries. Without it, ignored memories accumulate forever, contradictions stay unresolved until a conflicting query happens to trigger them, and stale memories retain high utility scores indefinitely.

The goal is a deterministic, auditable sweep runner that can execute as a cron job, a CLI command, or a background thread — and that leaves a clear audit trail in `GovernanceMeta` without touching Engram content.

---

## Scope

### 1. Freshness Decay Runner

**What it does:**
Applies time-based utility and trust decay to memories whose `last_used_at` is older than a configurable horizon. Updates `lifecycle_state` to `stale` when composite score drops below a floor.

**Why it is needed:**
`utility_score` only changes during reflect cycles. A memory that has not been cited in 90 days retains whatever score it earned during its last active period. Decay closes the gap between "was once relevant" and "is currently relevant."

**Proposed module:** `mnemos/governance/hygiene/decay_runner.py`

**Key design decisions:**

| Decision | Choice | Reason |
|---|---|---|
| Decay function | Linear per elapsed day | Predictable, auditable, easy to tune |
| Decay target | `utility_score` only (Wave 4); `trust_score` deferred | Trust decay requires more careful modelling of source reliability |
| Horizon | Configurable (default: 60 days since `last_used_at`) | Different deployments have different memory freshness needs |
| Floor | `utility_score ≥ 0.0` (existing clamp), `lifecycle_state = stale` when score < 0.20 | Keeps memories discoverable but ranked low |
| Trigger | Offline sweep (not in read-path) | Avoids adding latency to retrieval |

**Proposed config:**

```python
@dataclass
class DecayConfig:
    horizon_days: int = 60           # days of inactivity before decay begins
    decay_per_day: float = 0.005     # utility lost per day past horizon
    stale_threshold: float = 0.20    # lifecycle_state → stale below this
    min_utility: float = 0.0         # hard floor
```

**Algorithm (per Engram):**

```
elapsed = now - last_used_at  (days)
if elapsed < horizon_days:
    skip
delta = decay_per_day * (elapsed - horizon_days)
new_utility = max(min_utility, utility_score - delta)
if new_utility < stale_threshold:
    lifecycle_state = "stale"
utility_score = new_utility
```

**Acceptance criteria:**
- A memory last used 120 days ago with `utility_score=0.8` and `horizon_days=60` must have `utility_score ≤ 0.50` after one sweep.
- A memory used yesterday is untouched.
- Any memory with `utility_score < 0.20` post-sweep has `lifecycle_state == "stale"`.
- `utility_score` never goes below 0.0.
- Sweep emits a `DecayReport` dataclass: `{scanned, decayed, stale_promoted, skipped}`.

---

### 2. Prune Candidate Promotion

**What it does:**
Identifies memories whose composite governance score has fallen below a promotion floor and sets `lifecycle_state = prune_candidate`. Does not delete anything.

**Why it is needed:**
Deletion should be a human- or policy-gated action. Promotion to `prune_candidate` is a low-risk signal that a memory is eligible for review — it decouples the detection decision from the irreversible deletion action.

**Proposed module:** `mnemos/governance/hygiene/prune_promoter.py`

**Composite score for prune evaluation:**

```
composite = utility_score × trust_score × (1 - contradiction_modifier_penalty)
```

Where `contradiction_modifier_penalty = 1.0 - contradiction_modifier` (0.0 for non-contradicted memories, 0.75 for losers with modifier=0.25).

**Proposed config:**

```python
@dataclass
class PruneConfig:
    composite_floor: float = 0.05    # composite below this → prune_candidate
    min_usage_count: int = 0         # memories never used are also eligible
    respect_stale_state: bool = True # already-stale memories are always eligible
```

**Algorithm:**

```
for each engram:
    if lifecycle_state in ("deleted", "prune_candidate"):
        skip  # already processed
    composite = utility_score × trust_score × (1 - (1 - contradiction_modifier))
    if composite < composite_floor:
        lifecycle_state = "prune_candidate"
```

**Acceptance criteria:**
- A memory with `utility_score=0.02, trust_score=0.9` (composite ≈ 0.018) is promoted to `prune_candidate`.
- A memory with `lifecycle_state="deleted"` is skipped.
- A memory already at `prune_candidate` is not double-counted.
- Sweep emits a `PruneReport`: `{scanned, promoted, skipped}`.
- No memory content is modified; only `lifecycle_state` changes.

---

### 3. Contradiction Sweep

**What it does:**
Runs `ContradictionPolicy.detect_and_resolve()` over the full corpus (or a configurable entity-slot cluster), offline, updating contradiction state on memories whose conflicts were not caught at query time.

**Why it is needed:**
`ContradictionPolicy` runs in the read-path, so it only sees the candidates returned by a specific query. Two conflicting memories with different embedding distances from the query may never both be retrieved in the same context window. The offline sweep catches cross-document contradictions that retrieval doesn't surface.

**Proposed module:** `mnemos/governance/hygiene/contradiction_sweep.py`

**Key design decisions:**

| Decision | Choice | Reason |
|---|---|---|
| Clustering | Group by `(entity_key, attribute_key)` across all Engrams | Same grouping logic as ContradictionPolicy — consistent |
| Reuse | Call `ContradictionPolicy.detect_and_resolve()` directly | Avoids duplicating resolution logic |
| Scope | Entity-slot clusters only (skip `entity_key == ""`) | Generic memories with no entity binding cannot form meaningful contradiction pairs |
| Persistence | Update `GovernanceMeta` in-place; caller persists | Same contract as Wave 3 reflect — runner is stateless |

**Algorithm:**

```
clusters = group all engrams by (entity_key, attribute_key)
          where entity_key != "" and len(cluster) >= 2

for each cluster:
    fake_results = [SearchResult(engram=e, score=e.governance.utility_score) for e in cluster]
    fake_decisions = [GovernanceDecision from each result]
    contradiction_records = ContradictionPolicy().detect_and_resolve(fake_results, fake_decisions)
    apply modifier mutations back to engram.governance
    yield ContradictionSweepRecord(cluster_key, records)
```

**Acceptance criteria:**
- Two Engrams with same `(entity_key, attribute_key)` and different `normalized_value` have `conflict_status` set after a sweep.
- Engrams with `entity_key == ""` are not processed.
- Winner selection follows the same 5-level chain as the read-path: trust → recency → utility → source_authority → id.
- Sweep emits `ContradictionSweepReport`: `{clusters_scanned, contradictions_found, winners_set, losers_set}`.

---

## Proposed Module Layout

```
mnemos/governance/hygiene/
    __init__.py
    decay_runner.py          # DecayConfig, DecayReport, DecayRunner.run(engrams)
    prune_promoter.py        # PruneConfig, PruneReport, PrunePromoter.run(engrams)
    contradiction_sweep.py   # ContradictionSweepRunner.run(engrams)
```

All three runners share the same interface contract:

```python
class HygieneRunner(Protocol):
    def run(self, engrams: List[Engram], now_iso: str = None) -> HygieneReport: ...
```

A top-level `HygienePipeline` (in `__init__.py`) can chain them in order: decay → prune promotion → contradiction sweep.

---

## Test Plan

Each runner gets its own test file. All tests are in-process with no backend dependency, same pattern as `test_governance_drift_validation.py`.

| File | Tests | What it covers |
|---|---|---|
| `tests/test_hygiene_decay.py` | ~15 | Below-horizon untouched, decay formula correct, stale promotion, floor clamp, DecayReport counts |
| `tests/test_hygiene_prune.py` | ~10 | Composite score floor, already-deleted skip, prune_candidate promotion, PruneReport counts |
| `tests/test_hygiene_contradiction_sweep.py` | ~12 | Cluster grouping, empty entity_key skipped, winner selection parity with ContradictionPolicy, SweepReport counts |
| `tests/test_hygiene_pipeline.py` | ~8 | End-to-end pipeline, order of operations, idempotency |

Target: ~45 new tests, total suite ≥ 230.

---

## Implementation Order

1. `decay_runner.py` + `test_hygiene_decay.py` — standalone, no dependency on other hygiene modules
2. `prune_promoter.py` + `test_hygiene_prune.py` — reads `lifecycle_state` set by decay; should run after
3. `contradiction_sweep.py` + `test_hygiene_contradiction_sweep.py` — reuses `ContradictionPolicy` directly
4. `HygienePipeline` + `test_hygiene_pipeline.py` — integration wrapper

Each step is independently mergeable. Do not implement the pipeline until all three runners have passing tests.

---

## Explicitly Deferred

### Delete Cascade
Deleting a memory must update any Engram that references it via `derived_from`, `superseded_by`, or edge relationships. This requires graph traversal and cannot be done safely as a simple state flip. Deferred until a reference graph query API is available.

### Memory Consolidation
Merging two contradictory memories into a single canonical Engram requires content synthesis (LLM call or deterministic merge rules), schema decisions about how to preserve provenance, and edge rewiring. Deferred until the memory write-path is stable and consolidation policy is specified.

---

## Rollout Posture

Wave 4 follows the same advisory-before-enforced posture as Waves 1–3:

- **Wave 4 off (default):** Hygiene runners are importable but not called automatically.
- **Wave 4 advisory:** Runners return reports but do not mutate `GovernanceMeta`. Useful for dry-run audits.
- **Wave 4 enforced:** Runners mutate in-place; caller persists to backend.

The `governance_mode` parameter should be threaded into `HygienePipeline.run()` with the same `off / advisory / enforced` values used in `ReadPath` and `ReflectPath`.

---

## Acceptance Gate for Wave 4

Wave 4 is complete when:

1. All three runners have ≥ 10 passing tests each.
2. `HygienePipeline` has ≥ 5 integration tests.
3. A dry-run sweep over the MNEMOS governance benchmark corpus (any fixture set from `benchmarks/`) produces a non-empty `DecayReport` without errors.
4. `governor.stats()` includes hygiene counters: `total_decay_runs`, `total_stale_promoted`, `total_prune_candidates`, `total_contradiction_sweep_clusters`.
5. `/v1/mnemos/governance/stats` reflects the new counters.

---

*This brief is explicitly scoped. Any work on delete cascade, consolidation, or backend persistence of hygiene state is out of scope and should be tracked separately.*
