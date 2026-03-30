# MNEMOS MemArchitect Completion Program

**Status:** Active — governance completion lane
**Date:** 2026-03-30
**Source:** MemArchitect Completion Program v1

---

## Program Objective

Move governance from "feature-complete in code" to "credible, measurable, persistent, and safe enough to rely on."

MemArchitect is no longer the main innovation stream. Memory-Over-Maps is. But governance has a completion program that runs as a smaller stabilization lane in parallel.

---

## Foundation Already in Place

| Component | Status |
|---|---|
| Governance Waves 1–3 (veto, score, contradiction, reflect) | ✅ Complete |
| Governance Validation Pack v1 (33 tests, 10 named scenarios) | ✅ Complete |
| Wave 4 hygiene brief (decay, prune, contradiction sweep) | ✅ Briefed |
| Wave 4 hygiene implementation | ❌ Not yet implemented |
| Whitepaper 4.6 updated with reflect path + validation evidence | ✅ Complete |

---

## Phase 1 — Validation

**Priority:** Highest. No point persisting governance state that has not been proven to behave well.

### A1. Governance Benchmark Pack

Compare `governance=off` vs `advisory` vs `enforced` on real fixture data.

**Metrics:**
- Contradiction leakage rate (contradicted candidates reaching the answer in `off` mode that don't in `enforced`)
- Stale-memory leakage rate (high-age, low-utility candidates included vs suppressed)
- Irrelevant-memory inclusion rate (low-governed-score candidates passing in `off` that are suppressed in `enforced`)
- Over-pruning rate (legitimate candidates suppressed in `enforced` that were USED in reflect)
- Trust/utility drift quality (are reinforced memories ranking higher in subsequent queries?)
- Ranking delta after N reflect cycles

**Deliverable:** `benchmarks/governance_benchmark_pack/` — fixtures, runner, report table

**Current state:** Not started. Validation Pack v1 covers in-process drift behavior but does not compare modes on realistic fixture data.

---

### A2. Repeated-Scenario Drift Harness ✅

**Status: Complete** — `tests/test_governance_drift_validation.py`, 33/33 passing

Scenarios covered:
- Repeated winner cited (15 cycles) — utility/trust converge to ceiling
- Repeated distractor ignored (20 cycles) — utility decays monotonically
- Contradiction winner vs loser (10 cycles) — separation grows
- Stale value ignored (30 cycles) — utility drops from 0.8 to 0.5
- Short generic vs grounded — sub-3-char floor prevents false USED
- Overlap-heavy false-credit — threshold boundary documented, Scenarios F/G/I

Formal artifact: `benchmarks/TEMP/Governance_Validation_Pack_v1.md`

---

### A3. Explain-Payload Manual Review Pack

Human-readable audit of governance decisions from realistic queries.

**Required outputs:**
- For a set of N queries against real fixtures: print contradiction winners/losers and why
- Print veto reasons for suppressed candidates
- Print reflect outcomes after one real answer is fed back
- Human-readable "why did the system make this decision" for each non-obvious outcome

**Deliverable:** `benchmarks/governance_explain_pack/` — fixture queries, expected explain payloads, review script

**Current state:** The `/search?explain_governance=true` endpoint exists. No manual review pack built yet.

---

### A4. Threshold Tuning Report

Review and justify current parameter choices with evidence.

**Parameters to review:**
- Overlap threshold (default 0.15) — Scenarios F/G/J establish behavior; formal report not written
- Freshness half-life (default 180 days) — untested empirically
- Veto floor (default 0.0 — disabled) — no evidence for the right activation threshold
- Contradiction modifier values (winner 1.0, loser 0.25) — not benchmarked for over-aggressiveness
- Reinforcement deltas (±0.05/±0.01/±0.02/etc.) — drift validated in-process; not benchmarked against real query mix

**Deliverable:** `benchmarks/TEMP/Governance_Threshold_Tuning_Report.md` — parameter, current value, evidence, recommendation

**Current state:** Partially covered in Validation Pack v1 (Scenarios F/G/I/J). Formal report not written.

---

### Phase 1 Acceptance Gate

Phase 1 is complete when:
- [ ] Advisory mode demonstrates measurable improvement over raw retrieval on at least one scenario class
- [ ] Enforced mode behavior is explainable and not obviously over-aggressive
- [ ] Drift validation shows convergence, not runaway reinforcement (A2 ✅ already satisfied)
- [ ] Threshold choices are documented and justified

---

## Phase 2 — Persistence

**Priority:** Second. Governance is significantly weaker as a purely in-memory, per-request system.

### B1. Governance Persistence Shim

Backend-independent persistence layer for `GovernanceMeta` updates.

**Required:**
- In-memory mode (test parity with Waves 1–3 tests)
- Persisted production mode (Qdrant payload update, pgvector UPDATE)
- Clean serialize/deserialize roundtrip for all governance fields

**Fields to persist (minimum):**
`trust_score`, `utility_score`, `stability_score`, `retrievability_score`, `last_used_at`, `usage_count`, `lifecycle_state`, `conflict_group_id`, `contradiction_winner`, `conflict_status`, `deletion_state`

**Deliverable:** `mnemos/governance/persistence/shim.py`

---

### B2. Reflect Update Persistence

After `POST /v1/mnemos/governance/reflect`, governance mutations must persist.

Currently: reflect mutates `GovernanceMeta` in-memory; caller owns persistence. This is the gap.

**Deliverable:** `service/app.py` reflect endpoint calls shim to persist deltas. Test: reflect across two service requests, second request sees updated scores.

---

### B3. Hygiene Update Persistence

Decay/prune/contradiction-sweep outcomes written back to active backend after `HygienePipeline.run()`.

---

### B4. Conflict-State Persistence

Contradiction groups and resolved winners survive beyond one query. Currently re-evaluated per query.

---

### B5. Persistence Integrity Tests

- Save/reload roundtrip for all governance fields
- Repeated reflect cycles across persisted state (scores accumulate correctly)
- Contradiction winner retained after reload
- Lifecycle state (`stale`, `prune_candidate`) retained after hygiene pass

**Deliverable:** `tests/test_governance_persistence.py`

---

### Phase 2 Acceptance Gate

Phase 2 is complete when:
- [ ] Reflect updates survive a full service reload
- [ ] Hygiene outcomes survive reload
- [ ] Contradiction state survives reload where intended
- [ ] `GovernanceMeta` roundtrips cleanly through Qdrant payload and pgvector row

---

## Phase 3 — Deferred Lifecycle / Compliance

**Priority:** Third. Does not block core governance story. Required for enterprise credibility.

### C1. Delete Cascade

When a root memory is deleted, summaries/facts/derived artifacts derived from it must be invalidated or scheduled for re-derivation.

**Anti-pattern this prevents:** "Zombie memory" — a summary persists and is retrieved even after the source it was derived from has been removed.

---

### C2. Consolidation (Basic)

Promote repeated stable episodic memories into compact semantic facts.

**Minimum viable form:** Repeated episodic memories sharing the same entity/attribute/value triple over N cycles are promoted to `semantic` memory type with consolidated provenance.

---

### C3. Richer Contradiction Arbitration

After deterministic entity-slot logic:
- Source-type weighting (prefer user-asserted over inferred)
- Confidence-aware resolution
- Ambiguity class (three-way tie → "ambiguous" rather than forced winner)

---

### C4. Compliance-Oriented Deletion Semantics

Explicit support for: hard delete, soft delete / tombstone, delete preview with dependent artifact count, audit reason field.

---

### C5. Toxic / Poisoned Memory Write Guardrails

Optional write-path controls for prompt injection persistence risk and unsafe durable memory content.

---

### Phase 3 Acceptance Gate

Phase 3 is complete enough when:
- [ ] Delete cascade exists and is testable
- [ ] At least one consolidation flow works
- [ ] Derived artifacts can be invalidated from root memory deletion
- [ ] Compliance-style delete behavior is explicit and explainable

---

## "MemArchitect Complete" Definition

MemArchitect is complete as a product subsystem when all of the following are true:

| Domain | Criterion |
|---|---|
| **Core** | Query-time governance, contradiction handling, reflect path, hygiene path all work |
| **Proof** | Raw vs advisory vs enforced benchmarked; drift validated; decisions inspectable |
| **Durability** | Trust/utility/stability/lifecycle/contradiction state persists across requests and reloads |
| **Lifecycle** | Delete cascade exists; basic consolidation exists; derived-view invalidation supported |

---

## Parallel Streams

This program runs as a completion lane. The main architecture innovation stream is **Memory-Over-Maps**.

| Stream | Status | Priority |
|---|---|---|
| MemArchitect completion (this program) | Active — Phase 1 next | Stabilization lane |
| Memory-Over-Maps | Not started | Main innovation stream |

The two streams are independent. Memory-Over-Maps does not depend on governance completion. Governance completion does not depend on Memory-Over-Maps.

---

## Immediate Next Steps (in order)

1. **Wave 4 implementation** — `mnemos/governance/hygiene/` (decay_runner → prune_promoter → contradiction_sweep → HygienePipeline). Gate: ~45 tests passing, counters in `governor.stats()`.
2. **A1 Governance benchmark pack** — raw vs advisory vs enforced on fixture data.
3. **A3 Explain-payload review pack** — human-readable audit of real governance decisions.
4. **A4 Threshold tuning report** — formalize parameter choices with evidence.
5. **B1 Persistence shim** — backend-independent `GovernanceMeta` update layer.
6. **B2–B4 Persist reflect/hygiene/contradiction outcomes.**
7. **C1 Delete cascade** — first lifecycle/compliance deliverable.
