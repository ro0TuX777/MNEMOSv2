# MNEMOS MemArchitect Implementation Task Board

## Sprint Goal

Deliver a **benchmarkable, advisory-mode governance layer** inside MNEMOS that can:
- veto irrelevant memory
- detect and resolve basic contradictions
- update trust and utility from post-answer feedback
- run background hygiene for decay, pruning, consolidation, and delete cascade

---

## Epic 1 — Governance Data Model

### Task 1.1 — Extend memory schema with governance metadata
**Description**
Add fields for trust, utility, stability, retrievability, provenance, conflict state, and deletion state.

**Dependencies**
- none

**Acceptance criteria**
- schema supports all required governance fields
- migrations or storage updates apply cleanly
- existing reads/writes continue to work

### Task 1.2 — Add memory lineage support
**Description**
Support parent-child links for derived summaries and semantic facts.

**Dependencies**
- Task 1.1

**Acceptance criteria**
- derived memories can reference parent memory IDs
- lineage can be queried efficiently
- deletion preview can enumerate dependents

---

## Epic 2 — Governance Registry and Scoring

### Task 2.1 — Build policy registry
**Description**
Implement a registry that enables policies to be run in a stable order and turned on/off by config.

**Dependencies**
- Task 1.1

**Acceptance criteria**
- policies can be registered and invoked deterministically
- policy outputs are composable
- config can disable individual policies

### Task 2.2 — Implement governed score computation
**Description**
Implement explicit score composition from similarity, trust, utility, freshness, contradiction, and veto factors.

**Dependencies**
- Task 2.1

**Acceptance criteria**
- governed score is deterministic
- component scores are inspectable in debug/explain mode
- unit tests cover score composition rules

---

## Epic 3 — Read Path Governance

### Task 3.1 — Add governance mode to `/search`
**Description**
Support `governance=off|advisory|enforced` and `explain_governance=true|false`.

**Dependencies**
- Task 2.2

**Acceptance criteria**
- `/search` accepts governance flags
- advisory mode returns governed analysis without fully enforcing suppression
- enforced mode can suppress or veto candidates

### Task 3.2 — Implement relevance veto policy
**Description**
Add a post-retrieval veto gate to remove semantically similar but actually irrelevant candidates.

**Dependencies**
- Task 3.1

**Acceptance criteria**
- veto can drop low-support candidates
- explain payload shows pass/fail and reason
- unit/integration tests validate veto behavior

### Task 3.3 — Implement contradiction detection
**Description**
Detect conflicting memories about the same entity-slot pair.

**Dependencies**
- Task 1.1
- Task 3.1

**Acceptance criteria**
- conflicting candidate groups can be formed
- same-entity same-attribute inconsistencies are identified
- conflict IDs are emitted in explain mode

### Task 3.4 — Implement contradiction resolution
**Description**
Pick a winner among conflicting memories using trust, recency, utility, and optional source authority.

**Dependencies**
- Task 3.3

**Acceptance criteria**
- losing memories can be suppressed in enforced mode
- winning memory is surfaced clearly
- tests cover recency and trust tie-break behavior

### Task 3.5 — Add governed explain payload
**Description**
Return governed reasoning metadata in search responses.

**Dependencies**
- Tasks 3.2, 3.3, 3.4

**Acceptance criteria**
- explain payload includes retrieval score, governed score, trust, utility, retrievability, veto result, and conflict status
- disabled explain mode omits heavy governance details
- response schema is documented

---

## Epic 4 — Reflect Path

### Task 4.1 — Implement used-memory detection
**Description**
Detect which memories actually contributed to the final answer.

**Dependencies**
- Task 3.5

**Acceptance criteria**
- at least one heuristic path is implemented (citation, answer overlap, or support classifier)
- output distinguishes used vs ignored candidates
- tests cover representative cases

### Task 4.2 — Implement trust and utility updates
**Description**
Update trust/utility based on used, ignored, contradicted, or harmful memories.

**Dependencies**
- Task 4.1

**Acceptance criteria**
- used memories are reinforced
- ignored memories are mildly penalized
- harmful/contradicted memories can be penalized more strongly
- updated values are persisted

### Task 4.3 — Update recency/stability after reflection
**Description**
Refresh lifecycle state when memories are successfully used.

**Dependencies**
- Task 4.2

**Acceptance criteria**
- `last_used_at`, `stability_score`, and retrievability-related fields update correctly
- repeated usage improves retention state as expected

---

## Epic 5 — Background Hygiene Engine

### Task 5.1 — Build scheduled hygiene runner
**Description**
Implement a scheduler or callable job runner for governance hygiene passes.

**Dependencies**
- Task 1.1

**Acceptance criteria**
- hygiene jobs can run manually and on schedule
- runs are idempotent enough for repeated execution
- events are logged

### Task 5.2 — Implement decay pass
**Description**
Update retrievability/stability for all memories over time.

**Dependencies**
- Task 5.1

**Acceptance criteria**
- stale memories move toward prune eligibility
- active memories remain protected
- lifecycle state transitions are testable

### Task 5.3 — Implement prune pass
**Description**
Delete or archive low-retrievability, low-utility memories.

**Dependencies**
- Task 5.2

**Acceptance criteria**
- prune candidates are correctly selected
- soft-delete and hard-delete modes are both supported or clearly scoped
- over-pruning safeguards exist

### Task 5.4 — Implement consolidation pass
**Description**
Consolidate repeated episodic memories into stable semantic facts.

**Dependencies**
- Task 5.2

**Acceptance criteria**
- repeated stable memory clusters can be summarized into semantic facts
- derived memory references parents
- duplicate or replaced episodic items can be flagged appropriately

### Task 5.5 — Implement contradiction sweep
**Description**
Run background detection on unresolved contradictions and promote likely winners.

**Dependencies**
- Tasks 3.3, 3.4, 5.1

**Acceptance criteria**
- unresolved contradiction groups can be surfaced
- likely winners are promoted or flagged
- ambiguous groups can be left for manual review

---

## Epic 6 — Provenance and Delete Cascade

### Task 6.1 — Add delete preview API path
**Description**
Show what derived memories will be affected before deletion.

**Dependencies**
- Task 1.2

**Acceptance criteria**
- preview lists direct and indirect dependents
- preview supports root memory delete requests

### Task 6.2 — Implement delete cascade policy
**Description**
When a root memory is deleted, remove or tombstone derived dependents to prevent zombie memory.

**Dependencies**
- Task 6.1

**Acceptance criteria**
- cascades propagate through lineage correctly
- derived summaries/facts do not survive incorrectly after root deletion
- tests validate cascade correctness

---

## Epic 7 — Telemetry and Governance Stats

### Task 7.1 — Add governance event logging
**Description**
Emit events for veto, contradiction suppression, reinforcement, pruning, consolidation, and delete cascade.

**Dependencies**
- Epics 3–6

**Acceptance criteria**
- governance events are logged consistently
- enough detail exists for audit/debug
- event overhead is acceptable

### Task 7.2 — Add `/governance/stats`
**Description**
Expose aggregate governance metrics.

**Dependencies**
- Task 7.1

**Acceptance criteria**
- stats include veto rate, contradiction groups, decayed memories, pruned memories, consolidated memories, and delete cascades
- values are consistent with executed actions

---

## Epic 8 — Benchmark Track

### Task 8.1 — Create governance benchmark dataset
**Description**
Build a test set of contradictory, stale, temporal-update, preference-change, and deletion/compliance scenarios.

**Dependencies**
- none

**Acceptance criteria**
- dataset is committed to repo
- each scenario is labeled by class
- expected behavior is documented

### Task 8.2 — Add governance benchmark runner
**Description**
Benchmark raw retrieval vs raw+veto vs governed retrieval.

**Dependencies**
- Epics 3–6
- Task 8.1

**Acceptance criteria**
- benchmark produces raw JSON and summary report
- runs compare all required modes
- metrics are broken out by scenario class

### Task 8.3 — Add governance metrics/reporting
**Description**
Report contradiction leakage, stale leakage, irrelevant inclusion, context efficiency, retained useful memory, over-pruning, and delete cascade correctness.

**Dependencies**
- Task 8.2

**Acceptance criteria**
- summary report includes all required governance metrics
- success/failure criteria are explicit
- advisory vs enforced differences are visible

---

## Epic 9 — Docs and Operator Guidance

### Task 9.1 — Document governance modes
**Description**
Explain `off`, `advisory`, and `enforced`.

**Dependencies**
- Epic 3

**Acceptance criteria**
- docs explain behavior of each mode
- advisory mode is clearly recommended as initial rollout posture

### Task 9.2 — Document lifecycle, contradiction, and deletion policies
**Description**
Describe what each v1 policy does and how operators should interpret it.

**Dependencies**
- Epics 3–6

**Acceptance criteria**
- docs explain policy purpose and trade-offs
- known limitations are clearly stated

### Task 9.3 — Write rollout guidance
**Description**
Document recommended rollout:
1. advisory only
2. compare raw vs governed
3. enable enforced in controlled paths

**Dependencies**
- Task 8.3

**Acceptance criteria**
- clear rollout steps exist
- pass/fail thresholds for broader enforcement are documented

---

# Recommended Execution Order

## Wave 1
- Task 1.1
- Task 1.2
- Task 2.1
- Task 2.2
- Task 3.1
- Task 3.2

## Wave 2
- Task 3.3
- Task 3.4
- Task 3.5
- Task 4.1
- Task 4.2
- Task 4.3

## Wave 3
- Task 5.1
- Task 5.2
- Task 5.3
- Task 5.4
- Task 5.5
- Task 6.1
- Task 6.2

## Wave 4
- Task 7.1
- Task 7.2
- Task 8.1
- Task 8.2
- Task 8.3

## Wave 5
- Task 9.1
- Task 9.2
- Task 9.3

---

# Sprint Exit Criteria

The implementation phase is complete when:

- governance metadata exists in MNEMOS memory records
- `/search` supports advisory and enforced governance
- relevance veto works
- contradiction detection and basic resolution work
- reflect path updates trust and utility
- hygiene runner can decay, prune, consolidate, and sweep contradictions
- delete cascade works for derived memories
- governance benchmark reports raw vs governed outcomes
- operator docs and rollout guidance are complete

---

# Key Product Questions This Work Should Answer

1. Does governed memory materially reduce contradiction leakage?
2. Does it materially reduce stale-memory pollution?
3. Does it preserve useful memory without over-pruning?
4. Is advisory mode producing enough value to justify enforced mode later?
5. Does governance become the clearest differentiator for MNEMOS beyond retrieval substrate?
