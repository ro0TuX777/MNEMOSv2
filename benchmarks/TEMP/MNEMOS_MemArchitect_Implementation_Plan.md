# MNEMOS Governance Layer Implementation Plan
## MemArchitect-Inspired Adaptation

## 1. Objective

Build a **policy-driven memory governance layer** inside MNEMOS that sits above retrieval and storage and actively manages:

- stale memory
- contradictory memory
- low-utility memory
- unsafe or persistent poisoning risk
- deletion and compliance behavior
- memory budget competition

The goal is to evolve MNEMOS from a retrieval platform into a **governed memory system**.

---

## 2. Product Thesis

### Current issue
MNEMOS can retrieve, but retrieval alone does not solve:
- contradiction
- stale facts
- context pollution
- zombie memory
- memory quality drift over time

### Target state
MNEMOS should decide not only:
- **what can be found**

but also:
- **what should be allowed into context**
- **what should be decayed**
- **what should be consolidated**
- **what should be deleted**
- **what should be vetoed even if retrieved**

---

## 3. Scope

### In scope
- governance layer above storage and retrieval
- read / reflect / background execution paths
- staleness and retention scoring
- contradiction detection and resolution
- post-retrieval relevance veto
- memory provenance and dependency tracking
- deletion cascade behavior
- governance telemetry and policy benchmarks

### Out of scope
- changing model weights
- reopening Track 2 reranker
- replacing Core/Qdrant as default
- full legal/compliance certification work
- advanced learned policy optimization in v1

---

## 4. Architecture Concept

Adapt the MemArchitect workflow into MNEMOS as three coordinated paths:

### A. Read path
At query time:
1. retrieve candidate memories
2. apply policy scoring
3. apply relevance or entailment veto
4. resolve contradictions
5. fill context budget with governed memory set

### B. Reflect path
After answer generation:
1. detect which memories were actually used
2. reinforce useful memories
3. lower trust or utility of ignored or harmful memories
4. update recency and stability metrics

### C. Background hygiene path
During idle or scheduled cycles:
1. decay low-value memories
2. consolidate episodic memories into semantic facts
3. prune obsolete noise
4. apply deletion cascades
5. flag contradiction clusters for adjudication

---

## 5. Proposed MNEMOS Modules

```text
mnemos/governance/
├── governor.py
├── policy_registry.py
├── read_path.py
├── reflect_path.py
├── hygiene_path.py
├── policies/
│   ├── decay_policy.py
│   ├── utility_policy.py
│   ├── contradiction_policy.py
│   ├── relevance_veto_policy.py
│   ├── token_budget_policy.py
│   ├── deletion_cascade_policy.py
│   └── provenance_policy.py
├── models/
│   ├── memory_state.py
│   ├── governance_decision.py
│   └── contradiction_record.py
└── telemetry/
    ├── governance_metrics.py
    └── governance_events.py
```

---

## 6. Core Data Model Additions

Each memory item should gain governance metadata beyond embedding and content.

Recommended fields:

- `memory_id`
- `memory_type` (`episodic`, `semantic`, `derived`, `summary`)
- `created_at`
- `last_accessed_at`
- `last_used_at`
- `source_type`
- `source_id`
- `derived_from` (list of parent memory IDs)
- `trust_score`
- `utility_score`
- `stability_score`
- `retrievability_score`
- `conflict_group_id` (optional)
- `superseded_by` (optional)
- `deletion_state`
- `policy_flags` (toxic, stale, contradictory, protected, etc.)

This is critical because governance needs explicit state, not just vector entries.

---

## 7. Phase 1 Policies to Implement

Do not implement every MemArchitect policy at once. Start with the ones that map best to MNEMOS.

### Policy 1 — Lifecycle / decay
Goal:
- reduce stale and low-value memory accumulation

Implementation:
- compute a simple retrievability or staleness score from:
  - age
  - usage
  - recent reinforcement
- classify memory into:
  - keep
  - fading
  - prune candidate

### v1 output
- `retrievability_score`
- `stability_score`
- `lifecycle_state`

### Policy 2 — Utility / trust reinforcement
Goal:
- distinguish useful memories from distracting ones

Implementation:
- after generation, mark candidate memories as:
  - used
  - ignored
  - contradicted
- update `utility_score`
- update `trust_score`

### v1 output
- `utility_score`
- `trust_score`

### Policy 3 — Relevance veto gate
Goal:
- prevent semantically similar but actually irrelevant memory from entering context

Implementation:
- after initial retrieval, run a lightweight second-stage discriminator
- if candidate support or entailment score falls below threshold, drop it

### v1 output
- `veto_pass: bool`
- `veto_reason`

### Policy 4 — Contradiction detection
Goal:
- detect conflicting state claims before they pollute generation

Implementation:
- identify memory pairs about the same entity and attribute slot
- examples:
  - user marital status
  - current employer
  - city
  - preference value
- group conflicts into a `conflict_group`

### v1 rule
Use heuristic conflict detection:
- same entity
- same attribute
- mutually inconsistent value
- different timestamps or confidence

### v1 output
- `conflict_group_id`
- `conflict_status`
- `candidate_winner`

### Policy 5 — Contradiction resolution / supersession
Goal:
- choose what survives into context when conflicts are present

Implementation:
Simple v1 winner rule:
- prefer higher trust
- then newer timestamp
- then higher utility
- allow source authority override if configured

### v1 output
- `active_memory`
- `suppressed_memories`

### Policy 6 — Provenance and deletion cascade
Goal:
- support deletion and prevent zombie memories

Implementation:
- track parent-child memory lineage
- if a root memory is deleted:
  - find derived summaries
  - find derived facts
  - mark or remove dependents
- support:
  - hard delete
  - soft delete / tombstone
  - rederived-needed state

### Policy 7 — Adaptive token/context budgeting
Goal:
- make memories compete for scarce context space

Implementation:
- compute final governed score from:
  - retrieval similarity
  - trust
  - utility
  - staleness penalty
  - contradiction penalty
- fill context budget greedily from highest governed score downward
- reserve a configurable reasoning budget if retrieval confidence is high

---

## 8. Governance Score Model

Suggested v1 scoring:

```text
governed_score =
  similarity
  * trust_modifier
  * utility_modifier
  * freshness_modifier
  * contradiction_modifier
  * veto_modifier
```

Where:
- `trust_modifier` in `[0, 1.25]`
- `utility_modifier` in `[0.5, 1.25]`
- `freshness_modifier` in `[0, 1]`
- `contradiction_modifier` in `[0, 1]`
- `veto_modifier` is `0` or `1`

Keep this explicit and interpretable in v1.

---

## 9. Read Path Design

### Input
- query
- retrieval mode
- optional filters
- context token budget

### Flow
1. retrieve top-N raw candidates from current MNEMOS retrieval path
2. apply relevance veto
3. detect contradictions among surviving candidates
4. suppress losing contradictory candidates
5. rescore with governance modifiers
6. fill context budget
7. emit explanation payload if requested

### Output
- governed candidate set
- suppressed candidate set
- reasons
- optional explanation fields

### Response additions
For explain/debug mode, add:

```json
{
  "memory_id": "…",
  "retrieval_score": 0.81,
  "governed_score": 0.66,
  "trust_score": 0.74,
  "utility_score": 0.68,
  "retrievability_score": 0.52,
  "veto_pass": true,
  "conflict_status": "winner",
  "suppressed_reason": null
}
```

---

## 10. Reflect Path Design

### Trigger
Runs after answer generation.

### Inputs
- query
- governed candidate set
- answer or output
- optional answer citation or use detection

### Flow
1. detect which memories were actually used
2. reward used memories
3. mildly penalize retrieved-but-unused memories
4. strongly penalize contradicted or vetoed memories if they slipped through and caused issues
5. update:
   - utility
   - trust
   - last_used_at
   - stability

### v1 “used memory” detection
Use a simple heuristic:
- citations
- explicit chunk/source references
- overlap with grounded answer spans
- answer-support classifier if available

Keep this simple first.

---

## 11. Background Hygiene Path Design

Runs on schedule or idle cycle.

### Jobs
#### Job A — decay pass
- update retrievability for all memories

#### Job B — prune pass
- delete or archive low-retrievability, low-utility memories

#### Job C — consolidation pass
- compress multiple episodic memories into a semantic fact when:
  - repeated
  - stable
  - non-contradictory

#### Job D — contradiction sweep
- identify unresolved conflict groups
- promote candidate winner
- flag ambiguous cases

#### Job E — deletion cascade
- propagate deletes through derived memories

---

## 12. API Surface Changes

### `/search`
Add:
- `governance: off | advisory | enforced`
- `explain_governance: true | false`

Default:
- `governance=advisory` for internal/testing first
- later move to `enforced` in controlled paths

### `/memory/write`
Add:
- provenance fields
- memory type
- policy flags

### `/memory/delete`
Add:
- cascade mode
- delete reason
- preview dependents

### `/governance/stats`
Expose:
- active memories
- decayed memories
- pruned memories
- conflict groups
- veto rate
- deletion cascades executed
- governed vs raw retrieval difference rate

---

## 13. Benchmark Plan

### Benchmark question
Does governed memory improve context quality and contradiction safety enough to justify its added complexity?

### Compare
- raw retrieval
- raw retrieval + veto
- governed retrieval (full policy set v1)

### Query classes
1. contradictory user-state queries
2. stale fact queries
3. temporal-update queries
4. preference-change queries
5. long-horizon personalized recall
6. deletion/compliance scenarios

### Metrics
- answer accuracy
- contradiction leakage rate
- stale-memory leakage rate
- irrelevant-memory inclusion rate
- context token efficiency
- retained useful memory rate
- over-pruning rate
- delete cascade correctness

### Success criteria
Governance v1 passes if it:
- materially reduces contradiction and stale-memory leakage
- does not cause unacceptable over-pruning
- improves context quality on at least one long-horizon memory class
- preserves acceptable latency

---

## 14. LoE Estimate

For one strong AI Dev:

### Phase 1 — governance metadata + read path
1.5 to 2 weeks

### Phase 2 — reflect path + simple utility updates
4 to 6 days

### Phase 3 — hygiene jobs + deletion cascade
1 to 1.5 weeks

### Phase 4 — benchmark track + test set
1 to 1.5 weeks

### Phase 5 — docs + hardening
3 to 5 days

### Total estimated LoE
~4 to 6 weeks for a solid v1.

---

## 15. Risks

### Risk 1 — over-pruning
Governance can hurt recall if pruning is too aggressive, especially on temporal or raw recall tasks.

Mitigation:
- advisory mode first
- tunable thresholds
- benchmark over-pruning explicitly

### Risk 2 — policy complexity without clear win
Mitigation:
- ship only 5–7 core policies in v1
- benchmark every policy

### Risk 3 — contradiction detection too brittle
Mitigation:
- start with entity-slot heuristics
- defer complex NLI arbitration

### Risk 4 — provenance model too weak for delete cascade
Mitigation:
- enforce parent-child linkage at write time for summaries and derived facts

---

## 16. Recommended Implementation Order

### Wave 1
- data model extensions
- policy registry
- relevance veto
- governance scoring
- read path advisory mode

### Wave 2
- contradiction detection
- contradiction resolution
- explain payload

### Wave 3
- reflect path
- trust/utility updates

### Wave 4
- decay
- consolidation
- prune
- deletion cascade

### Wave 5
- benchmark track
- docs
- operator guidance

---

## 17. Acceptance Criteria

MemArchitect-inspired Governance v1 is complete when:

- MNEMOS supports a governance layer above retrieval
- read path can veto, suppress, and rescore memory
- contradiction groups can be detected and resolved
- trust/utility can be updated after generation
- background hygiene jobs can decay, prune, and consolidate memory
- delete cascades work for derived memories
- a governance benchmark track exists
- docs explain what governance mode does and when to use it

---

## 18. Product Posture After This Phase

If successful, MNEMOS can be described as:

- **Core**: fast retrieval substrate
- **Governance layer**: policy-driven memory adjudication
- **Memory system**: not just retrieve-and-stuff, but lifecycle-aware, contradiction-aware, provenance-aware memory

That is a much stronger differentiator than simply adding more retrieval backends.

---

## 19. Recommendation

Build this in **advisory mode first**, not enforced by default.

That lets the team compare:
- what raw retrieval would have done
- what governed retrieval would have changed

without destabilizing the product too early.
