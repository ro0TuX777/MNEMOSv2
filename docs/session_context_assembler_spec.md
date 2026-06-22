# MNEMOS Session Context Assembler Specification

Date: 2026-06-21

Status: **Proposed research lane only. Offline prototype, benchmark-first.
No production integration authorized by this spec.**

## Executive Boundary

No production integration, retrieval-ranking change, governance mutation,
authority mutation, promotion change, Engram mutation, or Resolution Engram
mutation is authorized by this plan.

## Objective

Evaluate whether a governed session-context assembler can reduce prompt
burden and improve continuity across long multi-turn sessions while
preserving source linkage, provenance, and prior-decision recall.

The design is inspired by EpiCache's episodic-session concept (see
`docs/adr/0007-session-context-assembler-shadow-only.md` for primary
sources), but does not adopt EpiCache as a runtime dependency or second
memory system.

## Core Principle

MNEMOS remains the durable governed-memory layer.

The session-context assembler is an ephemeral MNEMOS capability that selects
relevant session material and source-linked Engrams for a current task. It is
consumer-neutral; no application, agent framework, workflow, or operator
interface owns the capability.

```text
MNEMOS governed durable memory
-> source-linked Engrams and evidence bundles
-> session-context assembler
-> bounded context package
-> authorized-consumer read-only shadow adapter
-> external application, agent, workflow, or operator interface
```

Every selected session segment or episode summary must:

- be labeled `synthetic_context`
- retain parent Engram IDs and source IDs
- retain lineage metadata
- be non-authoritative
- be non-promotable
- be excluded from governance-state mutation
- be excluded from Resolution Engram creation unless independently
  re-grounded in source evidence

## Phase 0: Research Contract

Delivered by this spec and `docs/adr/0007-session-context-assembler-shadow-only.md`:

- explicit blocked list (see below)
- data-flow boundaries (see Core Principle)
- synthetic-context labeling rules (see Core Principle)
- source-ID preservation requirements (see Core Principle, Phase 4)
- no-write guarantees (see Phase 1, Phase 4)
- benchmark corpus requirements (see Phase 2)
- rollback and deletion behavior (see ADR 0007 Rollback)
- artifact-retention policy (see Phase 6)
- promotion requirements (see Phase 6, Per-Candidate Evaluation Gates in
  `docs/future_research_candidates.md`)

### Explicitly blocked in this lane

- writing summaries back as Engrams
- altering existing summaries
- altering retrieval ranking
- changing source authority
- changing contradiction state
- creating or modifying Resolution Engrams
- calling promotion paths
- persisting unlabelled synthetic summaries
- any production route or agent-facing memory write
- treating synthetic context as source truth

## Phase 1: Offline Prototype Only

**Status: complete (offline prototype + tests, 2026-06-21).** See
`docs/session_context_assembler_phase_1_notes.md`,
`prototype/session_context_assembler/`, and
`tests/test_session_context_assembler_prototype.py`. No Phase 3 baseline
comparison has run; nothing here is a benchmark claim.

An isolated prototype module with no production route wiring.

Suggested location:

```text
prototype/session_context_assembler/
```

Inputs:

```text
current_task
conversation_turns
eligible_source_linked_engrams
prior_decision_artifacts
session_metadata
```

Outputs:

```text
context_package
selected_episode_ids
selected_parent_engram_ids
selected_source_ids
synthetic_context_labels
selection_rationale
token_estimate
```

The prototype may:

- cluster prior conversation turns into candidate episodes
- select relevant prior episodes for a task
- attach eligible source-linked Engrams
- create bounded summaries of selected episodes
- emit a reproducible context package

The prototype may not:

- write summaries back as Engrams
- alter existing summaries
- alter retrieval ranking
- change source authority
- change contradiction state
- create or modify Resolution Engrams
- call promotion paths
- persist unlabelled synthetic summaries

## Phase 2: Benchmark Corpus

**Status: complete (R0, frozen 2026-06-21).** See
`docs/session_context_assembler_corpus_design.md`,
`benchmarks/truthsets/session_context_assembler_r0.json`, and
`benchmarks/truthsets/session_context_assembler_r0.manifest.json`.

**Phase 2R measurement revision: complete (R1, frozen 2026-06-22).** R1
keeps R0 frozen, introduces binding budgets and structured source links,
adds explicit contradiction labels and decision-artifact retention, and
adds five adversarial fixtures. The unchanged selector was replayed against
R1; its utility evidence does not pass Phase 4. See
`docs/session_context_assembler_phase_2r_notes.md`.

A small frozen replay corpus with 20 to 30 long-session cases.

Each case should include:

```text
session_id
task_id
conversation_history
required_prior_decision_ids
required_source_ids
known_irrelevant_history
expected_context_budget
```

Case families should include:

- prior architectural decision recall
- contradiction-aware follow-up
- source-specific follow-up question
- long-running implementation discussion
- topic shift and return
- stale-session material that should not be selected
- multiple similar prior decisions
- unresolved versus resolved decision distinction

## Phase 3: Baselines

**Status: complete (offline A/B/C replay harness, 2026-06-22).** See
`docs/session_context_assembler_phase_3_notes.md`,
`prototype/session_context_assembler/replay.py`,
`tools/run_session_context_assembler_replay.py`, and
`tests/test_session_context_assembler_replay.py`. All six required Phase 3
gates pass on the frozen r0 corpus. This authorizes review of the Phase 4
gates only — it is not a claim that quality is non-inferior to full
history or that token reduction meets the Phase 4 target, and it does not
authorize an external consumer shadow evaluation or production use.

Evaluate three conditions:

```text
A. Full conversation history
B. Naive sliding-window history
C. Governed episode-selected context package
```

All conditions must use the same model, task prompts, source material, and
token budget accounting.

## Phase 4: Required Evaluation Gates

The session-context assembler may advance only if all gates pass.

**Phase 4R status: S1 automated gates PASS (2026-06-22).** The bounded
offline S1 selector meets the frozen-R1 retention, source, contradiction,
token, provenance, labeling, and budget requirements. This authorizes Phase 5
human-review design only; no human review, consumer runtime integration, or production use
is authorized. See `docs/session_context_assembler_phase_4r_notes.md` and
`benchmarks/results/session_context_assembler_r1_s1_replay.md`.

```text
source_id_preservation_rate = 1.0
parent_engram_lineage_preservation_rate = 1.0
provenance_loss_count = 0
synthetic_context_label_coverage = 1.0
unauthorized_memory_write_count = 0
unauthorized_governance_mutation_count = 0
```

Quality and efficiency targets:

```text
prior_decision_recall >= baseline
required_source_recall >= baseline
contradiction_awareness >= baseline
prompt_token_reduction >= 25%
answer_quality_non_inferior_to_full_history
```

Safety failures that block advancement:

```text
missing source IDs
missing parent Engram IDs
synthetic context presented as source truth
summary promoted without evidence re-grounding
selection of blocked or ineligible artifacts
governance-state mutation
retrieval-ranking mutation
```

These gates correspond to the "EpiCache-inspired session assembler" row of
the per-candidate evaluation gates in `docs/future_research_candidates.md`:
source-ID preservation, prior-decision recall, prompt-token reduction, and
zero provenance loss — any failure on the last two disqualifies the approach
regardless of token savings.

## Phase 5: Human Review

**Design status: complete; study not run (2026-06-22).** The condition-masked
protocol, blank review form, reproducible frozen packet set, coordinator-only
manifest, response compiler, and acceptance tests are complete. This authorizes
review of the human-study materials only. Reviewer recruitment, packet
distribution, response collection, consumer runtime integration, and production use remain
unauthorized. See
`docs/session_context_assembler_phase_5_human_review_protocol.md`.

After automated gates pass, run a small reviewer evaluation.

### Phase 5A: non-human technical verification

**Status: technical PASS; owner pack prepared but unreviewed (2026-06-22).**
Because independent reviewer recruitment is unavailable, Phase 5A uses a
separate held-out R2 robustness lane. It does not replace or satisfy Phase 5
human review. All technical gates and mutation-sensitivity checks pass. This
authorizes a separate proposal for a read-only, consumer-neutral technical
shadow adapter only, not implementation. See
`docs/session_context_assembler_phase_5a_notes.md`.

**Adapter design review:** architecture and contract accepted; isolated
implementation authorized by ADR 0008. The design requires
artifact-local lineage, package integrity/replay controls, disclosure and
redaction checks, structured fail-closed errors, honest external-retention
limits, and explicit rollback/data-retention behavior. See
`docs/session_context_assembler_consumer_neutral_shadow_adapter_design.md`.

**Implementation ADR review:** ADR 0008 is accepted for isolated local shadow
implementation only. It defines policy-pinned replay, authenticated delivery
binding, content-free telemetry, and kill-switch requirements. No network
listener, consumer connection, live routing, SDK, deployment, or MNEMOS
authority-surface mutation is authorized.

**Isolated implementation status:** technical gates PASS. This authorizes
review of an authorized consumer-neutral shadow-evaluation proposal only; no
connection or active-path evaluation is authorized.

Reviewers should assess:

- whether required prior decisions were retained
- whether source references remain understandable
- whether selected context omitted critical information
- whether synthetic summaries were clearly distinguishable from source
  evidence
- whether the shorter context improved usability

No promotion claim should be made from synthetic evaluator responses.

## Phase 6: Decision Gate

Possible outcomes:

```text
PASS:
Create a consumer-neutral read-only shadow-adapter proposal.

PARTIAL:
Refine clustering, selection, or provenance packaging and rerun benchmark.

FAIL:
Archive prototype artifacts and retain findings in the research ledger.
```

A passing result does not authorize production integration automatically.

The proposal must preserve the architectural position:

```text
MNEMOS
-> governed durable memory
-> session-context assembler
-> read-only context package
-> authorized consumer adapter
-> external application, agent, workflow, or operator interface
```

SAM may later serve as one example test consumer. It is not the architectural
owner, default runtime, or product identity of the assembler.

Any later production proposal must include:

- a separate implementation ADR
- canary deployment plan
- observability and rollback design
- tenant and access-control review
- production latency and cost analysis
- explicit approval for any new route or consumer integration

## Sequencing

1. Approve this spec and `docs/adr/0007-session-context-assembler-shadow-only.md`.
2. Build the Phase 2 benchmark corpus (20-30 long-session cases).
3. Implement the Phase 1 offline prototype with no production route wiring.
4. Run Phase 3 baselines (A/B/C) under identical model, prompts, and token
   accounting.
5. Evaluate Phase 4 gates; block advancement on any zero-tolerance safety
   failure regardless of quality/efficiency gains.
6. Run Phase 5 human review only after automated gates pass.
7. Resolve Phase 6 decision gate. PASS produces a shadow integration
   proposal only, not a production change.

No runtime integration, Engram write path, retrieval-ranking change, or
agent-facing memory write begins until a separate implementation ADR is
approved and the gates above pass.
