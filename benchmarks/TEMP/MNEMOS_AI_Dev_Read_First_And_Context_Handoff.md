# MNEMOS AI Dev Handoff
## Read Order, Purpose, and Working Context

## 1. Why you are receiving this

You are joining an active architecture and benchmarking effort around **MNEMOSv2**.

Your job is **not** to invent a new direction from scratch.  
Your job is to continue a deliberate product and systems effort that has already produced:

- benchmarked retrieval comparisons
- profile/deployment decisions
- reranker gates and failure analysis
- hybrid retrieval experiments
- two new strategy plans:
  - **MemArchitect-inspired governance**
  - **Memory-Over-Maps-inspired lightweight memory architecture**

This handoff tells you:
1. what MNEMOS is trying to become
2. what conclusions are already frozen
3. what files to read first
4. what work should happen first

---

## 2. Core purpose of MNEMOS

MNEMOS is no longer being treated as “just a vector database wrapper” or “just another memory store.”

The target direction is:

> **MNEMOS should become a governed memory system with strong retrieval discipline, source grounding, lifecycle management, contradiction handling, and selective on-demand synthesis.**

That means we care about:
- what memory is retrieved
- what memory is allowed into context
- what memory should decay
- what memory should be consolidated
- what memory should be deleted
- what memory should remain source-grounded instead of being over-converted into derived structure

In practical terms, MNEMOS is evolving in this order:

### First
**MemArchitect-inspired governance layer**
- memory lifecycle
- contradiction resolution
- trust/utility scoring
- deletion cascade
- relevance veto

### Second
**Memory-Over-Maps-inspired lightweight memory architecture**
- source-grounded memory bank
- candidate narrowing before expensive reasoning
- on-demand derived views
- lightweight caching
- avoid overbuilt permanent structure

---

## 3. Frozen conclusions you should not re-litigate without new evidence

These are the current working truths from the benchmark program.

### A. Core/Qdrant is the default operational path
It consistently won on:
- ingest speed
- latency
- QPS

Do not assume another backend should replace Core by default unless new evidence proves it.

### B. Governance/pgvector is architecturally viable, but not a proven retrieval-quality winner
It has not yet shown a strong enough retrieval advantage to justify displacing Core as the broad default.

### C. The current reranker path is experimental / non-production
Track 2 and Gate B showed that the current ColBERT path is not trustworthy enough for production claims.
Do not quietly reintroduce it as a default path.

### D. Hybrid retrieval exists, but it did not justify a default switch
Gate C completed successfully as an implementation effort, but the latest real-corpus benchmark did **not** show a strong enough quality-class win to make hybrid the broad default.
Hybrid remains available, but not the default.

### E. Benchmark discipline matters
Do not promote features because they sound good.
Promote features only when:
- they have a clear benchmark win
- their trade-offs are understood
- the product message stays clean

---

## 4. What you should read first

Read in this order.

## Read 1 — Benchmark doc first
**File:** `benchmark.md`

This is the highest-priority context file.

Why:
- it contains the actual benchmark history
- it tells you what has already been tested
- it tells you what failed
- it tells you which conclusions are frozen
- it defines Gate A / Gate B / Gate C / Gate D logic

Do not start implementation work before reading this carefully.

When reading, focus especially on:
- Track 1 conclusions
- Track 2 / Gate A / Gate B reranker conclusions
- Gate C finalized operator guidance
- current frozen posture

---

## Read 2 — MemArchitect implementation plan
**File:** `MNEMOS_MemArchitect_Implementation_Plan.md`

Read this second.

Why:
- this is the next major product direction
- this explains the purpose of the governance layer
- this describes the read / reflect / background hygiene architecture
- this defines which policies should be implemented in v1

This document tells you the intended architecture and feature scope.

---

## Read 3 — MemArchitect task board
**File:** `MNEMOS_MemArchitect_Task_Board.md`

Read this third.

Why:
- this breaks the governance plan into concrete epics and tasks
- this tells you implementation order
- this defines acceptance criteria
- this should drive near-term execution

This is the practical working document for the first major phase.

---

## Read 4 — Memory Over Maps implementation plan
**File:** `MNEMOS_Memory_Over_Maps_Implementation_Plan.md`

Read this fourth.

Why:
- this is the second major architectural phase
- it tells you how to avoid overbuilding permanent intermediate structure
- it describes source-grounded memory, candidate narrowing, on-demand synthesis, and lightweight cache behavior

Important:
This comes **after** MemArchitect-inspired governance, not before.

---

## Read 5 — Memory Over Maps task board
**File:** `MNEMOS_Memory_Over_Maps_Task_Board.md`

Read this fifth.

Why:
- it breaks the lightweight memory architecture into actionable work
- it defines epics and sprint exit criteria
- it should only become active after the governance layer is sufficiently underway

---

## 5. Practical read sequence summary

If you want the shortest version:

1. `benchmark.md`
2. `MNEMOS_MemArchitect_Implementation_Plan.md`
3. `MNEMOS_MemArchitect_Task_Board.md`
4. `MNEMOS_Memory_Over_Maps_Implementation_Plan.md`
5. `MNEMOS_Memory_Over_Maps_Task_Board.md`

---

## 6. How to think about the product

MNEMOS is trying to become:

### Not this
- just a vector DB wrapper
- just a RAG pipeline
- just a memory log
- just another hybrid-search experiment

### More like this
- a **governed memory system**
- a **source-grounded retrieval system**
- a **policy-aware memory platform**
- a **memory lifecycle manager**
- a **selective synthesis architecture**

That means the product must eventually be able to answer:
- why this memory surfaced
- why another memory was suppressed
- which memories are stale
- which memory is the current winner among contradictions
- what was deleted and what derived artifacts were invalidated
- when a summary/view is grounded to sources vs synthesized temporarily

---

## 7. Immediate priority

Your immediate priority is:

> **MemArchitect-inspired governance layer first**

That means you should begin with the governance task board, not the Memory Over Maps task board.

The right first implementation area is:

### Advisory-mode governance
Start in advisory mode so the team can compare:
- raw retrieval behavior
- governed retrieval behavior

before enforcing suppression broadly.

### First features to prioritize
1. governance metadata on memory records
2. policy registry
3. relevance veto
4. governed scoring
5. contradiction detection
6. contradiction resolution
7. explain payload
8. reflect-path trust/utility updates
9. background hygiene jobs
10. delete cascade

---

## 8. What not to do

Do **not** do these unless explicitly re-opened by evidence:

### Do not re-open Track 2 as if it is solved
The reranker is still experimental/non-production.

### Do not make hybrid the default
Gate C did not justify that.

### Do not invent a brand new backend strategy
Core/Qdrant remains the default operational substrate.

### Do not turn derived views into the new always-on primary representation
That would conflict with the intended Memory Over Maps direction.

### Do not skip provenance
If lineage is weak, deletion cascade and grounded synthesis both become unreliable.

---

## 9. Engineering posture expected

The expected style of work is:

- benchmark-aware
- explicit about trade-offs
- source-grounded
- policy-driven
- conservative about defaults
- honest when a feature is still experimental

You are expected to prefer:
- explainable logic
- stable interfaces
- bounded expensive reasoning
- reversible rollout
- advisory mode before enforced mode

---

## 10. Suggested first execution path

### Phase 0 — orientation
Read the 5 files in the order above.

### Phase 1 — governance foundation
Start implementing from:
- `MNEMOS_MemArchitect_Task_Board.md`

Recommended first wave:
- governance schema
- policy registry
- governed score
- `/search` governance flags
- relevance veto
- contradiction detection

### Phase 2 — governance benchmarkability
Make sure the new governance layer can be measured:
- raw vs governed
- leakage reduction
- over-pruning risk
- contradiction suppression quality

### Phase 3 — deeper governance
Add:
- reflect path
- hygiene runner
- delete cascade
- governance stats

### Phase 4 — only then
Move into:
- `MNEMOS_Memory_Over_Maps_Task_Board.md`

---

## 11. Final instruction

Before proposing a new direction, ask:

1. Does this conflict with the benchmark conclusions?
2. Does this make the product clearer or muddier?
3. Does this reduce or increase ungrounded complexity?
4. Can this be benchmarked in a way that changes a product decision?
5. Does this preserve source grounding and governance discipline?

If the answer is weak on those questions, do not lead with it.

---

## 12. Working summary in one paragraph

MNEMOS is being evolved from a retrieval platform into a **governed, source-grounded memory system**. The benchmark program has already established that Core/Qdrant remains the operational default, the current reranker is still experimental, and hybrid retrieval is available but not broadly justified as a default. The next priority is to implement a **MemArchitect-inspired governance layer** with read, reflect, and hygiene paths, plus contradiction handling, relevance veto, lifecycle decay, provenance, and delete cascade. After that foundation is in place, the next phase is a **Memory-Over-Maps-inspired lightweight memory architecture**, where MNEMOS keeps source artifacts primary, narrows candidates before expensive reasoning, and performs on-demand synthesis only where justified.
