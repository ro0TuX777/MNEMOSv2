# ADR 0007: Session Context Assembler Remains Shadow-Only Research

Date: 2026-06-21

Status: Accepted

## Context

Long multi-turn sessions used by external applications, agents, workflows,
and operator interfaces accumulate conversation
history and source-linked Engrams faster than they can be cheaply carried in
a model's working context. EpiCache (Apple Research,
[arXiv:2509.17396](https://arxiv.org/abs/2509.17396),
[apple/ml-epicache](https://github.com/apple/ml-epicache)) demonstrates that
episodic clustering plus episode-specific KV-cache eviction can reduce prompt
burden while preserving long-conversation QA accuracy. The concept is
attractive for MNEMOS consumers, but EpiCache itself is research code
operating on the inference-time KV cache — a different layer than MNEMOS's
durable, governed memory.

There is a real risk in adapting the concept: a session-context assembler
that selects and summarizes prior conversation material could become a
second, ungoverned memory store if its output is ever treated as source
truth, written back as an Engram, or allowed to influence governance,
authority, or promotion state.

## Decision

A session-context assembler inspired by EpiCache's episode-selection concept
is approved as a shadow-only research lane, specified in
`docs/session_context_assembler_spec.md`. It is research and offline
prototype only. No production integration, retrieval-ranking change,
governance mutation, authority mutation, promotion change, Engram mutation,
or Resolution Engram mutation is authorized by this ADR.

The assembler is a MNEMOS capability, not a feature owned by any consumer.
Its architectural boundary is:

```text
MNEMOS governed durable memory
-> session-context assembler
-> read-only context package
-> authorized consumer adapter
-> external application, agent, workflow, or operator interface
```

SAM is one possible future consumer for testing only; it is not part of
MNEMOS's core architecture or product boundary.

## Alternatives Considered

- Adopt `apple/ml-epicache` directly as a runtime dependency and second
  caching layer. Rejected: it is research code tied to specific
  model/runtime versions, with no provenance or governance model of its own.
- Skip the session-context-assembler concept entirely and rely only on
  full-history or naive sliding-window context. Rejected: this leaves a
  real, named gap (prompt cost and continuity loss in long sessions)
  unevaluated.
- Allow selected episode summaries to be written back as Engrams to make the
  assembler "self-improving." Rejected: this would let an ephemeral,
  non-authoritative selection layer mutate durable governed memory, exactly
  the failure mode this ADR exists to prevent.

## Invariants

- MNEMOS remains the sole durable, governed-memory layer; the session-context
  assembler is an ephemeral, consumer-side layer only.
- Every adapter is consumer-neutral and read-only. A consumer receives bounded
  context and provenance, never authority to write Engrams, mutate governance
  or contradiction state, promote synthetic context, alter retrieval ranking,
  or treat package content as source truth.
- Any future adapter contract must preserve artifact-local lineage, package
  integrity and replay controls, disclosure/redaction enforcement, structured
  fail-closed errors, bounded retention, and rollback that leaves the live
  consumer path and MNEMOS state unchanged.
- Every selected session segment or episode summary must be labeled
  `synthetic_context`, retain parent Engram IDs and source IDs, retain
  lineage metadata, and remain non-authoritative and non-promotable.
- The assembler must be excluded from governance-state mutation and from
  Resolution Engram creation unless independently re-grounded in source
  evidence.
- No production route, retrieval-ranking change, or agent-facing memory
  write is authorized until a separate implementation ADR passes the gates
  defined in `docs/session_context_assembler_spec.md`.
- Synthetic evaluator or AI-generated review responses do not substitute for
  the human-review step required before any promotion claim.

## Rollback

If session-context-assembler code or artifacts are found writing Engrams,
altering retrieval ranking, mutating governance or contradiction state, or
being treated as source truth anywhere outside the isolated prototype
module, disable the path immediately and revert the lane to spec-only status
pending review.

## Evidence

- `docs/session_context_assembler_spec.md`
- `docs/session_context_assembler_corpus_design.md`
- `benchmarks/truthsets/session_context_assembler_r0.json`
- `benchmarks/truthsets/session_context_assembler_r0.manifest.json`
- `docs/session_context_assembler_phase_1_notes.md`
- `prototype/session_context_assembler/` (offline prototype, no production import path)
- `tests/test_session_context_assembler_prototype.py`
- `docs/session_context_assembler_phase_3_notes.md`
- `prototype/session_context_assembler/replay.py` (offline A/B/C replay harness, no production import path)
- `tools/run_session_context_assembler_replay.py`
- `tests/test_session_context_assembler_replay.py`
- `benchmarks/results/session_context_assembler_r0_replay.json`
- `benchmarks/results/session_context_assembler_r0_replay.md`
- `docs/session_context_assembler_corpus_design_r1.md`
- `docs/session_context_assembler_phase_2r_notes.md`
- `benchmarks/truthsets/session_context_assembler_r1.json`
- `benchmarks/truthsets/session_context_assembler_r1.manifest.json`
- `benchmarks/results/session_context_assembler_r1_replay.json`
- `benchmarks/results/session_context_assembler_r1_replay.md`
- `tests/test_session_context_assembler_r1.py`
- `prototype/session_context_assembler/selector_s1.py`
- `tests/test_session_context_assembler_selector_s1.py`
- `docs/session_context_assembler_phase_4r_notes.md`
- `benchmarks/results/session_context_assembler_r1_s1_replay.json`
- `benchmarks/results/session_context_assembler_r1_s1_replay.md`
- `docs/session_context_assembler_phase_5_human_review_protocol.md`
- `docs/session_context_assembler_phase_5_review_form.md`
- `tools/build_session_context_assembler_review_packets.py`
- `tools/compile_session_context_assembler_review_responses.py`
- `tests/test_session_context_assembler_human_review_protocol.py`
- `benchmarks/review_packets/session_context_assembler_phase_5/coordinator_manifest.json`
- `docs/session_context_assembler_phase_5a_protocol.md`
- `benchmarks/truthsets/session_context_assembler_r2.json`
- `benchmarks/truthsets/session_context_assembler_r2.manifest.json`
- `tools/run_session_context_assembler_r2_verification.py`
- `tests/test_session_context_assembler_phase_5a.py`
- `benchmarks/results/session_context_assembler_r2_verification.json`
- `benchmarks/results/session_context_assembler_r2_verification.md`
- `docs/session_context_assembler_phase_5a_notes.md`
- `docs/session_context_assembler_consumer_neutral_shadow_adapter_design.md`
- `docs/adr/0008-consumer-neutral-read-only-shadow-adapter-implementation.md` (accepted; isolated shadow implementation only)
- `prototype/session_context_assembler/shadow_adapter/` (isolated local shadow only)
- `docs/session_context_assembler_shadow_adapter_implementation_notes.md`
- `docs/future_research_candidates.md`
- `docs/adr/0004-ebir-shadow-only.md`
- `docs/associative_retrieval_a1_spec.md`
