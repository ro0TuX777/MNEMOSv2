# MNEMOS Future Research Candidates

Date: June 21, 2026

Status: **Mixed.** Most entries remain parking-lot candidates: not
scheduled, not branched, no code, no benchmark commitment (see
`docs/adr/0004-ebir-shadow-only.md` and
`docs/associative_retrieval_a1_spec.md` for the governing pattern). The
session-context assembler is the exception — it has an accepted Phase 0
research contract (`docs/adr/0007-session-context-assembler-shadow-only.md`,
`docs/session_context_assembler_spec.md`) and remains offline-prototype-only;
it has not been promoted to runtime integration.

Governing invariant for everything in this file:

```text
None of these candidates touch Engrams, Resolution Engrams, governance,
promotion, authority, or production retrieval ranking. They are either
ingestion adjuncts or inference-serving-plane optimizations.
```

---

## Candidate: PixelRAG (visual evidence retrieval)

Source: https://github.com/StarTrail-org/PixelRAG

What it is: renders web pages/PDFs/images into screenshot tiles, embeds them
with a vision-language model, and retrieves over the visual index. Preserves
layout-dependent content (tables, charts, diagrams) that text extraction
loses.

Where it could fit: a future multimodal evidence-intake lane, *upstream* of
MNEMOS's evidence bundle — not a replacement for governance.

```text
PDF / web page / dashboard screenshot
  -> PixelRAG render + visual retrieval
  -> source-linked visual evidence tile
  -> MNEMOS evidence bundle
  -> governance labeling
  -> retrieval / review packet
```

Caveats: hosted/FAISS-based index path and a patched headless Chrome
dependency — needs isolated evaluation before any embedding into the
runtime. No provenance/tenant/authority model of its own; MNEMOS would have
to supply all of that.

First question if revisited: does a visual-evidence lane improve
required-evidence recall on layout-dependent queries (tables, charts,
diagrams) versus text-only retrieval, without any governance regression?

---

## Candidate: KV-cache / session-context compression (TurboQuant, OSCAR, EpiCache)

Source (survey): https://www.marktechpost.com/2026/06/18/the-kv-cache-compression-race-turboquant-vs-oscar-vs-epicache/
Source (EpiCache paper): https://arxiv.org/abs/2509.17396
Source (EpiCache code): https://github.com/apple/ml-epicache

These operate on the ephemeral inference-time KV cache or session context —
a different layer than MNEMOS's durable, governed memory.

```text
Durable governed memory      MNEMOS (Engrams, lineage, evidence, contradiction,
                              authority/promotion) — unaffected by any of these.

Session / working-memory     EpiCache-style episode segmentation — closest
                              conceptual fit. Candidate for a governed
                              consumer-neutral session-context assembler: retrieve
                              source-linked Engrams, attach only the relevant
                              prior episode summary, label its provenance,
                              bound the working set. Any selected episode
                              summary or session segment MUST be labeled
                              `synthetic_context`, retain parent Engram/source
                              IDs, and remain non-authoritative and
                              non-promotable — this is what stops a session
                              optimization layer from quietly becoming a
                              second, ungoverned memory store.

                              SAM may be used as one future test consumer,
                              but is not the architectural owner, default
                              runtime, or product identity of the assembler.

Model serving layer          TurboQuant (data-oblivious, model-agnostic 1-bit
                              residual quantizer; covers both KV-cache and
                              ANN/vector search) / OSCAR (INT2 KV-cache,
                              attention-aware calibration, requires
                              vLLM/SGLang + a supported model) — VRAM/latency
                              optimization only, no truth or governance
                              change.
```

### EpiCache detail (Apple Research, training-free)

Method, per the primary paper/code: (1) **block-wise prefill** bounds peak
memory by processing conversation input in blocks rather than full-context at
once; (2) **episodic clustering** segments dialogue history into thematic
episodes; (3) **episode-specific eviction** prunes KV cache per-episode
(layer-sensitivity-aware budget allocation) rather than per-query, which is
what makes it robust across multi-turn conversations — query-specific
eviction schemes degrade turn-over-turn.

Reported results (LongMemEval, RealTalk, LoCoMo benchmarks): up to 30%
accuracy improvement over prior KV-eviction baselines, near full-cache
accuracy at 4-6x compression, up to 2.4x latency reduction and 3.7x peak
memory reduction.

Scope/caveats: this is research code to reproduce paper results, not a
production library — built on KVzip/AdaKV, requires CUDA 12.1 +
flash-attention 2.7.4, validated on Qwen/LLaMA at 3B-8B scale. Targets
long-form conversational QA specifically; the paper does not characterize
behavior on single-turn tasks or fragmented/ambiguous topic boundaries during
episode detection. Any MNEMOS-adjacent use would be evaluating the
*episode-segmentation concept* for a session-context assembler, not adopting
this repo's inference engine directly.

**Status update: promoted out of the parking lot.** The session-context
assembler has its own Phase 0 research contract:
`docs/adr/0007-session-context-assembler-shadow-only.md` and
`docs/session_context_assembler_spec.md`. It remains offline-prototype-only
and shadow-only; no production integration is authorized. TurboQuant and
OSCAR remain parking-lot-only candidates pending capacity.

Relative priority if revisited: EpiCache's episode-selection concept first
(reinforces "retrieve less, retrieve deliberately, keep provenance" — already
MNEMOS's posture), then TurboQuant on the serving plane (broadest
applicability), then OSCAR only once a supported long-context model/runtime
is settled. TurboQuant is also a candidate for a later, separate vector-store
compression benchmark against Qdrant-adjacent indexing — gated on
required-evidence recall, not speed alone.

---

## Per-candidate evaluation gates

Each candidate clears its own evidence bar — "human-value or recall
evidence" is not interchangeable across them:

- **PixelRAG**: required-evidence recall on layout-dependent queries, visual
  citation fidelity (does the cited tile actually support the claim), and
  lineage/tenant safety (no provenance loss, no cross-tenant leakage through
  the visual index).
- **EpiCache-inspired session assembler**: source-ID preservation (parent
  Engram/source IDs survive episode selection), prior-decision recall (does
  the assembler retrieve the episode that actually contains the relevant
  decision), prompt-token reduction, and zero provenance loss — any failure
  on the last two disqualifies the approach regardless of token savings.
- **TurboQuant / OSCAR**: answer agreement with baseline, citation fidelity
  under compression, VRAM/latency gains, and no governance-sensitive
  degradation (no dropped authority caveats, no lost source IDs) — speed
  gains alone do not clear the gate.

## Promotion path (if any candidate is picked up)

1. Confirm EBIR-R2 / A1 capacity allows new research-lane work.
2. Write a dedicated ADR + spec mirroring `docs/adr/0004-ebir-shadow-only.md`
   and `docs/associative_retrieval_a1_spec.md`: explicit blocked-list,
   shadow-only status, and a benchmark plan scoped to that candidate's gate
   above.
3. Run as an isolated `/prototype` session before any benchmark harness
   integration.
4. No live retrieval routing, ranking, or promotion changes until the
   candidate's own gate passes — not a generic bar borrowed from a different
   candidate.
