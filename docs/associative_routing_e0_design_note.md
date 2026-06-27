# MNEMOS Associative Routing View E0 — Design Note

Date: 2026-06-27

Status: **E0 implemented and passing its own completion gate on a small
frozen GateMem fixture pack. Offline, read-only, deterministic. No runtime
integration.**

## Relationship to Associative Retrieval A1

This is a different, narrower lane than
[`associative_retrieval_a1_spec.md`](associative_retrieval_a1_spec.md). A1 is
a deferred, PPR-based graph-projection benchmark over engram edges, blocked
behind EBIR-R2. E0 is a controlled-metadata Cue–Tag–Content routing view with
no PPR, no automatic relation extraction, and no LLM involvement anywhere in
its base path. The two lanes do not share code, do not share a corpus, and
neither depends on the other.

## What was built

```text
prototype/associative_routing_e0/
  models.py      Cue, Tag, ContentRef, RoutingPath, RoutingResponse, Abstention
  registry.py    loads + validates fixtures/{source_index,cue_registry,tag_registry}.json
  projection.py  deterministic snapshot builder + manifest (sha256 of every input/output stage)
  router.py      cue lookup -> typed-relationship traversal -> explanation -> abstention
  verify.py      independent re-derivation of every completion-gate check
  cli.py         python -m prototype.associative_routing_e0.cli {query|verify|manifest}
  fixtures/      9 Cues, 11 Tags, 6 Content references over real GateMem docs

tests/test_associative_routing_e0.py             24 tests
tools/run_associative_routing_e0_benchmark.py    E0-SMOKE evaluation, writes:
benchmarks/results/associative_routing_e0_benchmark.{json,md}
```

All Cue/Tag fixtures cite real, already-existing MNEMOS documents
(`docs/benchmarks/gatemem_program_status.md`, the GateMem G5 readiness
packet, and the GateMem G4 implementation/proposal docs) — no synthetic or
LLM-generated relationships.

## Completion gate

| Gate | Status |
|---|---|
| `ASSOCIATIVE_ROUTING_PROJECTION_BUILDS_DETERMINISTICALLY` | Pass — two independent builds from disk produce byte-identical manifests (`test_projection_rebuilds_deterministically`). |
| `ALL_TAGS_SOURCE_LINKED` | Pass — registry loader and `verify.py` both reject any Tag with empty `source_record_ids`. |
| `ALL_TAG_TARGETS_RESOLVE` | Pass — rejected at load time if `to_content_id` is unresolvable. |
| `NO_AUTHORITY_FIELDS_PRESENT` | Pass — closed field allowlist (`ALLOWED_CUE_FIELDS`/`ALLOWED_TAG_FIELDS`) plus an independent denylist scan in `verify.py`. |
| `READ_ONLY_QUERY_PATH_PASS` | Pass — `router.route()` only reads the projection; no write path exists in the package. |
| `NO_PATH_ABSTENTION_PASS` | Pass — out-of-domain and wrong-direction-relationship queries abstain with `NO_SUPPORTED_ASSOCIATIVE_PATH` rather than fabricating a path. |
| `CURRENT_STATE_AND_TEMPORAL_CASES_PASS` | Pass — `supersedes`/`superseded_by` resolve directionally; current-state queries prefer the live doc over the historical precursor. |
| `AMBIGUITY_AND_CONTRADICTION_CASES_PASS` | Pass — "What is the GateMem frozen baseline?" surfaces both the G2/G2A and G4 frozen artifacts as separate paths rather than picking one. |
| `NO_EXISTING_RETRIEVAL_REGRESSION` | Pass by construction — nothing in MNEMOS's retrieval path imports this package. |
| `E0_EVALUATION_REPORT_COMPLETE` | Pass on the frozen 10-query pack (see below); explicitly bounded in scope. |
| `NO_LLM_OR_RUNTIME_DEPENDENCY` | Pass — stdlib only (`hashlib`, `json`, `argparse`, `re`). |

Run the gate yourself:

```bash
python -m prototype.associative_routing_e0.cli verify
python -m pytest tests/test_associative_routing_e0.py
python tools/run_associative_routing_e0_benchmark.py
```

## Evaluation result (E0-SMOKE, 10-query frozen pack)

Comparator A (`semantic_keyword_baseline_proxy`) is a deterministic
bag-of-words ranker over the fixture documents' real text — an explicitly
declared **local proxy**, not MNEMOS's production semantic/hybrid retrieval.
No production-retrieval-quality claim is made anywhere in this report.

| Metric | Baseline proxy | Associative routing |
|---|---|---|
| All-required recall (mean) | 0.80 | 1.00 |
| Top-1 recall (mean) | 0.45 | 0.80 |
| False abstention count | — | 0 |
| Fallback/abstention correctness | — | 1.00 |
| Routing-path provenance completeness | — | 1.00 |

Full per-query detail: `benchmarks/results/associative_routing_e0_benchmark.md`.

This is evidence on a 10-query, single-domain (GateMem) fixture pack, not a
general retrieval-quality claim. The query pack is also the same one used to
design the fixtures, so the comparison demonstrates internal consistency
(correct directional/temporal/ambiguity handling) more than it demonstrates
generalization.

## Decision after E0

Recommended outcome: **(2) retain the associative projection as an optional,
read-only routing aid** — scoped narrowly to corpora where a small,
human-curated Cue/Tag registry is feasible to maintain (e.g. GateMem-style
phase-gated programs with a handful of canonical status documents). The
gains shown here come from precise relationship typing (supersedes vs.
superseded_by, blocked_by, frozen_baseline_of) that a keyword or embedding
baseline cannot represent, not from any graph-inference capability — so the
benefit is bounded by how much manual curation a corpus owner is willing to
do.

Not recommended: extending this into automatic Cue/Tag extraction, wiring it
into default retrieval, or treating it as a precursor to a first-class graph
store. None of those are supported by this evidence, and all are explicitly
excluded by the E0 authorization.
