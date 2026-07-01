# Associative Routing E1 — Shadow Integration Design Note

```text
ASSOCIATIVE_ROUTING_E1_SHADOW_ONLY
OPT_IN_EXPERIMENTAL
NO_DEFAULT_RETRIEVAL_CHANGE
NO_NEW_MEMORY_AUTHORITY
NO_GOVERNANCE_OR_AUTHORIZATION_CHANGE
NO_PRODUCTION_QUALITY_CLAIM
```

## Scope

E1 adds an opt-in, read-only shadow path that runs the E0 associative-routing
projection alongside normal MNEMOS retrieval and reports candidate paths for
comparison, without changing delivered results. This note records what was
built, what the offline evidence shows, what remains unanswered, and the
closeout decision among the four options in the E1 task spec.

## What was built

- `mnemos/retrieval/associative_shadow/` — `AssociativeShadowAdapter`, wrapping
  the frozen, unmodified E0 `AssociativeRouter`/`build_projection`/`load_corpus`
  over a new, expanded E1 fixture corpus (16 cues, 20 tags, 13 source documents
  spanning GateMem G4/G5, ADR 0013, R0 retrieval-hygiene, and the AI-developer
  memory trial). E0's own fixtures and 24 passing tests are untouched.
- Opt-in wiring in `service/app.py`: the `associative_routing_shadow` request
  field (default `false`) is threaded through `search_documents` and the
  `/v1/mnemos/search` route. When absent or `false`, the response body and
  meta are unchanged from today. When `true`, `meta.associative_routing_shadow`
  is attached additively (never reorders, suppresses, or injects into
  `results`). `RetrievalRouter.search` itself was deliberately left unmodified
  — the shadow path is composed at the service boundary only.
- Fail-closed behavior: `MNEMOS_DISABLE_ASSOCIATIVE_SHADOW=true` or any
  internal adapter error yields `status: "unavailable"` and never raises into
  the request path.
- `tests/test_associative_routing_e1_shadow.py` (11 tests, all passing):
  E1 projection verifies clean; kill switch; adapter resilience to internal
  errors; no authority/governance fields in the shadow payload; flag-off
  produces byte-identical results and omits the shadow key; flag-on attaches
  the block additively without altering `results`; request-layer validation
  of the new boolean field.
- `docs/experiments/associative_routing_e1_comparison_pack.json` — the frozen,
  hand-verified 30-query pack (10 direct-state, 6 temporal/supersession,
  6 dependency/blocker, 4 ambiguity/multi-path, 4 unrelated negative controls)
  required by the spec, plus a separate, disjoint
  `associative_routing_e1_fresh_verification_pack.json` for any future
  improvement claim.
- `tools/run_associative_routing_e1_comparison.py` — the four-condition runner
  (A semantic / B hybrid / C associative shadow / D associative-candidates-
  resolved-through-normal-retrieval), reusing the R0 hygiene-benchmark
  helpers for the live legs and the E0/E1 verifier for the offline ones.

## Evidence

Both blockers noted in the original draft of this note were resolved with
explicit authorization from the program owner: `mcp` was installed, and a
verified-idle MNEMOS instance at `localhost:8700` was reseeded (collection
`mnemos_ai_dev_e2_task_01` was unrelated foreign-task data, confirmed safe to
repoint) with all 13 E1 source documents via
`tools/seed_mnemos_repo_context.py`.

**Condition C** (associative shadow) ran against the full frozen 30-query
pack, offline and deterministically: zero false abstentions, zero scoring
misses against the pack's hand-verified expectations, sub-millisecond
latency (avg 0.10ms, max 0.80ms).
[benchmarks/results/associative_routing_e1_comparison_run_001.json](../benchmarks/results/associative_routing_e1_comparison_run_001.json)

**Conditions A, B, D** then ran as the full live 8-leg cold/warm matrix
(`tools/run_associative_routing_e1_comparison.py`, no `--offline-only`)
against the freshly seeded instance:
[benchmarks/results/associative_routing_e1_comparison_run_002.json](../benchmarks/results/associative_routing_e1_comparison_run_002.json)

| Metric (n=120 runs/path, n=30 for C) | A semantic | B hybrid | C associative |
|---|---:|---:|---:|
| Top-1 neighborhood accuracy | 0.40 | 0.37 | **1.00** |
| Top-3 neighborhood recall | 0.50 | 0.50 | **1.00** |
| Abstention accuracy | 0.77 | 0.77 | **1.00** |
| False abstention rate | 0.00 | 0.00 | 0.00 |
| Duplicate result rate | 0.00 | 0.00 | n/a |
| Latency | live | live | ~0.1ms avg |

**Condition D** (do C's candidates appear in A/B's actual top-3?): average
associative-candidate recall within normal retrieval results was **0.51**
across 176 resolved-query/leg pairs — roughly half of what associative
routing identifies as relevant is *also* found by semantic/hybrid retrieval
unprompted, and half is not.

All abstention misses on A/B came from the negative-control and no-supported-
relationship queries (e.g. "what superseded the G4 implementation lane" /
unrelated queries): the live `low_relevance_abstention` guard rarely
triggers, so semantic/hybrid almost always returns *something*, even when
nothing relevant exists. Associative routing abstained correctly every time.

**Held-out fresh-verification pack** (6 queries, not used to build the
projection or the comparison pack, run warm-only against the same seeded
instance):
[benchmarks/results/associative_routing_e1_fresh_verification_run_001.json](../benchmarks/results/associative_routing_e1_fresh_verification_run_001.json)

| Metric (n=6) | A semantic | C associative |
|---|---:|---:|
| Top-1 accuracy | 0.50 | **1.00** |
| Top-3 recall | 1.00 | 1.00 |

The pattern from the comparison pack holds directionally on the held-out
pack (top-3 recall saturates at 100% for both on this small, easy set —
expected given only 13 seeded documents — but top-1 still favors
associative routing 2:1).

All architecture-level safety gates pass on both runs: deterministic
projection, all cues/tags source-linked, no authority-field leak, default
retrieval unchanged with the flag off, shadow response isolated, no
duplicate candidate delivery.

## What this shows and what it doesn't

It shows, with both a disclosed comparison pack and an independent held-out
pack pointing the same direction, that on this corpus associative routing
finds the *correct* evidence neighborhood more reliably and abstains more
honestly than MNEMOS's real semantic/hybrid retrieval — and that roughly
half of what it finds is genuinely novel relative to what normal retrieval
surfaces on its own (condition D).

It does **not** show this generalizes beyond a 13-document, hand-curated
corpus with manually authored cue/tag relationships. The corpus is small,
the relationships were authored by the same person who wrote the queries
(disclosed non-independence on the comparison pack), and the result has not
been tested on a larger, less curated corpus, under production query
variation, or for downstream agent-task benefit, operational latency at
scale, or maintenance cost as the cue/tag registry grows. These remain the
real open questions for any integration beyond shadow mode.

## Closeout decision

**Option 3: propose E2 — a limited, opt-in candidate-expansion integration
behind a kill switch and an explicit evaluation profile.**

The evidence clears the bar this note's earlier draft set for considering
E2: non-inferiority is not just met but exceeded on top-1/top-3/abstention
across both the disclosed and held-out packs, with zero false abstentions
and zero correctness regressions in the shadow path. E2 should still be
scoped narrowly — candidate *expansion* (surfacing associative candidates
alongside, not replacing, normal results), opt-in, with the same kill switch
and isolation discipline as E1, and its own frozen evaluation profile rather
than reusing this one.

**Not chosen:**
- *Option 1 (offline-only)* — no longer fits; the shadow path has now shown
  measurable value live, not just architectural safety.
- *Option 2 (stay shadow/diagnostic-only)* — this was the right call before
  the live comparison ran; the comparison now supports moving past it.
- *Option 4 (reject)* — directly contradicted by the measured results.

## Required before E2 implementation

1. Scale the corpus beyond 13 hand-curated documents and confirm the pattern
   holds — the current result is real but small-sample.
2. A genuinely held-out comparison pack authored by someone other than the
   cue/tag registry author, to remove the disclosed non-independence.
3. A frozen E2 evaluation profile defining exactly how associative
   candidates may be surfaced (rank position, labeling, maximum count) and
   the specific kill-switch and rollback mechanism.
4. Sign-off through the same authorization process used for E0/E1 — this
   note does not itself authorize E2.

No default retrieval change, governance change, or production-quality claim
is made by E1 as implemented. The `associative_routing_shadow` flag remains
off by default; nothing in this evidence run altered delivered results for
any caller that did not explicitly opt in.
