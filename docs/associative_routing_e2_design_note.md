# Associative Routing E2 — Limited Opt-In Candidate Expansion Design Note

```text
ASSOCIATIVE_ROUTING_E2_OPT_IN_ONLY
KILL_SWITCH_REQUIRED_DEFAULT_DISABLED
CANDIDATE_ADDITION_ONLY
NORMAL_RETRIEVAL_REMAINS_PRIMARY
NO_DEFAULT_ROUTE_CHANGE
NO_GOVERNANCE_OR_AUTHORITY_CHANGE
NO_PRODUCTION_QUALITY_CLAIM
INDEPENDENT_AUTHORSHIP_PACK_NOT_PRODUCED
```

## Scope

E2 adds an opt-in, bounded mechanism that lets associative routing append a
small number of source-linked candidates to the *delivered* retrieval
results — unlike E1, which only ever attached an observational shadow
block. This note records what was built, the live comparison evidence, an
environment finding that shaped how the comparison was actually run, and the
closeout decision among the spec's four options.

## What was built

- `mnemos/retrieval/associative_expansion/` — `CandidateExpansionEngine`,
  reusing the frozen E0 `AssociativeRouter`/`build_projection` unchanged,
  over a new fixtures directory that is a superset of E1's 13 documents plus
  a third, genuinely unrelated documentation family (PIT-8/PIT-9B/PIT-10,
  the Memory-Over-Maps production-adjacent evaluation lane) for corpus
  fairness — 19 cues, 25 tags, 16 documents, verified clean
  (`verify_projection` passes all 8 checks).
- Double opt-in, fail-closed: a request must set
  `associative_candidate_expansion: true` **and** the global
  `MNEMOS_ASSOCIATIVE_CANDIDATE_EXPANSION_ENABLED` switch must be `true`
  (default `false`). Bounded by `MAX_PATHS=3`, `MAX_ADDED_CANDIDATES=3`,
  `MAX_EXPANSION_LATENCY_MS=10`; `MAX_TRAVERSAL_DEPTH=2` is satisfied by
  construction since E0's router is structurally a single cue→tag→content
  hop (tested directly).
- Sequencing in `service/app.py`: governance and the low-relevance
  abstention decision run **unchanged, on normal results only**, exactly as
  before E2. Expansion only runs afterward, and only if normal retrieval did
  not abstain — so expansion can never mask a deserved abstention. Injected
  candidates are governed as their own batch through the same
  `Governor.govern()` call normal candidates use, then appended (never
  interleaved by score) and labeled `candidate_origin`. The response field
  `candidate_origin`/expansion metadata only appears on requests that set
  the flag, so the default (flag-off) response is byte-for-byte unchanged.
- 21 tests (`tests/test_associative_routing_e2_expansion.py`), covering
  every item in the spec's "Required Tests" list: flag-off identity,
  kill-switch block, no normal-result suppression, max-candidate
  enforcement, traversal-depth (documented), latency-budget fallback
  (stops early, not mid-call — verified with an artificially slow resolver),
  dedup against the existing envelope, origin labeling, source-lineage
  completeness, inactive/unresolvable-target rejection, governance-rejection
  passthrough, no authority-field injection, no durable-write call. Plus all
  24 E0 and 13 E1 tests still pass unchanged (69 total across the three
  generations).
- Frozen verification pack (22 queries: direct-status, dependency/blocker,
  temporal/supersession, ambiguity, unrelated-negative-control, and the
  spec-required stale/superseded-source category) and a separate, informal
  33-query development pack, both hand-verified against the engine before
  freezing (0 mismatches on first pass for the frozen pack).

## Environment finding: the live container could not run today's code

Attempting the live before/after comparison surfaced a real environment
issue, distinct from E1's `mcp`/collection blockers (both already resolved):
the `mnemos-service` Docker image was built **46 hours before this code
existed** (`build: .` with no source bind mount — only `./data` is
volume-mounted), so recreating the container did not pick up any E1 or E2
code, including E1's own flag from earlier today. A direct exec check
confirmed `/app/service/app.py` inside the running container has zero
occurrences of `associative_candidate_expansion`.

Before discovering this, the container was briefly recreated twice while
diagnosing it (once to add the kill-switch env var, which had no effect
since the code wasn't present; once attempting to remove it, which
incorrectly reset `MNEMOS_QDRANT_COLLECTION` to the compose-file default).
Both were corrected immediately — the collection was restored to
`mnemos_ai_dev_e2_task_01` (the actual value found running, not the value in
the committed override file, which is stale) and confirmed healthy with all
21 seeded documents intact before proceeding. No `docker-compose.*.yml`
override files were left behind.

**Resolution**: rather than rebuild/redeploy the shared container (a larger,
slower, more disruptive action against an instance shared with another
task), the comparison ran **locally**: `MnemosRuntime` instantiated directly
in-process with today's actual code, pointed at the same `qdrant` and
`postgres` backends via their host-published ports
(`localhost:6333`/`localhost:5432`). This exercises the real E2 code against
the real, currently-seeded collection without touching the deployed
container at all. (The lexical/postgres tier was skipped for this run — the
host lacks `psycopg_pool` — so the comparison used semantic retrieval only,
which is what both conditions need.)

This also means **E1's own "live comparison" earlier today never actually
exercised the deployed container's HTTP-level shadow-flag code path either**
— its condition C ran the same way, as a local offline adapter call. E1's
correctness conclusions stand (they came from the adapter and isolation
tests, which are unaffected), but the "live flag smoke" framing in its
design note should be read as "verified via local in-process execution
against live retrieval data," not "verified against the deployed service."

Per the E2 authorization's environment-isolation note, this finding is
recorded rather than fixed this phase:

```text
DOCKER_IMAGE_CODE_DRIFT_KNOWN
NO_E2_FINDING_DEPENDS_ON_THE_DEPLOYED_CONTAINERS_CODE
LOCAL_IN_PROCESS_EXECUTION_USED_AGAINST_LIVE_BACKENDS_INSTEAD
```

## Fair-comparison manifest

| | Value |
|---|---|
| Corpus | Same 16-document E2 fixture corpus (GateMem/G4/G5 + ADR 0013 + R0 + AI-dev-trial + PIT-8/9B/10) |
| Collection | `mnemos_ai_dev_e2_task_01` (qdrant, 21→22 docs) — identical for both conditions |
| Embedding model | `nomic-ai/nomic-embed-text-v1.5`, identical for both |
| Retrieval mode | `semantic`, identical for both |
| top_k | 5, identical for both |
| Query pack | `associative_routing_e2_verification_pack.json` (22 queries), identical for both |
| Governance | off (default; governance-rejection path separately verified by unit test, not exercised live) |

## Evidence

Live run: [benchmarks/results/associative_routing_e2_live_comparison_run_001.json](../benchmarks/results/associative_routing_e2_live_comparison_run_001.json)

| Metric | Value |
|---|---|
| Queries run | 22 / 22 |
| Expansion triggered (`status: expanded`) | 2 / 22 |
| Candidates added | 2 (1 each on v13, v15) |
| Candidates deduplicated against existing top-5 | 18 |
| Candidates rejected (unresolvable/policy) | 0 |
| Latency-budget exceedances | 0 (only reproduced under an artificial 20ms/call stress test in unit tests) |
| Normal results suppressed by expansion | 0 (count only ever grew; order/content of normal candidates unchanged) |
| False abstentions introduced | 0 (v08/v09/v17/v18/v19/v21 still abstain on the associative side with zero effect on normal results) |

**Per-candidate classification** (both candidates ever added):
- v13 ("What superseded trial comparison 001?"): added
  `ai_dev_memory_trial_comparison_002.md` — **correct_and_needed**. Normal
  semantic top-5 returned the superseded document itself plus four unrelated
  docs and never surfaced the actual answer.
- v15 ("Tell me about Retrieval Hygiene R0."): added
  `retrieval_hygiene_r0_run_003.json` — **correct_and_needed**. Normal
  semantic top-5 found 3 of 4 R0 evidence documents but missed the
  regression-run evidence; expansion completed the set.

No `correct_but_redundant`, `irrelevant`, `stale_or_superseded`, or
`rejected_by_existing_policy` candidates were added in this run.

**Novel candidate rate** (2/22 ≈ 9%) is low at `top_k=5` against this
21-document corpus largely because the corpus is small enough that normal
semantic retrieval already covers most relevant material within 5 slots.
This is a corpus-size/`top_k` artifact, not evidence that the mechanism is
narrow — both observed additions were genuine, correct gap-fills exactly in
the failure mode E1 already identified (current-state and
supersession/evidence-completeness queries where semantic similarity alone
under-ranks the structurally-correct answer).

All architecture-level safety gates pass:

```text
DEFAULT_RETRIEVAL_UNCHANGED_WITH_EXPANSION_OFF   pass
KILL_SWITCH_PASS                                 pass
CANDIDATE_EXPANSION_BOUNDED                       pass
NO_NORMAL_RESULT_SUPPRESSION                      pass
ALL_EXPANSION_CANDIDATES_SOURCE_LINKED            pass
ALL_EXPANSION_CANDIDATES_ORIGIN_LABELED           pass
NO_AUTHORITY_OR_GOVERNANCE_CHANGE                 pass
NO_DURABLE_WRITE_SIDE_EFFECT                      pass
FAIR_COMPARISON_MANIFEST_COMPLETE                 pass (table above)
NOVEL_CANDIDATE_USEFULNESS_REPORTED               pass (both classified)
VERIFICATION_PACK_COMPLETE                        pass (22 queries, 0 mismatches)
NO_UNEXPLAINED_REGRESSION_IN_NEGATIVE_CONTROLS    pass
NO_UNEXPLAINED_CURRENT_STATE_OR_SUPERSESSION_ERROR pass
```

## Closeout decision

**Option 3: retain as opt-in experimental candidate expansion — useful for
selected query classes, behind the kill switch (default disabled).**

The mechanism is safe by construction (bounded, fail-closed, governed,
never suppresses normal results) and, on the limited evidence available,
adds genuinely correct and needed candidates specifically for current-state
and evidence-completeness queries where semantic similarity under-ranks the
structurally-correct document. It should remain opt-in and off by default;
nothing here supports broader rollout.

**Not chosen:**
- *Option 1 (reject)* — contradicted by the evidence; zero defects, two
  clean correct-and-needed additions, no regressions.
- *Option 2 (operator-diagnostics only)* — too conservative now that
  delivery value (not just explanation value) has been demonstrated, even
  on a small sample.
- *Option 4 (propose E3)* — explicitly not supported without an
  independently-authored evaluation pack, which this phase did not produce
  (allowed by the spec, but it caps the claim at "useful for selected query
  classes," not "broadly superior").

## Limitations and required follow-on before any E3 proposal

1. **No independent-authorship pack.** All packs (E0 through E2) were
   authored by the same process that built the cue/tag registries.
2. **Small, curated corpus.** 16 documents; the 9% novel-candidate rate is
   informative but not generalizable to a large, messy production corpus.
3. **Governance interaction not exercised live.** Verified only by unit
   test (`test_governance_rejection_preserved_for_expansion_candidates`)
   with a stub governor, not against the real `Governor` with live
   candidates.
4. **Docker image/code drift.** A separate, lower-priority task should
   either bind-mount source for faster iteration or establish a rebuild step
   before any future live verification, so future phases don't need the
   local-in-process workaround used here.
5. Per the explicit exclusion in the authorization header, the MCP/Python
   global-environment `starlette` conflict (flagged at E1 closeout) remains
   unresolved and is out of scope for E2.

No default retrieval change, governance change, or production-quality claim
is made by E2 as implemented. `associative_candidate_expansion` and
`MNEMOS_ASSOCIATIVE_CANDIDATE_EXPANSION_ENABLED` both remain off by default.
