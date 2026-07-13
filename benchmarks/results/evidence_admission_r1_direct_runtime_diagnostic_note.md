# Evidence Admission R1 — Direct-Runtime Diagnostic Note (run 001)

## ERRATUM (seeding fidelity defect, discovered during formal provisioning)

Run 001 was seeded with document IDs keyed by source **basename**, and four
corpus sources share the basename `README.md`. Upserts therefore overwrote 11
of the 684 retrieval units (collection held 673 points): in the diagnostic
collection, `docs/benchmarks/gatemem_g5/README.md` was entirely absent,
`docs/README.md` retained 3/5 units, and the root `README.md` retained 10/15.

Impact: 5 non-abstention queries had degraded drivers (`r1f-009`, `r1f-034`,
`r1f-035`, `r1f-038`, `r1f-039`); 3 of them scored "not covered" in every
condition (`r1f-009`, `r1f-034`, `r1f-039`) and those misses are plausibly
seeding artifacts, not retrieval failures — normal-baseline coverage may be
understated by up to 3 (31/42 → up to 34/42). The defect was identical across
all four conditions, so **cross-condition findings remain valid**: kill-switch
identity 54/54, shadow read-only equivalence, zero forbidden routes, the
abstention gap, and the route-collapse finding are unaffected. The absolute
coverage rates and the −4.76 pp delta should be treated as additionally
uncertain (they were already diagnostic-only due to the embedder divergence).

The runner's ID scheme is fixed (full-path-keyed, collision-free, verified
684/684) in the same commit as this erratum. Run 001's JSON artifact is left
as-committed for the record.

## Evidence class

- DIRECT_RUNTIME_ONLY_EVIDENCE
- FORMAL_CLAIM_PERMITTED=false
- NOMIC_EMBEDDER_DIAGNOSTIC_ONLY
- NOT_AGGREGATABLE_WITH_FORMAL_HTTP_RESULTS

Artifact: `benchmarks/results/evidence_admission_r1_direct_runtime_diagnostic_run_001.json`
Runner: `tools/run_evidence_admission_r1_comparison.py`
Exam: independently authored formal pack, 54 scored queries,
SHA-256 `f09651f3fc67b0bddf73b3981a0f635e21c58ff3d4ed50bc717d2886377c14cc`
(evaluation against a subsequently independently authored, frozen, hashed
formal pack — not strict pack-before-implementation preregistration).

This run seeds the frozen corpus content faithfully (41 sources,
word_window 120/20 → 684 units, zero per-file count mismatches vs the frozen
manifest) but uses `nomic-ai/nomic-embed-text-v1.5` — not the frozen profile's
`BAAI/bge-base-en-v1.5` — because the repository QdrantTier natively supports
only nomic at 768-dim. All numeric results are therefore diagnostic-only.
Diagnostic collection `evidence_admission_r1_frozen_corpus_diag` is retained
for local reproduction only and must not be reused for the formal run.

## Four-condition results (diagnostic numbers)

| Condition | Coverage (non-abstain) | Abstain served | Forbidden routes | Fallback |
|---|---|---|---|---|
| normal baseline | 31/42 (73.8%) | 1/12 | 0 | 0 |
| R0 shadow only | 31/42 (73.8%) | 1/12 | 0 | 0 |
| R1 enforcement | 29/42 (69.0%) | 1/12 | 0 | 6 |
| R1 gate disabled | 31/42 (73.8%) | 1/12 | 0 | 0 |

Kill-switch identity: 54/54 queries byte-order identical between
gate-disabled and normal. Diagnostic coverage delta normal→enforced:
−4.76 pp (exceeds the 2 pp preregistered margin, but the preregistered test
applies only to the formal BGE/HTTP run, not to this diagnostic).

## Structural findings (embedder-independent)

1. **Safety controls pass.** Kill switch exact-identity 54/54; R0 shadow
   read-only (condition 2 ≡ condition 1); zero forbidden route labels served
   in any condition.
2. **Out-of-corpus abstention expectation not met.** 11 of 12
   abstention-expected queries were served (7 bounded-semantic, 4 fallback)
   rather than abstained. By design, R1 only enforces pre-retrieval abstain
   for the service-scope-unknown reason; with cue/tag registries empty,
   content-level out-of-corpus queries never reach an enforceable abstain.
3. **Route collapse.** With empty cue/tag registries and no cache fixtures,
   R0 recommends semantic for nearly everything; enforcement reduced to
   BOUNDED_SEMANTIC_RETRIEVAL ×48 + NORMAL_RETRIEVAL_FALLBACK ×6.
   CUE_ONLY_LOOKUP and CACHE_ONLY were never exercised. The current
   experiment therefore mostly tests "semantic retrieval with a smaller
   candidate bound", not the intended route mix.
4. **Coverage-loss mechanism.** Both enforced coverage losses (`r1f-013`,
   `r1f-022`) occurred under BOUNDED_SEMANTIC_RETRIEVAL where the bound
   dropped the correct driver from top-k while R0 sufficiency still judged
   the bounded result SUFFICIENT, so the mandatory fallback never fired
   ("sufficient-but-lost-the-driver").

## Runtime configuration facts (recorded for formal-run attribution)

- `cue_registry_state`: empty_not_populated
- `tag_registry_state`: empty_not_populated
- `cache_fixture_state`: no_cache_fixtures_seeded

These facts distinguish a formal failure caused by unsafe R1 policy from a
formal failure caused by the frozen runtime not providing the declared
cue/cache mechanisms the pack was designed to exercise. Neither issue is to
be fixed before the formal run; the formal evaluation measures the currently
frozen system honestly.

## Decision state

- R1_STRUCTURAL_SAFETY_CONTROLS_PASS
- R1_DIRECT_RUNTIME_DIAGNOSTIC_COMPLETE
- FORMAL_BGE_HTTP_EVALUATION_NOT_YET_RUN
- R1_RETENTION_DECISION_PENDING_FORMAL_RESULT

R1 is **not retained** on this evidence. Next step is the frozen
four-condition formal evaluation on a separately seeded, separately
manifested BGE corpus behind a revision-proving HTTP service, with no changes
to R1 policy, thresholds, routes, registries, or corpus.
