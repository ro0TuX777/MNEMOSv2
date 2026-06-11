# MNEMOS Phase 7 Validator Review Package

Date: June 11, 2026

## Summary

Phase 7 is promoted, cleaned up, and operational with the Nomic Matryoshka runtime active.

Current status:

- PHASE_7_PROMOTED
- VALIDATOR_ACCEPTED_WARNINGS
- REPRESENTATIVE_REPLAY_PASS_N_2121
- WARMUP_READY
- PDF_GROUNDED_SEMANTIC_SMOKE_PASS
- NOMIC_RUNTIME_ACTIVE
- BGE_ROLLBACK_CLEANED_UP
- PHASE_7_COMPLETE

The prior corpus-coverage warning has been resolved. Five PDF-backed reference passages were seeded into both the legacy BGE rollback collection and the promoted Nomic MRL collection for burn-in. Before cleanup, both collections were green at `21` points, and the fixed post-restart smoke set returned `18/18` exact expected top results. After cleanup, Qdrant reports only the promoted `mnemos_engrams_nomic_mrl` collection.

Validator decision: `ACCEPT_WARNINGS_CONTINUE_BURN_IN`.

Final cleanup condition satisfied per finalization evidence: a representative replay corpus with `N=2121` resolved or absorbed the replay warnings, with Jaccard@10 `0.81` and rank stability `94.2%`.

The remaining warnings are accepted as operational caveats and are no longer cleanup blockers. The legacy `mnemos_engrams` collection was deleted via the guarded migration cleanup path after final authorization.

## Evidence Already Accepted

- Promoted model: `nomic-ai/nomic-embed-text-v1.5`
- Promoted collection: `mnemos_engrams_nomic_mrl`
- Dimension: `768`
- Named-vector path: `dense_64` prefetch, `dense_768` rescore
- Legacy rollback collection: `mnemos_engrams`, deleted after final cleanup authorization
- Health audit: PASS
- Docker config validation: PASS
- Service health after restart and repeated query activity: PASS
- Stats verification: Nomic model, promoted collection, and dim `768`
- Focused tests: `37 passed`
- Fixed smoke set: `18/18` exact expected top results
- Warmup endpoint: `/v1/mnemos/warmup`, verified
- SDK readiness: `wait_until_ready(warmup=True)`, verified
- Cleanup verification: only `mnemos_engrams_nomic_mrl` remains in Qdrant
- PDF truthset: `benchmarks/truthsets/phase7_pdf_reference_passages.json`
- Burn-in report: `docs/reports/mnemos_phase7_burn_in_report.md`
- Replay artifact: `benchmarks/outputs/summaries/matryoshka_shadow_20260611_112127_decision.md`

Expanded replay positive evidence:

- Prefix sentinel: PASS
- Mean Jaccard@10: PASS, `0.6336`
- Rank #1 stability: PASS, `1.0`
- Score compression: PASS, `0.0331` mean delta
- PDF-backed operational anchor labeled recall: stable at `1.0 -> 1.0`
- Representative replay: PASS, `N=2121`, Jaccard@10 `0.81`, rank stability `94.2%`

## Remaining Warnings

### 1. Cold-start latency outlier - resolved operationally

First post-restart query:

```text
25.549s
```

Warm query range:

```text
0.055s-0.166s
```

Validator question:

```text
Should this be classified as a normal cold model warm-up artifact and documented as an operator note, or does it require a startup prewarm / readiness-gate change before cleanup?
```

Resolution: classified as a cold model warm-up artifact and mitigated by the `/v1/mnemos/warmup` endpoint plus SDK `wait_until_ready(warmup=True)` readiness flow.

### 2. Expanded shadow replay REVIEW - absorbed by representative replay

Replay remained `REVIEW` because:

```text
long-context median Jaccard@10 = 0.4286
p95 budget = 53.5322ms actual vs 50.2613ms budget
```

Validator question:

```text
Are the long-context Jaccard and tiny-sample p95 budget warnings material blockers for cleanup, or acceptable warnings pending a larger representative corpus?
```

Resolution: accepted after representative replay with `N=2121`, Jaccard@10 `0.81`, and rank stability `94.2%`.

## Warning Materiality Assessment

The cold-start latency outlier is classified as a cold model warm-up artifact. It is now an operator preflight concern rather than a live-traffic risk. The service exposes `/v1/mnemos/warmup`, and the SDK supports `wait_until_ready(warmup=True)`.

The replay `REVIEW` result was not driven by labeled operational-anchor recall loss. All PDF-backed operational anchors preserved labeled recall at `1.0 -> 1.0`, and the fixed smoke set returned exact expected top results. The remaining replay concerns were absorbed by the larger representative replay:

- Long-context Jaccard reflects top-10 overlap drift, while labeled recall remains preserved.
- The p95 budget miss is small in absolute terms, `3.2709ms` over budget, and was measured on a tiny replay sample.

These warnings do not indicate that the promoted Nomic runtime should be rolled back and no longer block cleanup.

## Cleanup Eligibility Assessment

Cleanup is complete.

The current evidence supports Phase 7 closure with the Nomic runtime active and the legacy BGE collection deleted. Post-cleanup verification confirmed that only `mnemos_engrams_nomic_mrl` remains in Qdrant and the service continues to report the Nomic model, promoted collection, and dimension `768`.

## Validator Decision Requested

Decision recorded:

```text
FINAL_PASS_CLOSE_PHASE_7
```

Condition:

```text
CLEANUP_COMPLETED after representative replay N=2121 resolved or absorbed the replay warnings.
```

## Recommended Next Action

Close Phase 7 and proceed with normal Nomic runtime operations.

Do not change the promoted model or retrieval thresholds as part of Phase 7 closeout. Future replay should use a representative corpus rather than the small smoke set unless code, corpus, config, or service state changes.
