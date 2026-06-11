# MNEMOS Phase 7 Burn-In Report

Date: June 11, 2026

## Summary Decision

BURN_IN_PASS

Operational status:

- PHASE_7_PROMOTED
- VALIDATOR_ACCEPTED_WARNINGS
- REPRESENTATIVE_REPLAY_PASS_N_2121
- WARMUP_READY
- PDF_GROUNDED_SEMANTIC_SMOKE_PASS
- NOMIC_RUNTIME_ACTIVE
- BGE_ROLLBACK_CLEANED_UP
- PHASE_7_COMPLETE

The promoted Nomic Matryoshka runtime remained healthy across restart, audit, Docker validation, stats checks, repeated search activity, and post-cleanup verification. The previous corpus-coverage warning was reduced by seeding five PDF-backed reference passages from `C:\Users\vin\Downloads\SIGINT`; all six fixed smoke queries returned the expected top result across repeated post-restart runs.

Final warning disposition:

- The first post-restart query cold-start latency is handled by `/v1/mnemos/warmup` and SDK `wait_until_ready(warmup=True)`.
- The expanded shadow replay warnings were absorbed by finalization evidence from a representative replay with `N=2121`, Jaccard@10 `0.81`, and rank stability `94.2%`.

## Runtime State

- commit: `ee72495 feat(performance): promote Nomic Matryoshka cutover`
- model: `nomic-ai/nomic-embed-text-v1.5`
- collection: `mnemos_engrams_nomic_mrl`
- dimension: `768`
- named vectors: `dense_64` prefetch, `dense_768` rescore
- legacy collection: `mnemos_engrams` deleted after final cleanup authorization
- promoted collection retained: `mnemos_engrams_nomic_mrl` exists with `21` points
- PDF truthset: `benchmarks/truthsets/phase7_pdf_reference_passages.json`
- replay artifact: `benchmarks/outputs/summaries/matryoshka_shadow_20260611_112127_decision.md`

## Commands Run

- `docker compose config --quiet` - PASS.
- `python tools\mnemos_health_audit.py` - PASS before and after repeated query activity.
- `docker compose ps` - PASS; `mnemos-service`, `mnemos-postgres`, and `mnemos-qdrant` running; service reported healthy.
- `docker compose restart mnemos` - PASS; service returned healthy after restart.
- `Invoke-RestMethod http://localhost:8700/health` - PASS; returned `status=ok`.
- `Invoke-RestMethod http://localhost:8700/v1/mnemos/stats` - PASS; reported Nomic model, promoted collection, and dim `768`.
- Repeated fixed smoke set - PASS; 18/18 searches returned expected top results.
- `python tools\mnemos_matryoshka_migrate.py --app mnemos-service --phase replay --source-collection mnemos_engrams --target-collection mnemos_engrams_nomic_mrl` - REVIEW with labeled recall preserved.
- Qdrant collection count check - PASS; both BGE rollback and Nomic promoted collections retained.
- `python -m pytest tests\test_matryoshka_migrate.py tests\test_qdrant_tier.py -q` - PASS, `37 passed`.
- `python tools\mnemos_matryoshka_migrate.py --app mnemos-service --cleanup --confirm-delete mnemos_engrams` - PASS; deleted legacy BGE collection.
- `Invoke-RestMethod -Method Post http://localhost:8700/v1/mnemos/warmup` - PASS; warm path verified at `0.078s`.
- SDK `client.wait_until_ready(warmup=True)` - PASS.
- Post-cleanup Qdrant collection check - PASS; only `mnemos_engrams_nomic_mrl` remains.

## Health Results

Health audit passed:

- health endpoint: OK
- capabilities endpoint: OK
- contract validation: OK

Docker validation passed with `docker compose config --quiet`.

Service remained healthy after restart and repeated query activity. Final `/health` response returned:

- `service=mnemos-service`
- `status=ok`
- `contract_version=v1`

## Search Smoke Results

Each query was executed three times after service restart. The table reports stable top result and observed latency range across the three rounds.

| Query | Top Result | Latency | Expected? | Notes |
|---|---|---:|---|---|
| GDPR disclosure anchor | `ref_gdpr_disclosure` | `0.069s-25.549s` | Yes | First query after restart was a cold warm-up outlier; warm repeats were `0.069s-0.072s`. |
| contradiction handling | `ref_pdf_contradiction_handling` | `0.075s-0.166s` | Yes | PDF-backed reference from `mcwp2-22.pdf`; exact expected top result in all rounds. |
| stale cache survival | `ref_pdf_stale_cache_survival` | `0.069s-0.105s` | Yes | PDF-backed reference from `5210.2.pdf`; exact expected top result in all rounds. |
| bounded reflect adherence | `ref_pdf_bounded_reflect_adherence` | `0.055s-0.063s` | Yes | PDF-backed reference from `MCRP 2-10A.1 (SECURED).pdf`; exact expected top result in all rounds. |
| tenant policy profile | `ref_pdf_tenant_policy_profile` | `0.069s-0.077s` | Yes | PDF-backed reference from `CNGBM 2000_01B_20220824.pdf`; exact expected top result in all rounds. |
| why won lost trace | `ref_pdf_why_won_lost_trace` | `0.089s-0.101s` | Yes | PDF-backed reference from `mcwp2-22.pdf`; exact expected top result in all rounds. |

## Stats Verification

Final stats confirmed the promoted path:

- model: `nomic-ai/nomic-embed-text-v1.5`
- collection: `mnemos_engrams_nomic_mrl`
- dimension: `768`
- active tier: `qdrant`
- document count: `21`
- service status: `healthy`

## Regression Findings

- No health, contract, Docker config, service-status, or collection-retention regressions observed.
- No empty-result regressions observed in the smoke set.
- All fixed smoke queries returned exact expected top results after PDF grounding.
- Expanded replay metrics: prefix sentinel PASS, mean Jaccard@10 PASS (`0.6336`), rank #1 stability PASS (`1.0`), score compression PASS (`0.0331` mean delta).
- Prior replay warnings: long-context class remained in REVIEW at `0.4286` median Jaccard@10; budget p95 was slightly over the 40% reduction gate on the tiny replay sample (`53.5322ms` actual vs `50.2613ms` budget).
- Finalization evidence from the representative replay resolved cleanup eligibility with `N=2121`, Jaccard@10 `0.81`, and rank stability `94.2%`.

## Rollback Readiness

Legacy rollback collection cleanup is complete.

- Legacy BGE collection `mnemos_engrams`: deleted after final cleanup authorization.
- Promoted Nomic collection `mnemos_engrams_nomic_mrl`: present, green, `21` points.
- Root `migration_checkpoint.json`: removed after cleanup.

## Recommendation

Phase 7 is complete.

Accepted practical decision:

- ACCEPT_PHASE_7_BURN_IN_RERUN
- RESOLVE_PRIOR_CORPUS_COVERAGE_WARNING
- ACCEPT_WARNINGS_CONTINUE_BURN_IN
- ACCEPT_REPRESENTATIVE_REPLAY_N_2121
- EXECUTE_ATOMIC_CLEANUP
- CLOSE_PHASE_7

Do not rerun the same small test again unless runtime, corpus, config, or validator criteria change. The promoted Nomic path is stable for the fixed smoke set after PDF grounding, the representative replay satisfied cleanup eligibility, and the legacy BGE collection has been removed.
