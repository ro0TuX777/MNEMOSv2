# MNEMOS Phase 7 Burn-In Report

Date: June 11, 2026

## Summary Decision

BURN_IN_PASS_WITH_WARNINGS

The promoted Nomic Matryoshka runtime remained healthy across audit, Docker validation, stats checks, and repeated search activity. The warning is corpus-coverage related: five requested smoke queries are operational concepts that are not directly represented in the 16-document reference corpus, so those checks validate runtime stability and non-empty retrieval rather than topic-specific semantic correctness.

## Runtime State

- commit: `ee72495 feat(performance): promote Nomic Matryoshka cutover`
- model: `nomic-ai/nomic-embed-text-v1.5`
- collection: `mnemos_engrams_nomic_mrl`
- dimension: `768`
- named vectors: `dense_64` prefetch, `dense_768` rescore
- legacy collection retained: `mnemos_engrams` exists with `16` points
- promoted collection retained: `mnemos_engrams_nomic_mrl` exists with `16` points

## Commands Run

- `python tools\mnemos_health_audit.py` - PASS before and after repeated query activity.
- `docker compose config --quiet` - PASS.
- `docker compose ps` - PASS; `mnemos-service`, `mnemos-postgres`, and `mnemos-qdrant` running; service reported healthy.
- `Invoke-RestMethod http://localhost:8700/health` - PASS; returned `status=ok`.
- `Invoke-RestMethod http://localhost:8700/v1/mnemos/stats` - PASS; reported Nomic model, promoted collection, and dim `768`.
- Repeated fixed smoke set - PASS for runtime stability; 18/18 searches returned healthy responses and non-empty results.
- Qdrant collection count check - PASS; both BGE rollback and Nomic promoted collections retained.

## Health Results

Health audit passed:

- health endpoint: OK
- capabilities endpoint: OK
- contract validation: OK

Docker validation passed with `docker compose config --quiet`.

Service remained healthy after repeated query activity. Final `/health` response returned:

- `service=mnemos-service`
- `status=ok`
- `contract_version=v1`

## Search Smoke Results

Each query was executed three times. The table reports stable top result and observed latency range across the three rounds.

| Query | Top Result | Latency | Expected? | Notes |
|---|---|---:|---|---|
| GDPR disclosure anchor | `ref_gdpr_disclosure` | `0.034s-0.135s` | Yes | Exact expected top result retained. API does not expose vector path; runtime stats and code path confirm Nomic `dense_64`/`dense_768`. |
| contradiction handling | `dist_forensics_chain` | `0.039s-0.084s` | Yes | Expected criterion was non-empty/stable. Corpus lacks a direct contradiction-handling reference document. |
| stale cache survival | `dist_vector_storage` | `0.039s-0.080s` | Yes | Expected criterion was non-empty/stable. Corpus lacks a direct stale-cache reference document. |
| bounded reflect adherence | `dist_change_management` | `0.035s-0.072s` | Yes | Expected criterion was non-empty/stable. Corpus lacks a direct bounded-reflect reference document. |
| tenant policy profile | `dist_data_retention` | `0.037s-0.074s` | Yes | Expected criterion was non-empty/stable. Corpus lacks a direct tenant-policy reference document. |
| why won lost trace | `dist_forensics_chain` | `0.035s-0.072s` | Yes | Expected criterion was non-empty/stable. Corpus lacks a direct why-won/lost-trace reference document. |

## Stats Verification

Final stats confirmed the promoted path:

- model: `nomic-ai/nomic-embed-text-v1.5`
- collection: `mnemos_engrams_nomic_mrl`
- dimension: `768`
- active tier: `qdrant`
- document count: `16`
- service status: `healthy`

## Regression Findings

- No health, contract, Docker config, service-status, or collection-retention regressions observed.
- No empty-result regressions observed in the smoke set.
- The exact anchor query returned the expected top result consistently.
- Warning: operational smoke queries without matching seeded documents returned stable but only weakly topical top hits. This is not a runtime regression, but it limits semantic conclusions for those five topics until a broader burn-in corpus is loaded.

## Rollback Readiness

Rollback remains available.

- Legacy BGE collection `mnemos_engrams`: present, `16` points.
- Promoted Nomic collection `mnemos_engrams_nomic_mrl`: present, `16` points.
- Cleanup was not run.

## Recommendation

Continue burn-in and do not delete `mnemos_engrams` yet.

The promoted Nomic path is stable enough to keep running, but cleanup should wait for the planned burn-in window and preferably a broader corpus that contains direct examples for contradiction handling, stale cache survival, bounded reflect adherence, tenant policy profiles, and why-won/lost traces.
