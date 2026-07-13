# Evidence Admission and Budgeting R0 Closeout

## Scope And Authorization

Evidence Admission and Budgeting R0 is closed as a read-only, shadow-only
service-integration lane.

Authorization boundary:

```text
EVIDENCE_ADMISSION_AND_BUDGETING_R0_AUTHORIZED
READ_ONLY
SHADOW_ONLY
NO_DEFAULT_RETRIEVAL_CHANGE
NO_RETRIEVAL_TUNING
NO_GOVERNANCE_OR_AUTHORITY_CHANGE
NO_DURABLE_WRITES
```

R0 reports a non-authoritative admission recommendation and a separate
post-retrieval sufficiency assessment. It does not suppress, reorder, inject,
promote, delete, disclose, or govern normal retrieval results.

## Execution Modes

R0 comparison evidence is split into two non-interchangeable execution modes:

| Mode | Use | Claim boundary |
| --- | --- | --- |
| `direct_runtime` | Development-pack debugging, deterministic policy inspection, fast local diagnosis, and focused regression support | Not sufficient for a deployed-service or HTTP-path claim |
| `http_service` | Frozen formal evaluation, fresh verification, final comparison artifact, and service-level shadow integration claims | Requires verified service revision or image identity |

The runner requires explicit mode selection and never falls back from
`http_service` to `direct_runtime`.

## Frozen Packs

| Pack | Intended mode | Purpose |
| --- | --- | --- |
| `docs/experiments/evidence_admission_r0_development_pack.json` | `direct_runtime` | Development and diagnostic iteration |
| `docs/experiments/evidence_admission_r0_formal_evaluation_pack.json` | `http_service` | Formal HTTP service integration evaluation |
| `docs/experiments/evidence_admission_r0_fresh_verification_pack.json` | `http_service` | Fresh HTTP service verification |

Results from `direct_runtime` and `http_service` are not aggregated into one
metric.

## Recorded Artifacts

| Artifact | Mode | Gate | Decision label |
| --- | --- | --- | --- |
| `benchmarks/results/evidence_admission_r0_development_direct_runtime_run_001.json` | `direct_runtime` | enabled | `DIRECT_RUNTIME_ONLY_EVIDENCE` |
| `benchmarks/results/evidence_admission_r0_formal_http_service_run_001.json` | `http_service` | enabled | `FORMAL_HTTP_SERVICE_EVALUATION_COMPLETE` |
| `benchmarks/results/evidence_admission_r0_fresh_http_service_run_001.json` | `http_service` | enabled | `FORMAL_HTTP_SERVICE_EVALUATION_COMPLETE` |
| `benchmarks/results/evidence_admission_r0_formal_http_service_gate_disabled_run_001.json` | `http_service` | disabled | `FORMAL_HTTP_SERVICE_EVALUATION_COMPLETE` |

HTTP artifacts were generated against a locally launched service from the
current checked-out revision:

```text
git:bef472112a751436c7af35cf472e13ccfa3a2329
```

The local service reported this clean committed source revision during formal
HTTP evaluation.

## Observed Behavior

Verified:

- request flag absent or false omits `meta.evidence_admission_shadow`;
- request flag true with global gate disabled returns bounded unavailable
  status only;
- request flag true with global gate enabled returns additive shadow metadata;
- normal retrieval result ordering and stable response-contract fields remain
  behaviorally identical when R0 is not requested;
- admission recommendation reason codes remain `ADMISSION_*`;
- post-retrieval sufficiency reason codes remain `SUFFICIENCY_*`;
- telemetry records redact raw query text;
- runner manifests include `execution_mode`, `service_base_url`,
  `service_revision_or_image_identity`, `runner_commit`,
  `collection_or_corpus_snapshot`, `request_flag_state`, and
  `global_gate_state`;
- HTTP mode fails closed with `SERVICE_REVISION_UNVERIFIED` when service
  identity cannot be established.

## Known Limitations

- The local HTTP service used for recorded formal artifacts reported
  `mnemos_engrams:0`; this supports service-level shadow integration only,
  not retrieval-quality or corpus-performance claims.
- The direct-runtime development run observed a local vector-dimension
  mismatch in Qdrant during normal retrieval. That artifact is diagnostic and
  not a formal service claim.
- R0 budgets are non-authoritative recommendations only; they are not enforced
  against normal retrieval in this milestone.
- R0 does not authorize E3 work, retrieval tuning, governance changes, MCP/MSF
  changes, or production-readiness claims.

## Verification Commands

Focused R0 and runner tests:

```powershell
pytest -q tests/test_evidence_admission_and_budgeting_r0.py tests/test_evidence_admission_r0_runner.py
```

Service and adjacent-regression tests:

```powershell
pytest -q tests/test_service_hybrid_api.py tests/test_retrieval_hygiene_r0.py tests/test_associative_routing_e1_shadow.py tests/test_associative_routing_e2_expansion.py
```

Runner reproduction examples:

```powershell
python tools/run_evidence_admission_r0_comparison.py --execution-mode http_service --service-base-url http://localhost:8791 --pack-path docs/experiments/evidence_admission_r0_formal_evaluation_pack.json --result-path benchmarks/results/evidence_admission_r0_formal_http_service_run_001.json --request-flag-state true --global-gate-state enabled
python tools/run_evidence_admission_r0_comparison.py --execution-mode http_service --service-base-url http://localhost:8791 --pack-path docs/experiments/evidence_admission_r0_fresh_verification_pack.json --result-path benchmarks/results/evidence_admission_r0_fresh_http_service_run_001.json --request-flag-state true --global-gate-state enabled
python tools/run_evidence_admission_r0_comparison.py --execution-mode http_service --service-base-url http://localhost:8792 --pack-path docs/experiments/evidence_admission_r0_formal_evaluation_pack.json --result-path benchmarks/results/evidence_admission_r0_formal_http_service_gate_disabled_run_001.json --request-flag-state true --global-gate-state disabled
```

## Decision

```text
FORMAL_HTTP_SERVICE_EVALUATION_COMPLETE
EVIDENCE_ADMISSION_AND_BUDGETING_R0_COMPLETE
READ_ONLY_SHADOW_ONLY_RETAINED
GLOBAL_GATE_CONTROLLED
DEFAULT_RETRIEVAL_UNCHANGED
PRE_RETRIEVAL_RECOMMENDATION_SEPARATED_FROM_POST_RETRIEVAL_SUFFICIENCY
NO_DEPLOYED_CONTAINER_IMAGE_CLAIM
NO_RETRIEVAL_QUALITY_CLAIM_FROM_EMPTY_LOCAL_CORPUS
```
