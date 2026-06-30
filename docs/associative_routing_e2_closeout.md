# Associative Routing E2 Closeout

## Scope And Authorization

Associative Routing E2 is closed as an experimental, opt-in
candidate-expansion capability.

Authorization:

```text
ASSOCIATIVE_ROUTING_E2_CLOSEOUT_AUTHORIZED
EXPERIMENTAL_OPT_IN_ONLY
DEFAULT_RETRIEVAL_UNCHANGED
NO_E3_WORK
NO_RETRIEVAL_TUNING
NO_CORPUS_EXPANSION
NO_CUE_OR_TAG_REGISTRY_CHANGE
NO_GOVERNANCE_OR_AUTHORITY_CHANGE
NO_MCP_ENVIRONMENT_CHANGE
```

This closeout does not improve E2 metrics, tune routing logic, alter
retrieval ranking, expand the cue/tag registry, or begin E3.

## Architecture Boundary

E2 runs only after normal retrieval, normal governance processing, and the
low-relevance abstention decision have completed. It can append a small number
of source-linked candidates to the delivered result list when explicitly
requested and globally enabled.

E2 does not:

- change default retrieval behavior;
- suppress normal retrieval candidates;
- globally re-rank normal candidates;
- promote expansion candidates as authoritative;
- add authority, disclosure, deletion, or governance powers;
- write durable state;
- begin any E3 behavior.

## Request Opt-In And Kill-Switch Semantics

Request opt-in:

```json
{
  "associative_candidate_expansion": true
}
```

Global kill switch:

```text
MNEMOS_ASSOCIATIVE_CANDIDATE_EXPANSION_ENABLED=true
```

The environment variable defaults to disabled. A request flag alone is not
enough to activate expansion. When disabled, E2 returns a disabled metadata
block only for explicitly flagged requests and leaves normal results unchanged.

## Frozen Corpus And Query-Pack Scope

The E2 fixture corpus is frozen under:

```text
mnemos/retrieval/associative_expansion/fixtures/
```

The frozen verification pack is:

```text
docs/experiments/associative_routing_e2_verification_pack.json
```

Scope:

- 22 frozen verification queries.
- Small curated corpus.
- Query pack authored with knowledge of the corpus and registries.
- No independent-authored evaluation pack.
- No claim that results generalize beyond this controlled E2 evidence set.

The informal development pack is retained at:

```text
docs/experiments/associative_routing_e2_development_pack.json
```

It is not used as final claim evidence.

## Comparison Conditions

The recorded comparison used:

| Field | Value |
| --- | --- |
| Query pack | 22-query E2 verification pack |
| Retrieval mode | semantic for both baseline and expansion conditions |
| `top_k` | 5 |
| Governance | off for the live comparison; governance rejection is covered by focused tests |
| Backends | Existing live Qdrant/PostgreSQL backends via local in-process runtime |
| Deployed container | Not used for the E2 HTTP-path claim because the container image predated E1/E2 code |

The environment finding is recorded in the design note:

```text
DOCKER_IMAGE_CODE_DRIFT_KNOWN
NO_E2_FINDING_DEPENDS_ON_THE_DEPLOYED_CONTAINERS_CODE
LOCAL_IN_PROCESS_EXECUTION_USED_AGAINST_LIVE_BACKENDS_INSTEAD
```

## Live Comparison Results

Recorded artifact:

```text
benchmarks/results/associative_routing_e2_live_comparison_run_001.json
```

Summary:

| Metric | Result |
| --- | --- |
| Queries run | 22 / 22 |
| Expansion triggered | 2 / 22 |
| Candidates added | 2 |
| Correct-and-needed additions | 2 |
| Observed normal-result suppression | 0 |
| Observed normal-result global re-ranking | 0 |
| Observed frozen-evaluation regressions | 0 |
| Small-corpus limitation | Yes |
| Independent-authorship pack | No |

The two additions were:

- `benchmarks/results/ai_dev_memory_trial_comparison_002.md` for the query
  "What superseded trial comparison 001?"
- `benchmarks/results/retrieval_hygiene_r0_run_003.json` for the query
  "Tell me about Retrieval Hygiene R0."

Both were classified as `correct_and_needed` in the recorded comparison.

## Verification-Pack Result

The frozen verification pack contains direct status, dependency/blocker,
temporal/supersession, ambiguity, unrelated-negative-control, and
stale/superseded-source queries.

Verified closeout properties:

| Property | Verification source |
| --- | --- |
| Flag-off behavior is identical to normal retrieval | `tests/test_associative_routing_e2_expansion.py` |
| Kill switch prevents candidate expansion | `tests/test_associative_routing_e2_expansion.py` |
| Normal candidates are not suppressed | `tests/test_associative_routing_e2_expansion.py`; recorded comparison |
| Normal ranking is not globally changed | `tests/test_associative_routing_e2_expansion.py`; recorded comparison |
| Added candidates are bounded | `tests/test_associative_routing_e2_expansion.py` |
| Added candidates are deduplicated | `tests/test_associative_routing_e2_expansion.py`; recorded comparison |
| Added candidates carry origin labels | `tests/test_associative_routing_e2_expansion.py`; recorded comparison |
| Added candidates retain source lineage | `tests/test_associative_routing_e2_expansion.py` |
| Inactive or unresolved targets do not bypass controls | `tests/test_associative_routing_e2_expansion.py` |
| Policy-rejected targets do not bypass governance controls | `tests/test_associative_routing_e2_expansion.py` |
| No durable write occurs | `tests/test_associative_routing_e2_expansion.py` |
| No authority, disclosure, or governance behavior changed | `tests/test_associative_routing_e2_expansion.py`; code review |

Focused validation command:

```powershell
python -m pytest tests/test_associative_routing_e0.py tests/test_associative_routing_e1_shadow.py tests/test_associative_routing_e2_expansion.py tests/test_service_hybrid_api.py -q
```

Latest observed result during closeout:

```text
69 passed, 38 warnings
```

No separate E2-specific runner exists in `tools/`; the E2-specific gate is the
focused E2 expansion test suite plus the frozen verification/comparison
artifacts listed above.

## Known Limitations

- Small curated corpus.
- No independent-authored evaluation pack.
- E2 was not verified through the deployed container image because the image
  predates the E1/E2 implementation.
- Governance interaction was covered by focused tests, not by the recorded live
  comparison.
- The development pack is not claim evidence.
- No E3 design or implementation is authorized by this closeout.

## Public-Claim Boundary

Supported:

- E2 is experimental and opt-in.
- E2 has a global kill switch disabled by default.
- E2 is candidate-addition only.
- Normal retrieval remains primary.
- Normal results are not suppressed.
- Normal ranking is not globally re-ranked.
- Expansion candidates are origin-labeled, source-linked, bounded, and
  non-authoritative.
- The frozen 22-query comparison recorded 2 correct-and-needed additions and
  0 observed regressions.

Not supported:

- broad retrieval superiority;
- production readiness;
- authorization, deletion, or disclosure safety;
- independent human-authored evaluation evidence;
- default runtime behavior change;
- E3 readiness.

## Reproduction Commands

Run the focused regression suite:

```powershell
python -m pytest tests/test_associative_routing_e0.py tests/test_associative_routing_e1_shadow.py tests/test_associative_routing_e2_expansion.py tests/test_service_hybrid_api.py -q
```

Inspect the recorded comparison:

```powershell
python -c "import json; p='benchmarks/results/associative_routing_e2_live_comparison_run_001.json'; d=json.load(open(p)); print(len(d), sum(1 for r in d if r['expansion_block']['status']=='expanded'))"
```

Inspect the frozen verification pack:

```powershell
python -c "import json; p='docs/experiments/associative_routing_e2_verification_pack.json'; d=json.load(open(p)); print(d['pack_id'], len(d['queries']))"
```

## Rollback Behavior

Rollback can disable E2 without code removal by leaving
`MNEMOS_ASSOCIATIVE_CANDIDATE_EXPANSION_ENABLED` unset or setting it to a value
other than `true`. Because the request flag is also required, callers that do
not opt in remain on normal retrieval behavior.

Removing the E2 code path should only remove:

- `mnemos/retrieval/associative_expansion/`
- the `associative_candidate_expansion` request parsing and runtime invocation
  in `service/app.py`
- E2-specific tests and documentation/evidence artifacts

It should not touch E0, E1, governance, retrieval ranking, deletion, disclosure,
MCP/MSF configuration, or dependency files.

## Decision

```text
ASSOCIATIVE_ROUTING_E2_COMPLETE
EXPERIMENTAL_OPT_IN_CANDIDATE_EXPANSION_RETAINED
KILL_SWITCH_CONTROLLED
DEFAULT_RETRIEVAL_UNCHANGED
SMALL_CURATED_CORPUS_LIMITATION
NO_INDEPENDENT_AUTHORED_EVALUATION_PACK
NO_BROAD_RETRIEVAL_SUPERIORITY_CLAIM
NO_PRODUCTION_READINESS_CLAIM
```
