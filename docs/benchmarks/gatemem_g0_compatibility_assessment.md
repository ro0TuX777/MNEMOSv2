# GateMem G0 Compatibility Assessment

Date: 2026-06-24

Closeout: `GATEMEM_G0_PARTIAL_DELETION_GOVERNANCE_GAP_IDENTIFIED`

## Executive finding

GateMem is a useful external evaluation target for MNEMOS, but MNEMOS cannot
honestly claim full GateMem compatibility today. Provenance and auditability
are supported. Retrieval utility is partially testable. Multi-principal scope,
disclosure, and redaction are only shadow-testable in the isolated
session-context adapter. GateMem-grade active forgetting is unsupported.

No MNEMOS runtime, routing, retrieval, governance, authority, promotion,
consumer-adapter, or shadow-adapter code was changed by G0.

## Evaluation-integrity boundary

The following released checkpoint fields are scoring-only and prohibited from
future MNEMOS method logic:

```text
query_type
attack_type
expected_action
judge_spec
leak_targets
```

GateMem documentation states this boundary, but the current in-repository
agent interface does not enforce it structurally: `bench.eval.runner` builds a
`Checkpoint` carrying these fields and passes that object to the agent. The
offline stub also consumes `query_type` and `expected_action`. Therefore a
future MNEMOS adapter must not accept GateMem's native `Checkpoint` object and
must not use stub metrics as behavioral evidence.

The required clean-input projection is:

```text
episode identity
ordered turns no later than as_of_turn_id
authenticated requester principal identity
authenticated requester role
request query text
permitted benchmark-visible relationship/policy metadata only
```

The projection must be constructed by a boundary component with an explicit
allowlist. Scoring annotations stay in a separate evaluator process or data
structure. Prediction output may contain only the public normalized response
fields and non-sensitive audit identifiers.

## Compatibility findings

### Utility

MNEMOS can ingest source-linked evidence and return ranked, provenance-bearing
results. This supports an offline retrieval/evidence utility assessment.
GateMem evaluates final normalized actions and answers, however, and MNEMOS is
a memory service rather than an answer agent. Authorized utility is therefore
`PARTIAL`, not a current end-to-end claim.

### Access control

The production service's `_authorized()` check is a shared bearer-token check.
Search filters such as session or tenant metadata are caller-provided and are
not bound to an authenticated principal. Governance policy profiles tune
retrieval governance; they are not role/entitlement authorization.

The isolated session-context shadow adapter does provide useful mechanics:
authenticated local transport abstraction, consumer identity, tenant/session
scope, artifact/source/Engram allowlists, disclosure denial, redaction, policy
pinning, content-free telemetry, and fail-closed behavior. Those policy
snapshots are injected test fixtures and have no live authorization source or
consumer connection. Access-control evaluation without runtime changes is
therefore possible only as `SHADOW_TESTABLE` research.

### Deletion and active forgetting

MNEMOS exposes `DELETE /v1/mnemos/engrams/{id}` and calls delete on semantic
and lexical tiers. Qdrant and pgvector delete calls have focused unit tests.
Separately, governance metadata models `soft_deleted` and `tombstone`, and the
relevance veto policy excludes those states in governed reads.

These pieces do not form explicit, tested GateMem deletion semantics:

- no conversational deletion-operation parser or authorized target resolver;
- no supported transition API from active to soft-deleted/tombstone;
- no atomic all-tier success condition or rollback;
- backend exceptions become zero counts while the service returns HTTP 200;
- no verified negative read after deletion;
- no lineage cascade into Summary or Resolution Engrams, extracted facts,
  graph edges, derived views, caches, or other descendants;
- no durable tombstone preventing reingestion or rederivation;
- no proof covering replicas, backups, logs, external copies, or model prior;
- no answer-layer defense against confirmation or reconstruction.

The view-cache type understands a `source_artifact_deleted` invalidation event,
but the service delete path does not emit that event. Lineage metadata exists,
but the endpoint deletes only the requested ID.

For a future GateMem experiment, both mechanisms are needed and must not be
conflated:

1. an offline simulated deletion policy can exercise benchmark sequencing and
   produce a shadow characterization;
2. a true governed deletion/tombstone mechanism with authorization, target
   resolution, cascade, cache invalidation, reingestion defense, and negative
   verification is required before claiming active forgetting.

Results from the simulated path must be labeled shadow-only and cannot support
a production deletion claim.

## Required questions

1. **Can MNEMOS evaluate authorized utility under GateMem semantics?**
   Partially. Retrieval/evidence utility can be evaluated offline; final
   action/answer utility needs a separately authorized adapter and answer path.

2. **Can MNEMOS evaluate access-control enforcement without changing runtime?**
   Only in isolated shadow form using external policy snapshots and the
   consumer-neutral shadow mechanics. Production runtime enforcement is not
   present.

3. **Does MNEMOS have explicit, tested deletion semantics?**
   No, not at GateMem's semantic level. It has tested backend delete calls and
   tested read vetoes for pre-marked deletion states, without an end-to-end
   deletion lifecycle.

4. **Can MNEMOS prevent later retrieval, confirmation, or reconstruction after
   deletion?** No. The current implementation provides no such guarantee.

5. **Which dimensions are unsupported?** Production role enforcement,
   deletion-request interpretation, lineage-aware deletion cascade,
   post-deletion non-recoverability, and reconstruction/confirmation resistance.

6. **Would a future adapter need simulated deletion, true governed deletion,
   or both?** Both, with separate claims and gates. Simulation enables an
   honest shadow benchmark; the true mechanism is necessary for product claims.

7. **Can external `predictions.jsonl` scoring be used first?** Yes. GateMem
   explicitly supports external predictions and scoring, so no upstream edit or
   in-tree GateMem agent is necessary.

8. **What is the smallest next authorized implementation step?** A G1 proposal
   for an offline, original-MNEMOS clean-input projector and external
   `predictions.jsonl` normalizer. It should initially emit retrieval/disclosure
   shadow observations, treat active forgetting as unsupported or simulated,
   never receive scoring-only fields, and make no runtime or upstream changes.

## Licensing and artifact boundary

GateMem remains wholly external. Software is MIT licensed; dataset metadata in
`CITATION.cff` declares CC-BY-4.0. No dataset rows, prompts, evaluator code,
outputs, or hidden-field values are copied into MNEMOS. Any future adapter is
original MNEMOS code. Published use should cite Ren et al., “GateMem:
Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents” (2026),
<https://arxiv.org/abs/2606.18829>, and preserve the applicable notices.

## Advancement boundary

G0 does not authorize a GateMem MNEMOS agent, deletion subsystem, hosted-model
run, LLM judge, external scoring run of MNEMOS predictions, or public
leaderboard submission. The deletion gap must remain visible in any G1 design.

