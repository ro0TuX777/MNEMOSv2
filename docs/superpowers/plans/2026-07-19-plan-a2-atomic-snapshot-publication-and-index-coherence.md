# Plan A.2 - Atomic Snapshot Publication and Index Coherence

## Goal

Restore a strict publication invariant for Engram snapshots:

- a hybrid snapshot is not ACTIVE until its catalog artifacts and semantic points are both complete, verified, and bound to the same `snapshot_id`
- the previously published complete snapshot remains available until a replacement is promoted
- lexical-only operation is exposed only when explicitly configured or requested, never by accident

## Problem Summary

The current Engram flow writes a catalog snapshot, indexes semantic artifacts, marks the snapshot `complete`, and then runs semantic health and garbage collection against that same snapshot. In the observed failure mode, the catalog advanced to a new active snapshot with 6,305 artifacts while the semantic collection had zero points for that `snapshot_id`. This means Engram can advertise a hybrid-ready active snapshot whose semantic side is absent.

Today, the effective publication rule is "latest snapshot wins." That is too weak for hybrid retrieval because it conflates:

- build in progress
- build complete but unpublished
- published hybrid-active
- lexical-only usable
- inconsistent or partially failed

Plan A.2 separates snapshot construction from snapshot publication.

## Recommended Design

Use an explicit two-phase snapshot lifecycle with an atomic promotion step.

### Snapshot States

Add a publication model that distinguishes build state from serving state.

- `building`
  Catalog row exists and files/artifacts may be accumulating. Not queryable.
- `catalog_complete`
  Catalog/file/artifact write finished and manifest/count metadata is frozen. Not yet active.
- `semantic_indexing`
  Semantic write is in progress for the same `snapshot_id`. Not yet active.
- `ready_for_promotion`
  Catalog and semantic writes are both complete and verification passed, but active pointer has not yet moved.
- `active`
  Snapshot is the published serving target for the selected mode.
- `degraded_lexical`
  Snapshot is intentionally published for lexical-only service. This is valid only when explicitly enabled.
- `failed`
  Build or verification failed. Never active.
- `superseded`
  Snapshot was previously active and has been replaced, but may still be retained until GC completes.

The key distinction is that `active` is no longer inferred from recency. It is published through a dedicated promotion step.

### Publication Pointer

Introduce a project-scoped publication record rather than deriving activeness from `get_latest_snapshot(...)`.

Recommended shape:

- new table `published_snapshots`
- one row per project and serving mode
- fields:
  - `project_id`
  - `mode` with values like `hybrid` and `lexical`
  - `snapshot_id`
  - `published_at`
  - `publication_status`
  - `artifact_count`
  - `semantic_point_count`
  - `identity_instance_id`
  - `identity_logical_project_id`
  - `identity_canonical_path`

This lets Engram answer two separate questions cleanly:

- what snapshot is being built?
- what snapshot is currently published for serving?

## Build and Promotion Flow

### 1. Begin Pending Snapshot

When a refresh starts:

- create a new snapshot in `building`
- record project identity inputs at build start
- do not change any published pointer

### 2. Build Catalog First

Write files and extracted artifacts into the pending snapshot.

At catalog completion, persist frozen verification inputs:

- file count
- artifact count
- artifact ids or deterministic artifact manifest hash
- file manifest hash
- build identity fields used to scope semantic writes

Transition snapshot to `catalog_complete`.

### 3. Build Semantic Index for the Same Snapshot

Index semantic points with payload carrying:

- `snapshot_id`
- `local_instance_id`
- `logical_project_id`
- canonical path or equivalent scope field

The snapshot moves to `semantic_indexing` while this runs.

### 4. Verify Before Promotion

Before any active-pointer change, verify all of the following for the candidate snapshot:

- the catalog snapshot still exists and is still `catalog_complete` or `semantic_indexing`
- the semantic collection contains points for exactly that `snapshot_id`
- semantic point count matches the expected promoted count policy
- identity scope on those points matches the build identity
- optional sample identity check confirms returned points belong to the same project scope

Count policy should be explicit:

- preferred: semantic point count equals the number of semantic-eligible artifacts recorded for the snapshot
- acceptable only if intentionally documented: semantic point count equals total artifacts when every artifact is indexed semantically

If verification passes, move snapshot to `ready_for_promotion`.

### 5. Atomic Promotion

Promote using a single catalog transaction:

- update the project publication pointer from old snapshot to new snapshot
- mark new snapshot `active`
- mark previous active snapshot `superseded`
- persist promotion counts and timestamps

Only after that transaction commits should read paths observe the new snapshot as active.

### 6. Deferred Garbage Collection

Garbage collection must run after successful promotion, not before it.

Retention rule:

- always retain the currently published snapshot
- retain the immediately previous complete published snapshot until replacement promotion succeeds
- never delete points for the last good published snapshot while a replacement is still pending or failed

GC can then remove:

- superseded snapshots beyond the retention window
- failed snapshots after operator-safe grace rules
- orphan semantic points not referenced by any retained snapshot for the same project scope

## Degraded Operation

Lexical-only service must be explicit rather than accidental.

### Policy

- default behavior: if hybrid publication cannot be verified, continue serving the previous published hybrid snapshot
- if no published hybrid snapshot exists, report the project as not hybrid-ready
- lexical-only degraded publication is allowed only when explicitly enabled by config or an operator action

### Serving Modes

Recommended modes:

- `hybrid_required`
  Only published hybrid snapshots may serve bundle assembly.
- `allow_lexical_degraded`
  Lexical serving may continue from a verified catalog snapshot even when semantic publication is unavailable.

When degraded mode is active:

- status must state that semantic is unavailable for the active lexical snapshot
- retrieval instrumentation should indicate lexical-only mode
- this must not be reported as full hybrid readiness

## Failure and Restart Recovery

### Failure Cases

The plan must handle:

- semantic indexing exception after catalog completion
- process interruption during semantic indexing
- crash after semantic indexing but before promotion
- crash after promotion but before GC
- stale failed pending snapshots at process restart

### Recovery Rules

- a non-published pending snapshot must never displace the last good published snapshot
- restart logic should inspect unfinished snapshots and classify them as:
  - resumable
  - rebuild required
  - failed and quarantined
- if semantic completion cannot be proven for a pending snapshot, do not promote it
- if promotion completion cannot be proven, use the publication pointer as the source of truth and re-run verification before any repair action
- GC must be idempotent and safe to re-run after restart

Recommended restart strategy:

1. Read the published pointer first.
2. Enumerate newest non-published snapshots for the project.
3. For each pending snapshot, compare stored expected counts against semantic scoped counts.
4. If verified and unpublished, resume at promotion.
5. If unverifiable, mark `failed` or rebuild from scratch according to a bounded retry policy.

## Health and Status Model

Current status fields are too coarse for this failure mode. Add explicit health outputs that separate publication state from build progress.

### Project Status Fields

Recommended fields for `get_status` or a follow-up status surface:

- `project_root`
- `publication_mode`
- `active_snapshot_id`
- `active_snapshot_state`
- `active_snapshot_ready_for_hybrid`
- `active_artifact_count`
- `active_semantic_point_count`
- `pending_snapshot_id`
- `pending_snapshot_state`
- `pending_artifact_count`
- `pending_expected_semantic_point_count`
- `pending_observed_semantic_point_count`
- `previous_snapshot_id`
- `previous_snapshot_state`
- `status`
- `stale`

### Normalized Status Values

- `unindexed`
  No published snapshot exists.
- `pending`
  A build is in progress and the previous published snapshot remains authoritative.
- `complete`
  A verified hybrid snapshot is published and coherent.
- `degraded`
  A lexical-only snapshot is intentionally published.
- `inconsistent`
  Catalog and semantic state disagree for the candidate or advertised snapshot. This should page operators and block hybrid promotion.
- `failed`
  The newest pending snapshot failed and the system remains on the last good published snapshot if one exists.

### Index Health Fields

Recommended additions for index-health reporting:

- `scoped_status`
- `published_hybrid_snapshot_id`
- `published_lexical_snapshot_id`
- `pending_snapshot_id`
- `pending_semantic_verification`
- `expected_semantic_point_count`
- `observed_semantic_point_count`
- `identity_match`
- `catalog_semantic_coherent`
- `promotion_blocker`

## Count and Identity Verification

Promotion must verify both count and identity.

### Count Checks

- compare expected promoted semantic count from the catalog to the scoped vector count for the same `snapshot_id`
- reject zero-count promotion unless the snapshot truly contains zero semantic-eligible artifacts
- record both expected and observed counts in the snapshot or publication receipt

### Identity Checks

- verify `local_instance_id` matches the build identity for locally scoped projects
- if `logical_project_id` is present, verify it matches as well
- verify canonical path scoping where it participates in vector-store health decisions

This prevents cross-project or stale-snapshot semantic counts from satisfying promotion accidentally.

## Query Path Changes

Bundle assembly and freshness logic should target the published snapshot for the chosen serving mode, not the latest created snapshot.

That means:

- `build_bundle(...)` should retrieve against the published active snapshot
- refresh logic may create a pending replacement snapshot, but callers keep using the last published snapshot until promotion succeeds
- stale detection can still trigger rebuild, but not an immediate serving-snapshot swap

## Test Plan

Add focused tests around publication rather than generic retrieval only.

### Unit Tests

- snapshot lifecycle transitions from `building` to `catalog_complete` to `semantic_indexing` to `ready_for_promotion` to `active`
- promotion is rejected when semantic scoped count is zero for a non-empty snapshot
- promotion is rejected when identity fields do not match the pending snapshot
- lexical-only publication is rejected unless degraded mode is explicitly enabled
- GC retains the previous published snapshot until replacement promotion succeeds

### Failure and Recovery Tests

- semantic indexing failure leaves the previous active snapshot published and marks the candidate failed
- interruption after catalog completion leaves a pending snapshot that is not served as active
- restart with a fully indexed but unpublished snapshot resumes at verification and promotion
- restart with unverifiable semantic data marks the candidate failed or rebuild-required without affecting the published pointer
- crash after promotion but before GC preserves correct serving behavior and allows idempotent cleanup later

### Integration Tests

- successful refresh promotes a new coherent hybrid snapshot and then permits GC of older superseded snapshots
- hybrid retrieval uses the published snapshot even when a newer pending snapshot exists
- explicit degraded configuration publishes lexical-only status and instrumentation without claiming hybrid readiness
- health/status endpoints report `pending`, `complete`, `degraded`, and `inconsistent` accurately

## Suggested Work Sequence

1. Extend catalog schema to represent pending snapshots and explicit publication pointers.
2. Refactor scan/build flow so catalog completion, semantic indexing, verification, and promotion are distinct steps.
3. Change read paths to resolve the published snapshot instead of the latest snapshot.
4. Move GC behind successful promotion and add retention rules for the last good published snapshot.
5. Expand status and health surfaces to expose pending, degraded, and inconsistent states.
6. Add failure, restart, and promotion tests before cleanup/refinement.

## Acceptance Criteria

- Engram never advertises a hybrid-active snapshot whose semantic scoped count is absent or mismatched for the same `snapshot_id`
- a pending or failed snapshot cannot displace the last good published snapshot
- lexical-only service occurs only when explicitly enabled and is visible in status and instrumentation
- restart and recovery preserve publication correctness without manual index surgery
- GC does not delete the previous good snapshot until replacement promotion has succeeded
- tests cover semantic failure, interruption, restart, and successful promotion paths

## Deferred Retrospective Check

After snapshot coherence is restored in implementation, rerun the original MNEMOS task text through `engram_build_bundle` as a retrieval-only retrospective check. That follow-up should validate retrieval behavior only and must not repeat the earlier MNEMOS code change.
