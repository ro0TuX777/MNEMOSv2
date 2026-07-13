# Release / Documentation Review Checklist

A standing process checklist applied whenever a change adds, revises, or
reclassifies a capability claim (e.g. moving something between Core,
Experimental, and Research/Shadow in [README.md](../README.md) or the
[support matrix](support_matrix.md)).

## Required Repository Review

Before closing out a capability-claim change, review:

```text
docs/associative_routing_e2_design_note.md
benchmarks/results/associative_routing_e2_live_comparison_run_001.json
any existing E0/E1 associative-routing design notes or evidence artifacts
Docker Compose files and image-build documentation relevant to service-code deployment
```

## Required Internal Change-Note Item

Every change-note for this kind of update must state the current
associative-routing status, including the distinction between:

- E0/E1 research or shadow behavior;
- E2 opt-in candidate expansion;
- default retrieval behavior;
- deployed-container verification versus local in-process verification
  against live backends.

## Validation Requirement

For any documented experimental capability evaluated against a live
backend:

1. State whether the result came from:
   - deployed HTTP service execution;
   - local in-process execution against live backends; or
   - offline fixture evaluation.
2. Confirm the tested code revision or image build identity.
3. Do not label a result "live service verified" unless the deployed
   service actually executed the current implementation through its
   supported external interface.

## Commit Exclusion

Do not add, modify, or commit unowned, unexplained artifacts discovered in
the working tree during a review (for example, foreign PDFs or other files
not produced by the change in progress). Treat them as foreign
concurrent-process artifacts unless their ownership, relevance, licensing,
and intended repository role are separately confirmed.

## Public-Claims Deliverable

For every claim added or revised, identify:

- default behavior;
- enablement conditions;
- evidence type;
- scope of supported usefulness;
- explicit non-claims.

Expected public claim summary format:

```text
Added:
- <new capability statement>
- <bounding conditions: opt-in, kill switch, governance, no durable writes, etc.>
- <evidence summary: what was measured, on what corpus/scope>

Qualified:
- <evidence provenance caveat: e.g. local in-process vs deployed-service execution>
- <explicit scope limits: not a default change, not a broad superiority claim, etc.>

Removed or prohibited:
- <claims this change explicitly does NOT support>
- <any claim that the deployed service was verified with code it does not yet run>
- <any broad claim not backed by the recorded evidence artifact>
```
