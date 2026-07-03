# Evidence Admission and Budgeting R1 Closeout

## Status

FORMAL_HTTP_SERVICE_EVALUATION_COMPLETE.

R1 bounded enforcement is **rejected for retention in its current
configuration**. The formal HTTP evidence fails the preregistered primary
non-inferiority criterion by a narrow margin:

- normal retrieval baseline: 33/42 non-abstention queries covered with required lineage (78.57%)
- R1 enforcement enabled: 32/42 non-abstention queries covered with required lineage (76.19%)
- delta: -2.381 percentage points
- preregistered allowed drop: no more than -2.0 percentage points

The run does support the narrower positive finding that the opt-in gates and
route safety controls behaved as expected: R0 shadow was read-only, R1 gate-off
preserved served retrieval output, and no forbidden route label was served.

## Formal Artifacts

- Formal result: `benchmarks/results/evidence_admission_r1_formal_http_service_run_001.json`
- Formal pack: `docs/experiments/evidence_admission_and_budgeting_r1_formal_pack.json`
- Formal pack SHA-256: `f09651f3fc67b0bddf73b3981a0f635e21c58ff3d4ed50bc717d2886377c14cc`
- Frozen corpus collection: `evidence_admission_r1_frozen_corpus_formal`
- Service revision: `b80ef5d25f4b7e98a9e9c694483a96c106f5297c`
- Execution mode: `http_service`
- Embedding profile: `BAAI/bge-base-en-v1.5`, explicit dimension `768`

## Evidence Basis and Authorship Order

This result is **evaluation against a subsequently independently authored,
frozen, hashed formal pack** — not strict pack-before-implementation
preregistration. The R1 enforcement implementation predates the committed
54-query exam (it landed with an earlier, superseded pack in `475be22`); the
exam evaluated here (`f09651f3…`) was authored independently afterwards, so
the implementation could not have been tuned to these specific queries, but
the ordering does not satisfy strict preregistration. The non-inferiority
criterion, margin, conditions, and safety gates themselves were preregistered
before this evaluation ran.

Additional run facts not embedded in the result JSON's `run_manifest`:

- Freeze receipt commit: `33284e3` (`docs/evidence_admission_and_budgeting_r1_formal_pack_freeze_receipt.md`)
- Cache state: cold first-pass for every condition (fresh service instances,
  fresh collection; no pre-cognitive cache fixtures seeded)
- Corpus snapshot label: `evidence_admission_r1_frozen_corpus_formal:684`

The formal collection was seeded through the HTTP index path after fixing a
seed-ID collision defect. Verification:

```text
docs: 684 unique ids: 684 collisions: 0
collection dim: 768 | points: 684
SEED VERIFIED: 684/684, collision-free
```

## Results

| Condition | Coverage + lineage | Abstain served | Forbidden routes | Fallback |
|---|---:|---:|---:|---:|
| normal baseline | 33/42 (78.57%) | 0/12 | 0 | 0 |
| R0 shadow only | 33/42 (78.57%) | 0/12 | 0 | 0 |
| R1 enforcement | 32/42 (76.19%) | 0/12 | 0 | 7 |
| R1 gate disabled | 33/42 (78.57%) | 0/12 | 0 | 0 |

Safety and gate checks:

- R0 shadow served output matched normal retrieval for 54/54 queries.
- R1 gate disabled served output matched normal retrieval for 54/54 queries.
- R1 gate-disabled metadata correctly recorded `requested=true` and `globally_enabled=false` for 54/54 queries.
- R1 enforcement served zero forbidden route labels.

R1 enforcement routes:

- `BOUNDED_SEMANTIC_RETRIEVAL`: 47 queries
- `NORMAL_RETRIEVAL_FALLBACK`: 7 queries

The single source-level primary-metric regression versus normal was:

- `r1f-004`: normal top source was `docs/benchmarks/gatemem_g5/README.md`; enforced top source was `docs/benchmarks/gatemem_g4_offline_reference_implementation.md`; final route was `BOUNDED_SEMANTIC_RETRIEVAL`; fallback did not trigger.

## Interpretation

The formal result repeats the structural shape seen in the diagnostic run:
R1's kill switches and forbidden-route controls are mechanically sound, but
the current runtime does not meaningfully exercise the intended cue-only and
cache-only lower-cost route mix. With empty cue/tag registries and no cache
fixtures, enforcement mostly collapses to bounded semantic retrieval plus
fallback. The formal failure is therefore a retention failure for this R1
configuration, not evidence for a positive bounded-enforcement deployment
claim.

The `r1f-004` regression is the key source-level failure mode for a future R2
proposal: R1 judged a bounded result sufficient even though the bounded result
displaced the required source and did not fall back. That should be addressed
in a new preregistered redesign cycle, not by tuning this R1 cycle after the
formal result.

Out-of-corpus abstention remains unmet: all 12 abstention-expected queries were
served by retrieval rather than abstained. This is not counted in the primary
coverage denominator, but it blocks any broader safety claim about content-level
out-of-corpus abstention.

## Decision

- `R1_FORMAL_HTTP_SERVICE_EVALUATION_COMPLETE`
- `R1_PRIMARY_NON_INFERIORITY_FAIL`
- `R1_SAFETY_GATE_CONTROLS_PASS`
- `R1_POSITIVE_RETENTION_CLAIM_BLOCKED`
- `R1_ENFORCEMENT_NOT_RETAINED`

No R1 policy, thresholds, route mappings, formal pack content, or corpus
content should be tuned in place in response to this result. Any revised R1
candidate requires a new preregistered evaluation cycle, with the affected
artifacts refrozen before another formal claim.
