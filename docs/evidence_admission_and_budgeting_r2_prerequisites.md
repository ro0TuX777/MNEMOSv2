# Evidence Admission and Budgeting R2 — Design Prerequisites

## Status

R2_NOT_OPENED. This document is not an R2 design, preregistration, or work
authorization. It records the prerequisites any future R2 cycle must satisfy
before design work begins, so the closed R1 lane is never reopened to hold
them.

Posture until R2 opens: R0_SHADOW_OBSERVABILITY_RETAINED — observe admission
and sufficiency signals; do not enforce them into retrieval behavior.
R1_ENFORCEMENT_NOT_RETAINED per
`docs/evidence_admission_and_budgeting_r1_closeout.md`.

## Primary redesign target (from R1 formal evidence)

The sharply defined failure mode, observed as the sole primary-metric
regression (`r1f-004`, formal run 001):

```text
bounded result judged sufficient
→ required source absent
→ fallback not triggered
```

Post-retrieval sufficiency assessment can approve a bounded result that
displaced the required evidence, so the mandatory fallback never fires. Any
R2 enforcement design must close this gap before anything else.

Secondary targets, both structural and embedder-independent (R1 diagnostic +
formal):

- Content-level out-of-corpus abstention was never delivered (0/12 formal;
  11/12 served in the diagnostic): pre-retrieval abstain only fires for
  service-scope-unknown reasons while cue/tag registries are empty.
- Route collapse: with empty cue/tag registries and no cache fixtures,
  enforcement reduces to bounded semantic retrieval (+ fallback); the
  CUE_ONLY_LOOKUP and CACHE_ONLY routes are structurally unexercisable.
  An R2 evaluation must either provision these mechanisms or preregister
  their absence.

## R2_TOOLING_REQUIREMENT

HTTP run manifests must emit, per condition/pass:

- `formal_pack_hash`
- `freeze_receipt_commit`
- `corpus_snapshot`
- `cache_state` (cold/warm, per condition and per pass)
- `service_revision`
- `runner_revision`
- embedding model and dimension

The R1 runner (`tools/run_evidence_admission_r1_comparison.py`) does not emit
`freeze_receipt_commit`/`cache_state` for HTTP runs. Deliberately deferred:
R1 is closed, and patching its tooling would reopen a closed lane without
changing the decision. The requirement binds R2 tooling instead.

## Process prerequisites

- New corpus manifest, formal pack, and preregistration cycle — no reuse or
  repair of R1 artifacts. R1 evidence stands as durable negative evidence.
- Formal pack authored by an `independent_non_implementation_author` and
  frozen/hashed **before** R2 enforcement implementation (strict
  pack-before-implementation ordering — the R1 evidence-order limitation
  documented in the R1 closeout must not recur).
- Formal evaluation over a revision-proving HTTP service from a clean
  committed checkout; direct-runtime evidence remains diagnostic-only and
  non-aggregatable.
