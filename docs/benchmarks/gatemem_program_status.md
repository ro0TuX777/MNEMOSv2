# GateMem Program Status

Updated: 2026-06-25

## Closed milestones

```text
GATEMEM_G0_ENVIRONMENT_AND_GAP_ASSESSMENT_COMPLETE
GATEMEM_G1_CLEAN_INPUT_PROJECTION_COMPLETE
GATEMEM_G2_OFFLINE_BASELINE_COMPLETE
GATEMEM_G2A_CROSS_DOMAIN_CHARACTERIZATION_COMPLETE
GATEMEM_G3_DESIGN_COMPLETE
GATEMEM_G4_OFFLINE_REFERENCE_IMPLEMENTATION_COMPLETE
```

Canonical program closeout:

```text
GATEMEM_G0_ENVIRONMENT_AND_GAP_ASSESSMENT_COMPLETE
GATEMEM_G1_CLEAN_INPUT_PROJECTION_COMPLETE
GATEMEM_G2_G2A_FROZEN_BASELINE_COMPLETE
GATEMEM_G3_AUTHORIZATION_DISCLOSURE_DESIGN_COMPLETE
GATEMEM_G4_OFFLINE_REFERENCE_CONFORMANCE_COMPLETE
```

G3 acceptance record:

```text
GATEMEM_G3_DESIGN_ACCEPTED
AUTHORIZATION_DISCLOSURE_SEMANTICS_COMPLETE
NO_RUNTIME_IMPLEMENTATION
NO_POLICY_TUNING
NO_NEW_BENCHMARK_CLAIM
NO_DELETION_ENGINEERING
```

| Milestone | Closed outcome | Claim boundary |
|---|---|---|
| G0 | External environment pinned; compatibility and deletion gaps assessed | No adapter or runtime claim |
| G1 | Clean projection, hidden-field isolation, and external normalizer complete | Offline plumbing only |
| G2 | Medical offline baseline frozen under composite SHA-256 `4211bc91e7dbe53f588a8ecb00a04e6e44d2fb775ca1e888d058680acb5ad209` | Mechanics/provenance pass; disclosure inadequate |
| G2A | Same frozen baseline replayed across all four GateMem domains | Characterization only; no tuning or fresh held-out claim |
| G3 | Principal-bound authorization/disclosure semantics and evaluation governance accepted | Design only; no implementation, tuning, benchmark, or deletion claim |
| G4 | Local authorization/disclosure reference implementation passed inspectable synthetic development cases | Reference-contract conformance only; no production, held-out, benchmark, runtime, or deletion claim |

G2/G2A is a successful benchmark-baseline closeout, not a successful
governance-performance result. Across 2,218 checkpoints, the frozen baseline
produced:

- auxiliary utility `0.2775`;
- privacy end-to-end leakage `0.1788`;
- over-refusal `0.4052`;
- provenance integrity `1.0000`; and
- deletion-refusal consistency `0.9738`, with active forgetting unscored.

## Latest completed milestone

```text
ADR_0013_ACCEPTED
GATEMEM_G4_OFFLINE_REFERENCE_IMPLEMENTATION_COMPLETE
REFERENCE_CONTRACT_CONFORMANCE_ON_INSPECTABLE_SYNTHETIC_DEVELOPMENT_CASES
SYNTHETIC_DEVELOPMENT_ONLY
SEALED_EVALUATION_STILL_BLOCKED
NO_RUNTIME_OR_DELETION_CHANGE
```

G4 implemented a local offline reference over 36 MNEMOS-owned, inspectable
synthetic development cases. All expected outcomes and reference gates passed.
This is contract-conformance evidence, not authorization security, production
readiness, benchmark performance, or held-out evaluation.

The exact G4 source/corpus composite
`ed3b5c7672e591b039183eaa2d8c7c7630a655575bfc7866558d53f2eb874c52`
is now frozen as a regression-only reference baseline.

## GateMem pause and external blocker

```text
GATEMEM_G4_FROZEN_REFERENCE_CONTRACT_BASELINE
REGRESSION_TESTING_ONLY
SEALED_EVALUATION_STILL_BLOCKED
NO_FURTHER_INTERNAL_PROTOTYPE_AUTHORIZED
```

Further GateMem policy and implementation work is paused. The next meaningful
GateMem milestone requires an independent sealed-evaluation custodian, a newly
sealed or independent corpus, completed preregistration, a frozen candidate
policy artifact, and one-shot evaluation under evaluator-only label access.

Without that arrangement, additional policy work is development evidence only
and cannot support a fresh generalization claim.

## Independent evaluation continuation packet

```text
GATEMEM_G5_PACKET_READY_FOR_EXTERNAL_HANDOFF
SEALED_EVALUATION_BLOCKED_EXTERNAL_INPUTS_REQUIRED
NO_EVALUATION_RUN_AUTHORIZED
```

The continuation index is `docs/benchmarks/gatemem_g5/README.md`. It identifies
the files used by the independent custodian, evaluator operator, release
reviewer, and MNEMOS policy group:

- custodian charter and independence attestation;
- sealed-evaluation preregistration;
- custodian-controlled evaluator protocol;
- one-shot and invalidation rules;
- ordered handoff checklist;
- frozen G4 candidate nomination; and
- machine-readable handoff state and readiness verifier.

The packet nominates the existing frozen G4 composite; it does not modify G4.
No custodian is appointed, no sealed corpus is present, no preregistration is
frozen, and no evaluation or performance claim is authorized. Those fields must
be completed by named external parties before work can advance.

## MNEMOS roadmap handoff

The separately authorized post-GateMem backlog was executed without reopening
the frozen GateMem lane:

| Item | Outcome |
|---|---|
| Model-assisted A/B/C1 answer fidelity | Completed as fixed-model surrogate evidence; not human or generalizable |
| Narrow focused GitHub Actions gates | Added with G4 frozen-verifier-first ordering and assembler-focused tests |
| Release SBOM and dependency hygiene | SPDX generation implemented; external release remains blocked on an exact/hash-pinned dependency set and approved vulnerability scan |

See `docs/reports/post_gatemem_authorized_backlog_closeout.md`.

## Persistent red lines

- The four released GateMem domains are frozen historical baseline data, not a
  fresh future held-out set.
- Candidate filtering is not content-safe disclosure.
- Provenance integrity is not authorization correctness.
- Refusal is not deletion or active forgetting.
- Authorization design must not be folded into durable deletion engineering.
- Any future policy evaluation requires a preregistered development/evaluation
  split and a newly sealed or independent evaluation corpus.
- G4 is a frozen regression baseline; changes require a new development
  iteration rather than rewriting the frozen result.
