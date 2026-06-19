# ADR 0004: EBIR Remains Shadow-Only Until Human-Value Evidence Passes

Date: 2026-06-20

Status: Accepted

## Context

Evidence-Bounded Iterative Reconciliation (EBIR) can refine contradiction
handling offline, but its current evidence base does not justify authoritative
promotion or live memory mutation.

## Decision

EBIR-R1 and EBIR-R2 remain shadow-only evaluation lanes. They must not write
live memory, alter ranking, promote authority, mutate parent evidence, or
change production retrieval behavior.

## Alternatives Considered

- Promote EBIR outputs into Resolution Engrams automatically.
- Use synthetic or AI-generated reviewer responses as human-value evidence.
- Hide EBIR until every validation step is complete.

## Invariants

- `blocked_from_authoritative_resolution_promotion` remains the promotion
  posture.
- Reviewer-facing R2 packets must be blinded and free of gold labels.
- Human-value claims require independent reviewers who do not see the truthset,
  condition mapping, or MNEMOS internals.
- Parser, compiler, and scoring validation is not a substitute for human-value
  evidence.

## Rollback

If an EBIR path leaks into live memory, ranking, governance, or production API
behavior, disable the path and rerun the EBIR preflight and CI gates before
resuming evaluation.

## Evidence

- `docs/ebir_r1_acceptance.md`
- `docs/ebir_r2_trial_protocol.md`
- `tools/run_ebir_refinement_benchmark.py`
- `tools/run_ebir_r2_preflight.py`
- `tools/score_ebir_r2_gold_report.py`

