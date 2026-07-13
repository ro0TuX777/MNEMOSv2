# GateMem G5 One-Shot and Invalidation Rules

Status: proposed rules; custodian/reviewer must adopt them before unsealing.

## One-shot definition

The claim-eligible attempt begins when the evaluator first decrypts, mounts,
streams, or otherwise exposes any sealed case to the candidate execution
environment. From that moment, the candidate, configuration, thresholds,
scorer, aggregation, and corpus are immutable.

## Before unsealing

A preflight may be repeated only when the custodian certifies that:

- no sealed case or label was exposed;
- no prediction was produced from sealed data;
- candidate and corpus commitments remain unchanged; and
- the failed preflight and retry are recorded.

## After unsealing

Any of the following makes the attempt final or invalid:

- candidate/policy/configuration change;
- threshold, metric, denominator, or claim change;
- selective case retry or omission;
- annotation exposure to candidate or policy developers;
- missing, duplicate, reordered contrary to protocol, or malformed predictions;
- scorer/aggregation defect;
- output loss, partial overwrite, or unverifiable manifest;
- undeclared network/model/service dependency;
- corpus commitment mismatch or development overlap;
- accidental human or automated disclosure of sealed cases to policy developers.

An invalid attempt may be debugged only as retrospective/development evidence.
A fresh held-out claim requires a new untouched corpus and new registration.

## Infrastructure failures

The preregistration must choose one rule before unsealing:

1. **Strict:** any post-unseal infrastructure failure invalidates the run; or
2. **Checkpointed:** resume is allowed only from a custodian-signed append-only
   checkpoint, with the identical candidate/environment, no case replay unless
   preregistered, and proof that no intermediate result reached policy
   developers.

The default is strict if no rule is selected.

## Result access

- Policy developers receive no intermediate per-case feedback.
- The custodian decides whether row-level artifacts are ever released.
- Aggregate results are released once, after exception review and claim
  classification.
- If cases or labels are later disclosed, that corpus becomes retrospective for
  all future policy changes.

## Exception log

Each exception record must include time, operator, affected component/cases,
whether unsealing occurred, exposure assessment, action, signatures, and final
classification. Silence is not evidence that no exception occurred.
