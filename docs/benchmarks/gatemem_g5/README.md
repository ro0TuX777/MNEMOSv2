# GateMem G5 Independent Evaluation Readiness Packet

Status: `READY_FOR_EXTERNAL_HANDOFF` — sealed evaluation remains blocked.

This directory is the starting point for continuing GateMem work after the
frozen G4 reference baseline. It prepares an independent evaluation handoff; it
does not appoint a custodian, create a sealed corpus, freeze a preregistration,
or authorize a run.

## File map

| File | Used by | Purpose |
|---|---|---|
| [custodian_charter.md](custodian_charter.md) | Independent custodian and release reviewer | Independence, custody, access, conflict, disclosure, and signing duties |
| [preregistration.md](preregistration.md) | Custodian, evaluator, policy group, release reviewer | Fields and decisions that must be completed and frozen before unsealing |
| [evaluator_protocol.md](evaluator_protocol.md) | Evaluation operator/custodian | Clean projection, candidate execution, prediction freeze, scoring join, and reporting interface |
| [one_shot_rules.md](one_shot_rules.md) | Custodian and release reviewer | Retry, invalidation, exposure, exception, and post-run change rules |
| [handoff_checklist.md](handoff_checklist.md) | All parties | Ordered handoff and signatures required to move from blocked to executable |
| [candidate nomination JSON](../../../benchmarks/evaluation/gatemem_g5_candidate_nomination.json) | Custodian/evaluator | Machine-readable nomination of the frozen G4 candidate |
| [handoff state JSON](../../../benchmarks/evaluation/gatemem_g5_handoff_state.json) | Readiness verifier | Explicit completed and externally blocked state |
| [readiness verifier](../../../tools/verify_gatemem_g5_readiness.py) | Policy group and custodian | Checks packet integrity without accessing any sealed corpus |

## What can happen now

1. Send this packet to a person/team/process outside the policy-development
   group.
2. The prospective custodian reviews and signs the charter/conflict statement.
3. The custodian supplies or controls a newly sealed/independent corpus and
   publishes only a commitment hash and allowed metadata.
4. All parties complete and freeze the preregistration before unsealing.
5. The custodian accepts the nominated candidate hash or rejects it before the
   run.
6. The evaluator executes the one-shot protocol under custodian control.

## What remains prohibited

- policy developers creating, inspecting, or holding sealed cases or labels;
- using the four already-observed GateMem domains as fresh held-out data;
- changing G4 and rewriting its frozen baseline;
- executing a claimed held-out run before all external signatures are present;
- production authorization, legal compliance, deletion, or active-forgetting
  claims.

The readiness command is:

```text
python tools/verify_gatemem_g5_readiness.py
```

Expected pre-custodian result:

```text
GATEMEM_G5_PACKET_READY_FOR_EXTERNAL_HANDOFF
SEALED_EVALUATION_BLOCKED_EXTERNAL_INPUTS_REQUIRED
```
