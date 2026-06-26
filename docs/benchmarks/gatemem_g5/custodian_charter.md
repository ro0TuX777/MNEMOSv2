# GateMem G5 Independent Custodian Charter

Status: unsigned template. Completion requires a real external custodian.

## Mission

The custodian protects the independence of a newly sealed or independent
authorization/disclosure evaluation. The custodian controls corpus access,
accepts the frozen candidate and preregistration, authorizes at most one
claim-eligible run, and releases only preregistered results.

## Independence requirements

The custodian must attest that neither the custodian nor the corpus authors:

- designed or tuned the nominated G4 policy/reference implementation;
- inspected G4 failure cases for the purpose of optimizing the sealed corpus;
- provided sealed cases, labels, leak targets, or expected actions to the policy
  group before candidate freeze;
- share credentials, storage, workspace, logs, or model conversation history
  with the policy-development group; or
- have an undisclosed financial, organizational, or authorship conflict that
  could influence corpus construction, execution, or reporting.

Automation may execute the run, but a person/team outside policy development
must control its secrets, approvals, corpus, and exception decisions.

## Custodian duties

1. Verify identity and role separation for policy developers, corpus authors,
   evaluator operators, release reviewers, and custodian staff.
2. Hold the sealed corpus and evaluator-only annotations outside the MNEMOS
   development workspace.
3. Publish a cryptographic corpus commitment before accepting the candidate,
   without disclosing cases, labels, domains if sealed, or sensitive counts.
4. Verify corpus novelty/non-overlap against development and released GateMem
   data using a method unavailable to policy code.
5. Verify and accept the candidate, dependency, configuration, and
   preregistration hashes before unsealing.
6. Enforce one-way data flow: clean policy inputs to the candidate; predictions
   to the evaluator; annotations never to the candidate.
7. Authorize or deny the one-shot run and record start/end times, operator,
   environment, exceptions, and exposure events.
8. Retain row-level sealed artifacts; release only preregistered aggregate or
   approved audit evidence.
9. Classify the run as held-out, retrospective, development, or invalid.
10. Refuse any post-unseal policy change, selective rerun, threshold change, or
    unregistered claim.

## Required controls

- separate private storage/repository and credentials;
- least-privilege evaluator service identity;
- append-only execution/audit log;
- candidate and corpus hash verification;
- network and output allowlists;
- disabled interactive policy-developer access during execution;
- sealed row-level outputs and bounded retention;
- documented incident and accidental-exposure procedure.

## Appointment and attestations

```yaml
custodian_organization_or_person:
custodian_contact:
custodian_signing_identity:
appointment_reference:
appointed_at:
conflict_disclosures: []
independence_attestation_signed: false
corpus_custody_accepted: false
one_shot_authority_accepted: false
signature_or_attestation_digest:
```

No blank/false field may be interpreted as approval. The current MNEMOS policy
group and the assistant that helped develop G3/G4 cannot sign as the independent
custodian.
