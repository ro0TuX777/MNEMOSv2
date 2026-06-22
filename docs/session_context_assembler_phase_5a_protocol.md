# Session Context Assembler — Phase 5A Technical and Owner Verification

Status: authorized non-human-verification lane. Phase 5 independent human
review remains frozen and unrun.

## Evidence boundary

Phase 5A may support technical robustness, artifact integrity, provenance,
budget-abstention behavior, and a non-generalizable product-owner finding. It
may not support human usability, general preference, operator productivity,
production readiness, authority, promotion, governance, or durable-memory
claims.

All outputs must carry these boundaries:

```text
TECHNICAL_VERIFICATION_ONLY
NOT_HUMAN_VALUE_EVIDENCE
NO_RUNTIME_INTEGRATION
NO_PRODUCTION_READINESS_CLAIM
```

## Workstream A — frozen held-out R2

R2 contains ten new synthetic cases, one for each authorized adversarial
class: old decisive evidence versus lexical distraction, multiple unresolved
contradictions, mixed related decisions, split source evidence, incident
interruption, mandatory overflow, ineligible source linkage, within-episode
reordering, irrelevant injection, and task paraphrase.

Every case carries a scoring-only `verification_expectations` block. The S1
selector must never read that block. It receives only the existing runtime-like
task, session turns, eligibility, decision artifacts, and source links. R0 and
R1 remain independently frozen.

## Workstream B — deterministic verification

Run:

```text
python tools/run_session_context_assembler_r2_verification.py
python -m pytest -q tests/test_session_context_assembler_phase_5a.py
```

The verifier evaluates A full history, B sliding window, and C1 S1 against the
same case budget. Advancement is gated on:

- 1.0 required-artifact retention for budget-feasible cases;
- explicit abstention for every infeasible mandatory set;
- zero silent required-artifact omissions;
- 100% budget compliance;
- zero provenance loss and 1.0 lineage/label preservation;
- zero ineligible-source selection violations;
- fixed-seed determinism;
- unchanged, hash-valid R1 and R2; and
- no scoring-field, runtime, filesystem-write, governance, authority,
  promotion, retrieval, or memory mutation path in S1.

Mutation sensitivity is required. Tests and verifier self-checks must detect
mandatory-order bypass, parent-source removal, synthetic-label removal,
abstention suppression, and scoring-only-field access. These checks establish
that critical gates are not vacuous.

## Workstream C — optional model-assisted surrogate

Not run under this closeout. Any future fixed-model answer-fidelity evaluation
must be separately labeled `MODEL_ASSISTED_SURROGATE_EVALUATION`, use identical
model/prompt settings across A/B/C1, and report only grounded agreement,
source-ID preservation, contradiction handling, abstention acknowledgement,
and unsupported-claim rate. It cannot substitute for human evidence.

## Workstream D — product-owner pack

The verifier prepares ten masked tasks at:

```text
benchmarks/review_packets/session_context_assembler_phase_5a_owner_review.json
```

The restricted condition key is stored separately in
`session_context_assembler_phase_5a_owner_manifest.json`. The owner must not
use that key on first pass. The pack visibly retains S1 provenance and
abstention disclosures and carries:

```text
PRODUCT_OWNER_REVIEW
NOT_INDEPENDENT_HUMAN_STUDY
NOT_GENERALIZABLE
```

Preparation is not execution. No owner findings are recorded by the technical
verification run.

## Advancement boundary

A Phase 5A PASS authorizes a separate proposal for a read-only,
consumer-neutral technical shadow adapter.

It does not authorize live routing, production use, memory writes, governance
mutation, promotion behavior, or a human-value claim. SAM is one possible
future consumer for testing only; it is not part of MNEMOS's core architecture
or product boundary.
