# GateMem G4 Synthetic Development Corpus Proposal

Date: 2026-06-24

Status: implemented as inspectable synthetic development data; never held out.

## Purpose

The G4 development corpus would exercise the G3 contracts without GateMem data,
production memory, personal data, or a hidden evaluation set. Every case is
MNEMOS-owned synthetic development material and may be inspected during policy
work. No result from this corpus is a held-out performance claim.

## Generator contract

The future deterministic generator should accept only:

```yaml
generator_request:
  schema_version: immutable version
  generator_version: immutable version
  seed: integer
  base_case_count: positive integer
  mutation_profile: registry identifier
  output_root: declared local path
```

It should emit:

```yaml
development_corpus_manifest:
  corpus_id: identifier
  license: MNEMOS-owned synthetic research data
  schema_version: immutable version
  generator_version: immutable version
  seed: integer
  case_count: integer
  mutation_profile: registry identifier
  file_sha256: mapping
  composite_sha256: sha256
  contains_gatemem_data: false
  contains_production_data: false
  held_out_eligible: false
```

Case files should separate decision-path inputs from evaluator expectations.
Decision components must never load expectation files.

## Case families

Base cases must cover:

- principal-owned session evidence;
- authorized scoped delegation;
- same-tenant but unrelated principal;
- cross-session and cross-tenant attempts;
- suspended/revoked/expired membership;
- missing, expired, or wrong-scope role assignment;
- role without matching entitlement;
- purpose, operation, relationship, artifact-class, source-class, and
  classification mismatches;
- explicit denial overriding a broad grant;
- raw, summary, resolution, derived, and synthetic artifact lineage;
- permitted disclosure, required redaction, and total denial;
- uniform not-found/out-of-scope/denied behavior; and
- valid replay and replay after every G3 drift dimension.

## Adversarial mutations

Each applicable base case should be mutated one dimension at a time:

- caller tenant/session/principal widening;
- query text claiming consent, urgency, role, relationship, or authority;
- forged/unknown issuer, bad signature, altered envelope, expiry, or replay;
- role inflation and delegation widening;
- entitlement removal, conflicting denial, or snapshot drift;
- session tenant swap or membership revocation;
- artifact/source reclassification and incomplete/changed lineage;
- unknown policy or obligation identifier;
- redaction span overlap, missing label, transform failure, or residue;
- package budget overflow;
- cache/package reuse after identity, policy, entitlement, descriptor, or
  redaction-version change;
- audit unknown-field injection and protected-content canaries; and
- evaluator-field injection into a decision-path schema.

## Synthetic secret canaries

Protected fields should contain unique generator-owned canaries. Gates scan
disclosure, denial, error, diagnostic, and audit artifacts for canaries that are
not explicitly authorized. Canaries are test strings, not credentials or real
personal data.

## Development measurements

Report exact counts, not a broad security score:

- expected permits, redacted permits, and denials matched;
- unauthorized disclosure count;
- required-disclosure omission count;
- redaction residue and verification-failure count;
- caller-widening and query-authority permit count;
- stale replay acceptance count by drift dimension;
- provenance/decision fingerprint mismatch count;
- prohibited audit/error field or canary count; and
- deterministic manifest/rerun mismatch count.

Any nonzero unauthorized disclosure, redaction residue, caller/query authority
permit, stale replay acceptance, or prohibited audit/error leakage fails the
reference gate. Passing demonstrates conformance to authored development cases,
not production safety or generalization.

## Corpus governance

- Generator source, templates, labels, seeds, manifests, and hashes are versioned.
- Generated data remains outside GateMem upstream.
- GateMem medical, office, education, and household data are prohibited inputs.
- Production MNEMOS records, copied user text, and external personal data are
  prohibited inputs.
- Policy developers may inspect and change development cases, but every change
  creates a new corpus version and composite hash.
- This corpus is permanently marked `held_out_eligible: false`.
- A fresh evaluation requires an independently held, newly sealed corpus under
  the G3 preregistration protocol.
