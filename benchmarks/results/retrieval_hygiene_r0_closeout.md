# Retrieval Hygiene R0 Closeout

Date: 2026-06-27

Status:

```text
RETRIEVAL_HYGIENE_AND_REPRODUCIBILITY_R0_COMPLETE
R0_REGRESSION_EVIDENCE_COMPLETE
FRESH_VERIFICATION_EVIDENCE_COMPLETE
VERIFICATION_PACK_NOT_HELD_OUT
NO_FORMAL_QUERY_ALIAS_ARTIFACT_NEEDED_AT_THIS_TIME
NO_GOVERNANCE_OR_AUTHORITY_CHANGE
```

## Scope

This closeout records the retrieval hygiene and reproducibility milestone for
source-linked MNEMOS summary cards over the current seeded repository corpus.

It covers:

- idempotent summary seeding;
- retrieval-side duplicate suppression;
- seed snapshot derivation;
- executed-route fingerprint reporting;
- low-relevance abstention for unrelated queries; and
- direct-service versus MCP-path comparison under a fixed 8-leg cold/warm run matrix.

It does not authorize any change in memory authority, governance semantics, or
artifact type promotion.

## Evidence bundle

Primary artifacts:

- [benchmarks/results/retrieval_hygiene_r0_run_003.json](G:\MNEMOS\benchmarks\results\retrieval_hygiene_r0_run_003.json)
- [benchmarks/results/retrieval_hygiene_r0_fresh_verification_run_001.json](G:\MNEMOS\benchmarks\results\retrieval_hygiene_r0_fresh_verification_run_001.json)
- [benchmarks/results/retrieval_hygiene_r0_fresh_verification_pack_freeze_001.json](G:\MNEMOS\benchmarks\results\retrieval_hygiene_r0_fresh_verification_pack_freeze_001.json)
- [docs/experiments/retrieval_hygiene_r0_frozen_alias_benchmark.json](G:\MNEMOS\docs\experiments\retrieval_hygiene_r0_frozen_alias_benchmark.json)
- [docs/experiments/retrieval_hygiene_r0_fresh_verification_pack.json](G:\MNEMOS\docs\experiments\retrieval_hygiene_r0_fresh_verification_pack.json)

Supporting tooling:

- [tools/run_retrieval_hygiene_benchmark.py](G:\MNEMOS\tools\run_retrieval_hygiene_benchmark.py)
- [tools/run_retrieval_fresh_verification.py](G:\MNEMOS\tools\run_retrieval_fresh_verification.py)
- [tools/freeze_retrieval_fresh_verification_pack.py](G:\MNEMOS\tools\freeze_retrieval_fresh_verification_pack.py)
- [tools/seed_mnemos_repo_summaries.py](G:\MNEMOS\tools\seed_mnemos_repo_summaries.py)
- [tools/seed_mnemos_repo_context.py](G:\MNEMOS\tools\seed_mnemos_repo_context.py)

## Supported claims

The combined regression and verification evidence supports these bounded claims:

1. Source-linked summary cards can improve retrieval access to repository
   knowledge.
2. Idempotent seeding prevents repeated summary ingestion from silently
   expanding the active corpus.
3. Retrieval-side duplicate suppression prevents duplicate candidates from
   consuming delivered top-k space.
4. Seed snapshots and executed-route fingerprints make retrieval runs more
   reproducible and inspectable.
5. The low-relevance abstention guard rejected unrelated negatives in the tested
   corpus without suppressing the tested weak-but-relevant query.
6. Direct and MCP retrieval paths behaved consistently under the tested fixed
   seed snapshot and 8-leg cold/warm matrix.

## Regression result

R0 regression rerun artifact:
[retrieval_hygiene_r0_run_003.json](G:\MNEMOS\benchmarks\results\retrieval_hygiene_r0_run_003.json)

Summary:

- direct-service top-1 accuracy: `1.0`
- MCP-path top-1 accuracy: `1.0`
- direct/MCP agreement: `1.0`
- cold/warm consistency: `1.0`
- duplicate result rate: `0.0`
- q15 abstention: present on both paths
- executed-route fingerprints: present

q08 was retained as an ambiguity-aware benchmark case rather than a forced
single-neighborhood accuracy gate. This reflects shared wording across G4 and
G5 evidence neighborhoods rather than a retrieval defect.

## Fresh verification result

Fresh verification artifact:
[retrieval_hygiene_r0_fresh_verification_run_001.json](G:\MNEMOS\benchmarks\results\retrieval_hygiene_r0_fresh_verification_run_001.json)

Summary:

- direct-service top-1 accuracy: `1.0`
- MCP-path top-1 accuracy: `1.0`
- abstention accuracy: `1.0`
- false abstention rate: `0.0`
- direct/MCP agreement: `1.0`
- cold/warm consistency: `1.0`
- duplicate result rate: `0.0`

Verification probes included:

- unrelated negatives;
- weak-but-relevant positives;
- a relevant-low-score query that must not abstain; and
- G4/G5 disambiguation cases.

## Boundaries

This evidence is bounded in two important ways.

1. It is retrieval behavior evidence for the current seeded corpus, query packs,
   seed snapshot, and retrieval configurations. It is not a broad retrieval
   quality claim.
2. The fresh verification pack is a separate verification artifact, not a
   held-out evaluation artifact, because it was created and reviewed within the
   same development effort.

## Conclusion

At this milestone, ordinary summaries are sufficient for the tested retrieval
need when they are:

- source-linked;
- versioned;
- idempotently seeded;
- clearly non-authoritative; and
- paired with duplicate hygiene, abstention guardrails, and executed-route
  fingerprinting.

No `query_alias` or `retrieval_anchor` primitive is needed at this time.

The benchmark and verification packs should be retained as regression assets.
A formal retrieval-only alias artifact should be reconsidered only if later
evidence shows that ordinary summaries cannot represent a retrieval need
cleanly.
