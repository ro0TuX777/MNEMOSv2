# Evidence Admission and Budgeting R1 Formal Pack — Freeze Receipt

## Status

FROZEN. This receipt closes the independent-authorship step required by the R1
preregistration and formal-pack template before any enforcement implementation
work may begin.

This receipt supersedes the prior receipt that described an earlier pack
(55 scored queries, SHA-256 `338999ce…`). That earlier pack was replaced by the
independently authored pack recorded below; the earlier hash and receipt are no
longer valid for any formal R1 claim.

## Artifact

- Path: `docs/experiments/evidence_admission_and_budgeting_r1_formal_pack.json`
- `pack_id`: `evidence_admission_and_budgeting_r1_formal_pack_v1`
- `derived_from_template`: `evidence_admission_and_budgeting_r1_formal_pack_template_v1`
- `corpus_manifest_id`: `evidence_admission_and_budgeting_r1_corpus_69ad546ba30ed71a`
- Scored query count: 54 (minimum required: 50)
- Byte count: 74770
- SHA-256: `f09651f3fc67b0bddf73b3981a0f635e21c58ff3d4ed50bc717d2886377c14cc`
- Authored against repository baseline commit: `e7b86c65d3fd0298f915efc00cc8cf6fd7fb7f7e` (branch `codex/gatemem-governance-reference-baseline`)
- Committed in: `8385b27`
- Machine-readable freeze record: `docs/experiments/evidence_admission_and_budgeting_r1_formal_pack_freeze.json`
- Freeze date: 2026-07-02

Recompute with:

```
python -c "import hashlib; print(hashlib.sha256(open(r'docs/experiments/evidence_admission_and_budgeting_r1_formal_pack.json','rb').read()).hexdigest())"
```

Any content change to the pack after this receipt invalidates the freeze and
requires a new hash and a new receipt before the pack can be used for a formal
R1 claim.

## Author identity / role statement

This pack was authored in the `independent_non_implementation_author` role
required by the corpus manifest, the formal-pack template, and the R1
preregistration. The author did not write, review, or modify any R1
enforcement implementation code, policy, or threshold, and holds no stake in
enforcement outcomes beyond producing a correctly specified, corpus-grounded
exam. The `authorship` and `provenance` blocks embedded in the pack JSON carry
the same statement together with the SHA-256 of each authoring input.

## Confirmation of authoring inputs

The pack was authored only from:

- the committed, frozen corpus manifest (`docs/evidence_admission_and_budgeting_r1_corpus_manifest.json`), which enumerates the 41 frozen sources with their family/role/hash metadata;
- the formal-pack template (`docs/experiments/evidence_admission_and_budgeting_r1_formal_pack_template.json`), which fixed the required schema, allowed/forbidden route labels, and non-scored schema examples;
- the R1 preregistration (`docs/evidence_admission_and_budgeting_r1_preregistration.md`);
- the R1 design note (`docs/evidence_admission_and_budgeting_r1_design_note.md`).

Queries were grounded in the frozen corpus manifest's declared sources (by
path, family, and role). No development-pack content and no runtime system
behavior were consulted.

The author explicitly did **not** read, inspect, or use:

- `docs/experiments/evidence_admission_and_budgeting_r1_development_pack.json` (development-pack results) or any of its contents;
- any R1 enforcement implementation code, route-classification logic, or runtime behavior;
- any prior R1 policy or threshold value, for the purpose of tuning query difficulty or expected outcomes.

## Authoring constraints observed

- R1 policy, thresholds, and implementation were not modified.
- Development-pack results were not inspected, aggregated, or referenced.
- Queries were not tuned against implementation behavior — none was observed.
- No enforcement code was created.
- Development evidence and formal evidence were not aggregated; this pack contains only formal-pack queries.

## Coverage summary

- 54 scored queries: 42 non-abstention and 12 abstention-expected.
- Enforced-route eligibility declared per query: `CUE_ONLY_LOOKUP` 21, `CACHE_ONLY` 2, `BOUNDED_SEMANTIC_RETRIEVAL` 5, forced `NORMAL_RETRIEVAL_FALLBACK` 14, and `ABSTAIN_OR_REQUEST_SCOPE` 12. No query selects any label in the template's `forbidden_enforced_route_labels` set.
- Source coverage: 38 of the 41 frozen corpus sources appear in at least one accepted evidence neighborhood. The 3 sources that never appear are `negative_control_material`.
- Accepted-evidence neighborhood role references: `current_state_record` 40, `superseded_record` 8, `dependency_blocker_record` 5, `duplicate_or_near_duplicate_condition` 5, and `negative_control_material` 6. Only `current_state_record` (40), `dependency_blocker_record` (4), and `duplicate_or_near_duplicate_condition` (2) references are admissible drivers; the `superseded_record` and `negative_control_material` references appear solely as non-driving context that must not satisfy lineage.
- Lineage families exercised by non-abstention queries: `gatemem_governance_status` 15, `retrieval_hygiene_associative_evidence_admission` 19, `unrelated_mnemos_documentation` 8.
- Abstention: all 12 abstention-expected queries are fully out-of-corpus (9 `out_of_corpus_current_state`, 3 `out_of_corpus_scope`) and carry an empty accepted evidence neighborhood, matching the abstention framing in the preregistration.
- Negative-control rejection is tested by 5 substitution-guard queries (`r1f-013`, `r1f-031`, `r1f-032`, `r1f-041`, `r1f-042`): each has a real answer from a `current_state_record` driver while a `negative_control_material` source is present as non-driving context, and every query sets `reject_if_only_negative_control: true`. Two of these (`r1f-031`, `r1f-032`) specifically guard the direct-runtime development diagnostic (`benchmarks/results/evidence_admission_r0_development_direct_runtime_run_001.json`), which is never admitted as a driver.
- 4 queries are compound two-driver cross-lineage lookups (`r1f-005`, `r1f-027`, `r1f-031`, `r1f-038`).
- 6 supersession-disambiguation queries pair a `current_state_record` driver with superseded context and set `may_include_superseded_context: true`; the current-state record must drive the answer.
- Schema validation against the template's `query_schema`, `accepted_evidence_neighborhood_schema`, and `lineage_requirement_schema` required fields passed with zero errors; every `allowed_lower_cost_route_labels` value is drawn from the template's `allowed_enforced_route_labels` set and none from `forbidden_enforced_route_labels`. Each accepted-evidence `source_path` exists in the frozen manifest and its declared `allowed_roles` matches the manifest role for that source.

## Handback

The pack is frozen and hashed above. Per the preregistration and template, the
implementation team (AI Dev) may now run R1 enforcement evaluation
against the fixed corpus, this preregistration, and this scored exam, but may
not edit the scored queries or tune R1 policy/thresholds/implementation in
response to them. Any need to change the pack requires refreezing and
invalidates prior claim evidence per the corpus manifest's rules.
