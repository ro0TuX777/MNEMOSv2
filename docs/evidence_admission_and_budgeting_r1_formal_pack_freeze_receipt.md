# Evidence Admission and Budgeting R1 Formal Pack — Freeze Receipt

## Status

FROZEN. This receipt closes the independent-authorship step required by the R1
preregistration and formal-pack template before any enforcement implementation
work may begin.

## Artifact

- Path: `docs/experiments/evidence_admission_and_budgeting_r1_formal_pack.json`
- `pack_id`: `evidence_admission_and_budgeting_r1_formal_pack_v1`
- `derived_from_template`: `evidence_admission_and_budgeting_r1_formal_pack_template_v1`
- `corpus_manifest_id`: `evidence_admission_and_budgeting_r1_corpus_69ad546ba30ed71a`
- Scored query count: 55 (minimum required: 50)
- Byte count: 66964
- SHA-256: `338999ce626d7035935155f488b99b97deafd5334d76ec3773ce1cde66230554`
- Authored against repository commit: `e7b86c65d3fd0298f915efc00cc8cf6fd7fb7f7e` (branch `codex/gatemem-governance-reference-baseline`)
- Freeze date: 2026-07-01

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
exam.

## Confirmation of authoring inputs

The pack was authored only from:

- the committed, frozen corpus manifest (`docs/evidence_admission_and_budgeting_r1_corpus_manifest.json`), which enumerates the 41 frozen sources with their family/role/hash metadata;
- the formal-pack template (`docs/experiments/evidence_admission_and_budgeting_r1_formal_pack_template.json`), which fixed the required schema, allowed/forbidden route labels, and non-scored schema examples;
- the R1 preregistration (`docs/evidence_admission_and_budgeting_r1_preregistration.md`);
- the R1 design note (`docs/evidence_admission_and_budgeting_r1_design_note.md`);
- the actual text of the 41 frozen corpus source documents themselves (read via their manifest-declared paths), which was necessary to write real, content-grounded queries rather than filename-only guesses. These are corpus content, not R1 implementation code and not R1 enforcement behavior.

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

- All 41 corpus sources are accounted for: 33 appear as accepted evidence in at least one scored query (roles: `current_state_record` 29 refs, `superseded_record` 7 refs, `dependency_blocker_record` 4 refs, `duplicate_or_near_duplicate_condition` 5 refs across 41 non-abstention queries), and the remaining 8 (`negative_control_material`) are deliberately excluded from any accepted evidence neighborhood and instead drive abstention-expected queries that test rejection of negative-control-only evidence.
- 16 of 55 queries are `abstention_expected: true` (7 negative-control-driven, 9 fully out-of-corpus), matching the abstention/negative-control zero-tolerance framing in the preregistration.
- 1 query is a negative-control substitution guard: it has a real answer (via the two HTTP-service current-state records) and specifically tests that the direct-runtime diagnostic negative control cannot substitute for it.
- 5 queries are compound (two-source) cross-lineage lookups.
- Schema validation against the template's `query_schema`, `accepted_evidence_neighborhood_schema`, and `lineage_requirement_schema` required fields passed with zero errors; all `allowed_lower_cost_route_labels` values are drawn from the template's `allowed_enforced_route_labels` set and none from `forbidden_enforced_route_labels`.

## Handback

The pack is frozen and hashed above. Per the preregistration and template, the
implementation team (AI Dev) may now begin R1 enforcement implementation
against the fixed corpus, this preregistration, and this scored exam, but may
not edit the scored queries or tune R1 policy/thresholds/implementation in
response to them. Any need to change the pack requires refreezing and
invalidates prior claim evidence per the corpus manifest's rules.
