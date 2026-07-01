# Evidence Admission and Budgeting R1 Development Checkpoint

## Status

R1 development enforcement wiring and regression coverage are present for bounded, opt-in enforcement.

State labels:

- `R1_DEVELOPMENT_ENFORCEMENT_IMPLEMENTED`
- `R1_REGRESSION_COVERAGE_PRESENT`
- `DEVELOPMENT_ONLY_EVIDENCE`
- `INDEPENDENT_FORMAL_PACK_STILL_REQUIRED`
- `FORMAL_R1_EVALUATION_NOT_RUN`
- `NO_R1_POSITIVE_CLAIM`

This is not an R1 closeout. It is development-pack-level evidence only and must not be used as proof of retrieval-quality improvement, cost reduction, retention, or safety on unseen corpora.

## Implemented Boundary

R1 enforcement is additive to R0 and remains behind the two-part opt-in:

- global kill switch: `MNEMOS_EVIDENCE_ADMISSION_R1_ENFORCEMENT_ENABLED=true`
- per-request flag: `evidence_admission_enforce=true`

When either gate is absent, false, malformed, or unsupported, retrieval behavior remains on the R0/default path. R1 never enforces forbidden route labels.

Allowed enforced route labels:

- `CUE_ONLY_LOOKUP`
- `CACHE_ONLY`
- `BOUNDED_SEMANTIC_RETRIEVAL`
- `ABSTAIN_OR_REQUEST_SCOPE`
- `NORMAL_RETRIEVAL_FALLBACK`

Forbidden labels remain non-enforceable in R1:

- `HYBRID_RETRIEVAL`
- `ASSOCIATIVE_EXPANSION_ELIGIBLE`
- `graph_hybrid_experimental`
- `derived_facts`
- `summary_inclusion`
- `governance_override`

## Freeze Boundary

This checkpoint should be committed as the policy-frozen development baseline before the formal-pack template is handed to an independent author.

After that handoff, enforcement logic, thresholds, fallback rules, policy fixtures, development fixtures, and route-label mappings must not be altered in response to formal-pack content. Any required change after formal-pack authorship invalidates the affected formal evidence unless the corpus, formal pack, and preregistration are refrozen together.

## Regression Coverage

Dedicated tests are present in `tests/test_evidence_admission_and_budgeting_r1.py`.

Covered behaviors:

- pure enforcement decision mapping for allowed, fallback, unavailable, abstention, and forbidden R0 recommendation routes;
- kill switch off/on permutations at the service layer;
- request flag enabled while the global gate is disabled;
- global gate enabled while the request flag is absent;
- malformed global gate values;
- bounded `CUE_ONLY_LOOKUP` and `BOUNDED_SEMANTIC_RETRIEVAL` retrieval overrides;
- insufficient bounded attempts forcing `NORMAL_RETRIEVAL_FALLBACK`;
- final served results coming from fallback when fallback is triggered;
- forbidden route labels never appearing as enforced retrieval modes;
- no router write/index path called by R1 enforcement.

## Verification

Targeted R1 regression command:

```powershell
pytest tests/test_evidence_admission_and_budgeting_r1.py -q
```

Observed result:

```text
27 passed, 15 warnings in 0.99s
```

Repo-local suite command:

```powershell
pytest -q tests
```

Observed result:

```text
14 failed, 1288 passed, 1 skipped, 952 warnings in 61.55s
```

The 14 observed failures are non-R1 suite failures. They are excluded from the scoped R1 result pending separate baseline attribution from the same branch and environment:

- `tests/test_gatemem_g5_readiness.py::test_packet_contains_no_sealed_corpus_or_labels`
- `tests/test_governance_explainability.py::test_search_explain_governance_includes_trace_and_suppressed_summary`
- `tests/test_hierarchy_lineage.py::test_summary_edges_resolve_to_leaf_engrams_via_api`
- `tests/test_hierarchy_lineage.py::test_root_summary_recursive_lineage`
- `tests/test_hybrid_runtime_response.py::test_runtime_hybrid_explain_fields_are_typed_and_present`
- `tests/test_hybrid_runtime_response.py::test_runtime_hybrid_explain_false_omits_explain_fields`
- `tests/test_hybrid_runtime_response.py::test_runtime_exposes_complexity_and_routing_metadata`
- `tests/test_hybrid_runtime_response.py::test_runtime_phase2_candidate_envelope_meta_exposed_when_enabled`
- `tests/test_hybrid_runtime_response.py::test_runtime_phase3_derived_views_generated_when_enabled`
- `tests/test_hybrid_runtime_response.py::test_runtime_phase4_derived_view_cache_hit_and_invalidate`
- `tests/test_memory_over_maps_phase1.py::test_search_explain_includes_lineage_when_phase1_enabled`
- `tests/test_memory_over_maps_phase1.py::test_search_explain_omits_lineage_when_phase1_disabled`
- `tests/test_retrieval_router_telemetry.py::test_telemetry_emission_and_health`
- `tests/test_vfr7_api.py::test_gate_5_audit_log_integrity`

An unscoped `pytest -q` also collects nested `turbovec/turbovec-python` tests and stops during collection because the local environment does not expose the expected `turbovec` Python package symbols. Use `pytest -q tests` for the MNEMOS repo-local suite baseline.

## Claim Boundary

This checkpoint supports only the following development claim:

R1 bounded enforcement wiring and regression tests are present, opt-in gated, additive to R0, and locally verified against the development test suite shape above.

It does not support a formal R1 positive claim. A formal claim still requires a live corpus run against the frozen independently authored formal pack, with normal retrieval baseline, R0 shadow, R1 enforcement enabled, and R1 global gate disabled results kept separate under the preregistered protocol.

## Required Follow-Up

1. Commit the R1 enforcement implementation, R1 tests, this checkpoint, and the pre-formal baseline manifest together as the policy-frozen development baseline.
2. Record the 14-failure suite baseline separately with commit, interpreter, OS, command, failure IDs, and timestamp.
3. Hand the committed corpus manifest and formal-pack template to the independent author.
4. Do not inspect or tune against the scored pack until it is returned, validated, frozen, and hashed.
5. Run formal R1 evaluation with normal retrieval baseline, R0 shadow, R1 enforcement enabled, and R1 global gate disabled conditions kept separate.
6. Require fresh independent verification before any positive R1 retention decision.
