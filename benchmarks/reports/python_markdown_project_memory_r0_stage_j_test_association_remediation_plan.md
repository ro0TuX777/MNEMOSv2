# Stage J Objective: Test Association Remediation R0

Purpose: address the remaining packet-critical blocker identified in Stage I: `test-association-01` fails in `L`, `S`, `H`, and `H+E`.

Query under analysis:

- `test-association-01`
- text: `tests for LocalShadowAdapter replay handling`

## 1. Expected artifact

Expected sealed artifact:

- artifact_id: `sha256:13e170272bb2f8f3491ec30a96d7e0e0a4e47223500cb27cab0c0770e29cb57d`
- qualified name: `tests.test_session_context_assembler_shadow_adapter.test_replay_policy_drift_fails_closed`
- file path: `tests/test_session_context_assembler_shadow_adapter.py`
- artifact_type: `python_symbol`
- symbol_kind: `test_function`

This artifact exists in the active sealed snapshot and is not stale-rejected.

## 2. Sealed metadata on the expected artifact

Observed sealed evidence on the expected replay test artifact:

- test function name: `test_replay_policy_drift_fails_closed`
- file path: `tests/test_session_context_assembler_shadow_adapter.py`
- artifact_type: `python_symbol`
- symbol_kind: `test_function`
- parent_symbol: `None`
- test_marker: `True`
- imports include `prototype.session_context_assembler.shadow_adapter.adapter`
- imports include `LocalShadowAdapter` via imported names
- explicit outgoing relationships:
  - `contains` -> parametrize decorator artifact
  - `exact_test_reference` -> `prototype.session_context_assembler.shadow_adapter.adapter`
  - `exact_test_reference` -> `_run`
  - `exact_test_reference` -> `_policy`
  - `exact_test_reference` -> `_case`
  - `exact_test_reference` -> `r2_cases`
  - `imports_name` -> adapter module
  - `imports_name` -> `validate_response_contract`
- incoming explicit relationship:
  - `decorated_by` from its parametrize decorator artifact

Observed content evidence:

- contains `replay`
- does not contain literal `LocalShadowAdapter`
- does not contain literal `process`
- does not contain literal `build_response`

Key implication: the expected artifact is connected to the adapter slice by sealed explicit relationships, but its own local text is mostly about `replay_policy_drift`, not about the user-facing phrase `LocalShadowAdapter replay handling`.

## 3. Failure cause analysis

The failure is not caused by absence from corpus.

It is also not caused by stale validation, leakage filtering, or mutation/execution boundaries.

### Retrieval evidence

In Stage H2:

- `L`: expected artifact ranked 6, just below top_k=5
- `H`: expected artifact ranked 6, just below top_k=5
- `H+E`: expected artifact ranked 11 because expansion spent early slots on artifacts reachable from higher-ranked but broader replay seeds
- `S`: expected artifact was not competitive in top_k

Top lexical/hybrid winners were broader replay-related artifacts such as:

- `tests.test_session_context_assembler_selector_s1.test_s1_replay_is_deterministic_and_has_four_conditions`
- `tests.test_session_context_assembler_selector_s1`
- `tests.test_session_context_assembler_shadow_adapter.test_happy_path_is_shadow_only_and_artifact_local`
- `tests.test_session_context_assembler_shadow_adapter`
- `tests.test_session_context_assembler_shadow_adapter.test_redaction_payload_drift_changes_replay_fingerprint`

This shows the remaining blocker is primarily ranking/discoverability, not ingestion.

### Cause classification

Most likely causes:

1. weak test-name/token matching
   - expected test name contains `replay`, but not `LocalShadowAdapter` or `handling`
   - competing replay tests receive similar or stronger lexical weight

2. weak file-path matching
   - the expected file path is a strong shadow-adapter test file signal, but that file-level evidence is not enough to pull the specific replay test into top_k

3. weak relationship directionality
   - the expected test already has explicit sealed links to adapter artifacts
   - current retrieval does not sufficiently reward those explicit links when the query is phrased from production behavior toward tests

4. weak semantic retrieval
   - semantic-only does not appear packet-critical here; it behaves as a conceptual lane and does not recover the expected replay test

Not supported by the evidence:

- missing explicit test-reference relationship
  - explicit sealed relationships already exist

- overly strict oracle
  - the expected artifact is discoverable from sealed evidence and does belong to this query

## 4. Smallest bounded remediation

The smallest bounded remediation is:

1. keep the existing one-hop-only sealed relationship boundary
2. keep no inferred caller/callee behavior
3. keep no GraphRAG, semantic-neighborhood, or multi-hop traversal
4. improve test-association discoverability inside the isolated R0 prototype by:
   - strengthening test artifact identifier matching over:
     - test function name
     - test file path
     - qualified test symbol
   - adding explicit reverse discoverability only for already sealed explicit test-reference relationships
   - optionally adding a narrow test-association query classifier inside isolated R0 retrieval so queries beginning with `tests for ...` prioritize test artifacts and explicit sealed test-reference edges

This is narrower than general retrieval remediation because it only targets test-association queries and only uses sealed metadata/relationships already present in the corpus.

## 5. Proposed remediation shape

Recommended bounded implementation target:

- when a query is classified as test-association:
  - prioritize test artifacts over modules/documents when they share the same strong identifier/path evidence
  - boost test artifacts that have explicit sealed references to adapter/module/function artifacts matched by the query
  - allow reverse discovery from matched production artifact -> explicitly linked test artifact, but only one hop and only for sealed `exact_test_reference`-style edges
  - do not infer unstated test ownership
  - do not traverse beyond one hop

## 6. Preservation checks

Stage J remediation planning must preserve:

- no inferred caller/callee behavior
- no repository source execution/import/eval/compile/runtime-load/dynamic introspection
- no GraphRAG behavior
- no semantic-neighborhood traversal
- no multi-hop expansion
- zero leakage gates
- stale-evidence gates
- no runtime/default retrieval changes
- no memory/governance mutation

## 7. Recommendation

IMPLEMENT_TEST_ASSOCIATION_REMEDIATION

Closeout labels:

- PYTHON_MARKDOWN_PROJECT_MEMORY_R0_STAGE_J_PLAN_COMPLETE
- TEST_ASSOCIATION_BLOCKER_ANALYZED
- NO_CONTEXT_PACKET_AUTHORIZED
- NO_RUNTIME_BEHAVIOR_CHANGE
- NO_CODE_EXECUTION
- NO_CROSS_COLLECTION_LEAKAGE
