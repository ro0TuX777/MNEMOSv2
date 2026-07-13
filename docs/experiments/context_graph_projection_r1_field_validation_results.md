# Context Graph Projection R1 Field Validation Results

Date: 2026-07-13

Status: **Research-only validation result. No implementation authorized.**

## Executive Summary

This validation inspected representative existing MNEMOS artifacts to determine
whether the minimal evidence-to-decision trace path can be projected from
explicit fields:

```text
source_artifact
  -> source_engram
  -> retrieval_result_set
  -> evaluation_result or decision record
  -> handoff_package or context_package
```

Recommendation:

```text
NARROW_SCOPE
```

The field evidence is promising but not sufficient for a full `GO`. Existing
artifacts show explicit Engram IDs, source paths, selected parent Engram IDs,
selected source IDs, receipt IDs, retrieval fingerprints, content hashes,
package lineage flags, and digest validation results. However, the inspected
artifacts do not yet prove a uniform durable retrieval-result-set identity,
uniform evidence refs across cognitive-cycle and evaluation records, uniform
handoff package refs, or direct forensic ledger-event correlation.

The future projection should therefore remain conditional and narrow:

```text
GO only for artifact families that expose explicit refs.
NO_GO for any path segment requiring inference, semantic guessing, or new
authority semantics.
```

## Boundary

This result does not authorize:

- graph code
- graph storage
- graph database adoption
- GraphRAG
- retrieval changes
- governance changes
- promotion changes
- context assembly changes
- Engram schema changes
- authority changes

## Artifacts Inspected

Representative samples inspected:

| Artifact family | Sample artifact path or source |
|---|---|
| Evidence receipts / retained search responses | `logs/evidence_receipts/chatcmpl-mnemos-0719a54feb5347f6aef9db992c4874da.json`; `logs/evidence_receipts/chatcmpl-mnemos-2c02345faa2c4dd7b0363e65ad77e7d9.json` |
| Evaluation artifacts | `benchmarks/results/evidence_admission_r1_formal_http_service_run_001.json`; `docs/experiments/evidence_admission_and_budgeting_r1_development_pack.json` |
| Session Context Assembler packages and results | `benchmarks/results/session_context_assembler_r1_replay.json`; `benchmarks/results/session_context_assembler_r2_verification.json`; `benchmarks/results/session_context_assembler_shadow_adapter_gate.json` |
| Review / package artifacts | `benchmarks/review_packets/session_context_assembler_phase_5/packets/task-001.json` |
| Handoff artifacts | `benchmarks/evaluation/gatemem_g5_handoff_state.json`; `benchmarks/evaluation/ai_dev_memory_quality_e2_task_01_starter_repo/docs/handoff_notes.md` |
| Cognitive-cycle evidence | `docs/reports/coala_cycle_operational_validation.md` |
| Digest / verification artifacts | `benchmarks/results/session_context_assembler_shadow_adapter_gate.json`; `benchmarks/results/session_context_assembler_shadow_adapter_gate.md`; `benchmarks/results/session_context_assembler_r2_verification.json` |

## Validation Results

| Check | Artifact family inspected | Sample artifact path or source | Required fields | Observed fields | Explicit refs? | Inference needed? | Disclosure concern | Result | Projection impact |
|---|---|---|---|---|---|---|---|---|---|
| FV-001 Source artifact key normalization | Evidence receipts, evaluation results, Session Context Assembler results | `logs/evidence_receipts/chatcmpl-mnemos-0719a54feb5347f6aef9db992c4874da.json`; `benchmarks/results/evidence_admission_r1_formal_http_service_run_001.json`; `benchmarks/results/session_context_assembler_r2_verification.json` | Stable source ID or deterministic source key; source URI/path/label; Engram refs | `source`, `source_path`, `selected_source_ids`, Engram IDs embedded in receipt citations and benchmark top results | Yes, but not uniformly | No for sampled refs; yes if source labels must be normalized across lanes | Source paths may reveal sensitive document identity | PARTIAL | Source-to-Engram projection is viable only where source IDs or deterministic source paths are present. |
| FV-002 Retrieval result-set identity | Evidence receipts, formal HTTP evaluation result | `logs/evidence_receipts/chatcmpl-mnemos-0719a54feb5347f6aef9db992c4874da.json`; `benchmarks/results/evidence_admission_r1_formal_http_service_run_001.json` | Request/receipt/query ID; returned Engram IDs; rank/score; timestamp; retrieval metadata | `receipt_id`, `created`, `retrieval_fingerprint`, `result_count`, `citations[].engram_id`, `citations[].score`, query IDs, condition names, top result entries with `engram_id`, `source_path`, rank, score | Yes in richer receipts and formal result entries | No for sampled richer receipts; possible yes for receipts lacking `engram_id` | Retrieval metadata may expose query or source relationship | PARTIAL | Projection can use receipt IDs or query/condition-local IDs, but durable result-set identity is not uniform enough for full GO. |
| FV-003 Explicit evidence refs in CognitiveCycleRecord outputs | Cognitive-cycle validation report | `docs/reports/coala_cycle_operational_validation.md` | Cycle ID; action/grounding records; evidence Engram refs; governance summaries; forensic ledger refs | Report states `CognitiveCycleRecord` validates redaction, evidence-derived attention, representative path coverage, `forensic_ledger_refs`, and advisory pattern candidate supporting IDs | Partially; report-level evidence, not sampled serialized cycle JSON | Yes if projecting from report prose rather than serialized records | Cycle records must not expose raw prompts, private reasoning, or raw Engram content | PARTIAL | Cognitive-cycle nodes remain plausible, but field validation needs actual serialized cycle records before use in the minimal path. |
| FV-004 Explicit evidence refs in evaluation records | Formal HTTP evaluation, development pack, agent navigation study | `benchmarks/results/evidence_admission_r1_formal_http_service_run_001.json`; `docs/experiments/evidence_admission_and_budgeting_r1_development_pack.json`; `benchmarks/results/mnemos_agent_navigation_trial_001_memory_assisted.json` | Evaluation ID; result status; claim/query ID; source Engram IDs or source artifact refs; accepted/rejected status | `pack_id`, `claim_status`, per-query IDs, condition scores, top result entries with `engram_id` and `source_path`, `memory_ids_retrieved`, `accepted_memory_ids`, `rejected_memory_ids`, `evidence_paths_used`, `boundary_decision` | Yes for sampled evaluation and navigation artifacts | No for sampled IDs; yes if interpreting prose-only evaluation claims | Evaluation artifacts may include query text and source paths | PARTIAL | Evaluation trace is viable for selected artifact families, but schema variance prevents full GO. |
| FV-005 Explicit artifact refs in handoff packages | Review packets, handoff state, handoff notes | `benchmarks/review_packets/session_context_assembler_phase_5/packets/task-001.json`; `benchmarks/evaluation/gatemem_g5_handoff_state.json`; `benchmarks/evaluation/ai_dev_memory_quality_e2_task_01_starter_repo/docs/handoff_notes.md` | Handoff ID/path; included Engram/source/context/evaluation refs; timestamp or scope | Review packet artifacts include `parent_engram_ids`, `parent_source_ids`, `non_authoritative`, `non_promotable`; G5 handoff state includes `packet_index`, `required_documents`, `candidate_nomination`; markdown handoff is prose-only | Yes for review packet and G5 document refs; no for prose-only handoff notes | Yes for markdown handoff notes | Handoff document lists can reveal project state and dependency relationships | PARTIAL | Handoff projection should be limited to structured handoff/review artifacts with explicit refs. |
| FV-006 Package digest availability | Session Context Assembler gate and verification outputs | `benchmarks/results/session_context_assembler_shadow_adapter_gate.json`; `benchmarks/results/session_context_assembler_shadow_adapter_gate.md`; `benchmarks/results/session_context_assembler_r2_verification.json` | Package ID/key; digest value; digest validity; parent Engram/source IDs; lineage completeness | `digest_valid`, `digest_verification_rate_1_0`, `artifact_local_lineage_complete`, `r1_file_sha256`, `r2_file_sha256`, `selected_parent_engram_ids`, `selected_source_ids`, lineage loss counts | Yes for validation flags and file hashes; digest value not always attached to each package record | No for gate status; yes if per-package digest must be reconstructed | Digest metadata may prove package existence even when content is redacted | PARTIAL | Digest-backed verification exists, but per-package digest fields need confirmation before full projection. |
| FV-007 Ledger-event correlation | Evidence receipts, cognitive-cycle report, architecture docs | `logs/evidence_receipts/chatcmpl-mnemos-0719a54feb5347f6aef9db992c4874da.json`; `docs/reports/coala_cycle_operational_validation.md` | Ledger event ID; operation type; target refs; timestamp; tenant/scope marker | Evidence receipts expose `receipt_id`, `created`, `content_hash`; CoALA report states `forensic_ledger_refs` exist in cycle records | Partially; receipt IDs and content hashes are explicit, but direct ledger event samples were not found | Yes if treating receipts as ledger events | Ledger refs can leak operation timing and evidence relationships | PARTIAL | `audited_by` should remain optional until direct forensic ledger event artifacts are sampled. |
| FV-008 Disclosure behavior for relationship edges | Session Context Assembler gate, GateMem disclosure references, review packets | `benchmarks/results/session_context_assembler_shadow_adapter_gate.md`; `benchmarks/results/session_context_assembler_shadow_adapter_gate.json`; `benchmarks/review_packets/session_context_assembler_phase_5/packets/task-001.json` | Endpoint authorization; tenant/scope; redaction labels; edge type; authority class; caller entitlement | Gate reports authorization bypass and redaction bypass detection; review packets include `non_authoritative`, `non_promotable`; SCA records include content-free telemetry and lineage checks | Partially; policy gates exist but not edge-specific disclosure rules | Yes for graph-specific edge visibility until rules are formalized | Relationship-only leakage remains the main unresolved risk | PARTIAL | Future projection must wait for edge visibility rules before any multi-user or sensitive export. |
| FV-009 `lineage_incomplete` labeling rules | Session Context Assembler replay and verification outputs | `benchmarks/results/session_context_assembler_r1_replay.json`; `benchmarks/results/session_context_assembler_r2_verification.json`; `benchmarks/results/session_context_assembler_shadow_adapter_gate.json` | Rules for omission vs labeling; missing-lineage counters; lineage completeness flags | `source_lineage_loss_count`, `decision_lineage_loss_count`, `provenance_loss_count`, `artifact_local_lineage_complete`, `missing_required_artifact_ids`, `silent_required_artifact_omission`, `selection_abstention_reason` | Yes for counters and flags | No for observed counters; yes if deriving per-edge labels without explicit rules | Missing-lineage labels can reveal omitted protected relationships | PARTIAL | The ingredients exist, but graph-specific omission and `lineage_incomplete` rules must be written before implementation. |

## Minimal Trace Assessment

| Trace segment | Result | Evidence | Projection impact |
|---|---|---|---|
| `source_artifact -> source_engram` | PARTIAL | Receipts and formal results expose `source`, `source_path`, and `engram_id`; SCA exposes `selected_source_ids` and `selected_parent_engram_ids`. | Viable where source keys are explicit; source normalization remains required. |
| `source_engram -> retrieval_result_set` | PARTIAL | Rich evidence receipts expose `receipt_id`, `retrieval_fingerprint`, `citations[].engram_id`, and scores; formal results expose query/condition top entries with `engram_id`. | Viable for retained receipts and benchmark outputs; durable result-set identity remains inconsistent. |
| `retrieval_result_set -> cognitive_cycle_record` | PARTIAL | CoALA report documents cycle records and ledger refs, but no sampled serialized cycle JSON was validated. | Needs actual cycle record artifacts before projection. |
| `retrieval_result_set -> evaluation_result` | PARTIAL | Evidence admission formal result connects query IDs, condition outputs, scores, top Engram IDs, source paths, and claim status. | Viable for selected evaluation families, but schema variance requires narrow scope. |
| `evaluation_result -> context_package` | PARTIAL | SCA outputs carry selected parent Engram/source IDs; evaluation artifacts carry Engram/source refs, but direct evaluation-to-package refs were not observed. | Must be projected only when explicit shared refs exist. |
| `cognitive_cycle_record -> context_package` | PARTIAL | SCA context packages expose selected parent/source refs; CoALA report describes cycle records, but direct shared refs were not sampled. | Needs serialized cycle artifacts. |
| `context_package -> handoff_package` | PARTIAL | Review packets include synthetic context artifacts with parent refs; handoff state includes required documents; markdown handoff is prose-only. | Restrict to structured handoff/review packets. |
| `any projected node -> ledger_event` | PARTIAL | Receipts have IDs, timestamps, content hashes; CoALA report says ledger refs exist. Direct forensic ledger event samples were not validated. | Keep audit links optional until direct ledger refs are sampled. |
| `any projected edge -> caller-visible graph` | PARTIAL | SCA gates cover redaction and authorization bypass, but graph-specific relationship-edge disclosure rules do not yet exist. | Blocks broad GO; requires disclosure model before implementation. |

## Go / No-Go Recommendation

```text
NARROW_SCOPE
```

### Why Not GO

Full `GO` is not justified because:

- durable retrieval-result-set identity is not uniform across sampled artifacts
- actual serialized `CognitiveCycleRecord` outputs were not validated
- evaluation artifacts expose useful refs but vary by lane and serialization
- handoff artifacts are mixed between structured refs and prose-only notes
- direct forensic ledger event samples were not validated
- graph-specific relationship disclosure rules are not yet defined
- `lineage_incomplete` counters exist, but graph edge omission rules are not yet formalized

### Why Not NO_GO

`NO_GO` would be too pessimistic because:

- retained evidence receipts can expose `receipt_id`, retrieval metadata,
  `engram_id`, source, score, and content hash
- formal evaluation results can expose query IDs, top retrieved Engram IDs,
  source paths, and claim status
- Session Context Assembler records expose selected parent Engram IDs, selected
  source IDs, lineage loss counts, and package verification signals
- structured review packets expose non-authoritative synthetic context labels
  with parent Engram/source refs

### Narrow Scope Allowed For Future Consideration

A future JSON-only projection may be considered only for artifact families with
explicit refs, such as:

- retained evidence receipts with `citations[].engram_id`
- formal benchmark/evaluation records with top entries containing `engram_id`
  and `source_path`
- Session Context Assembler records with `selected_parent_engram_ids` and
  `selected_source_ids`
- structured review packets with `parent_engram_ids` and `parent_source_ids`

The projection must omit or label all records that lack explicit refs.

## Required Before Any Implementation

Before code is authorized, MNEMOS must still define or verify:

- canonical source artifact key normalization
- durable or projection-local retrieval result-set identity rules
- actual serialized `CognitiveCycleRecord` field samples
- selected evaluation artifact families and accepted field contracts
- structured handoff artifact requirements
- direct forensic ledger event correlation
- relationship-edge disclosure rules
- graph-specific `lineage_incomplete` and omission rules

## Acceptance Statement

This validation result is acceptable only under the following interpretation:

```text
Existing MNEMOS artifacts contain enough explicit refs to justify continued
research toward a narrow JSON-only projection.

They do not yet justify implementation.

Future implementation remains blocked until the remaining field contracts and
disclosure rules are validated without inference, semantic guessing, or new
authority semantics.
```

## Closeout Labels

```text
CONTEXT_GRAPH_PROJECTION_R1_FIELD_VALIDATION_RESULTS_COMPLETE
RESEARCH_ONLY
NARROW_SCOPE_RECOMMENDED
NO_IMPLEMENTATION_AUTHORIZED
FIELD_LEVEL_VALIDATION_PARTIAL
EXPLICIT_REFS_REQUIRED
NO_GRAPH_STORAGE
NO_GRAPHRAG
NO_RETRIEVAL_GOVERNANCE_PROMOTION_CONTEXT_ASSEMBLY_ENGRAM_SCHEMA_OR_AUTHORITY_CHANGE
```
