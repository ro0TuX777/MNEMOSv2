# GateMem G0 Capability Matrix

Date: 2026-06-24

Allowed labels: `SUPPORTED_NOW`, `SHADOW_TESTABLE`, `PARTIAL`, `UNSUPPORTED`,
and `OUT_OF_SCOPE`.

| GateMem dimension | Classification | Evidence and boundary |
|---|---|---|
| Authorized utility | `PARTIAL` | MNEMOS supports evidence ingestion, governed retrieval, provenance, and citations. It does not currently turn an authenticated GateMem checkpoint into a normalized answer action. |
| Multi-principal scope enforcement | `SHADOW_TESTABLE` | The isolated session-context adapter validates consumer, tenant, session, artifact, source, and Engram scope against injected immutable policy snapshots. The production API has no first-class principal policy engine. |
| Cross-tenant isolation | `SHADOW_TESTABLE` | Tenant mismatch fails closed in the isolated adapter. Runtime search accepts metadata filters but does not derive or enforce tenant scope from authenticated identity. |
| Cross-session isolation | `SHADOW_TESTABLE` | Session mismatch fails closed in the isolated adapter. Runtime `session_id` is primarily correlation/filter metadata rather than an authenticated isolation boundary. |
| Role enforcement | `UNSUPPORTED` | The service bearer token is global and no runtime GateMem-style actor-role authorization decision exists. |
| Entitlement/disclosure enforcement | `SHADOW_TESTABLE` | Deny-by-default source/Engram disclosure and artifact-class tests pass only in the isolated adapter; policy snapshots are injected and not connected to a live authorization service. |
| Redaction correctness | `SHADOW_TESTABLE` | Pre-assembly redaction and replay-fingerprint drift are tested in the isolated adapter only. |
| Provenance | `SUPPORTED_NOW` | The supported evidence contract, Engram source metadata, parent lineage, and non-authoritative synthetic labels provide traceable evidence. |
| Auditability | `SUPPORTED_NOW` | MNEMOS has supported forensic ledgers and content-bounded adapter telemetry. This does not by itself prove access-control or deletion correctness. |
| Explicit deletion request interpretation | `UNSUPPORTED` | MNEMOS does not interpret a conversational deletion operation, resolve its target set, or authorize the requester under GateMem semantics. |
| Per-ID backend deletion | `PARTIAL` | `DELETE /v1/mnemos/engrams/{id}` calls each configured tier and backend delete methods have unit coverage. Failures are reduced to zero counts and the route still returns HTTP 200. |
| Tombstone/revocation lifecycle | `PARTIAL` | `GovernanceMeta.deletion_state` and relevance vetoes model `soft_deleted`/`tombstone`, but no supported mutation workflow connects an explicit deletion request to those states. |
| Delete cascade through lineage | `UNSUPPORTED` | Summary, Resolution, extracted, derived, graph, cache, and other descendants are not atomically discovered and removed or rederived by the delete endpoint. |
| Post-deletion non-recoverability | `UNSUPPORTED` | No cross-tier negative verification, durable tombstone, cache purge proof, backup policy, or descendant sweep establishes non-recoverability. |
| Reconstruction/confirmation resistance | `UNSUPPORTED` | MNEMOS does not currently prevent an answer layer from confirming or reconstructing deleted facts from surviving evidence, derived artifacts, conversation context, or model knowledge. |
| Over-refusal risk | `PARTIAL` | GateMem can score normalized predictions, but MNEMOS has no current answer/action adapter whose refusals can be measured. |
| GateMem output compatibility | `PARTIAL` | External `predictions.jsonl` requires only checkpoint ID, action, answer, structured answer, and used record IDs. A clean MNEMOS projection/normalizer is not yet implemented. |
| Hosted LLM judge | `OUT_OF_SCOPE` | Explicitly prohibited in G0. |
| Public leaderboard submission | `OUT_OF_SCOPE` | Explicitly prohibited in G0. |

## Concept mapping

| GateMem concept | Honest MNEMOS mapping today |
|---|---|
| Principal / role | No equivalent production authorization principal. Candidate representations are Engram metadata, request context outside MNEMOS, or isolated adapter consumer/policy snapshots. |
| Memory ingestion turn | A turn may become one or more source-linked Engrams through the index boundary. GateMem `memory_ops` are not interpreted by current MNEMOS ingestion. |
| Authorized checkpoint query | MNEMOS search plus governance/evidence output, followed by an external authorization and answer layer that does not yet exist. |
| Access-control request | Isolated scope/disclosure/redaction policy evaluation is shadow-testable; runtime filters and a global bearer token are not equivalent. |
| Deletion request | A target-resolved per-ID physical delete is partially available. Governed deletion/tombstone/cascade semantics are unsupported. |
| Leak target | Prohibited source, Engram, selected artifact, derived artifact, or answer content. Leak targets belong only to the evaluator. |
| Output action | A future external normalizer must produce `answer`, `answer_redacted`, `refuse`, or `no_memory`; current MNEMOS search does not produce these actions. |

