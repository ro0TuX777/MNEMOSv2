# 04 Control-to-Evidence Traceability Matrix

| `control_id` | `control_name` | `control_objective` | `track_source` | `enforcing_mechanism` | `evidence_artifact_id` | `evidence_location` | `owner` | `cadence` | `status` | `last_verified_at` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| VFR-CONTROL-001 | Sidecar-only Fact Eval | Ensure Sidecar isolation | VFR-10 | Governed Sidecar Read Path | ART-VFR-10 | docs/reports/vfr_10 | VFR Lead | Continuous | ACTIVE | 2026-06-07 |
| PIT-CONTROL-001 | Variant B Sideband UI | Ensure UI aids are isolated | PIT-5 | Sideband Isolation Logic | ART-PIT-5 | docs/reports/pit_5 | PIT Lead | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-001 | Unified Gov Ledger | Cryptographic Non-repudiation | GOV-1 | Ledger Append-only Log | ART-GOV-1 | docs/reports/gov_1 | Gov Admin | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-002 | Gov Review Console | Governed manual review | GOV-2 | Role-Based Access | ART-GOV-2 | docs/reports/gov_2 | Gov Admin | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-003 | Evidence Bundle Gen | Bundle without raw payload | GOV-4 | Payload Scrubber | ART-GOV-4 | docs/reports/gov_4 | Sec Auditor | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-004 | Break-glass Routing | Secure redaction routing | GOV-3 | Break-glass Workflow | ART-GOV-3 | docs/reports/gov_3 | DPO | On-Demand | ACTIVE | 2026-06-07 |
| PIT-CONTROL-002 | Triple Opt-In | Validate opt-in checks | PIT-0 | Metadata Validation | ART-PIT-0 | docs/reports/pit_0 | Privacy | Continuous | ACTIVE | 2026-06-07 |
| VFR-CONTROL-002 | Marked sidecar exports | Reject prod ingestion | VFR-10 | `production_ingestion_allowed=false` | ART-VFR-10 | docs/reports/vfr_10 | Sec Auditor | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-005 | Raw Payload Ext (Cond Auth) | Dual-control extraction | GOV-3 | Dual-signature approval | ART-GOV-3 | docs/reports/gov_3 | DPO/GovAdmin | On-Demand | ACTIVE | 2026-06-07 |
| PIT-CONTROL-003 | Block Default Retrieval | Prevent prod integration | PIT-0 | Retrieval Gateway Reject | ART-PIT-0 | docs/reports/pit_0 | PIT Lead | Continuous | ACTIVE | 2026-06-07 |
| VFR-CONTROL-003 | Block Graph Traversal | Prevent `graph_hybrid_experimental` | VFR-10 | Sidecar Read-only Path | ART-VFR-10 | docs/reports/vfr_10 | VFR Lead | Continuous | ACTIVE | 2026-06-07 |
| PIT-CONTROL-004 | Block Prod EchoFrame | Prevent derived inclusion | PIT-0 | Schema Validation | ART-PIT-0 | docs/reports/pit_0 | PIT Lead | Continuous | ACTIVE | 2026-06-07 |
| PIT-CONTROL-005 | Block Derived Embeddings | Prevent indexing | PIT-0 | Indexing Rejection Rule | ART-PIT-0 | docs/reports/pit_0 | PIT Lead | Continuous | ACTIVE | 2026-06-07 |
| PIT-CONTROL-006 | Block Metadata Ranking | Prevent ranking | PIT-5 | Ranker Isolation | ART-PIT-5 | docs/reports/pit_5 | PIT Lead | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-006 | Block Auto-Promotion | Enforce human-in-loop | GOV-1 | Ledger Validation | ART-GOV-1 | docs/reports/gov_1 | Gov Admin | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-007 | Block Auto-Contradiction | Enforce human-in-loop | GOV-1 | Ledger Validation | ART-GOV-1 | docs/reports/gov_1 | Gov Admin | Continuous | ACTIVE | 2026-06-07 |
| VFR-CONTROL-004 | Block Sidecar Prod Ingest | Enforce marked export | VFR-10 | Ingestion Gateway Reject | ART-VFR-10 | docs/reports/vfr_10 | Sec Auditor | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-008 | Block Raw Rendering | Scrub gov console view | GOV-2 | UI Scrubber | ART-GOV-2 | docs/reports/gov_2 | Gov Admin | Continuous | ACTIVE | 2026-06-07 |
| GOV-CONTROL-009 | Block Silent Deletion | Enforce non-repudiation | GOV-1 | Ledger Append-only WORM | ART-GOV-1 | docs/reports/gov_1 | Sec Auditor | Continuous | ACTIVE | 2026-06-07 |
