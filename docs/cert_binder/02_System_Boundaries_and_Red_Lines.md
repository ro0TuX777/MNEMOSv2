# 02 System Boundaries and Red Lines

This document strictly defines the operational bounds of the MNEMOS system.

## AUTHORIZED
- Sidecar-only Fact-Aware Evaluation Mode
- Variant B metadata sideband UI aid for authorized roles
- Unified Governance Ledger
- Governance Review Console
- Evidence Bundle generation without raw payloads
- Break-glass redaction request routing only
- Triple Opt-In Framework
- Marked non-production sidecar evaluation exports (with `production_ingestion_allowed=false`)

## CONDITIONALLY AUTHORIZED
- Raw payload extraction only through separate redaction/export process with dual control, legal basis, Data Privacy Officer approval, Governance Admin approval, Security Auditor awareness/attestation, and sanitized ledger receipt.

## BLOCKED (Production Red Lines)
- Default retrieval integration
- `graph_hybrid_experimental` FactNode traversal
- Production EchoFrame inclusion of derived material
- Derived text embedding/indexing/reranking
- Metadata-based ranking
- Automatic promotion
- Automatic contradiction resolution
- Production ingestion of sidecar exports
- Raw payload rendering/storage in Governance Console or Governance Ledger
- Silent ledger deletion
