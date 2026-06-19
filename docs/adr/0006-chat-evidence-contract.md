# ADR 0006: Chat Integrations Use The Evidence Contract For Citations

Date: 2026-06-20

Status: Accepted

## Context

Downstream chat and RAG systems can blur the boundary between retrieved MNEMOS
evidence and model prior knowledge. Without a formal evidence contract, systems
may invent citations, lose chunk metadata, or claim grounding when retrieval
was empty or weak.

## Decision

Chat integrations must use `results[].evidence` and `meta.evidence_summary` as
the citation authority for MNEMOS-grounded answers. Integrators should preserve
rank, score, chunk, source, and document metadata through adapters.

## Alternatives Considered

- Let downstream models generate citation text from prompt context.
- Expose only source filenames without per-result evidence.
- Treat audit records as answer citations.

## Invariants

- Citations must come from MNEMOS evidence fields, not model invention.
- Null page or span fields mean unknown, not zero.
- Empty or low-confidence retrieval must not be described as grounded.
- Adapter layers must not flatten away rank, score, source, or chunk metadata.

## Rollback

If an integration cannot preserve evidence metadata, it must not claim MNEMOS
grounding until the adapter is fixed and validated.

## Evidence

- `docs/chat_integration_evidence_contract.md`
- `tests/test_service_hybrid_api.py`

