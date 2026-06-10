# DFE-21: Controlled Human Operator Value Trial for Quarantined Derived Facts

## Decision Record
**Previous Phase (DFE-20):** `ACCEPT_DFE_20_TECHNICAL_CLOSEOUT`
**Context:** DFE-20 validates the Option B feature-flagged API architecture for controlled operator exposure of Derived Facts. It does not yet validate operator usefulness, retrieval quality improvement, or production promotion.

## Goal
Determine whether quarantined Derived Facts improve human document understanding, triage, and synthesis while preserving source fidelity and production retrieval isolation.

## Objective
Evaluate whether allowlisted human operators gain measurable value from Derived Facts when answering document-grounded questions, without weakening source fidelity, evidence discipline, or default retrieval safety.

## Scope & Constraints
- Use allowlisted operators only.
- Use `enable_derived_facts=true`.
- Keep `config.derived_enabled` kill-switch active.
- Keep default retrieval invariant active.
- Keep Derived Facts restricted to `derived_lane_meta`.
- Run against 5–10 curated document sets (mix of technical PDFs, policy docs, RFP material, architecture reports, benchmark reports, and documents with ambiguous/overlapping claims).
- Capture human ratings and evidence-quality outcomes.
- **DO NOT promote Derived Facts to production evidence.**

## Trial Mechanics
Use three modes for testing:
* **Mode A:** Default retrieval only (No Derived Facts).
* **Mode B:** Derived Facts visible in `derived_lane_meta` (clearly separated).
* **Mode C:** Blind comparison (Operators compare two answers without knowing which used Derived Facts).

## Required Metrics
| Metric | How to measure |
|--------|----------------|
| Time to useful answer | Did Derived Facts reduce review time? |
| Evidence correctness | Did the answer cite the right source? |
| Unsupported-claim rate | Did Derived Facts introduce claims not supported by source excerpts? |
| Operator trust rating | Did the human trust the result? |
| Operator usefulness rating | Did it help them think or decide? |
| Escalation count | How often did Derived Facts require manual inspection? |
| Confusion rate | Did users mistake Derived Facts for source truth? |

## Exit Criteria
**The trial will PASS if:**
- At least 20 evaluated operator tasks are completed.
- Operators find Derived Facts useful (usefulness score meets agreed threshold).
- Source fidelity remains intact; evidence correctness remains equal to or better than default retrieval.
- Derived Facts improve synthesis or triage.
- Humans understand the quarantine boundary.
- 0 production contamination events occur.
- 0 unsupported Derived Facts are accepted as source truth.
- Kill-switch and allowlist remain effective during the trial.

**The trial will FAIL if:**
- Operators confuse Derived Facts with source facts.
- Derived Facts appear outside `derived_lane_meta`.
- Unsupported Derived Facts survive filtering.
- The experimental lane increases hallucinated confidence.
- Users rely on Derived Facts without checking excerpts.
- Default retrieval begins surfacing Derived Facts.
- Source citations become weaker.
