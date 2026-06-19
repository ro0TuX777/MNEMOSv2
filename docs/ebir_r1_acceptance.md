# EBIR-R1 Technical Acceptance

Date: 2026-06-18

Decision: **EBIR-R1 technically accepted for shadow-only burn-in.**

Promotion status: **authoritative promotion remains blocked.**

## Acceptance Boundary

EBIR is accepted only as a shadow research and evaluation lane. This decision
does not authorize changes to default retrieval, governance scoring, Phase 10
consensus behavior, parent engrams, or any automatic promotion path.

The R1 adversarial pack remains CI-gated through:

```bash
python tools/run_ebir_refinement_benchmark.py
```

The generated result artifact must retain:

- `promotion_status = blocked_from_authoritative_resolution_promotion`
- zero EBIR regressions
- zero EBIR safety violations
- passing packet hash, parent-mutation, side-effect, and promotion-block assertions

## R2 Requirement

The next stage is **EBIR-R2 Shadow Burn-In and Human Review Value Trial**.

R2 must compare:

- one-pass reconciliation
- EBIR refinement
- raw evidence review

Representative difficult conflict packets should be evaluated for:

- correct resolution
- correct abstention or escalation
- evidence-supported decision quality
- reviewer confidence calibration
- review time
- unsupported-claim detection
- latency and token cost
- trigger selectivity

No product promotion is approved until EBIR demonstrates that it improves real
human review outcomes, not merely benchmarked reconciliation quality.
