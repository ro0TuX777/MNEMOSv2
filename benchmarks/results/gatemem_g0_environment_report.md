# GateMem G0 Environment Report

`GATEMEM_G0_PARTIAL_DELETION_GOVERNANCE_GAP_IDENTIFIED`

| Gate | Result |
|---|---|
| Upstream pinned to exact SHA | PASS |
| Upstream clone clean before/after runs | PASS |
| Dedicated Python 3.11 environment | PASS |
| Requirements installed | PASS |
| Core imports | PASS |
| Exact example command | EXPLAINED FAILURE — default OpenAI router requires prohibited key |
| Exact long-context command | EXPLAINED FAILURE — default OpenAI router requires prohibited key |
| Offline example stub diagnostic | PASS — 579 checkpoints |
| Offline long-context stub diagnostic | PASS — 579 checkpoints |
| Environment evidence | PASS |
| Compatibility assessment | PASS |
| Capability matrix | PASS |
| Scoring-field isolation specified | PASS |
| No MNEMOS runtime or policy mutation | PASS |
| No benchmark agent implementation | PASS |
| No hosted judge or public submission | PASS |
| GateMem-grade deletion semantics | GAP IDENTIFIED |

Pinned upstream: `603f9f4b4ba4b77f043c20f85687fa016fd720b0`.

The toolkit is environment-compatible with Python 3.11.9. The required command
failures are fully explained by a contradiction between GateMem's default
OpenAI provider and G0's no-key rule; distinct offline stub diagnostics prove
the local pipeline operates. Stub scores are not behavioral evidence because
the upstream stub consumes scoring annotations.

See [environment](../../docs/benchmarks/gatemem_g0_environment.md),
[assessment](../../docs/benchmarks/gatemem_g0_compatibility_assessment.md), and
[matrix](../../docs/benchmarks/gatemem_g0_capability_matrix.md).

