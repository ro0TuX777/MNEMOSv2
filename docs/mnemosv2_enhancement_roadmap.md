# MNEMOSv2 Enhancement Roadmap

Date baseline: March 30, 2026
Planning horizon: 30/60/90 days
Primary objective: move from "benchmark-proven architecture" to "operationally enforced and production-tunable system."

---

## Status Snapshot

| Item | Status | Last updated |
|---|---|---|
| 1. Enforce phase-gate and governance evidence in CI/CD | Completed | March 30, 2026 |
| 2. Operationalize Wave 4 hygiene as continuous control loops | Completed (dry-run control loop + CI gate) | March 30, 2026 |
| 3. Improve reflect precision and reduce false reinforcement | Completed (precision guards + benchmark gate metric) | March 30, 2026 |
| 4. Tenant-aware governance policy profiles | Completed (profile model + API wiring + validation + tests) | March 30, 2026 |
| 5. Strengthen explainability output | Completed (per-result why won/lost traces + suppressed-candidate summary) | March 30, 2026 |
| 6. Optimize candidate-envelope and cache economics | Completed (runtime economics counters + API telemetry + tests) | March 30, 2026 |
| 7. Promote SLO-driven operations and rollback discipline | Completed (SLO reliability gate runner + CI enforcement + rollback signal) | March 30, 2026 |
| 8. Consolidate benchmark + promotion documentation into an operator playbook | Completed (`docs/mnemos_operator_playbook.md` published; benchmark + whitepaper linked) | March 30, 2026 |

---

## Strategic Priorities

1. Enforce phase-gate and governance evidence in CI/CD.
2. Operationalize Wave 4 hygiene as continuous control loops.
3. Improve reflect precision and reduce false reinforcement.
4. Add tenant-aware governance policy profiles.
5. Strengthen explainability and operator observability.
6. Optimize candidate-envelope and cache economics.
7. Promote SLO-driven operations and rollback discipline.
8. Consolidate benchmark + promotion documentation into an operator playbook.

---

## Workstreams

| ID | Workstream | Outcome |
|---|---|---|
| W1 | Gate Enforcement | Benchmark and phase-gate regressions block promotion automatically |
| W2 | Wave 4 Hygiene | Scheduled stale-memory, contradiction-sweep, and trust-recovery loops |
| W3 | Reflect Precision | Lower false-positive `USED` labeling and safer reinforcement behavior |
| W4 | Tenant Policies | Policy-by-tenant controls (thresholds, half-life, penalties) |
| W5 | Explainability | "Why won/lost" retrieval decision traces for dev and audit users |
| W6 | Performance Economics | Better cost/latency via envelope sizing and cache invalidation tuning |
| W7 | SLO + Reliability | Explicit latency/quality/reliability SLOs with rollback triggers |
| W8 | Documentation Ops | Single operational playbook for release and incident response |

---

## 30/60/90 Plan

## Day 0-30 (through April 29, 2026)

### Sprint goals

- W1: wire Memory Over Maps Phase 1-5 gate checks into CI as required checks for merge/promotion.
- W2: implement hygiene runner scaffold with dry-run mode and artifact emission.
- W3: add minimum-content and token-floor guards to overlap labeling.
- W5: add compact explain payload for governance outcomes per result.
- W8: publish operator playbook skeleton and link to benchmark + whitepaper.

### Exit criteria

- CI fails on any phase-gate regression.
- Hygiene dry-run produces reproducible reports on schedule.
- Reflect precision guards are benchmarked and do not regress trust recovery behavior.
- Explain payload is available behind an explicit flag.

## Day 31-60 (through May 29, 2026)

### Sprint goals

- W2: promote hygiene runner from dry-run to enforceable action mode with safe limits.
- W4: introduce tenant policy profiles with defaults and override validation.
- W6: instrument envelope size, cache hit ratio, invalidation fanout, and per-query cost metrics.
- W7: define SLOs for p50/p95 latency, stale-cache survival, suppression drift, and contradiction correctness.

### Exit criteria

- Hygiene actions are auditable, bounded, and reversible.
- Tenant policy profiles are versioned and validated at load time.
- Performance economics metrics are visible in benchmark artifacts.
- SLO targets and breach policies are documented and testable.

## Day 61-90 (through June 28, 2026)

### Sprint goals

- W1 + W7: attach SLO and phase-gate checks to promotion pipeline (canary to full rollout).
- W3 + W5: add semantic assist option to reflect labeling and expand explain traces.
- W6: tune cache invalidation + envelope defaults from measured distributions.
- W8: finalize operator playbook with incident, rollback, and promotion runbooks.

### Exit criteria

- Promotion is blocked automatically on gate or SLO breach.
- Reflect precision false-positive rate is reduced against March 30 baseline.
- Cache and envelope defaults are evidence-backed and documented.
- Operators can run deploy, promote, and rollback using one playbook.

---

## Ownership Model

| Area | Owner |
|---|---|
| W1 Gate Enforcement | Platform/CI engineer |
| W2 Hygiene Runner | Governance engineer |
| W3 Reflect Precision | Governance + applied NLP engineer |
| W4 Tenant Policies | Backend API engineer |
| W5 Explainability | API + frontend/SDK engineer |
| W6 Performance Economics | Performance engineer |
| W7 SLO + Reliability | SRE/operations engineer |
| W8 Documentation Ops | Tech lead + developer enablement |

---

## Dependencies and Risks

### Key dependencies

- Stable benchmark inputs and artifact schema for CI checks.
- Governance validation pack expansion for Wave 4 scenarios.
- Metrics pipeline for cache/economics/SLO data.
- Clear API contract versioning for explain and tenant-policy extensions.

### Primary risks

- Over-constraining CI can slow iteration if thresholds are noisy.
- Hygiene action mode may over-prune without conservative caps.
- Reflect precision tuning can trade recall for precision too aggressively.
- Tenant policy complexity can increase support burden.

### Mitigations

- Start with warning mode then enforce after two stable cycles.
- Require dry-run parity before hygiene action mode.
- Track precision and recall jointly and gate on balanced thresholds.
- Provide profile presets plus strict schema validation.

---

## Success Metrics

| Metric | Baseline (Mar 30, 2026) | 90-day target |
|---|---:|---:|
| Memory Over Maps phase-gate pass rate | 5/5 phases passing | 5/5 sustained in CI |
| Stale cache survival rate | 0.0000 | <= 0.0005 sustained |
| Bounded reflect adherence | 1.0000 | >= 0.995 sustained |
| Contradiction handling success (concurrent scenarios) | 1.0000 | >= 0.995 sustained |
| CI promotion blocked on gate/SLO breach | Manual policy | 100% automated |
| Explainability coverage for governed results | Partial | >= 95% governed results with explain trace |

---

## Recommended Sequence

1. Enforce existing evidence first (W1) before adding new behavior.
2. Promote hygiene safely (W2) with dry-run parity.
3. Improve reflect precision and explainability together (W3 + W5).
4. Add tenant controls after stable baseline telemetry (W4).
5. Finalize economics and SLO-governed promotions (W6 + W7).
6. Lock operating model with the playbook (W8).
