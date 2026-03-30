# MNEMOS Operator Playbook

Date baseline: March 30, 2026  
Scope: deployment, promotion, rollback, and incident response for MNEMOSv2.

---

## 1. Purpose

This playbook is the operational source of truth for MNEMOS promotion and reliability workflows.

Use this document to:
- deploy MNEMOS in containerized runtime mode,
- validate benchmark and governance gates,
- promote safely through canary stages,
- rollback deterministically on gate or SLO breach,
- capture evidence artifacts for auditability.

---

## 2. Runtime Model

- Runtime services are deployed with Docker Compose (containerized serving model).
- Benchmarks, tools, and test runners are generally executed from host Python.
- Promotion decisions are evidence-driven and blocked automatically on failing gates.

References:
- `docs/benchmark.md`
- `docs/whitepaper.md`
- `docs/mnemosv2_enhancement_roadmap.md`

---

## 3. Preflight Checklist

1. Confirm runtime health:
   - `python tools/mnemos_health_audit.py`
2. Confirm contracts parse and pass:
   - `python tools/mnemos_ci_gates.py --contract-dir service`
3. Confirm required benchmark stack services are up when benchmarking:
   - `docker compose -f benchmarks/docker-compose.bench.yml up -d`
4. Confirm CI gate suite baseline:
   - `python tools/mnemos_ci_gates.py --run-memory-over-maps-gates --run-governance-evidence-gates --run-wave4-hygiene-gate --run-slo-reliability-gate`

Exit condition: all checks pass.

---

## 4. Promotion Runbook

### 4.1 Generate rollout scaffold

- `python tools/mnemos_cutover_scaffold.py --app <app_name>`

This creates a staged cutover manifest with shadow, canary (`5/25/50`), and full stages.

### 4.2 Stage promotion sequence

1. `shadow`: dual-write; reads remain on old backend.
2. `canary_5`: route 5% reads to MNEMOS.
3. `canary_25`: route 25% reads to MNEMOS.
4. `canary_50`: route 50% reads to MNEMOS.
5. `full`: route 100% reads to MNEMOS.

At each stage:
1. Run CI and gate validation:
   - `python tools/mnemos_ci_gates.py --run-health-audit --smoke-spec tools/mnemos_smoke_spec.json --run-memory-over-maps-gates --run-governance-evidence-gates --run-wave4-hygiene-gate --run-slo-reliability-gate`
2. Require pass status before advancing.
3. Preserve generated raw/report artifacts under `benchmarks/outputs`.

Promotion policy:
- Do not advance stage on any failed phase gate, governance evidence gate, hygiene gate, or SLO reliability gate.

---

## 5. Rollback Runbook

Rollback trigger conditions:
- Any CI gate failure.
- SLO reliability gate breach.
- Manual operator decision based on live incident risk.

Immediate rollback actions:
1. Stop promotion at current stage.
2. Re-route reads to the previous stable canary stage.
3. Keep dual-write if needed to avoid data-loss windows.
4. Re-run health audit and smoke checks:
   - `python tools/mnemos_health_audit.py`
   - `python tools/mnemos_ci_gates.py --smoke-spec tools/mnemos_smoke_spec.json`
5. Open incident record with failing metric and stage details.

Evidence to capture:
- latest `slo_reliability_*_raw.json`
- latest gate reports in `benchmarks/outputs/summaries`
- relevant service logs and deployment change record

---

## 6. Incident Response Runbook

### 6.1 Triage

1. Confirm if breach is functional, quality, or latency/SLO.
2. Identify first failing gate and timestamp.
3. Identify whether failure is reproducible locally via gate runner.

### 6.2 Stabilize

1. Roll back traffic to last known-good stage.
2. Freeze further promotion.
3. Maintain read availability over progression speed.

### 6.3 Diagnose

Run targeted checks:
- SLO gate:
  - `python tools/run_slo_reliability_gate.py --stage canary_25 --fail-on-breach`
- Wave 4 hygiene:
  - `python tools/run_wave4_hygiene.py --mode dry-run --input benchmarks/truthsets/wave4_hygiene_seed.json --fail-on-gate`
- Memory Over Maps phase-gates:
  - `python -m pytest -q tests/test_memory_over_maps_benchmark_runner.py`
- Governance evidence:
  - `python -m pytest -q tests/test_governance.py tests/test_governance_contradictions.py tests/test_governance_reflect.py tests/test_governance_drift_validation.py`

### 6.4 Recover

1. Apply fix.
2. Re-run failing gate(s) and full CI gate command.
3. Resume canary from last stable stage only after all gates pass.

---

## 7. Artifact and Audit Requirements

For every promotion decision (advance/hold/rollback), retain:
- gate command used,
- stage and timestamp,
- raw artifact path(s),
- report path(s),
- pass/fail decision with operator name.

Recommended storage:
- retain `benchmarks/outputs/raw/*` and `benchmarks/outputs/summaries/*` as immutable release evidence.

---

## 8. Ownership and Escalation

- Platform/CI engineer: gate automation and pipeline enforcement.
- Governance engineer: governance evidence and hygiene controls.
- SRE/operations engineer: SLO breaches, rollback execution, incident command.
- Tech lead: final promotion authorization when all gates pass.

Escalation rule:
- If production risk is non-trivial, prioritize rollback first and diagnosis second.

---

## 9. Definition of Done for Operations

Operational posture is complete when:
1. Promotion is blocked automatically on gate/SLO breach.
2. Canary-to-full runbook is followed without manual ambiguity.
3. Rollback is deterministic and stage-based.
4. Artifacts are sufficient for external audit or postmortem review.
5. This playbook remains the single operator entry point for release and incident response.
