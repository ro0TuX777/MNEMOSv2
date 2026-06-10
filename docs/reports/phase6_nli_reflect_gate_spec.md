# Phase 6 Gate Spec — Entailment-Grounded Reflect (NLI Critic)

Date: June 10, 2026
Workstreams: W3 (Reflect Precision) + W5 (Explainability)
Status: Prototype implemented; gate runnable; production wiring pending gate PASS

---

## 1. Capability

Replace the lexical word-overlap signal in the reflect path's `UsageDetector` with a
stateless NLI cross-encoder (`NLIUsageDetector`, `mnemos/governance/nli_usage_detector.py`).

| Design principle | Compliance |
|---|---|
| #1 Application-agnostic | NLI is general semantic entailment; no domain knowledge |
| #7 Graceful degradation | `health()` probe; reflect path falls back to lexical detector when the model is unavailable |
| #12 Non-destructive | Scores updated in `GovernanceMeta` only; raw engrams untouched |

### Classification logic

- **Premise = answer sentence, hypothesis = memory content.** (Direction matters:
  an answer synthesised from many memories is never entailed by one memory, but a
  used memory *is* entailed by the answer span that restates it.) Per memory, take
  max P(entailment) over answer sentences.
- P(entailment) ≥ threshold (default 0.5) → `USED`; otherwise → `IGNORED`.
- **NLI contradiction does NOT override read-path state.** `CONTRADICTED` remains
  exclusively the entity-slot contradiction outcome from the read path — Validation
  Pack v1 guarantees that conflict state outranks reflect-time signals, and a
  reflect-time critic must not be able to relabel read-path decisions. Max
  P(contradiction) is recorded as telemetry only; a future `REFUTED` label would
  require its own gate.
- Signals 0/0b/1 (veto, contradiction state, explicit citation) are byte-identical
  to the lexical detector and short-circuit before NLI inference.

### Model

`cross-encoder/nli-deberta-v3-xsmall` (default; ~70M params, shares the CUDA runtime
with the existing rerank lane via `sentence_transformers.CrossEncoder` — same
dependency, no new packages).

## 2. Benchmark Gate

Runner: `benchmarks/run_phase6_nli_gate.py` (gate runners live in `benchmarks/`,
matching `run_memory_over_maps_benchmarks.py`; `tools/` is reserved for operational
tooling).

Truthset: `benchmarks/truthsets/reflect_usage_truthset_v1.json` — 10 scenarios /
24 labeled judgments across five categories. Note: no JSON truthset previously
existed for reflect labeling (Governance Validation Pack v1 is a markdown scenario
document); this truthset is new and versioned. The adversarial categories encode the
whitepaper's documented precision boundaries, with one correction: the lexical
detector already enforces `min_memory_tokens_for_overlap=3` (W3 Day 0–30 token-floor
guard), so 2-token memories never reach the overlap path. The adversarial cases
therefore use **3–5 token generic memories** that pass the floor but carry no
substantive claim.

| Criterion | Threshold | Rationale |
|---|---|---|
| `used_precision_uplift` | ≥ +25% relative vs. lexical baseline | Headline precision goal |
| `no_recall_regression` | NLI USED recall ≥ lexical USED recall | Roadmap mitigation: "track precision and recall jointly and gate on balanced thresholds" |
| `adversarial_accuracy` | 1.0 on the gold-IGNORED judgments of `short_token_generic` + `proper_noun_coincidence` | Fixes the documented boundaries (false-suppression check; gold-USED controls in those scenarios count toward recall) |
| `latency_p95_ms_per_10_candidates` | < 150 ms | Enforced on CUDA (target hardware) only; informational on CPU |

Artifacts follow the phase-gate family:
`benchmarks/outputs/raw/phase6_nli_reflect_<ts>_raw.json`, plus `_report.md` and
`_decision.md` under `benchmarks/outputs/summaries/`. Decision renders PASS/HOLD.

## 3. GovernanceMeta extension (wiring phase, post-gate)

Additive, nullable fields — contract-safe for legacy engrams:

```json
{
  "_governance": {
    "reflect_precision_mode": "nli-deberta-v3-xsmall",
    "entailment_score": 0.92,
    "last_reflect_timestamp": "2026-06-10T17:45:00Z"
  }
}
```

`NLIUsageDetector.detect_with_scores()` already returns per-memory entailment /
contradiction probabilities for this purpose.

## 4. Promotion condition

A Phase 6 PASS qualifies the NLI detector for reflect-path wiring behind a
`reflect_precision_mode` policy-profile field (per-tenant, consistent with
`GovernancePolicyProfile`). It is **necessary but not sufficient** for
`MNEMOS_GOVERNANCE_MODE=enforced` promotion: enforced-mode suppression risk is
dominated by read-path veto/contradiction behavior, and the roadmap's open gaps
(enforced-mode drift divergence, trust recovery, concurrent reflect safety) still
require their own evidence. The accurate claim: Phase 6 removes false-positive
reinforcement as a blocker, shrinking the enforced-mode risk surface to the
read-path items.

## 5. Risks

- Truthset v1 is small (24 judgments); uplift numbers will be coarse. Expand to
  ≥100 judgments before treating the precision delta as load-bearing evidence.
- NLI models can mark generic noun-phrase hypotheses as entailed; if
  `adversarial_accuracy` fails, raise `entailment_threshold` before changing models.
- xsmall checkpoint trades accuracy for latency; `--model` flag allows gating
  larger checkpoints under the same criteria.
- **Known recall boundary (measured June 10, 2026):** when the answer wraps a
  claim in attribution framing ("the report indicates X…") and the memory is
  more specific than X, entailment fails in both directions (measured
  P(entailment) ≈ 0.004 — not a threshold issue). Mitigation path: claim
  extraction on answer sentences before pairing. Bidirectional scoring already
  recovers the plain over-specific-memory case.
