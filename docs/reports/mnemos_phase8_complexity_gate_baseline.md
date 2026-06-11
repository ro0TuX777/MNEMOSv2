# MNEMOS Phase 8 Complexity Gate Baseline

Date: June 11, 2026

## Summary Decision

ZERO_SHOT_NLI_BASELINE_FAIL

The Phase 8 query-complexity truthset and shadow classifier path are implemented, but the reused NLI cross-encoder does not meet the proposed gate for production routing.

## Truthset

- file: `benchmarks/truthsets/query_complexity_v1.json`
- query count: `60`
- class balance: `20` CLASS_A, `20` CLASS_B, `20` CLASS_C
- domains: SIGINT and GDPR operational references

## Classifier

- implementation: `mnemos/retrieval/complexity.py`
- model: `cross-encoder/nli-deberta-v3-xsmall`
- mode: zero-shot NLI entailment
- router integration: shadow metadata only via `complexity_shadow=true`

## Gate Result

Command:

```bash
python tools\run_phase8_complexity_gate.py
```

Metrics:

- overall accuracy: `0.4000`
- CLASS_A accuracy: `2/20` (`0.1000`)
- CLASS_B accuracy: `18/20` (`0.9000`)
- CLASS_C accuracy: `4/20` (`0.2000`)
- p95 latency: `71.2064ms`

Gate thresholds:

- accuracy: `> 0.85` - FAIL
- p95 latency: `< 25ms` - FAIL

## Interpretation

The NLI reuse strategy is useful as a measurable baseline, but not as the Phase 8 router brain. It over-predicts the multi-hop class and under-recognizes simple factoid and global synthesis queries.

## Recommendation

Proceed to a dedicated low-latency complexity classifier or a lightweight hybrid heuristic/model gate. Keep the current classifier path in shadow mode only until accuracy and p95 latency pass the Phase 8 thresholds.
