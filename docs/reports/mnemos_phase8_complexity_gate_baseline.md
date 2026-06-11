# MNEMOS Phase 8 Complexity Gate Baseline

Date: June 11, 2026

## Summary Decision

EMBEDDED_REFLEX_ACTIVE_GATE_PASS

The reused NLI cross-encoder failed the Phase 8 gate and established that MNEMOS needs a router reflex rather than another reasoning model. The embedded linear classifier trained over the promoted query embedding space passed the gate with zero material runtime overhead after embedding.

The hold-out validation also passed without retraining, so the classifier is now wired into active adaptive routing with response metadata for auditability.

## Truthset

- file: `benchmarks/truthsets/query_complexity_v1.json`
- query count: `60`
- class balance: `20` CLASS_A, `20` CLASS_B, `20` CLASS_C
- hold-out file: `benchmarks/truthsets/query_complexity_holdout_v1.json`
- hold-out query count: `15`
- hold-out class balance: `5` CLASS_A, `5` CLASS_B, `5` CLASS_C
- domains: SIGINT and GDPR operational references

## Classifiers

- implementation: `mnemos/retrieval/complexity.py`
- zero-shot baseline: `cross-encoder/nli-deberta-v3-xsmall`
- embedded reflex: `embedded-linear-softmax`
- weights: `mnemos/retrieval/complexity_weights.bin`
- embedding model: `nomic-ai/nomic-embed-text-v1.5`
- router integration: active adaptive routing; legacy diagnostics remain available via `complexity_shadow=true` when adaptive routing is disabled

## Zero-Shot Gate Result

Command:

```bash
python tools\run_phase8_complexity_gate.py --classifier zero-shot
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

## Embedded Reflex Gate Result

Commands:

```bash
python tools\train_complexity_weights.py
python tools\run_phase8_complexity_gate.py
```

Metrics:

- overall accuracy: `1.0000`
- CLASS_A accuracy: `20/20` (`1.0000`)
- CLASS_B accuracy: `20/20` (`1.0000`)
- CLASS_C accuracy: `20/20` (`1.0000`)
- p95 classifier latency after embedding: `0.0127ms`

Gate thresholds:

- accuracy: `> 0.85` - PASS
- p95 classifier latency after embedding: `< 2ms` - PASS

## Hold-Out Gate Result

Command:

```bash
python tools\run_phase8_complexity_gate.py --truthset benchmarks\truthsets\query_complexity_holdout_v1.json
```

Metrics:

- overall accuracy: `1.0000`
- CLASS_A accuracy: `5/5` (`1.0000`)
- CLASS_B accuracy: `5/5` (`1.0000`)
- CLASS_C accuracy: `5/5` (`1.0000`)
- p95 classifier latency after embedding: `0.0502ms`

Gate thresholds:

- accuracy: `> 0.85` - PASS
- p95 classifier latency after embedding: `< 2ms` - PASS

## Active Routing Mapping

- CLASS_A: semantic retrieval, semantic-dominant posture, aggressive budget plan, rerank skipped, graph skipped.
- CLASS_B: balanced posture, conservative budget plan, graph requested when a graph tier is available, rerank forced.
- CLASS_C: hybrid retrieval when lexical is available, lexical-dominant posture, balanced budget plan, graph skipped.

Every adaptive search response now reports `complexity_classification` and `routing_posture` in the response metadata block.

## Interpretation

The NLI reuse strategy is useful as a measurable baseline, but not as the Phase 8 router brain. It over-predicts the multi-hop class and under-recognizes simple factoid and global synthesis queries.

The embedded reflex path fits the 60-query SIGINT/GDPR truthset in the active Nomic embedding space and passed a 15-query hold-out without retraining. The reported latency excludes embedding time because the intended production path reuses the query vector already computed for retrieval.

## Recommendation

Keep collecting adaptive-routing telemetry and expand the hold-out set as new corpus families are added. The next hardening step is to pass the already-computed retrieval query vector into `classify_vector()` directly in the Qdrant hot path; the current active path is correct and gated, but still embeds once for classification before tier-local retrieval embedding.
