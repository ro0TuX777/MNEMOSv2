# Associative Routing View E0 — Benchmark Report

Status: `pass` (phase `E0-SMOKE`)

semantic_keyword_baseline_proxy is a local deterministic bag-of-words ranker over fixture documents, not MNEMOS's production semantic/hybrid retrieval. This benchmark makes no claim about production retrieval quality.

## Summary

| Metric | semantic_keyword_baseline_proxy | associative_routing |
|---|---|---|
| All-required recall (mean) | 0.800 | 1.000 |
| Top-1 recall (mean) | 0.450 | 0.800 |

- Query count: 10
- False abstention count (routing): 0
- Fallback/abstention correctness rate (routing): 1.000
- Routing-path provenance completeness: 1.000

## Per-query results

| Query | Baseline recall | Routing recall | Routing result | Note |
|---|---|---|---|---|
| Why is GateMem work paused? | 1.00 | 1.00 | resolved | Positive routing — required quoted question. |
| What is frozen for regression testing only? | 1.00 | 1.00 | resolved | Positive routing — required quoted question. |
| What blocks a fresh GateMem evaluation? | 0.00 | 1.00 | resolved | Positive routing — required quoted question. |
| What superseded the G4 implementation lane? | 1.00 | 1.00 | abstained | Positive routing — required quoted question; correct answer is abstention (G4 is the frozen latest baseline; nothing supersedes it). |
| What is the current state of the G5 handoff? | 0.00 | 1.00 | resolved | Positive routing — required quoted question. |
| What is the current status of GateMem G4? | 1.00 | 1.00 | resolved | Temporal — current state over historical precursor. |
| What superseded the G4 implementation proposal? | 1.00 | 1.00 | resolved | Temporal — superseded_by resolves in passive direction. |
| What did the G4 implementation lane supersede? | 1.00 | 1.00 | resolved | Temporal — supersedes resolves in active direction. |
| What is the GateMem frozen baseline? | 1.00 | 1.00 | resolved | Ambiguity — two genuinely distinct frozen baselines must both surface. |
| What is the capital of France? | 1.00 | 1.00 | abstained | Out-of-domain — no cue should match; must abstain. |
