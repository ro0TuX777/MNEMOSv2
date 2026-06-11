# MNEMOS Phase 9 Hierarchy Activation Report

Date: June 11, 2026

## Summary Decision

PHASE_9_SCAFFOLD_READY

The RAPTOR-lite hierarchy lane is implemented in action-capable form, but the hierarchy gate is not yet passing in the current local environment because the active Qdrant collection contains only the 21-point smoke corpus and no summary engrams have been indexed.

## Implemented

- `HierarchicalClusteringRunner` supports dry-run and action mode.
- Action mode generates summary engrams with:
  - `metadata.is_summary_engram = true`
  - `metadata.cluster_id`
  - `metadata.depth = 1`
  - `edges` containing source member engram ids
  - `source = derived://hierarchy/<cluster_id>`
- Synthesis uses the `SMC_LLM_*` OpenAI-compatible endpoint when configured, with deterministic join-and-truncate fallback.
- `tools/run_phase9_hierarchy_dry_run.py --apply` can index summary engrams into the active Qdrant collection.
- CLASS_C adaptive routing now searches the summary layer first via `metadata.is_summary_engram = true`, limits summary retrieval to at most 5 results, and falls back to flat lexical-dominant retrieval if the summary layer is empty.
- CLASS_C truthset rows now include `golden_summary` fields for hierarchy evaluation.
- `tools/run_phase9_hierarchy_gate.py` evaluates summary hit rate, cosine similarity against golden summaries, and summary-layer p95 latency versus flat p95 latency.

## Local Smoke Evidence

Command:

```bash
python tools\run_phase9_hierarchy_dry_run.py --limit 2121 --output benchmarks\outputs\raw\hierarchy_report.json
```

Result:

- collection: `mnemos_engrams_nomic_mrl`
- engrams scanned: `21`
- clusters: `1`
- summary writes: `0`

Gate command:

```bash
python tools\run_phase9_hierarchy_gate.py --no-artifacts
```

Result:

- queries: `20`
- summary hit rate: `0.0000`
- mean similarity: `0.0000`
- summary p95: `565.9091ms`
- flat p95: `38.2087ms`
- gate: `FAIL`

## Interpretation

This is an expected environment failure, not a code-path failure. The hierarchy gate requires populated summary engrams. The local collection currently has no summary layer and does not contain the 2,121-engram representative corpus.

## Next Operator Step

Load or restore the representative corpus, then run:

```bash
python tools\run_phase9_hierarchy_dry_run.py --limit 2121 --apply
python tools\run_phase9_hierarchy_gate.py
```

Promotion should wait until the hierarchy gate passes on the representative corpus.
