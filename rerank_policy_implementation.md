# MNEMOS Conditional Rerank Policy Implementation Note

This document summarizes the architectural integration of the new Conditional Cross-Encoder Policy and provides operations with the necessary runbook for maintaining the feature flags across Shadow and Production modes.

## Implementation Details

* **Family Classification:** Evaluates intents using a pluggable wrapper supporting `heuristic` or `zero_shot` modes, driven by `rerank_policy.yaml`. The fallback for unrecognized classifications sits securely at extremely low confidence (`0.20`), eliminating previous misrepresentations.
* **Eligibility Evaluation:** Triggered directly following Stage-1. Safely checks candidate pools, circuit breakers, and queries a strictly bounded `.health()` probe directly validating the HuggingFace CrossEncoder model array.
* **Telemetry Emission:** All required `telemetry_fields` are extracted inline into a JSON dictionary and instantly appended locally to a robust `logs/retrieval_telemetry.jsonl` sink enabling subsequent off-system analytics ingestion.
* **Shadow mode Safety Checks:** Operations strictly distinguish between `hard_skip_reasons` (where evaluation guarantees abort sequence: e.g. timeouts or budgets) and `soft_skip_reasons` (insufficient candidates or excluded families where experimental reranking computes purely for analytics).

---

## Ops Runbook 

### 1. Enable Rerank (from Shadow to Full Enforcement)
1. Locate `mnemos/retrieval/policies/rerank_policy.yaml`.
2. Locate the root `shadow_mode` dictionary.
3. Modify `enabled_initially: true` to `enabled_initially: false`.
4. Ensure `mode:` remains `conditional`.
5. Restart the `RetrievalRouter` service container to reload the YAML config matrix into memory.

### 2. Disable Procedure & Emergency Rollback
If top-tier latency (P95) regresses drastically, or if rerank CPU isolation destabilizes the host matrix:
1. Open `rerank_policy.yaml`.
2. Switch `mode: conditional` to `mode: dense_only`.
3. Clear `enabled_query_families` or leave it empty `[]`.
4. Run configuration reload (or service restart).
5. Add the deployment rollback note into the ops changelog.

### 3. Health Checks & Alerting Thresholds
Monitor the new payload telemetry matrix for these triggers:
* **`timeout_occurred`:** High severity. Indicates the HuggingFace CrossEncoder model layer is stalling under request throughput.
* **`rerank_skip_reason = circuit_breaker_open`:** Critical priority. Means timeouts or native exceptions have peaked >1%.
* **`budget_exceeded`:** Warning. Means p95 latencies for a specific family have breached constraints and conditional rendering is being dropped dynamically.

### 4. Adjusting the Rerank Depth Parameters
If top-1 churn is low (i.e. reranking Top-50 rarely changes the 1st result compared to Top-20):
* Access `depth_by_family` inside the YAML and decrement the active family from `50` to `20` to linearly drop CPU/GPU cost by 60%.
