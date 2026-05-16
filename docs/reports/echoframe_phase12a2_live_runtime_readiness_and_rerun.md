# Phase 12A.2 — Live Runtime Readiness and Rerun

## Objective
To ensure the local Ollama daemon is active, a reasoning model is loaded, and the Phase 12A.1 benchmark is successfully executed against live inference. This document serves as an operational checklist and rerun package to prevent ambiguous "try again later" outcomes.

## 1. Required Checks Before Rerun

Run the following checks in your PowerShell environment to verify the daemon is healthy on the correct port (**7777**):

1. **Check loaded models:**
   ```powershell
   ollama list
   ollama ps
   ```

2. **Verify API connectivity:**
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:7777/api/tags" -Method Get
   ```

3. **Tiny Direct Generation Smoke Test:**
   Confirm the model can respond with the required `READY` payload:
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:7777/api/generate" `
     -Method Post `
     -ContentType "application/json" `
     -Body '{"model":"<MODEL_ID>","prompt":"Return only the word READY.","stream":false}'
   ```
   *(Ensure you replace `<MODEL_ID>` with your primary active model.)*

## 2. Rerun Command

Once the endpoint responds successfully to the smoke test, execute the full benchmark run:

```powershell
python benchmarks/echoframe_phase12a/run_phase12a_placement_benchmark.py `
  --model-provider ollama `
  --model-id <MODEL_ID> `
  --base-url http://localhost:7777 `
  --layouts A,B,C,D,E `
  --cases all `
  --out benchmarks/outputs/raw
```

## 3. Decision Rule (Unchanged)

The rerun should only recommend `SHADOW_LAYOUT_D` or `SHADOW_LAYOUT_E` if the live model shows:
- **PASS**: zero source pointer loss
- **PASS**: zero protected numeric/date span loss
- **PASS**: zero negation/exception regression
- **PASS**: zero contradiction suppression
- **PASS**: zero evidence-gap suppression
- **PASS**: zero governance warning suppression
- **PASS**: composite placement_quality_score improves over Layout A
- **PASS**: token and latency overhead remain within threshold

Otherwise, the decision must remain `NO_CHANGE`, `INSUFFICIENT_EVIDENCE`, or `FAIL_SAFETY_REGRESSION`.
