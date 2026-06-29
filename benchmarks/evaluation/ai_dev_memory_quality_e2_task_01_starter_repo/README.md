# AI Dev Memory Quality E2 Task 01 Starter Repo

This starter repo supports the E2 durable-context benchmark:

```text
AI_DEV_MEMORY_QUALITY_E2
DURABLE_CONTEXT_AND_STALE_GUIDANCE_REJECTION
```

The app is intentionally incomplete. The agent must repair the local Release
Review Queue by using current local evidence and rejecting superseded archived
guidance.

Important files:

- `TASK_BRIEF.md` - task objective and boundaries.
- `ACCEPTANCE_CRITERIA.md` - frozen acceptance contract.
- `task_control_manifest.json` - task identity and MNEMOS seed metadata.
- `docs/` - current contract, ADR, known regression, handoff notes, and stale archived guidance.
- `src/logic.js` - intentionally flawed implementation.
- `acceptance/acceptance.test.js` - frozen acceptance tests.

