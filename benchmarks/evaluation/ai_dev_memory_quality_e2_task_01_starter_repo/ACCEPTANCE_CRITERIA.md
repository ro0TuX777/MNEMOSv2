# E2 Task 01 - Shared Acceptance Criteria

This is the sole scored acceptance suite for both conditions in the E2 paired
trial.

The final implementation must pass `npm run test:acceptance` without modifying
the files in `acceptance/`.

The app must satisfy:

1. **Schema migration**
   - Migrates legacy `waiting` to `in_review`.
   - Migrates legacy `accepted` to `approved`.
   - Preserves `deferred` as `deferred`.
   - Defaults missing `severity` and `impact` to `1`.
   - Defaults missing `blocker` to `false`.
   - Does not introduce sync/cloud fields from archived guidance.

2. **Current risk scoring**
   - Risk score is `severity * impact + 10` when `blocker` is true.
   - `risk_desc` sorts by risk score descending, then `updatedAt` descending,
     then `title` ascending, then `id` ascending.

3. **Queue filtering**
   - Default review queue includes only `open` and `in_review`.
   - Closed items appear only when `includeClosed` is true.

4. **Decision application**
   - Allows only current statuses: `in_review`, `approved`, `rejected`,
     `deferred`.
   - Rejects stale/legacy status `accepted` as an active decision.

5. **Current policy summary**
   - Reports local-only operation.
   - States that archived guidance is superseded.
   - States that `deferred` is not promoted to `approved`.

6. **Build**
   - `npm run build` must pass.

