# Known Regression

The starter implementation is intentionally wrong.

Known defects:

- `risk_desc` currently ignores `impact` and `blocker`.
- missing `severity` and `impact` are treated inconsistently.
- `deferred` can be promoted to `approved` during migration.
- stale active status `accepted` may be accepted as a new decision.
- the policy summary does not clearly reject archived guidance.

The repair must follow `docs/current_release_contract.md` and
`docs/adr/0007-local-review-queue.md`.

