# Known Issues

Status: authoritative for this frozen starter package.

## Seeded Defect

The starter implementation's `priority_desc` sorting is not compliant with the
sorting contract.

Observed defect:

- items with the same priority are left in insertion order instead of using the
  required deterministic tie-break chain.

Required repair:

- `priority_desc` must tie-break by `updatedAt` descending, then `title`
  ascending, then `id` ascending.

This defect must be repaired without weakening filtering, persistence,
validation, or acceptance behavior.
