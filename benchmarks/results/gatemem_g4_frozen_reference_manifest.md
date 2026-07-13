# GateMem G4 Frozen Reference-Contract Baseline

```text
GATEMEM_G4_FROZEN_REFERENCE_CONTRACT_BASELINE
REGRESSION_TESTING_ONLY
SYNTHETIC_DEVELOPMENT_ONLY
SEALED_EVALUATION_STILL_BLOCKED
NO_GENERALIZATION_OR_PRODUCTION_CLAIM
```

| Frozen item | Value |
|---|---|
| Implementation/corpus composite | `ed3b5c7672e591b039183eaa2d8c7c7630a655575bfc7866558d53f2eb874c52` |
| Synthetic corpus composite | `b0cb2522f2d21fd90705cac921a5c533d4cb569e1f5ad267f6e35b8b52cd01e3` |
| Inspectable development cases | 36 |
| Exact expected outcomes | 36/36 |
| Reference gates | 33/33 passed |

This manifest freezes G4 for regression testing only. Any source or corpus
change creates a new development iteration and must not overwrite this result.

Further GateMem implementation work is paused. A fresh evaluation requires an
independent sealed-evaluation custodian, a newly sealed or independent corpus,
preregistration, a frozen candidate artifact, and one-shot evaluation.
