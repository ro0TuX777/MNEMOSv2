# AI Developer Memory Quality Lane Result

```text
LOCAL_DEVELOPMENT_EVIDENCE_ONLY
NO_GENERAL_MEMORY_CLAIM
```

## Task Outcome

| Metric | MNEMOS enabled | No memory |
|---|---:|---:|
| Task completion rate | 1.0 | 1.0 |
| Acceptance-test pass rate | 1.0 | 1.0 |
| Time to passing tests (s) | 20.0 | 37680.0 |
| Required constraints satisfied | True | True |
| Regressions introduced | 0 | 0 |

## Workflow Efficiency

| Metric | MNEMOS enabled | No memory |
|---|---:|---:|
| Total estimated tokens | None | None |
| Logged tool calls | 8 | 2 |
| Failed-test count | 0 | 0 |
| Wrong-turn count | 1 | 0 |
| Files changed | 2 | 2 |
| Rework after first implementation | 0 | 0 |

## Memory Quality

| Metric | MNEMOS enabled | No memory |
|---|---:|---:|
| Correct source / decision neighborhood | True | None |
| Provenance retained | 1.0 | None |
| Irrelevant-context rate | 0.0 | None |
| Retrieval precision | 0.8571428571428571 | None |
| False-abstention rate | None | None |
| Retrieved-context usefulness | 0.8571428571428571 | None |

## Retrieval-Integrity Controls

| Control | MNEMOS enabled | No memory |
|---|---|---|
| Seed snapshot | fresh_workspace | None |
| Executed-route fingerprint | None | None |
| Cache state | None | None |
| Duplicate suppression count | None | None |
| Retrieval profile / path | mnemos_memory_assisted | none |

## Pairwise Summary

- Both completed: True
- Acceptance parity: True
- Constraint satisfaction parity: True
- Token estimate delta, MNEMOS minus no-memory: 0
- Wrong-turn delta, MNEMOS minus no-memory: 1
- Repo-activity delta, MNEMOS minus no-memory: -1
- MNEMOS required memory tools used: True
- MNEMOS retrieved-context usefulness: 0.8571428571428571

## Interpretation

This paired artifact now records task outcome, workflow efficiency, memory quality, and retrieval-integrity controls in one contract. That makes future comparisons more inspectable and reduces the risk of reading performance differences without the retrieval conditions that produced them.
