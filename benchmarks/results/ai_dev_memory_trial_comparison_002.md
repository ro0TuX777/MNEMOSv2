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
| Time to passing tests (s) | 37149.0 | 557.182571 |
| Required constraints satisfied | True | True |
| Regressions introduced | 0 | 0 |

## Workflow Efficiency

| Metric | MNEMOS enabled | No memory |
|---|---:|---:|
| Total estimated tokens | 5000 | 27000 |
| Logged tool calls | 26 | 6 |
| Failed-test count | 0 | 0 |
| Wrong-turn count | 0 | 4 |
| Files changed | 0 | 14 |
| Rework after first implementation | 0 | 1 |

## Memory Quality

| Metric | MNEMOS enabled | No memory |
|---|---:|---:|
| Correct source / decision neighborhood | True | None |
| Provenance retained | 0.5 | None |
| Irrelevant-context rate | 0.23076923076923078 | None |
| Retrieval precision | 0.5 | None |
| False-abstention rate | None | None |
| Retrieved-context usefulness | 0.5 | None |

## Retrieval-Integrity Controls

| Control | MNEMOS enabled | No memory |
|---|---|---|
| Seed snapshot | None | None |
| Executed-route fingerprint | None | None |
| Cache state | None | None |
| Duplicate suppression count | None | None |
| Retrieval profile / path | mnemos_memory_assisted | none |

## Pairwise Summary

- Both completed: True
- Acceptance parity: True
- Constraint satisfaction parity: True
- Token estimate delta, MNEMOS minus no-memory: -22000
- Wrong-turn delta, MNEMOS minus no-memory: -4
- Repo-activity delta, MNEMOS minus no-memory: -19
- MNEMOS required memory tools used: True
- MNEMOS retrieved-context usefulness: 0.5

## Interpretation

This paired artifact now records task outcome, workflow efficiency, memory quality, and retrieval-integrity controls in one contract. That makes future comparisons more inspectable and reduces the risk of reading performance differences without the retrieval conditions that produced them.
