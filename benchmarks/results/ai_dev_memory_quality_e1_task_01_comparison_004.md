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
| Time to passing tests (s) | 216.651834 | 194.046631 |
| Timing comparable | True | True |
| Required constraints satisfied | True | True |
| Regressions introduced | 0 | 0 |

## Workflow Efficiency

| Metric | MNEMOS enabled | No memory |
|---|---:|---:|
| Total estimated tokens | 57000 | 51000 |
| Token counts comparable | True | True |
| Logged tool calls | 14 | 5 |
| Raw failed-test count | 5 | 0 |
| Failed-test metric status | not_comparable_due_to_harness_failures | not_comparable_due_to_harness_failures |
| Harness/environment failures | 1 | 1 |
| Expected RED acceptance failures | 4 | 0 |
| Agent-caused test failures | 0 | 0 |
| Wrong-turn count | 2 | 2 |
| Files changed | 7 | 8 |
| Rework after first implementation | 3 | 3 |

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
| Seed snapshot | 0a578569ef136afa | None |
| Seed snapshot layer | task_seed_manifest_hash | None |
| Collection snapshots from executed route | mnemos_ai_dev_e1_task_01:2437be792647c500 |  |
| Collection snapshot layer | retrieval_index_snapshot | None |
| Executed-route fingerprint | semantic\|balanced\|lexical_top_k=25\|semantic_top_k=25 \| mnemos_ai_dev_e1_task_01:2437be792647c500; semantic\|none\|lexical_top_k=25\|semantic_top_k=25 \| mnemos_ai_dev_e1_task_01:2437be792647c500; hybrid\|balanced\|lexical_top_k=25\|semantic_top_k=25 \| not_applicable_write | None |
| Cache state | None | None |
| Duplicate suppression count | 0 | None |
| Retrieval profile / path | mnemos_memory_assisted | none |
| Observed execution path | mcp | no_memory_control |

## Pairwise Summary

- Both completed: True
- Acceptance parity: True
- Constraint satisfaction parity: True
- Timing comparable between conditions: True
- Token estimate delta, MNEMOS minus no-memory: 6000
- Token counts comparable between conditions: True
- Wrong-turn delta, MNEMOS minus no-memory: 0
- Repo-activity delta, MNEMOS minus no-memory: -13
- Failed-test metric status: not_comparable_due_to_harness_failures
- Agent-caused test failure delta, MNEMOS minus no-memory: 0
- MNEMOS required memory tools used: True
- MNEMOS retrieved-context usefulness: 0.8571428571428571
- Observed MNEMOS execution path: mcp

## Interpretation

This paired artifact now records task outcome, workflow efficiency, memory quality, and retrieval-integrity controls in one contract. That makes future comparisons more inspectable and reduces the risk of reading performance differences without the retrieval conditions that produced them.
