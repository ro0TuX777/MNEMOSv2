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
| Time to passing tests (s) | None | 198.147754 |
| Timing comparable | False | True |
| Required constraints satisfied | True | True |
| Regressions introduced | 0 | 0 |

## Workflow Efficiency

| Metric | MNEMOS enabled | No memory |
|---|---:|---:|
| Total estimated tokens | None | None |
| Token counts comparable | False | False |
| Logged tool calls | 6 | 8 |
| Failed-test count | 5 | 5 |
| Wrong-turn count | 1 | 2 |
| Files changed | 5 | 4 |
| Rework after first implementation | 1 | 2 |

## Memory Quality

| Metric | MNEMOS enabled | No memory |
|---|---:|---:|
| Correct source / decision neighborhood | False | None |
| Provenance retained | 0.6666666666666666 | None |
| Irrelevant-context rate | 0.0 | None |
| Retrieval precision | 0.0 | None |
| False-abstention rate | None | None |
| Retrieved-context usefulness | 0.0 | None |

## Retrieval-Integrity Controls

| Control | MNEMOS enabled | No memory |
|---|---|---|
| Seed snapshot | mnemos_claude_repo_seed:2437be792647c500 | None |
| Executed-route fingerprint | collection_snapshot=mnemos_claude_repo_seed:2437be792647c500; configured=semantic\|balanced\|lexical_top_k=25\|semantic_top_k=25; executed=hybrid\|lexical_dominant\|lexical_top_k=25\|semantic_top_k=25 | None |
| Cache state | None | None |
| Duplicate suppression count | 0 | None |
| Retrieval profile / path | mnemos_memory_assisted | none |
| Observed execution path | mcp | no_memory_control |

## Pairwise Summary

- Both completed: True
- Acceptance parity: True
- Constraint satisfaction parity: True
- Timing comparable between conditions: False
- Token estimate delta, MNEMOS minus no-memory: None
- Token counts comparable between conditions: False
- Wrong-turn delta, MNEMOS minus no-memory: -1
- Repo-activity delta, MNEMOS minus no-memory: -5
- MNEMOS required memory tools used: True
- MNEMOS retrieved-context usefulness: 0.0
- Observed MNEMOS execution path: mcp

## Interpretation

This paired artifact now records task outcome, workflow efficiency, memory quality, and retrieval-integrity controls in one contract. That makes future comparisons more inspectable and reduces the risk of reading performance differences without the retrieval conditions that produced them.
