# AI Developer Memory Trial Comparison

```text
LOCAL_DEVELOPMENT_EVIDENCE_ONLY
NO_GENERAL_MEMORY_CLAIM
```

| Metric | MNEMOS enabled | No memory |
|---|---:|---:|
| Total estimated tokens | 5000 | 27000 |
| Memory calls | 26 | 0 |
| Route log rows | 8 | 12 |
| Repo activity rows | 6 | 25 |
| Wrong-turn rows | 0 | 4 |
| Test runs | 2 | 3 |
| Failed test runs | 0 | 2 |

## Observed Pattern

- Both completed: True
- MNEMOS required memory tools used: True
- MNEMOS had recovery overhead: True
- No-memory run had more logged wrong turns: True
- No-memory run had more repo activity: True
- Token estimate delta, MNEMOS minus no-memory: -22000

## Interpretation

This single paired trial suggests MNEMOS can be integrated into an AI developer workflow and used during app construction. The MNEMOS run shows lower logged repo activity, fewer logged wrong turns, and lower estimated token use, but it also had infrastructure recovery overhead from early unavailable/misrouted calls. This is local development evidence only and should not be treated as a general memory-performance claim.
