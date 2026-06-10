# Counterfactual Explainability — Spec & Implementation Record

Date: June 10, 2026
Workstream: W5 (Explainability)
Status: Implemented, unit + service-path tested

---

## 1. Capability

Extends the `explain_governance: true` response block with deterministic
"distance to rank 1" counterfactuals for the top 3 non-rank-1 candidates
(suppressed candidates included). Pure arithmetic over the governed-score
product — no models, no retrieval calls, no stored state:

```
governed_score = retrieval_score × trust × utility × freshness
               × contradiction × veto
```

For each analysed candidate and each modifier *m* with current value *c*:

```
required_m = c × rank1_score / governed_score
```

(the value of *m* that ties rank 1, holding all other factors fixed).
`achievable` is checked against the modifier's valid policy range; when
unreachable, the response reports the best rank attainable at the range
maximum instead.

| Modifier | Valid range | Source |
|---|---|---|
| trust / utility | [0.75, 1.25] | `UtilityPolicy` mapping |
| freshness | (0, 1.0] | exponential decay |
| contradiction | {0.25, 1.0} | loser / winner |

## 2. Veto handling

- **Score-floor veto** (the only threshold-reversible veto): emits a policy
  hint with the effective per-profile threshold —
  *"Current min_score veto threshold (0.6000) suppressed this result; a
  threshold of 0.5500 or lower would have included it."*
  The threshold comes from `Governor.effective_min_score(profile)` (new
  method), so per-tenant profile overrides are reported correctly.
- **State vetoes** (`deletion_state`, `toxic` flag) are declared
  non-counterfactable — no modifier or threshold change can include them.
- **Contradiction losers** get the win-the-group counterfactual via the
  contradiction modifier entry plus a hint naming the group winner.

## 3. Response shape

`meta.governance_explain.counterfactuals` (alongside the existing
`suppressed_candidates`):

```json
[
  {
    "engram_id": "...",
    "current_rank": 2,
    "governed_score": 0.36,
    "rank1_gap": 0.54,
    "counterfactuals": [
      {
        "modifier": "freshness",
        "current": 0.4,
        "required_for_rank1": 1.0,
        "achievable": true,
        "statement": "If freshness_modifier were 1.0000 instead of 0.4000, governed_score would be 0.9000 and this candidate would tie rank 1."
      }
    ],
    "policy_hints": []
  }
]
```

`current_rank: null` means the candidate is excluded from ranked results
(vetoed or suppressed).

## 4. Files

| File | Change |
|---|---|
| `mnemos/governance/counterfactuals.py` | New — pure functions |
| `mnemos/governance/governor.py` | New `effective_min_score(profile)` method |
| `service/app.py` | `counterfactuals` added to `governance_explain` block |
| `tests/test_counterfactuals.py` | 7 unit tests (achievable/unachievable, both veto classes, contradiction loser, top-n) |
| `tests/test_governance_explainability.py` | `_StubGovernor` tracks the new Governor method |

## 5. Boundaries

- Statements are *ceteris paribus* — one modifier varied at a time. Joint
  multi-modifier counterfactuals are out of scope (combinatorial, and single
  -factor statements are what policy tuning needs).
- "Tie rank 1" is exact equality; surpassing requires any epsilon above the
  reported value.
- ~~Freshness counterfactuals are expressed at modifier level~~ **Resolved
  (June 10, 2026): Modifier-to-Age Inversion.** When the caller supplies
  `created_at_by_id` (the service passes it from the result set) the freshness
  counterfactual is restated in age terms by inverting the read-path decay:
  `max_age = half_life × log2(1 / required_modifier)`, using the effective
  per-profile half-life (`Governor.effective_freshness_half_life(profile)`).
  Output adds `current_age_days` and `max_age_days_for_rank1`; the statement
  becomes e.g. *"This result lost ground to freshness decay (200 days old);
  it would tie rank 1 if it were younger than 105 days."* Falls back to the
  modifier-level statement when the timestamp is missing or unparseable —
  including enforced-mode suppressed candidates, whose engrams are removed
  from the result set and therefore have no timestamp available at the
  explain site.
