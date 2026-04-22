# Governance Validation Pack v1

**Status:** Complete
**Date:** 2026-03-30
**Test file:** `tests/test_governance_drift_validation.py`
**Result:** 33/33 passing

This is a formal internal milestone. It is not a unit test suite for correctness; it is a behavioral proof artifact. Each scenario is a named claim about how the governance layer behaves under stress, and the test is the evidence for that claim.

---

## Milestone Summary

| Claim | Evidence |
|---|---|
| Reinforcement converges — used memories rise, unused memories fall | Scenarios A, B (deterministic monotonic drift) |
| Contradiction adjudication separates winner from loser over time | Scenario C (growing separation, 10 cycles) |
| Ignored stale memories decay without backend intervention | Scenario D (30-cycle utility decay from 0.8 to 0.5) |
| Word-set floor prevents zero-content false positives | Scenario E (sub-3-char content always IGNORED) |
| Overlap threshold is a tunable precision/recall dial | Scenarios F, J (boundary documentation) |
| Contradiction state beats overlap signal | Scenario H (loser stays CONTRADICTED despite lexical match) |
| Known false-positive cases are documented, not hidden | Scenarios G, I (explicit behavior documentation) |

---

## Scenario Findings

### Scenario A — Repeated Winner Drift

**What it proves:** A memory cited in every answer will have its utility and trust converge to the ceiling (1.0) within a bounded number of cycles. Stability also rises monotonically.

**Failure mode protected against:** Score inflation without ceiling — a runaway winner that accumulates unbounded advantage. The clamp at 1.0 prevents this. After cycle 3, utility and trust are both 1.0 and remain there.

**Cycles to ceiling at default deltas:** utility in 1 cycle (starts 0.95), trust in 3 cycles (starts 0.93 → 0.95 → 0.97 → 0.99 → 1.0), stability in ~25 cycles (starts 0.5, +0.02/cycle).

---

### Scenario B — Repeated Distractor Drift

**What it proves:** A memory never cited across 20 reflect cycles loses 0.01 utility per cycle. Starting from 0.5, it reaches 0.3 after 20 cycles. Trust is unchanged (trust_ignored = 0.0 by design).

**Failure mode protected against:** Utility score stagnation — memories that are retrieved but never useful maintaining the same score as memories that are consistently cited. The ignore penalty creates natural separation.

**Design note:** Trust is intentionally not penalized for IGNORED. Only active contradiction evidence penalizes trust. This is the precision-first design choice.

---

### Scenario C — Contradiction Winner vs Loser Separation

**What it proves:** When a contradiction winner is cited repeatedly and its loser is never cited, their utility scores diverge at 0.08 per cycle. After 10 cycles, winner is at 1.0 and loser is at 0.2 — an 0.8 separation.

**Failure mode protected against:** Contradiction resolution without reinforcement — the winner and loser could have the same utility after resolution if reflect cycles are not applied. The combined effect of the 0.25 loser modifier (from `ContradictionPolicy`) and utility decay creates strong and growing separation.

---

### Scenario D — Stale Value Ignored Repeatedly

**What it proves:** A memory whose content has no lexical overlap with any answer in 30 reflect cycles loses 0.01 utility per cycle (−0.30 total), dropping from 0.8 to 0.5. It never gains.

**Failure mode protected against:** Stale memory retention — a memory that was once relevant but is now superseded maintaining high utility because it happens to be retrieved. The ignore penalty is the only in-process decay mechanism before Wave 4 runs.

**Boundary:** At 30 cycles, utility is 0.5, well above the prune threshold (0.05). Wave 4 decay runner is needed to push genuinely obsolete memories to the stale/prune-candidate state without waiting for 100+ reflect cycles.

---

### Scenario E — Short Generic vs Grounded Content

**What it proves:** Content composed entirely of tokens shorter than 3 characters produces an empty word set and always gets an overlap score of 0.0, regardless of how many of those words appear in the answer. A memory matching on zero tokens is classified IGNORED every cycle.

**Failure mode protected against:** Short-content false positives — a memory with content like `"it on"` matching answers that contain `"it"` and `"on"`, receiving USED classification and unearned utility gains.

**Implementation boundary documented:** The 3-character floor is in `UsageDetector._word_set()`. It is a hard floor, not a threshold.

---

### Scenario F — Long Answer Multi-Candidate Surface Overlap

**What it proves:** A candidate with partial lexical overlap (`"business quarterly forecast budget goals"` — 2/5 words in the answer) is classified USED at threshold=0.15 (40% > 15%) and IGNORED at threshold=0.50 (40% < 50%). The threshold is a meaningful precision dial.

**Failure mode protected against:** Mass false-positive USED classification in long answers. A 300-word answer that shares individual words with many candidates will classify all of them as USED at low thresholds. Raising the threshold above the expected incidental overlap rate restores precision.

**Threshold guidance:** At default=0.15, expect false positives when candidate content is short relative to a long answer. For high-precision deployments, 0.30–0.50 is a reasonable starting range. Document your corpus-specific tuning in benchmark runs.

---

### Scenario G — Short Generic Memory (Known False Positive)

**What it proves:** A 2-token memory like `"system status"` achieves either 0% or 100% word overlap with any answer that contains both tokens. At default threshold=0.15, 100% > 15% → USED. This is a known precision gap.

**This is not a bug — it is a documented system boundary.** The lexical overlap detector is recall-oriented by design. Short, generic content at the tail of a memory corpus is the operator's responsibility to prevent (content length validation upstream, or threshold tuning).

**Mitigation path:** Minimum content token count enforced at write-time (Wave 4+), or raise threshold above 0.50.

---

### Scenario H — Contradiction Loser with Phrasing Overlap

**What it proves:** Signal priority ordering holds under pressure. A contradiction loser whose content shares 3 words with the answer (`"project"`, `"active"`, `"systems"`) is classified CONTRADICTED, not USED. The contradiction check runs before the overlap check.

**Failure mode protected against:** Contradiction loser receiving USED classification because it happens to share words with the winning answer. Without priority ordering, contradicted memories could accumulate positive utility reinforcement while their contradiction state causes a 0.25 modifier penalty — an incoherent combination.

---

### Scenario I — Semantic Distractor (Known False Positive)

**What it proves:** The overlap detector is purely lexical. A topically irrelevant memory (`"eiffel tower france server infrastructure"`) achieves 75% overlap (3/4 tokens: `tower`, `france`, `eiffel`) with an answer about Paris travel, and is classified USED at threshold=0.15.

**This is the primary known precision limitation of the current detector.** Proper nouns, place names, and entity names shared between topically unrelated memories and answers will fire the overlap signal.

**Mitigation path:** Semantic re-ranking (Wave 5+) or proper-noun filtering in `_word_set()`. Not addressed in Wave 4.

---

### Scenario J — Threshold Sensitivity

**What it proves:** Documents the classification outcome across a range of thresholds (0.10 → 0.75) for a high-overlap candidate (`"system deployed production"` against an answer about system deployment). High-overlap candidates stay USED at all thresholds because they are genuinely matched content, not incidental overlap.

**Use:** Threshold tuning reference. Run this scenario against your own candidate corpus to find the threshold that gives the desired precision/recall trade-off.

---

## What Remains Untested

These are known gaps, not oversights. They are documented here to scope Wave 5+ work.

| Gap | Risk | Path to coverage |
|---|---|---|
| Semantic relevance vs lexical overlap | The lexical detector attributes false positives (Scenarios G, I) | Semantic re-ranking layer in reflect path (Wave 5) |
| Long-horizon decay | 30 cycles of IGNORED gives −0.30 utility; Wave 4 decay runner handles calendar-based decay | `test_hygiene_decay.py` (Wave 4) |
| Contradiction sweep coverage | Contradictions only detected at read-time if both candidates are in the same result set | `test_hygiene_contradiction_sweep.py` (Wave 4) |
| Multi-session trust recovery | A penalized memory that starts being cited again has no tested recovery path | New drift scenario after Wave 4 |
| Advisory vs enforced mode drift divergence | All drift scenarios run in advisory mode; enforced mode suppresses losers and may change which memories accumulate reflect cycles | Enforced-mode variants of Scenarios A–D |
| Concurrent reflect cycles | `ReflectMetrics` is thread-safe; `GovernanceMeta` mutation is not tested under concurrent access | Threaded stress test (Wave 5) |

---

## Competitive Significance

This validation pack is the evidence layer for the following product claim:

> MNEMOS does not just retrieve memory — it maintains memory quality over time through reinforcement validation, contradiction adjudication, and hygiene state transitions.

Each scenario above maps to a specific failure mode that an unvalidated retrieval system would silently produce. The test suite proves that MNEMOS detects and handles those failure modes deterministically, with tunable parameters and explicit documentation of known boundaries.

This is what "trustworthy memory" means as an engineering claim, not a marketing claim.

---

*Wave 4 implementation should be gated on this document remaining accurate. If hygiene runner behavior changes the drift patterns shown above, update the scenarios and re-run.*
