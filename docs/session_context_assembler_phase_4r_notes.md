# Session Context Assembler — Phase 4R Selector S1

Status: `PASS — PHASE_5_HUMAN_REVIEW_DESIGN_ONLY`.

Phase 4R adds one bounded offline selector and leaves the frozen R1 corpus,
manifest, gold labels, measurement rules, C0 selector, and all MNEMOS runtime
paths unchanged.

## S1 policy

S1 selects atomic episodes in this order:

1. eligible prior-decision artifacts;
2. eligible unresolved or mixed contradiction artifacts;
3. eligible source-linked evidence;
4. task-relevant supporting episodes;
5. optional semantic fill.

Within a tier, relevance is computed from artifact-bearing turns when
available. This prevents a nearby, lexically attractive follow-up from making
an episode containing the wrong decision outrank the correct decision artifact.
Eligibility gates both inline IDs and structured links.

The implementation reads only runtime-available task text, eligible session
turns, extracted decision/source artifacts, structured source links,
contradiction-language signals, and deterministic session order. It does not
read R1 scoring-only fields. This boundary is AST-tested.

## Budget and abstention behavior

Mandatory candidates are allocated before semantic fill. Episodes remain
atomic and C1 never exceeds the same per-case R1 budget used by B and C0. If an
additional mandatory runtime-visible candidate cannot fit, the package emits:

```text
context_budget_insufficient = true
omitted_required_artifact_types = [...]
selection_abstention_reason = "..."
```

Five cases conservatively emit this partial-abstention signal. In each, the
R1-scored required artifact is retained; the omitted candidate is an
additional runtime-visible decision or contradiction candidate. No required
artifact is silently omitted.

## Frozen-R1 comparison (seed 7)

| Condition | Decision retention | Source recall | Contradiction awareness | Token reduction |
|---|---:|---:|---:|---:|
| A full history | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| B sliding window | 0.4118 | 0.1839 | 0.1250 | 0.5043 |
| C0 relevance-led | 0.1176 | 0.1379 | 0.0000 | 0.6126 |
| C1 selector S1 | 1.0000 | 1.0000 | 1.0000 | 0.4513 |

C1 also records zero provenance loss, 1.0 synthetic-context label coverage,
1.0 source and parent-Engram lineage preservation, zero unauthorized writes or
governance mutations, and budget compliance on 29/29 cases. It retains the
required decision on all three R1 adversarial cases and matches the authored
category on all eight contradiction cases.

## Outcome

All Phase 4R automated advancement requirements pass. This authorizes Phase 5
human-review design only.

```text
SESSION_CONTEXT_ASSEMBLER_PHASE_4R_S1_PASS
PHASE_5_HUMAN_REVIEW_DESIGN_AUTHORIZED
NO_HUMAN_VALUE_CLAIM
NO_CONSUMER_RUNTIME_INTEGRATION
NO_PRODUCTION_USE
NO_MEMORY_OR_GOVERNANCE_MUTATION
```

The replay is structural and fixture-local. Human review has not run, and this
result does not authorize consumer runtime integration, production routing, durable-memory
writes, retrieval-ranking changes, or any governance/promotion change.
