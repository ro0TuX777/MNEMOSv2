# Phase 12B — EchoFrame Packet DSL / Format Benchmark

## Objective
With Phase 12A confirming that **Layout D (Top-and-Bottom Constraints)** is the best placement candidate, Phase 12B will hold the Layout D ordering constant and test packet syntax/format variants.

## Core Research Question
Given Layout D placement, which packet format gives the best balance of token efficiency, model readability, source fidelity, governance retention, and parser reliability?

## Scope
**In scope:**
- Build format renderers adhering to Layout D positioning.
- Test against live local endpoints.
- Evaluate token/latency overhead for each format syntax.

**Out of scope:**
- Altering the placement ordering (Phase 12A is locked).

## Candidate Formats
1. **Current compact Markdown tags:** Baseline
2. **Layout D with current tags:** Placement winner baseline
3. **Ultra-compact tags (G/F/E/C):** Token-minimal test
4. **YAML-lite:** Structured but less verbose than JSON
5. **Minified JSON:** Control case from prior JSON failure
6. **Markdown table:** Readability/source-locality test
7. **TOON-inspired rows:** Token-oriented structured format test
8. **Source-table + fact rows:** Citation fidelity test

## Decision Criteria
The rerun should only recommend `SHADOW_FORMAT_VARIANT` or `KEEP_CURRENT_TAGGED_D` if the live model shows:
- **PASS:** zero source pointer loss
- **PASS:** zero protected numeric/date span loss
- **PASS:** zero negation/exception regression
- **PASS:** zero contradiction suppression
- **PASS:** zero evidence-gap suppression
- **PASS:** zero governance warning suppression
- **PASS:** improved or equal composite score versus baseline
- **PASS:** token and latency overhead remain within threshold

## Potential Decisions
- `KEEP_CURRENT_TAGGED_D`
- `SHADOW_FORMAT_VARIANT_<X>`
- `NO_CHANGE`
- `FAIL_FORMAT_SAFETY_REGRESSION`
