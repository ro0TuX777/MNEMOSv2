# Evidence Admission and Budgeting R1 Design Note

## Status

R1 is a bounded enforcement evaluation design. It does not begin enforcement implementation, formal evaluation, or policy tuning.

Required controls:

- NO_DEFAULT_ENFORCEMENT
- NO_HYBRID_ROUTE_ENFORCEMENT_IN_R1
- NORMAL_RETRIEVAL_FALLBACK
- OPT_IN_KILL_SWITCH_ONLY
- READ_ONLY_POLICY_INPUTS

## Objective

R1 tests whether a small set of R0 recommendation outputs can be enforced behind an explicit opt-in kill switch without losing provenance, current-state correctness, or acceptable retrieval quality.

R1 must not claim that evidence admission generally improves route choice, cost, or quality. It may only claim bounded behavior for the preregistered corpus, frozen formal pack, and explicitly enabled enforcement labels.

## Enforcement Boundary

Allowed enforced route labels:

- CUE_ONLY_LOOKUP
- CACHE_ONLY
- BOUNDED_SEMANTIC_RETRIEVAL
- ABSTAIN_OR_REQUEST_SCOPE
- NORMAL_RETRIEVAL_FALLBACK

Forbidden enforced route labels:

- HYBRID_RETRIEVAL
- ASSOCIATIVE_EXPANSION_ELIGIBLE
- graph_hybrid_experimental
- derived_facts
- summary_inclusion
- governance_override

R1 enforcement must be additive to existing request controls. If the opt-in kill switch is absent, false, globally disabled, malformed, or unsupported, retrieval behavior remains behaviorally identical for stable retrieval and response-contract fields.

## Evidence Separation

Pre-retrieval recommendations remain separate from post-retrieval sufficiency assessment. Enforcement may consume only the preregistered recommendation fields for the allowed labels above. Post-retrieval sufficiency records may audit outcomes, but must not retroactively justify the route chosen for a request.

## Fallback and Abstention

NORMAL_RETRIEVAL_FALLBACK is mandatory whenever an allowed lower-cost route cannot satisfy the formal pack's accepted evidence neighborhood, lineage, freshness, or abstention expectations.

ABSTAIN_OR_REQUEST_SCOPE is valid only when the query is preregistered as abstention-expected or when the accepted evidence neighborhood requires a missing or unavailable current-state lineage that normal retrieval cannot supply.

## Corpus and Pack Boundary

The R1 corpus manifest freezes a non-empty corpus before enforcement implementation or policy tuning. The development pack is diagnostic only and cannot support formal claims.

The formal pack template contains schema, validation rules, allowed labels, lineage requirements, fallback expectations, and non-scored examples only. The actual 50-query formal pack must be independently authored after the corpus and preregistration are committed.

## Non-Goals

R1 does not tune hybrid retrieval. R1 does not add new authority semantics. R1 does not train, fine-tune, or modify embeddings. R1 does not use development-pack outcomes as formal evidence.
