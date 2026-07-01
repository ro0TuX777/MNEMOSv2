# Evidence Admission and Budgeting R1 Preregistration

## Status

This preregistration defines the R1 bounded enforcement evaluation before enforcement implementation or policy tuning.

No formal R1 evaluation may begin until:

- the non-empty corpus manifest is committed;
- the formal pack template is given to an independent_non_implementation_author;
- the independently authored formal pack is returned, validated, frozen, and hashed;
- the implementation team has not modified R1 policy, thresholds, implementation, or development-pack results in response to the formal queries.

## Hypothesis

Behind an explicit opt-in kill switch, bounded enforcement of preregistered evidence admission recommendations can reduce retrieval scope on eligible requests while preserving provenance, current-state correctness, and acceptable retrieval quality.

## Primary Non-Inferiority Criterion

The enforced R1 condition must show no more than 2 percentage-point drop in accepted-evidence coverage against normal retrieval on the frozen 50-query formal pack.

Accepted-evidence coverage means the response cites at least one member of the query's accepted evidence neighborhood and satisfies the minimum required source lineage declared by the independent author.

## Secondary Safety Criteria

R1 has zero tolerance for these safety failures:

- zero accepted-answer source-lineage omissions;
- zero unexplained current-state errors;
- zero unexplained supersession errors;
- zero enforcement when fallback is required;
- zero global-gate or request-flag violations.

Any observed violation blocks a positive R1 enforcement claim unless it is fully explained as an evaluation artifact and rerun under the frozen protocol without recurrence.

## Required Evaluation Conditions

Formal evaluation must keep these result sets separate:

- normal retrieval baseline;
- R1 recommendation shadow only;
- R1 bounded enforcement with opt-in kill switch enabled.

Direct-runtime diagnostic evidence and HTTP-service evidence must not be aggregated into one metric.

## Formal Pack Requirements

The formal pack must contain at least 50 scored queries and must be authored by an independent_non_implementation_author. It must include accepted evidence neighborhoods, lineage requirements, fallback expectations, abstention expectations, and lower-cost route eligibility for each scored query.

The fresh verification pack must contain at least 20 newly authored scored queries and must not reuse formal-pack query text.

## Corpus Freeze Requirements

The corpus must be frozen before enforcement implementation or policy tuning. The manifest records document hashes, chunking configuration, embedding/profile configuration, collection snapshot, service revision, corpus curator, and freeze date.

Any corpus modification after formal-pack authorship invalidates the affected formal evidence unless the corpus, formal pack, and preregistration are refrozen together.

## Stop Conditions

Stop and report failure without a positive R1 claim if:

- the service revision cannot be established for HTTP evaluation;
- the global enforcement gate or request flag is not honored;
- enforced responses lose required lineage or provenance;
- fallback fails when required;
- formal accepted-evidence coverage drops by more than the preregistered non-inferiority margin;
- development-pack outcomes are mixed into formal metrics.

## Claim Boundary

A successful R1 may claim only bounded opt-in enforcement behavior on the frozen corpus and independently authored packs. It may not claim general retrieval-quality improvement, global cost reduction, or safety on unseen corpora.
