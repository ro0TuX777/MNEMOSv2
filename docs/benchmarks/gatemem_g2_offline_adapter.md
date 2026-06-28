# GateMem G2 Offline Governed Retrieval/Disclosure Adapter

Date: 2026-06-24

Status: `GATEMEM_G2_BASELINE_COMPLETE`

```text
OFFLINE_ADAPTER_MECHANICS_VERIFIED
PROVENANCE_INTEGRITY_VERIFIED
DISCLOSURE_POLICY_INADEQUATE_FOR_PRODUCTION
ACTIVE_FORGETTING_NOT_EVALUATED
GATEMEM_G2_OFFLINE_BASELINE_FROZEN
MEDICAL_DOMAIN_ONLY
NO_POLICY_TUNING_ON_MEDICAL_RESULTS
```

```text
GATEMEM_G2_OFFLINE_RETRIEVAL_DISCLOSURE_ADAPTER
OFFLINE_ONLY
CLEAN_G1_PROJECTIONS_ONLY
NO_RUNTIME_ROUTE_OR_AUTH_CHANGE
NO_HOSTED_JUDGE
NO_LEADERBOARD_SUBMISSION
DELETION_CASES_EXPLICITLY_REFUSED
```

## Outcome

G2 successfully connects the G1 injected probes to an original offline lexical
retriever and conservative disclosure policy. It produces real GateMem external
predictions while preserving the evaluator boundary and exact used-record
provenance.

This is a successful benchmark-baseline closeout, not a successful
governance-performance result. On the
medical domain, auxiliary utility is `0.2762`, privacy end-to-end leakage is
`0.4427`, and over-refusal is `0.5286`. These values demonstrate that the
metadata-poor G2 policy is not production-grade access control.

## Data flow

```text
external G1 clean projections
        |
        v
original deterministic lexical retrieval
        |
        v
conservative visible-input disclosure policy
        |
        v
G1 shadow observation + normalizer
        |
        v
frozen external predictions.jsonl
        |
        | evaluator-only join after freeze
        v
GateMem auxiliary rule scorer (no hosted judge)
        |
        v
aggregate MNEMOS report only
```

The adapter imports neither MNEMOS runtime code nor GateMem. The external
GateMem scorer is invoked only after predictions are frozen and cannot feed
annotations back into the adapter.

## Retrieval

Each projected turn becomes an ephemeral record with a deterministic ID:

```text
g2:{episode_id}:{turn_id}
```

Retrieval uses original standard-library tokenization, per-projection inverse
document frequency, query coverage, phrase/recency signals, and small visible
requester/adjacency bonuses. Defaults are:

```text
top_k = 8
minimum_score = 0.08
max_disclosed_records = 4
max_answer_characters = 2200
```

No embedding model, LLM, network provider, MNEMOS tier, or GateMem baseline is
used.

## Disclosure policy

The policy permits a retrieved record when at least one clean condition holds:

- the requester authored it;
- it is an adjacent agent reply to the requester;
- it visibly refers to the requester;
- its principal or role is explicitly allowed in G1 `permitted_metadata`.

Explicit denied principal/role metadata takes precedence. A query visibly
naming another principal constrains eligible records to that principal before
ordinary requester allowances are considered.

In the frozen medical G1 projections, `permitted_metadata` is empty. The policy
therefore cannot represent GateMem's detailed relationship entitlements. This
is the principal reason G2 cannot support production role-enforcement claims.

Denied lower-ranked candidates are filtered silently. `answer_redacted` is used
when the highest-ranked evidence is denied but eligible lower-ranked evidence
can still support a bounded response. If nothing is eligible, G2 refuses.

Only disclosed text enters `output.memory_audit.prompt_context.text`. This lets
GateMem's auxiliary scorer measure prompt-context leakage rather than receiving
an artificially empty audit surface.

## Evaluator isolation

The adapter input loader accepts only the exact G1 schema and verifies every
projection digest. Predictions contain no `query_type`, `attack_type`,
`expected_action`, `judge_spec`, or `leak_targets` key.

After predictions are frozen, a separate compiler aggregates GateMem rule-score
rows. Row-level scores remain external. The report contains aggregate counts
and rates only. GateMem's stub is never used; no hosted judge is run.

## Medical result

| Dimension | Result |
|---|---:|
| Checkpoints | 579 |
| Utility checkpoints | 210 |
| Privacy checkpoints | 192 |
| Deletion checkpoints | 177 |
| `answer` / `answer_redacted` / `refuse` | 218 / 90 / 271 |
| Authorized retrieval utility (auxiliary) | 0.2762 (58/210) |
| Privacy action correctness | 0.2500 |
| Expected-refusal correctness | 0.2743 |
| Privacy answer/context/e2e leakage | 0.4427 / 0.4427 / 0.4427 |
| Expected-redaction correctness | 0.0000 (0/17) |
| Cross-principal candidates blocked | 0.9165 (1701/1856) |
| Over-refusal | 0.5286 |
| Used-record provenance integrity | 1.0000 (579/579) |
| Deletion refusal consistency | 1.0000 (177/177) |
| Active-forgetting score | **NOT SCORED** |

The visible-only deletion detector identifies all 177 deletion checkpoints in
the evaluator-side audit and conservatively marks 32 non-deletion checkpoints.
Those false positives remain reflected in over-refusal.

Blocking 91.65% of cross-principal candidates does not imply disclosure safety:
the official auxiliary leak scan still finds 44.27% privacy leakage. Candidate
filter counts and content-level leakage measure different failure surfaces.

## Deletion treatment

Visible deletion or historical-reference queries with a prior deletion signal
are refused before retrieval. Their prompt context is empty and no record IDs
are disclosed.

The official auxiliary scorer reports zero deletion leakage for these
refusals, but G2 explicitly excludes that value. A refusal does not demonstrate
that data was deleted, cascaded, made unrecoverable, or protected against later
confirmation. G2 reports refusal consistency only.

## External artifacts

All row-level artifacts remain outside MNEMOS and the pristine GateMem clone:

```text
G:\MNEMOS-research\gatemem_g2_artifacts\medical_predictions.jsonl
G:\MNEMOS-research\gatemem_g2_artifacts\medical_diagnostics.jsonl
G:\MNEMOS-research\gatemem_g2_artifacts\medical_run_summary.json
G:\MNEMOS-research\gatemem_g2_artifacts\medical_rule_score\
```

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| predictions | 1,040,393 | `be06dd67a72fc61b8abb21a423acc5830a149dcc7c7665ca2333d03b77cf9c73` |
| diagnostics | 779,259 | `a81d43bfd103353c43b81f9b7b059542a7ad2411aba045ae192d85d70a47729f` |
| run summary | 807 | `71ed33f77cbbd06bf303a5030d60068078ead4c99ac8e12b7e817dddabdd31bc` |
| evaluator scores | 404,912 | `028cd02633ec5c1ccedc5374da3d2581c1529bcc9243d57673f7a26bfba5fa38` |
| evaluator summary | 1,820 | `979c0f3691673e54c1edfa11e15d8d30b100bbfd3e9e1ed2fa48d2d940e74606` |

## Reproduction

```powershell
python tools/run_gatemem_g2_offline.py `
  --projections G:\MNEMOS-research\gatemem_g1_artifacts\medical_clean_projections.jsonl `
  --predictions G:\MNEMOS-research\gatemem_g2_artifacts\medical_predictions.jsonl `
  --diagnostics G:\MNEMOS-research\gatemem_g2_artifacts\medical_diagnostics.jsonl `
  --run-summary G:\MNEMOS-research\gatemem_g2_artifacts\medical_run_summary.json

G:\MNEMOS-research\gatemem_upstream\.venv\Scripts\python.exe `
  G:\MNEMOS-research\gatemem_upstream\bench\scripts\score_predictions.py `
  --data_dir G:\MNEMOS-research\gatemem_upstream\bench\data\medical `
  --predictions G:\MNEMOS-research\gatemem_g2_artifacts\medical_predictions.jsonl `
  --out_dir G:\MNEMOS-research\gatemem_g2_artifacts\medical_rule_score

python tools/compile_gatemem_g2_report.py `
  --run-summary G:\MNEMOS-research\gatemem_g2_artifacts\medical_run_summary.json `
  --diagnostics G:\MNEMOS-research\gatemem_g2_artifacts\medical_diagnostics.jsonl `
  --scores G:\MNEMOS-research\gatemem_g2_artifacts\medical_rule_score\scores.jsonl `
  --official-summary G:\MNEMOS-research\gatemem_g2_artifacts\medical_rule_score\summary.json `
  --output-json benchmarks/results/gatemem_g2_offline_report.json `
  --output-md benchmarks/results/gatemem_g2_offline_report.md
```

## Advancement boundary

G2 does not authorize runtime integration, role/auth changes, hosted judging,
leaderboard submission, or deletion engineering. Any true governed deletion
lifecycle remains a separate ADR and authorization.
