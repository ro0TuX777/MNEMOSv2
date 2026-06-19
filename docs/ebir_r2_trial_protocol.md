# EBIR-R2 Shadow Burn-In And Human Review Value Trial Protocol

Date: 2026-06-19

Status: **Pilot complete. Packet wording, response schema, compiler behavior,
and scoring fields are frozen for full R2 setup. Full reviewer trial remains
pending full-corpus preflight and distribution.**

## Boundary

EBIR-R2 is a human-review value trial around existing EBIR-R1 shadow behavior.
It is not a new reconciliation algorithm, retrieval feature, authority layer,
promotion path, Context Atlas runtime, or Graph Tier change.

The R2 harness may generate blinded offline reviewer packets from synthetic or
sanitized truthsets. It must not write to live MNEMOS memory, retrieval indexes,
governance state, EBIR promotion state, Resolution Engrams, Context Atlas
artifacts, or production routes.

Promotion remains:

```text
blocked_from_authoritative_resolution_promotion
```

## Reviewer Task Schema

Each reviewer-facing packet asks a reviewer to inspect the same parent evidence
under one of three blinded conditions:

- raw evidence only
- one-pass reconciliation candidate
- EBIR refinement candidate

Reviewer-facing packets must use the same schema across conditions:

```json
{
  "packet_id": "blind packet id",
  "reviewer_id": "reviewer id",
  "case_blind_id": "blind case id",
  "condition_blind_id": "blind condition id",
  "task": {},
  "parent_evidence": [],
  "candidate": {},
  "reviewer_questions": [],
  "scoring_rubric": {}
}
```

Condition labels, gold labels, expected outcomes, EBIR internals, packet hashes,
pass records, critique traces, and promotion metadata must be absent from
reviewer-facing packets.

## Gold Labels And Expected Outcomes

Gold labels live only in the frozen truthset and admin manifest. They include:

- expected resolution state
- expected resolved value, if any
- expected abstention or escalation behavior
- required parent evidence IDs
- prohibited unsupported claims
- expected reviewer-relevant critique categories

Gold labels must never be copied into reviewer-facing packets.

## Conditions

The harness builds three conditions for each case:

| Condition | Meaning |
|---|---|
| `raw_evidence` | Parent evidence only; no synthetic candidate supplied. |
| `one_pass_reconciliation` | Existing one-pass `ReconciliationRunner` dry-run output normalized for review. |
| `ebir_refinement` | Existing EBIR-R1 `RepFusionRefiner` final shadow candidate normalized for review. |

Parent evidence must be byte-identical after normalization across all three
conditions for the same underlying case.

## Blinding Rules

- Reviewer-facing packet IDs are deterministic from case ID, condition, seed, and
  reviewer assignment.
- Reviewer-facing packets include blind case and condition identifiers only.
- Reviewer-facing packets do not include condition keys or labels.
- EBIR-specific internals are removed.
- Gold labels and expected outcomes are removed.
- Admin manifests may retain the condition mapping for later scoring.

## Assignment Rules

- Each underlying case produces exactly three reviewer-facing packets.
- No reviewer may receive the same underlying case more than once.
- Assignment must be deterministic for a fixed seed.
- Condition exposure must be balanced across reviewers.
- The first preflight slice requires at least three reviewers.
- Reviewer identities and assignment mappings must remain separate from analysis
  exports. Scoring and analysis operate on pseudonymous reviewer IDs so
  reviewer-level effects can be measured without unnecessarily exposing
  identities.

## Reviewer Questions

Each packet asks reviewers to score:

- correct resolution
- correct abstention or escalation
- evidence-supported decision quality
- reviewer confidence calibration
- unsupported-claim detection
- clarity of remaining uncertainty

Each packet must also ask blinding-integrity questions:

- Did this packet appear to include a synthesized recommendation?
  - Yes
  - No
  - Unsure
- How confident are you in that impression?
  - 1 to 5

These questions measure condition recognition and must not reveal the study
condition, mention EBIR, or explain the trial arms. Their answers are used only
to determine whether perceived packet type correlates with scoring behavior.

Reviewer timing, latency, and token-cost analysis are recorded outside the first
preflight slice.

## Scoring Rubric

Scores use a 0-2 ordinal rubric unless otherwise specified:

| Score | Meaning |
|---|---|
| `0` | Incorrect, unsupported, misleading, or unsafe. |
| `1` | Partially correct but incomplete, unclear, or weakly supported. |
| `2` | Correct, evidence-supported, appropriately cautious, and reviewable. |

Reviewer confidence is recorded on a 1-5 scale.

## Acceptance Criteria

R2 may advance from preflight to a small pilot review only when:

- all cases have raw, one-pass, and EBIR variants
- parent evidence is identical across conditions
- gold labels are absent from reviewer-facing packets
- condition labels and EBIR-specific internals are removed
- no reviewer receives the same underlying case twice
- assignment is balanced across conditions
- packet schema is normalized across conditions
- EBIR remains shadow-only
- no retrieval, governance, ranking, promotion, or memory writes occur

The preflight gates have passed for the synthetic nine-packet pilot. The small
pilot review is authorized as an instrument test only. It must not be scored as
evidence that EBIR improves review outcomes.

Pilot review must check:

- unclear wording
- packet formatting imbalance
- obvious condition recognition
- reviewer confusion about resolution versus escalation
- missing or unusable evidence identifiers
- scoring-rubric ambiguity
- whether synthesized-recommendation impression correlates with scoring

A versioned pilot report must be frozen before changing wording, packet
normalization, assignment rules, or rubric text. At most one controlled protocol
revision may be made after the pilot before freezing the full R2 corpus.

Full blinded reviewer-trial acceptance is deferred until after pilot execution,
pilot report freeze, full corpus freeze, and scoring analysis.

## Post-Pilot Freeze

The small pilot completed as an instrument test. The full R2 setup must preserve
the pilot packet wording, response schema, compiler behavior, and scoring fields.
Any later change to those fields requires an intentional protocol version bump,
a change-control note, and regeneration of reviewer-facing packets.

Frozen components:

- reviewer-facing Markdown headings and response prompts
- response checkbox options and ordinal score ranges
- blinding-integrity questions
- compiler validation rules
- condition-masked three-arm structure
- restricted admin-only unblinding section

Full R2 preparation may expand the truthset and reviewer pool, but must not
change runtime behavior, retrieval behavior, EBIR-R1 behavior, governance,
promotion, Context Atlas, A1, Graph Tier, stores, routes, or production APIs.
