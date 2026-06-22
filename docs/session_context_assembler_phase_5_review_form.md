# Session Context Assembler — Phase 5 Review Form

Status: **blank design template; no review responses have been collected**.

Use only a coordinator-issued pseudonym. Do not enter your name, email,
employee ID, or other direct identifier. Review each masked package
independently before comparing them. Package codes do not identify how the
context was constructed.

## Header

```text
Reviewer pseudonym: REV-________________
Task code: TASK-___
```

## Complete once for each PACKAGE-1, PACKAGE-2, and PACKAGE-3

1. Did the package preserve the prior decision needed to answer the task?
   `yes / no / uncertain / not_applicable`
2. Were source references understandable? `1 / 2 / 3 / 4 / 5 / not_applicable`
3. Were source references sufficient to verify the answer?
   `1 / 2 / 3 / 4 / 5 / not_applicable`
4. Where relevant, was the contradiction represented correctly?
   `correct / incorrect / unclear / not_applicable`
5. Did the package omit material you considered necessary?
   `yes / no / uncertain`
6. Was synthetic context clearly distinguishable from source evidence?
   `1 / 2 / 3 / 4 / 5 / not_applicable`
7. If a budget-abstention warning appeared, was its meaning clear and
   appropriately cautious? `1 / 2 / 3 / 4 / 5`. If no warning appeared, use
   `not_applicable`.
8. Relative to the alternatives, did this package make the task
   `easier / harder / no_different`?
9. Give a short rationale (10–1000 characters), naming any confusing source,
   contradiction, omission, synthetic label, or warning.

Likert anchors: `1 = strongly no/unclear/insufficient`, `3 = mixed`,
`5 = strongly yes/clear/sufficient`.

## Response JSON template

Create one response file per reviewer/task. Repeat the review object for all
three package codes shown in the packet.

```json
{
  "schema": "sca_phase5_review_response_v1",
  "study_id": "SCA-PHASE5-R1-S1",
  "reviewer_pseudonym": "REV-EXAMPLE-01",
  "task_code": "TASK-001",
  "package_reviews": [
    {
      "condition_code": "PACKAGE-1",
      "prior_decision_preserved": "uncertain",
      "source_references_understandable": 3,
      "source_references_sufficient": 3,
      "contradiction_representation": "not_applicable",
      "necessary_material_omitted": "uncertain",
      "synthetic_context_distinguishable": "not_applicable",
      "abstention_clear_and_cautious": "not_applicable",
      "task_effect": "no_different",
      "short_rationale": "Replace this example with the human reviewer's own rationale."
    }
  ]
}
```

The example is a schema illustration only. It is not a reviewer response and
must never be compiled or cited as human-value evidence.
