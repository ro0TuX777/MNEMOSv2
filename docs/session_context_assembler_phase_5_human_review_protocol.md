# Session Context Assembler — Phase 5 Human-Review Protocol

Status: **design complete; study not run**.

This protocol evaluates whether bounded S1 context packages are understandable
and useful to humans. It is not production validation, authority validation,
or evidence that synthetic context is durable memory. It authorizes review of
the study materials only. It does not authorize recruiting reviewers, running
the study, consumer runtime integration, runtime routing, or any memory write.

## Research questions

For the same task and eligible session material, assess whether reviewers can:

- identify the prior decision needed for the task;
- understand and verify source references;
- correctly interpret resolved, unresolved, and mixed contradictions;
- notice materially missing context;
- distinguish governed synthetic context from source evidence;
- understand an S1 budget-abstention warning; and
- judge whether a package makes the task easier, harder, or no different.

## Design

The frozen study contains all 29 R1 tasks. Each task packet presents three
within-task packages:

- full eligible conversation history;
- naive sliding-window context; and
- S1 governed episode-selected context.

The task prompt and underlying eligible material are identical across the
three construction paths. Packages receive opaque `PACKAGE-1..3` codes. Their
order is deterministically shuffled per task using the frozen packet seed.
Reviewers are not told the mapping. The required S1 provenance labels remain
visible, so reviewers may recognize that one package is synthetic; they are
still not told its condition name or expected performance.

The design is within-task and condition-masked, not fully presentation-blind.
Reviewers first score each package independently, then make the comparative
ease judgment. Each task should receive at least three independent human
reviews. Work should be assigned in balanced blocks to limit fatigue; a
reviewer must not submit more than one response for the same task.

No study execution is authorized by this document. Reviewer recruitment,
consent language, compensation, scheduling, and the final assignment roster
require a separate approval before any packet is distributed.

## Frozen materials and custody

Generate and verify the packet set with:

```text
python tools/build_session_context_assembler_review_packets.py
python tools/build_session_context_assembler_review_packets.py --verify
```

The builder writes 29 reviewer-facing files under:

```text
benchmarks/review_packets/session_context_assembler_phase_5/packets/
```

It also writes `coordinator_manifest.json`, containing packet hashes and the
condition key. That file is coordinator-only and must never be included in the
reviewer distribution. Any changed packet, protocol, form, selector, corpus
hash, builder, seed, or condition key invalidates verification and requires an
explicitly versioned design revision. Do not edit a frozen packet in place.

## Reviewer-visible content

Ordinary history packages show only whitelisted turn ID, speaker, content, and
structured source links. S1 artifacts additionally show:

```text
synthetic_context
non_authoritative
non_promotable
parent_engram_ids
parent_source_ids
```

When present, the warning is visually and structurally distinct:

```text
context_budget_insufficient
omitted_required_artifact_types
selection_abstention_reason
```

The warning describes runtime-visible artifact types only. It does not reveal
benchmark requirements or expected answers.

Reviewer packets must never contain corpus case IDs, condition names, required
ID lists, authored contradiction categories, episode hints, irrelevant-turn
annotations, recall scores, gate outcomes, selection rationales, or expected
answers. Packet generation uses explicit output whitelists and is covered by
leakage tests.

## Reviewer identity and response handling

Use coordinator-issued pseudonyms matching `REV-[A-Z0-9-]`, stored separately
from any identity/contact roster. Response files must not contain names,
emails, phone numbers, employee IDs, or platform user IDs. Access to the
condition key and identity roster should be separated where staffing allows.

Reviewers complete the structured form in
`docs/session_context_assembler_phase_5_review_form.md`, including a short
rationale for every package. AI-generated, AI-completed, simulated, or
synthetic reviewer responses are prohibited as evidence of human value.

## Compilation and analysis plan

Packet generation and response compilation are separate programs. After an
independently authorized human run, compile only actual human response files:

```text
python tools/compile_session_context_assembler_review_responses.py \
  --manifest benchmarks/review_packets/session_context_assembler_phase_5/coordinator_manifest.json \
  --responses-dir <approved-human-response-directory> \
  --output <approved-compiled-output.json>
```

The compiler verifies every frozen packet hash before validating responses and
unblinding aggregate condition labels. It rejects malformed pseudonyms,
identity fields, missing package reviews, inappropriate abstention ratings,
and duplicate reviewer/task submissions. It never fabricates missing reviews.

Pre-registered summaries are per-condition categorical counts and mean Likert
ratings. Paired task-level comparisons should report effect sizes and
uncertainty, not only pooled means. Free-text rationales should be reviewed for
recurring comprehension failures and necessary omissions while retaining only
pseudonymous identifiers.

## Interpretation boundary

A completed design authorizes review of these materials only. A later human
study result must be separately reviewed against the Phase 6 decision gate.
Neither this design nor any eventual descriptive score authorizes durable
memory, authority, promotion, consumer integration, runtime routes, or production
use.
