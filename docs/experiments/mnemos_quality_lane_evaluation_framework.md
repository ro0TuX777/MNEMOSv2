# MNEMOS Quality Lane Evaluation Framework

Date: 2026-06-27

Status: `MNEMOS_QUALITY_LANE_EVALUATION_FRAMEWORK_READY`

## Purpose

This document defines the standard scorecard for the next MNEMOS quality lane
and related follow-on agent-memory evaluations.

The framework separates:

- measured task outcomes;
- workflow efficiency signals;
- memory-quality signals; and
- retrieval-integrity controls used to interpret the results honestly.

The first three groups are the measured outcomes. The fourth group is the
interpretability record. It lets us determine whether an apparent win or loss
was caused by the memory system itself or by an uncontrolled retrieval
condition.

## Metric Groups

### 1. Task outcome

These metrics answer whether the work actually got done correctly.

| Metric | Meaning |
|---|---|
| `task_completion_rate` | Whether the assigned task was completed |
| `acceptance_test_pass_rate` | Fraction of required acceptance tests that passed |
| `time_to_passing_tests` | Elapsed time until the required test state first passed |
| `required_constraints_satisfied` | Whether stated red lines and constraints were preserved |
| `regressions_introduced` | New breakages or backward steps introduced during the run |

### 2. Workflow efficiency

These metrics answer how costly the work was to complete.

| Metric | Meaning |
|---|---|
| `estimated_tokens` | Estimated model input/output token usage |
| `tool_calls` | Total tool or MCP calls made during the run |
| `failed_test_count` | Count of failed test executions before success |
| `wrong_turn_count` | Count of meaningful missteps, reversals, or avoidable detours |
| `files_changed` / `churn` | Breadth of edits and approximate modification volume |
| `rework_after_first_implementation` | Additional work required after the first working version |

### 3. Memory quality

These metrics answer whether retrieved memory was relevant, trustworthy, and
actually useful in the task.

| Metric | Meaning |
|---|---|
| `correct_source_or_decision_neighborhood` | Whether retrieval landed in the correct evidence neighborhood |
| `provenance_retained` | Whether cited memory preserved source and traceability |
| `irrelevant_context_rate` | Frequency of distracting or unrelated retrieved context |
| `retrieval_precision` | Share of surfaced memory that was meaningfully relevant |
| `abstentions_correct` | Correct abstentions on unrelated or unsupported queries |
| `abstentions_missed` | Cases where the system should have abstained but did not |
| `false_abstention_rate` | Cases where the system abstained despite relevant evidence existing |
| `retrieved_context_usefulness` | Whether retrieved context materially influenced a justified action |

### 4. Retrieval-integrity controls

These are not outcome metrics. They are the run conditions required to make the
results reproducible and interpretable.

| Control | Meaning |
|---|---|
| `seed_snapshot` | Corpus identity used for the run |
| `executed_route_fingerprint` | Truthful retrieval-path fingerprint for the delivered result |
| `cache_state` | Whether the run was cold or warm, and what was observed |
| `duplicate_suppression_count` | Number of candidates removed by duplicate hygiene |
| `retrieval_profile_or_path` | Direct service vs MCP path, plus retrieval-profile details |

## Why The Two New Metrics Matter

Two additions are especially important for the next lane:

- `false_abstention_rate`
- `retrieved_context_usefulness`

A system that abstains safely but fails to retrieve useful context is not
delivering practical value. A system that retrieves the right documents but
does not change the agent's justified decisions is also not delivering much
real benefit. We need both measures.

## Recording Guidance

Each future quality-lane artifact should record:

1. the task and evaluation scope;
2. the metric values for the three measured-outcome groups;
3. the retrieval-integrity controls for every run leg;
4. a short interpretation of whether memory changed the outcome, cost, or
   decision quality; and
5. explicit claim boundaries.

The current paired AI-developer comparison runner emits this structure directly
through:

- `tools/compare_ai_dev_memory_trials.py`
- `benchmarks/results/ai_dev_memory_quality_lane_result_template.json`

## Claim Boundary

This framework does not itself prove that MNEMOS improves agent performance.
It makes future evaluations more credible by ensuring that performance,
memory-quality, and retrieval-condition evidence are recorded together.
