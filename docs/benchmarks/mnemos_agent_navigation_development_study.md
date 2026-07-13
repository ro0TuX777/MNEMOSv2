# MNEMOS Agent Navigation Development Study

Date: 2026-06-26

Status: `MNEMOS_AGENT_NAVIGATION_DEVELOPMENT_STUDY_READY`

Classification:
`LOCAL_REPO_AGENT_ORIENTATION_AND_BOUNDARY_RECALL_DEVELOPMENT_STUDY`

## Purpose

This lane uses MNEMOS itself as a navigation maze for an agent. The study asks
whether a MNEMOS-backed memory layer helps an agent orient through repo-local
evidence, preserve claim boundaries, reject stale or unsupported memory, and
reach primary evidence with fewer wrong turns.

The first subject is Codex operating in this repository. The first maze is the
existing MNEMOS documentation, benchmark evidence, manifests, tools, and tests.

## Authorization

```text
MNEMOS_AGENT_NAVIGATION_DEVELOPMENT_STUDY_READY
LOCAL_REPO_TASKS_ONLY
AGENT_ORIENTATION_AND_BOUNDARY_RECALL_EVIDENCE_ONLY
NO_GATEMEM_REOPENING
NO_SEALED_EVALUATION
NO_GENERAL_MEMORY_CLAIM
NO_RUNTIME_MEMORY_INTEGRATION
```

This study may create task fixtures, memory-card fixtures, trial logs, scoring
tools, and local development results. It does not authorize any change to the
frozen GateMem G4 baseline or any claim that GateMem policy work has advanced.

## Method

Each navigation task is run in paired modes:

- `baseline_repo_search`: the agent may use normal repository inspection only.
- `mnemos_memory_assisted`: the agent may use repository inspection plus the
  supplied MNEMOS memory cards and retrieval notes.

Both modes must produce a trial log containing:

- task id and mode;
- files opened or searched;
- memory cards retrieved, if any;
- memories accepted, rejected, or treated as stale;
- evidence paths used in the final answer;
- final answer summary;
- claim-boundary decision;
- wrong turns or corrections; and
- elapsed turn or operation counts when available.

## Seed tasks

The initial task set focuses on GateMem because it is a dense, high-constraint
maze with frozen evidence, historical baselines, explicit blockers, and strong
red lines. This is intentional: it tests boundary recall without reopening the
GateMem lane.

Seed tasks are stored in:

```text
benchmarks/evaluation/mnemos_agent_navigation_protocol.json
```

The first tasks ask an agent to:

- identify the current GateMem status and external blocker;
- decide whether G4 policy logic may be modified;
- trace the frozen G4 evidence chain;
- separate development evidence from held-out or production claims; and
- reject stale memory that contradicts the current status document.

## Scoring

The development scorer checks structure and boundary safety first. It can also
score completed trial logs against required evidence and forbidden claims.

Metrics:

| Metric | Meaning |
|---|---|
| `required_evidence_recall` | Required primary evidence paths cited by the trial |
| `forbidden_claim_avoidance` | Forbidden overclaims absent from the final answer |
| `boundary_decision_match` | Trial boundary decision matches the expected decision |
| `memory_skepticism` | Stale or contradictory memory is rejected when supplied |
| `path_efficiency_observed` | File/search counts are present for later comparison |

The scorer does not decide whether MNEMOS generally improves agent memory. It
only records development evidence for these repo-local tasks.

## Next Quality Lane

The next MNEMOS quality lane should use the canonical evaluation structure in:

`docs/experiments/mnemos_quality_lane_evaluation_framework.md`

That framework extends the original navigation-study metrics with:

- task-outcome measures;
- workflow-efficiency measures;
- memory-quality measures, including false abstention and retrieved-context
  usefulness; and
- retrieval-integrity controls such as seed snapshot, executed-route
  fingerprint, cache state, duplicate suppression, and retrieval path.

This keeps the measured outcomes separate from the run-interpretability record.

## Claim Boundary

This study may support narrow development observations about agent orientation
inside MNEMOS repo tasks. It is not:

- a GateMem benchmark run;
- a GateMem G4 development iteration;
- a sealed evaluation;
- authorization security evidence;
- production readiness evidence;
- durable deletion evidence; or
- a general claim that MNEMOS improves all agent memory behavior.

Any future public or generalizable claim requires a separately designed
evaluation protocol with independent task custody and preregistered scoring.
