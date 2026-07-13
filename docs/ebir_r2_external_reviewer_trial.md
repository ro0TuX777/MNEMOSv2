# EBIR-R2 External Reviewer Trial Kit

This guide is for external developers or evaluators who want to run the
independent EBIR-R2 human review trial without learning MNEMOS internals.

## Boundary

EBIR-R2 remains an offline, shadow-only evaluation. The trial kit does not write
MNEMOS memory, alter retrieval, change governance state, promote Resolution
Engrams, or change production APIs.

Human-value claims remain blocked until independent blinded reviewers complete
the full R2 protocol and scoring analysis.

## Quick Start

Prepare reviewer and admin bundles:

```bash
python tools/ebir_r2_trial.py prepare --full --output-dir eval_results/ebir_r2_external
```

Validate reviewer/admin separation:

```bash
python tools/ebir_r2_trial.py validate --output-dir eval_results/ebir_r2_external --fail-on-gate
```

Give reviewers only:

```text
eval_results/ebir_r2_external/reviewer_bundle/
```

Do not give reviewers:

```text
eval_results/ebir_r2_external/admin_bundle/
```

After completed responses are returned, place them in:

```text
eval_results/ebir_r2_external/responses/
```

Compile the frozen response set:

```bash
python tools/ebir_r2_trial.py compile --output-dir eval_results/ebir_r2_external --fail-on-gate
```

Score after all responses are frozen:

```bash
python tools/ebir_r2_trial.py score --output-dir eval_results/ebir_r2_external --fail-on-gate
```

## Output Layout

```text
eval_results/ebir_r2_external/
  reviewer_bundle/
    README_FOR_REVIEWERS.md
    RETURN_CHECKLIST.md
    packets/
      reviewer_R01.md
      reviewer_R02.md
      reviewer_R03.md
    manifest_public.json
  admin_bundle/
    assignment_manifest.json
    preflight_report.json
    trial_admin_notes.md
  responses/
    reviewer_R01_completed.md
    reviewer_R02_completed.md
    reviewer_R03_completed.md
  compiled/
    ebir_r2_external_report.md
  scoring/
    ebir_r2_gold_scoring.json
    ebir_r2_gold_scoring.md
```

## Reviewer Rules

Reviewers should:

- use only their assigned Markdown file;
- complete every checkbox and free-text field;
- avoid outside knowledge;
- avoid inferring condition identity;
- return only their completed Markdown response.

Reviewers must not receive:

- the frozen truthset;
- gold labels;
- assignment manifest;
- condition mappings;
- admin scoring outputs;
- MNEMOS internals.

## Admin Rules

The trial admin should:

- freeze generated reviewer packets before distribution;
- keep admin files separate from reviewer files;
- collect responses before unblinding;
- compile responses before scoring;
- treat missing responses as a blocked trial, not partial evidence;
- keep EBIR human-value claims blocked until scoring completes.

## Commands

Use `--pilot` for the small instrument-test truthset and `--full` for the full
R2 reviewer truthset:

```bash
python tools/ebir_r2_trial.py prepare --pilot
python tools/ebir_r2_trial.py prepare --full
```

The default is `--full`.

## Claim Boundary

Passing preflight, rendering, compilation, or parser checks does not prove EBIR
improves human review outcomes. Only completed independent blinded reviewer
responses scored under the frozen protocol can support a human-value claim.

