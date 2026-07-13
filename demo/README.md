# MNEMOS Static Demo Trace Package

This directory contains precomputed, public-safe demo traces for MNEMOS.

The package is designed for a static website or guided interactive demo that
shows how MNEMOS records evidence-shaped answers, decisions, handoffs, and
evaluations.

## What This Is

- A set of static JSON traces under `demo/traces/`.
- A small `demo_index.json` manifest for a frontend to discover available
  traces.
- Public-safe summaries of committed MNEMOS artifacts, including artifact
  paths, claim states, decision boundaries, and trace labels.

## What This Is Not

- It is not a live chatbot.
- It does not run live inference.
- It does not accept user uploads.
- It does not implement context graph projection.
- It does not add graph storage, GraphRAG, retrieval behavior, governance
  behavior, promotion behavior, context assembly behavior, Engram schema
  behavior, or authority behavior.

## Included Traces

| Trace | Purpose | File |
| --- | --- | --- |
| Context Graph Projection R1 | Shows research-only preregistration and narrow-scope/no-implementation boundaries. | `traces/context_graph_r1.json` |
| Evidence Admission R1 | Shows formal non-inferiority failure and honest non-retention. | `traces/evidence_admission_r1.json` |
| Source-grounded retrieval | Shows evidence summaries and citation-ready search fields. | `traces/source_grounded_retrieval.json` |
| Session context handoff | Shows selected parent/source refs, lineage counters, and shadow-only package boundaries. | `traces/session_context_handoff.json` |
| Research intake OCR | Shows pypdf-first extraction, Docling OCR fallback, metadata, and limitations. | `traces/research_intake_ocr.json` |

No recommended trace was omitted. Each included trace is limited to claims
supported by existing committed artifacts.

## Static Frontend Use

A frontend can:

1. Load `demo_index.json`.
2. Render each entry in `traces[]` as a selectable scenario.
3. Load the selected `path`.
4. Render `demo_panels.answer_panel`, `demo_panels.evidence_panel`,
   `demo_panels.decision_boundary_panel`, and `demo_panels.trace_panel`.
5. Show `excluded_or_unsupported_claims` and `limitations` beside the demo so
   the user sees what MNEMOS does not claim.

The JSON is intentionally UI-neutral. It can power cards, timelines, source
drawers, decision-boundary panels, or a future interactive walkthrough without
requiring a live MNEMOS service.

## Static Frontend And Pages Hosting

The checked-in static frontend lives in `demo/frontend/`.

Local preview from the repository root:

```bash
python -m http.server 8765
```

Then open:

```text
http://127.0.0.1:8765/demo/frontend/
```

GitHub Pages deployment is handled by
`.github/workflows/demo-pages.yml`. The workflow validates the trace package,
stages `demo/frontend/` at the Pages site root, and copies `demo_index.json`
plus `traces/*.json` beside it so the first public URL can be:

```text
https://ro0TuX777.github.io/MNEMOSv2/
```

For a first deployment, open the repository **Actions** tab and run
`Deploy static MNEMOS demo to Pages` on `main`. The workflow configures GitHub
Pages with `enablement: true` on first run. A custom domain is optional later
and is not required for the GitHub Pages URL.

## Manual Review Or Regeneration

This package was built manually from committed artifacts. To review it:

1. Open `demo/demo_index.json`.
2. For each trace, inspect `evidence_used[].artifact_path`.
3. Confirm the trace does not include private paths, raw logs, secrets, or
   unsupported production claims.
4. Confirm `decision_state`, `limitations`, and
   `excluded_or_unsupported_claims` match the cited artifacts.

If traces are regenerated later, keep the same evidence standard: summarize
existing committed artifacts only, update `provenance.content_hash`, and rerun
the lightweight validation checks listed below.

## Public-Safety Rules

- Use repository-relative paths only.
- Do not include raw private content, credentials, secrets, local absolute
  paths, private logs, or unpublished artifacts.
- Do not claim production readiness.
- Do not present research-only lanes as runtime features.
- Do not claim live demo behavior unless a separate live demo exists and is
  verified.

## Lightweight Validation

The package should pass:

- all JSON files parse successfully;
- required trace fields are present;
- `demo_index.json` paths resolve;
- no repository-external absolute local paths appear;
- no obvious secrets or credentials appear;
- no trace claims implementation for research-only lanes.
