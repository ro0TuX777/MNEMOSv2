# MNEMOS Static Demo Frontend Prototype

This is a static-only prototype that renders the precomputed demo traces in
`demo/demo_index.json` and `demo/traces/*.json`. It includes three tabs:

- `Trace Explorer` for selectable precomputed scenarios.
- `Where MNEMOS helps` for eight static, public-safe use cases spanning
  personal documents, enterprise evidence work, and professional review.
- `Research Intake + Receipt` for a static copy of the PDF intake/receipt flow.

It does not run live inference, accept uploads, run OCR, implement graphs, or
change MNEMOS runtime behavior.

## Local Preview

From the repository root:

```bash
python -m http.server 8765
```

Open:

```text
http://127.0.0.1:8765/demo/frontend/
```

Serving from the repository root is important because the frontend fetches:

```text
../demo_index.json
../traces/<trace>.json
```

The same frontend also works when staged at a site root for GitHub Pages. In
that layout it fetches:

```text
demo_index.json
traces/<trace>.json
```

## GitHub Pages Deployment

The repository includes `.github/workflows/demo-pages.yml`, which publishes the
static demo to GitHub Pages when `demo/**` changes on `main` or when the
workflow is run manually.

The workflow stages the site as:

```text
_site/
  index.html
  app.js
  styles.css
  demo_index.json
  traces/*.json
```

Expected first deployment URL:

```text
https://ro0TuX777.github.io/MNEMOSv2/
```

First deployment:

1. Open the repository **Actions** tab.
2. Run `Deploy static MNEMOS demo to Pages` on `main`.
3. The workflow configures GitHub Pages with `enablement: true` on first run.
4. A custom domain is optional later; it is not required for the GitHub Pages
   URL.

## What The UI Shows

- Scenario cards from `demo/demo_index.json`
- Selected trace question and short answer
- Evidence artifacts and claim status
- Decision state and boundary warnings
- Trace path timeline
- Excluded or unsupported claims
- Limitations
- Demo receipt and provenance fields
- Eight static use-case cards with full detail panels and authority boundaries
- In-page links from each use case back to the precomputed Trace Explorer
- Static Research Intake page styled after the local MNEMOS intake UI
- Precomputed PDF Q/A receipt walkthrough using `research_intake_ocr`

## Validation

The prototype can be validated without MNEMOS services:

```bash
python demo/frontend/validate_static_demo.py
```
