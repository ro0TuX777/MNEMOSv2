# Where MNEMOS Helps Tab Design

## Purpose

Add an approachable, static use-case experience to the MNEMOS GitHub Pages
demo so non-technical and professional visitors can understand where
source-grounded memory may help.

The core framing is:

> MNEMOS is AI memory for work where the source matters.
>
> It helps connect answers, decisions, handoffs, and evaluations back to the
> documents and evidence that shaped them.

The experience describes where MNEMOS can help. It does not claim that MNEMOS
fully solves these workflows today.

## User Experience

Add a third top-level tab named **Where MNEMOS helps** beside the existing
Trace Explorer and Research Intake + Receipt tabs.

The tab begins with the heading **AI memory for work where the source
matters.** and a short explanation of source-backed answers and decisions. It
then presents eight use-case cards in a responsive grid:

1. Personal document understanding — bill
2. Compliance / audit / governance
3. Legal document review
4. Research and academic work
5. Business proposal
6. Healthcare documents
7. Education
8. Journalism / fact-checking

Personal documents, compliance / audit, and legal review receive stronger
visual emphasis to make the range from everyday use through enterprise and
professional review immediately visible.

Each card is a normal in-page link to a semantic detail panel lower in the
same tab. This keeps navigation accessible and functional without adding a
client-side router or additional application state.

## Use-Case Detail Template

Every detail panel contains:

- Audience
- Typical question
- Documents involved
- What MNEMOS shows
- What MNEMOS does not claim
- Example evidence trace
- Why this matters
- A call to action back to the Trace Explorer

Each evidence trace is a short three-step example displayed as a directional
sequence. The examples are illustrative positioning content, not live traces
or claims of runtime execution.

## Content Boundaries

The copy uses public-safe, capability-bounded language such as **Where MNEMOS
can help** and **MNEMOS can show**.

The use cases include these explicit boundaries:

- Personal document understanding: no financial, tax, or legal advice.
- Compliance / audit / governance: no compliance certification and no
  replacement for an auditor.
- Legal document review: no legal advice and no replacement for attorney
  judgment.
- Research and academic work: no claim of settled truth.
- Business proposals: no submission approval.
- Healthcare documents: no medical advice, diagnosis, or treatment guidance.
- Education: no replacement for teacher judgment or academic integrity rules.
- Journalism / fact-checking: no replacement for editorial judgment.

No use case claims legal, medical, tax, compliance, production, or guaranteed
correctness authority.

## Implementation Shape

The change remains within the existing static frontend:

- `demo/frontend/index.html` receives the third tab, overview cards, and detail
  panels.
- `demo/frontend/styles.css` receives responsive card, emphasis, detail-panel,
  and three-step trace styles consistent with the existing visual language.
- `demo/frontend/app.js` reuses the existing tab setup and adds only the small
  behavior needed for Trace Explorer calls to action to activate the existing
  tab.
- `demo/frontend/validate_static_demo.py` checks the eight use-case anchors,
  their matching detail panels, and the retained static boundaries.
- Demo documentation is updated only where needed to describe the third tab.

No new data-fetching path, route, API, backend, upload, inference, graph, or
runtime integration is introduced.

## Navigation and Data Flow

1. A visitor selects **Where MNEMOS helps** using the existing tab controls.
2. The browser reveals the static use-case tab panel.
3. A use-case card follows an in-page anchor to its matching detail panel.
4. A detail-panel call to action activates the existing Trace Explorer tab.
5. The Trace Explorer continues to load `demo/demo_index.json` and the five
   existing precomputed trace files exactly as before.

The use-case content is embedded in semantic HTML and requires no additional
JSON or network requests.

## Error Handling and Accessibility

Because the use-case content is static HTML, it remains readable even if trace
JSON loading fails. Cards use normal anchor links, detail panels have unique
IDs, headings follow a logical hierarchy, and tab controls retain their
existing button behavior.

No new runtime error states are required.

## Validation

Validation must confirm:

1. All eight use-case card anchors resolve to unique detail panels.
2. All five existing trace JSON files still load through the demo manifest.
3. All three tab controls activate their matching panels.
4. The static Pages staging layout builds successfully.
5. The source-tree local preview renders successfully.
6. No local absolute paths, private logs, or private artifacts appear.
7. No unsupported runtime, legal, medical, tax, compliance, production, or
   guaranteed-correctness claims appear.
8. No backend, upload, live inference, graph implementation, or runtime
   behavior change is added.

## Acceptance Criteria

- A non-technical visitor can identify at least one relevant use case quickly.
- Professional visitors can see evidence-sensitive applications and explicit
  authority boundaries.
- The three emphasized use cases visibly span personal, enterprise, and
  professional-review audiences.
- Every use case follows the approved detail template.
- The existing Trace Explorer and Research Intake + Receipt experiences remain
  functional.
- The GitHub Pages artifact remains static and repository-relative.
