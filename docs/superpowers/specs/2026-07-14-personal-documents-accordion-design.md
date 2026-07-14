# Personal Documents Accordion Design

## Purpose

Turn the static **Where MNEMOS helps** tab into a focused, single-open
accordion experience. The first fully themed use case demonstrates how a
person could understand a bill while preserving the source trail behind the
answer.

The design must help a visitor or investor imagine MNEMOS inside a document
workflow without implying live upload, inference, financial authority, or a
complete production solution.

## Approved Direction

The approved direction combines:

- **Accordion stack:** one use case is expanded while the other seven remain
  compact rows directly underneath.
- **MNEMOS hybrid theme:** a calm document surface, navy evidence workspace,
  blue comparison cues, and restrained green source indicators.
- **Progressive theming:** Personal Documents receives the first dedicated
  experience. The remaining use cases retain their existing generic detail
  content until separate visual references are approved.

Personal Documents is expanded by default.

## Reference Use

The visual and content hierarchy is informed by these public references:

- One NZ bill walkthrough:
  `https://one.nz/help/understand-my-bill/`
- BroadConnect strategic billing guide:
  `https://broadconnect.com.au/understanding-your-business-phone-bill-a-strategic-audit-guide/`

MNEMOS may borrow general patterns such as numbered document explanation,
audit framing, visible takeaways, and a professional navy/blue tone. It must
not copy logos, branded bill content, original article copy, screenshots, or
page structure.

## Information Architecture

The **Where MNEMOS helps** tab keeps its existing introduction and becomes:

1. Introductory source-matters framing.
2. One accordion stack containing eight use cases.
3. Personal Documents expanded by default.
4. Seven compact use-case triggers beneath the expanded panel.

Selecting a compact trigger:

1. Collapses the currently expanded use case.
2. Expands the selected use case in the same stack position.
3. Preserves the selected use-case hash in the URL.
4. Leaves all other use cases compact.

The current eight-card overview grid and the duplicated full-detail stack are
replaced by this single accordion structure.

## Personal Documents Experience

### Header

The expanded panel begins with:

- Audience label: **Personal document understanding — bill**
- Title: **Understand a bill without losing the source**
- Question: **Why did my bill increase this month?**
- Boundary badge: **Illustrative static example**

### Fictional Bill

Use a clearly fictional, provider-neutral bill comparison:

| Item | Previous bill | Current bill | Change |
| --- | ---: | ---: | ---: |
| Monthly plan | $89.00 | $94.00 | +$5.00 |
| Out-of-plan usage | $0.00 | $8.40 | +$8.40 |
| Total | $89.00 | $102.40 | +$13.40 |

No real provider name, account number, customer information, or copied bill
layout appears.

### Source-Backed Answer

The static answer is:

> The bill increased by $13.40. The statement shows a $5.00 monthly plan-rate
> adjustment and $8.40 in out-of-plan usage. The documents establish the
> charges, but they do not establish why the extra usage occurred.

The answer must visually distinguish:

- supported charge changes;
- the source lines that support them;
- the unsupported cause of the extra usage;
- the professional-authority boundary.

### Evidence Receipt

The receipt is illustrative and static. It shows:

- Question: `Why did my bill increase this month?`
- Sources: `current-month-statement.pdf`, `prior-month-statement.pdf`, and
  `rate-change-notice.pdf`
- Evidence used: monthly plan line, usage line, and total-due line
- Supported result: `+$13.40 total increase`
- Unsupported result: `cause of out-of-plan usage`
- Decision state: `illustrative_static_example`
- Boundary: `No financial, tax, or legal advice`

These filenames are public-safe fictional labels, not repository paths or live
documents.

### Evidence Path

Show this three-step path:

```text
Bill statement → charge comparison → bounded explanation
```

### Calls to Action

The expanded panel keeps:

- **Explore the evidence demo** → activates the existing Trace Explorer tab.
- **Inspect demo JSON** → uses the existing repository-relative manifest link.

Neither CTA starts a live workflow.

## Remaining Use Cases

The remaining use cases appear as compact accordion rows:

1. Compliance / audit / governance
2. Legal document review
3. Research and academic work
4. Business proposal
5. Healthcare documents
6. Education
7. Journalism / fact-checking

Selecting one expands its existing generic fields, three-step trace, boundary,
why-it-matters copy, and Trace Explorer CTA. No additional themed content is
invented before the user provides and approves a visual reference.

## Visual System

The Personal Documents panel uses scoped custom properties so later use cases
can receive separate themes without changing the global MNEMOS palette:

```css
--bill-navy: #112337;
--bill-deep-blue: #00194c;
--bill-blue: #204ce5;
--bill-source-green: #0b7f66;
--bill-source-soft: #d9f99d;
--bill-paper: #f3f8f6;
```

Application:

- Navy header and receipt surface communicate evidence review.
- White or soft-paper panels keep the bill readable and approachable.
- Blue marks document comparisons and changed amounts.
- Green marks admitted source evidence and receipt boundaries.
- Warning colors remain reserved for unsupported claims and no-authority copy.

The panel remains consistent with the existing MNEMOS type system and border
radii. It does not reproduce either reference site’s branding.

## Interaction and Accessibility

Each accordion trigger is a native `<button>` with:

- `aria-expanded="true|false"`;
- `aria-controls="<panel-id>"`;
- a visible expanded/collapsed indicator;
- a unique use-case label.

Each panel:

- has a matching `id`;
- uses `role="region"`;
- references its trigger through `aria-labelledby`;
- is hidden when collapsed;
- retains its existing use-case hash target.

Native button behavior provides keyboard activation. Focus remains on the
selected trigger when a panel changes. Motion is limited and disabled through
`prefers-reduced-motion`.

On initial load:

- a valid use-case URL hash activates the **Where MNEMOS helps** top-level tab
  and opens that use case;
- a missing or invalid hash opens Personal Documents;
- no hash produces no error or network request.

## Static Data Flow

1. The browser loads the existing static HTML, CSS, and JavaScript.
2. Personal Documents is present as fictional static markup.
3. The accordion controller reads the selected trigger and toggles local DOM
   state only.
4. The Trace Explorer continues loading the existing manifest and five
   precomputed traces exactly as before.
5. No use-case action sends, uploads, stores, or infers data.

## Error Handling

The use-case content requires no network request. If JavaScript is unavailable,
Personal Documents remains visible and the other use-case labels remain
readable; expanding the other panels requires JavaScript. If the hash is
unknown, the controller falls back to Personal Documents.

Existing trace-loading errors remain unchanged and isolated to the Trace
Explorer and Research Intake views.

## Files in Scope

- `demo/frontend/index.html`
- `demo/frontend/styles.css`
- `demo/frontend/app.js`
- `demo/frontend/validate_static_demo.py`
- `demo/frontend/README.md`
- `demo/README.md`

No trace JSON, workflow, runtime, retrieval, governance, promotion, context
assembly, Engram schema, graph, or authority file changes are required.

## Validation

Validation must confirm:

1. Eight accordion triggers map to eight unique panels.
2. Exactly one panel is expanded at a time when JavaScript is active.
3. Personal Documents is the default panel without a valid hash.
4. A valid use-case hash opens its matching panel.
5. Invalid hashes fall back safely to Personal Documents.
6. All triggers expose correct `aria-expanded` and `aria-controls` state.
7. The fictional bill totals and receipt fields remain internally consistent.
8. All five existing demo traces load in source-tree and Pages-root layouts.
9. Trace Explorer and Research Intake + Receipt behavior remains unchanged.
10. No local paths, private data, real customer identifiers, copied brand
    assets, or unsupported authority claims appear.
11. No backend, upload, live inference, graph implementation, or runtime
    behavior is added.
12. Desktop and mobile layouts keep the selected panel readable and compact
    rows operable.

## Acceptance Criteria

- A visitor can understand the bill example without technical MNEMOS context.
- The selected panel demonstrates document, evidence, answer, unsupported
  claim, and receipt relationships in one view.
- The page is materially shorter because only one use case is expanded.
- Personal Documents feels distinct without breaking the MNEMOS visual system.
- Other use cases remain accessible and ready for later themed redesigns.
- The example remains clearly illustrative, static, provider-neutral, and
  non-authoritative.
