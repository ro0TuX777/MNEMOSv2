# Business Proposals RFP Review Design

## Purpose

Give the static **Business Proposals** accordion panel a dedicated experience
that shows how MNEMOS can help proposal teams inspect whether claims are
supported by approved evidence before submission.

The example must help a visitor understand MNEMOS's evidence-first value
without implying live document processing, proposal approval, guaranteed
delivery outcomes, or a complete production workflow.

## Approved Direction

The approved direction combines:

- an **RFP response review** scenario;
- a **claim review matrix** as the primary interaction surface;
- a fictional **IT service-platform migration** proposal;
- a **MNEMOS navy hybrid** visual theme;
- one deliberately unsupported guarantee to make the review boundary visible;
- static, public-safe evidence labels and an illustrative receipt.

The existing single-open accordion behavior remains unchanged. Business
Proposals continues to open only when its accordion trigger or valid hash is
selected.

## Reference Use

The visual and information hierarchy is informed by HubSpot's business
proposal guide:

`https://blog.hubspot.com/sales/how-to-write-business-proposal`

MNEMOS may borrow general proposal concepts such as client need, proposed
solution, qualifications, timeline, pricing awareness, and next steps. It may
also use an approachable editorial hierarchy with generous spacing and clear
section cues.

MNEMOS must not copy HubSpot branding, logos, article copy, screenshots,
illustrations, templates, page structure, or proprietary examples. The final
panel uses MNEMOS colors and original fictional content.

## Scenario

The fictional scenario is an RFP response for an IT service-platform
migration. A proposal team is reviewing a draft before submission and asks:

> Which proposal claims are supported by approved past-performance evidence?

The expanded panel title is:

> Review proposal claims before submission

The panel remains an illustrative static example. It does not accept an RFP,
proposal, or supporting document from the visitor.

## Information Architecture

The expanded Business Proposals panel contains:

1. Experience header and illustrative-static-example badge.
2. Claim review matrix.
3. Review summary with state counts.
4. Public-safe fictional evidence list.
5. Illustrative evidence receipt.
6. Three-step evidence path.
7. Submission and authority boundary.
8. Why-this-matters explanation.
9. Links to the Trace Explorer and demo JSON.

The content stays inside the existing accordion panel. No new route, backend,
upload control, API, graph view, or runtime integration is added.

## Claim Review Matrix

The matrix contains exactly four fictional proposal claims:

| Review state | Proposal claim | Evidence result | Required action |
| --- | --- | --- | --- |
| Supported | The team has completed three comparable service-platform migrations. | Two approved case studies support comparable delivery experience. | Keep the claim attached to both case studies. |
| Needs qualification | The migration can be completed within six weeks. | One transition plan supports an eight-week baseline; no approved source supports six weeks. | Qualify the timeline or obtain approved six-week evidence. |
| Unsupported | The migration will cause zero service disruption. | No approved artifact establishes a zero-disruption guarantee. | Remove the guarantee or replace it with bounded transition language. |
| Evidence gap | The transition team provides 24/7 support throughout cutover. | The draft makes the claim, but the approved evidence set contains no staffing or coverage record. | Obtain an approved support plan or remove the claim. |

The matrix visually distinguishes the four states without implying that MNEMOS
approves or rejects the proposal. Status language describes evidence support
only.

## Review Summary

The summary displays fixed counts derived from the four matrix rows:

- `1 supported`
- `1 needs qualification`
- `1 unsupported`
- `1 evidence gap`

It also states that human proposal review remains required. The summary is not
a readiness score, compliance score, win probability, or submission decision.

## Fictional Evidence Set

Use these repository-safe fictional labels:

- `rfp-requirements.pdf`
- `draft-technical-proposal.pdf`
- `approved-case-study-alpha.pdf`
- `approved-case-study-beta.pdf`
- `transition-plan.pdf`

The labels are display content only. They do not resolve to uploaded files,
repository files, private paths, customer data, or live documents.

## Illustrative Receipt

The receipt records:

- Question: `Which proposal claims are supported by approved past-performance evidence?`
- Supported result: `three comparable migrations`
- Qualified result: `six-week transition timeline`
- Unsupported result: `zero service disruption`
- Evidence gap: `24/7 cutover support`
- Decision state: `illustrative_static_example`
- Boundary: `Human proposal review required; no submission approval or delivery guarantee`

The receipt is static presentation content. It is not generated at runtime.

## Evidence Path

Show this three-step path:

```text
Proposal claim → approved past-performance evidence → support or gap flag
```

## Visual System

Use a scoped MNEMOS navy hybrid theme:

- navy header and review-summary surfaces;
- white or cool-neutral claim rows;
- blue navigation and action cues;
- green supported-evidence indicators;
- amber qualification and evidence-gap indicators;
- restrained red or rust for the unsupported guarantee;
- clear black or dark-ink text on light surfaces;
- high-contrast light text on navy surfaces.

The panel should feel like a professional proposal review packet rather than a
generic dashboard. It must remain visually distinct from the Personal
Documents bill while preserving the overall MNEMOS design language.

## Responsive Behavior

On screens wider than 900 pixels, use a two-column layout with the matrix as
the primary content and the review summary as a narrower sidebar. At 900
pixels and below, use one column and:

- the summary moves below the matrix;
- each matrix row becomes a readable stacked card;
- source labels wrap without clipping;
- CTAs wrap into a compact vertical or two-row layout;
- status meaning remains available through text, not color alone.

No animation is required. Existing reduced-motion behavior remains sufficient.

## Boundaries

The panel must state:

- MNEMOS can help organize proposal claims and supporting evidence.
- MNEMOS does not approve proposals or submissions.
- MNEMOS does not guarantee delivery timelines or service outcomes.
- MNEMOS does not replace proposal-owner, legal, commercial, or executive
  review.
- The example performs no upload, retrieval, inference, or document processing.

Avoid production-readiness, compliance-certification, guaranteed-correctness,
or autonomous-authority claims.

## Static Data Flow

All panel content is embedded in repository-relative static HTML and CSS. The
existing JavaScript accordion controller only shows or hides the panel and
maintains the selected hash. The existing demo-index loader updates the
secondary demo JSON link, as it does for Personal Documents. No new JavaScript
controller or data-loading behavior is required.

No trace JSON schema change is required. No new trace is required. No runtime
request is introduced beyond the existing static demo manifest and trace
loads.

## Validation

Validation must confirm:

1. Business Proposals remains one of eight accessible accordion panels.
2. The panel opens from its trigger and `#use-case-proposal` hash.
3. Exactly four claim rows and the four approved review states appear.
4. Summary counts match the matrix rows.
5. The unsupported zero-disruption guarantee remains visible.
6. All five fictional evidence labels appear.
7. Submission, guarantee, and human-review boundaries remain visible.
8. Both Trace Explorer and demo JSON links remain repository-relative.
9. Existing five trace demos and Research Intake + Receipt remain unchanged.
10. Desktop and mobile layouts remain readable.
11. No local absolute paths, private data, raw logs, secrets, or reference-site
    branding appear.
12. No backend, upload, live inference, graph behavior, or runtime behavior is
    added.

## Out of Scope

- Proposal uploads or document ingestion
- Live claim extraction or evidence retrieval
- Proposal editing or submission workflow
- Approval routing or signatures
- Pricing recommendations
- Win-probability scoring
- Compliance or legal certification
- Delivery guarantees
- Graph implementation
- Runtime behavior changes

## Acceptance Criteria

The design is successful when a first-time visitor can quickly understand:

1. which proposal claims have supporting evidence;
2. which claims need qualification, removal, or additional evidence;
3. which fictional documents shaped each review state;
4. why an evidence receipt is useful before proposal submission; and
5. where MNEMOS stops and human authority begins.
