# Business Proposals RFP Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the generic Business Proposals accordion content with a static, fictional RFP claim-review matrix that shows which proposal claims are supported, qualified, unsupported, or missing evidence.

**Architecture:** Keep the existing eight-panel accordion and static loader unchanged. Add semantic proposal-review HTML inside `#use-case-proposal`, scope a responsive MNEMOS navy hybrid theme to `.proposal-use-case`, and extend the existing Python validator with exact content, state-count, boundary, and style contracts. No new JavaScript controller, route, trace, schema, or runtime request is required.

**Tech Stack:** Static HTML5, CSS, existing vanilla JavaScript accordion, Python standard-library validator, PowerShell verification commands.

## Global Constraints

- Static content only; no backend, API, upload, live inference, retrieval, document processing, or graph behavior.
- Preserve the existing Trace Explorer, Research Intake + Receipt, five precomputed traces, accordion behavior, and hash behavior.
- Use fictional, repository-safe evidence labels only.
- Do not copy HubSpot branding, logos, article copy, screenshots, illustrations, templates, page structure, or proprietary examples.
- Do not imply proposal approval, submission authority, guaranteed timelines, guaranteed service outcomes, legal authority, compliance certification, or autonomous decision-making.
- Keep all links and assets repository-relative.
- Do not touch MNEMOS retrieval, governance, promotion, context assembly, Engram schema, or authority behavior.

---

### Task 1: Add the Static RFP Claim Review

**Files:**
- Modify: `demo/frontend/validate_static_demo.py`
- Modify: `demo/frontend/index.html`

**Interfaces:**
- Consumes: existing `USE_CASE_IDS`, `StaticDemoHTMLParser`, `data-tab-link="trace-demo"`, `data-demo-index-link`, and `#use-case-proposal` accordion region.
- Produces: four `.proposal-claim-row[data-proposal-claim-state]` elements in a fixed order, one static review summary, five fictional source labels, and one illustrative receipt.

- [ ] **Step 1: Add failing proposal-content contracts**

Add these constants after `REQUIRED_BILL_FRAGMENTS` in
`demo/frontend/validate_static_demo.py`:

```python
PROPOSAL_CLAIM_STATES = (
    "supported",
    "needs_qualification",
    "unsupported",
    "evidence_gap",
)

REQUIRED_PROPOSAL_FRAGMENTS = (
    "Review proposal claims before submission",
    "Which proposal claims are supported by approved past-performance evidence?",
    "three comparable service-platform migrations",
    "six-week transition timeline",
    "zero service disruption",
    "24/7 cutover support",
    "rfp-requirements.pdf",
    "draft-technical-proposal.pdf",
    "approved-case-study-alpha.pdf",
    "approved-case-study-beta.pdf",
    "transition-plan.pdf",
    "Human proposal review required",
    "no submission approval or delivery guarantee",
)
```

Extend `StaticDemoHTMLParser.__init__()` with:

```python
self.proposal_claim_states: list[str] = []
```

Extend `handle_starttag()` with:

```python
if proposal_state := attributes.get("data-proposal-claim-state"):
    self.proposal_claim_states.append(proposal_state)
```

Add these checks after the existing bill-fragment loop:

```python
    if parser.proposal_claim_states != list(PROPOSAL_CLAIM_STATES):
        errors.append(
            f"proposal claim-state order mismatch: {parser.proposal_claim_states}"
        )

    for fragment in REQUIRED_PROPOSAL_FRAGMENTS:
        if fragment.lower() not in normalized_html:
            errors.append(f"missing fictional proposal fragment: {fragment}")

    for state in PROPOSAL_CLAIM_STATES:
        summary_fragment = f"1 {state.replace('_', ' ')}"
        if summary_fragment not in normalized_html:
            errors.append(f"proposal summary count mismatch: {summary_fragment}")
```

- [ ] **Step 2: Run the validator and verify RED**

Run:

```powershell
python demo/frontend/validate_static_demo.py
```

Expected: exit `1` with `proposal claim-state order mismatch` and missing
fictional proposal fragment errors. Existing trace validation must still report
no missing or invalid traces.

- [ ] **Step 3: Add the proposal experience header and review layout**

In `demo/frontend/index.html`, change the Business Proposals item wrapper to:

```html
<section class="use-case-item proposal-use-case">
```

Keep the existing trigger ID, `data-use-case-target`, `aria-expanded`, and
`aria-controls` unchanged. Replace only the body of `#use-case-proposal` with:

```html
<div class="proposal-experience-header">
  <div>
    <p class="eyebrow">Business proposal — RFP response review</p>
    <h3>Review proposal claims before submission</h3>
    <p class="question">Which proposal claims are supported by approved past-performance evidence?</p>
  </div>
  <span class="proposal-static-badge">Illustrative static example</span>
</div>

<div class="proposal-review-layout">
  <section class="proposal-matrix" aria-labelledby="proposal-matrix-heading">
    <p class="eyebrow">Fictional IT service-platform migration</p>
    <h4 id="proposal-matrix-heading">Claim review matrix</h4>

  </section>

  <aside class="proposal-summary" aria-labelledby="proposal-summary-heading">
    <p class="eyebrow">Review summary</p>
    <h4 id="proposal-summary-heading">Evidence state counts</h4>
    <ul>
      <li><span class="proposal-status supported">Supported</span><strong>1 supported</strong></li>
      <li><span class="proposal-status qualification">Needs qualification</span><strong>1 needs qualification</strong></li>
      <li><span class="proposal-status unsupported">Unsupported</span><strong>1 unsupported</strong></li>
      <li><span class="proposal-status gap">Evidence gap</span><strong>1 evidence gap</strong></li>
    </ul>
    <p>Human proposal review remains required. These counts are not a readiness score, win probability, or submission decision.</p>
  </aside>
</div>
```

- [ ] **Step 4: Add exactly four semantic claim rows**

Inside `.proposal-matrix`, add the rows in this exact order:

```html
<article class="proposal-claim-row" data-proposal-claim-state="supported">
  <div class="proposal-claim-heading">
    <span class="proposal-status supported">Supported</span>
    <strong>The team has completed three comparable service-platform migrations.</strong>
  </div>
  <dl>
    <dt>Evidence result</dt>
    <dd>Two approved case studies support comparable delivery experience.</dd>
    <dt>Sources</dt>
    <dd><code>approved-case-study-alpha.pdf</code> and <code>approved-case-study-beta.pdf</code></dd>
    <dt>Required action</dt>
    <dd>Keep the claim attached to both approved case studies.</dd>
  </dl>
</article>

<article class="proposal-claim-row" data-proposal-claim-state="needs_qualification">
  <div class="proposal-claim-heading">
    <span class="proposal-status qualification">Needs qualification</span>
    <strong>The migration can be completed within a six-week transition timeline.</strong>
  </div>
  <dl>
    <dt>Evidence result</dt>
    <dd>The approved transition plan supports an eight-week baseline; no approved source supports six weeks.</dd>
    <dt>Source</dt>
    <dd><code>transition-plan.pdf</code></dd>
    <dt>Required action</dt>
    <dd>Qualify the timeline or obtain approved six-week evidence.</dd>
  </dl>
</article>

<article class="proposal-claim-row" data-proposal-claim-state="unsupported">
  <div class="proposal-claim-heading">
    <span class="proposal-status unsupported">Unsupported</span>
    <strong>The migration will cause zero service disruption.</strong>
  </div>
  <dl>
    <dt>Evidence result</dt>
    <dd>No approved artifact establishes a zero-disruption guarantee.</dd>
    <dt>Source reviewed</dt>
    <dd><code>draft-technical-proposal.pdf</code></dd>
    <dt>Required action</dt>
    <dd>Remove the guarantee or replace it with bounded transition language.</dd>
  </dl>
</article>

<article class="proposal-claim-row" data-proposal-claim-state="evidence_gap">
  <div class="proposal-claim-heading">
    <span class="proposal-status gap">Evidence gap</span>
    <strong>The transition team provides 24/7 cutover support.</strong>
  </div>
  <dl>
    <dt>Evidence result</dt>
    <dd>The draft makes the claim, but the approved evidence set contains no staffing or coverage record.</dd>
    <dt>Requirement source</dt>
    <dd><code>rfp-requirements.pdf</code></dd>
    <dt>Required action</dt>
    <dd>Obtain an approved support plan or remove the claim.</dd>
  </dl>
</article>
```

- [ ] **Step 5: Add the static source packet and receipt**

After `.proposal-review-layout`, add:

```html
<div class="proposal-evidence-grid">
  <section class="proposal-sources" aria-labelledby="proposal-sources-heading">
    <p class="eyebrow">Evidence reviewed</p>
    <h4 id="proposal-sources-heading">Public-safe fictional proposal packet</h4>
    <ul>
      <li><code>rfp-requirements.pdf</code><span>Client requirements and cutover-support request</span></li>
      <li><code>draft-technical-proposal.pdf</code><span>Claims under review before submission</span></li>
      <li><code>approved-case-study-alpha.pdf</code><span>Comparable migration evidence</span></li>
      <li><code>approved-case-study-beta.pdf</code><span>Additional past-performance evidence</span></li>
      <li><code>transition-plan.pdf</code><span>Approved eight-week delivery baseline</span></li>
    </ul>
  </section>

  <section class="proposal-receipt" aria-labelledby="proposal-receipt-heading">
    <p class="eyebrow">Illustrative receipt</p>
    <h4 id="proposal-receipt-heading">What is supported—and what still needs review</h4>
    <dl>
      <dt>Question</dt><dd>Which proposal claims are supported by approved past-performance evidence?</dd>
      <dt>Supported result</dt><dd><code>three comparable migrations</code></dd>
      <dt>Qualified result</dt><dd><code>six-week transition timeline</code></dd>
      <dt>Unsupported result</dt><dd><code>zero service disruption</code></dd>
      <dt>Evidence gap</dt><dd><code>24/7 cutover support</code></dd>
      <dt>Decision state</dt><dd><code>illustrative_static_example</code></dd>
      <dt>Boundary</dt><dd><code>Human proposal review required; no submission approval or delivery guarantee</code></dd>
    </dl>
  </section>
</div>

<h4>Evidence path</h4>
<ol class="mini-trace"><li>Proposal claim</li><li>Approved past-performance evidence</li><li>Support or gap flag</li></ol>
<p class="use-case-boundary">MNEMOS can help organize proposal claims and supporting evidence. It does not approve submissions or proposals, and it does not guarantee delivery timelines or service outcomes.</p>
<p class="use-case-why"><strong>Why this matters:</strong> Teams can qualify or remove unsupported language while preserving the evidence behind claims that remain in the response.</p>
<div class="use-case-actions">
  <a class="use-case-cta" href="#trace-demo" data-tab-link="trace-demo">Explore the evidence demo</a>
  <a class="use-case-secondary-cta" href="demo_index.json" data-demo-index-link>Inspect demo JSON</a>
</div>
```

- [ ] **Step 6: Run validation and verify GREEN**

Run:

```powershell
python demo/frontend/validate_static_demo.py
git diff --check
```

Expected:

```text
STATIC_DEMO_FRONTEND_VALIDATION_OK
traces=5
```

`git diff --check` produces no output.

- [ ] **Step 7: Commit the semantic proposal experience**

```powershell
git add -- demo/frontend/index.html demo/frontend/validate_static_demo.py
git diff --cached --check
git commit -m "feat: add static RFP claim review"
```

---

### Task 2: Add the Scoped Navy Proposal Theme

**Files:**
- Modify: `demo/frontend/validate_static_demo.py`
- Modify: `demo/frontend/styles.css`

**Interfaces:**
- Consumes: `.proposal-use-case`, `.proposal-review-layout`, `.proposal-claim-row`, `.proposal-summary`, `.proposal-sources`, `.proposal-receipt`, and status classes from Task 1.
- Produces: a two-column desktop review layout, one-column responsive layout, text-visible status styling, and explicit dark text on light surfaces.

- [ ] **Step 1: Add failing scoped-style contracts**

After `bill_text_rules` in `validate_static_demo.py`, add:

```python
    proposal_style_rules = (
        r"\.proposal-use-case\s*\{[^}]*--proposal-navy:\s*#112337;",
        r"\.proposal-review-layout\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s*320px;",
        r"\.proposal-claim-row\s*\{[^}]*color:\s*#000000;",
        r"@media\s*\(max-width:\s*900px\)[\s\S]*\.proposal-review-layout\s*\{[^}]*grid-template-columns:\s*1fr;",
    )
    for rule in proposal_style_rules:
        if not re.search(rule, styles_css, re.DOTALL):
            errors.append(f"frontend styles missing proposal rule: {rule}")
```

- [ ] **Step 2: Run the validator and verify RED**

Run:

```powershell
python demo/frontend/validate_static_demo.py
```

Expected: exit `1` with four `frontend styles missing proposal rule` errors.

- [ ] **Step 3: Add the scoped theme and desktop layout**

Add after the bill-theme rules in `demo/frontend/styles.css`:

```css
.proposal-use-case {
  --proposal-navy: #112337;
  --proposal-blue: #204ce5;
  --proposal-green: #0b7f66;
  --proposal-amber: #b45309;
  --proposal-rust: #a33a2b;
  --proposal-paper: #f5f7fa;
  --proposal-line: #dbe3ea;
}

.proposal-use-case > .use-case-trigger,
.proposal-use-case.is-expanded > .use-case-trigger,
.proposal-use-case > .use-case-trigger:hover,
.proposal-use-case > .use-case-trigger:focus-visible {
  background: var(--proposal-navy);
  color: #ffffff;
}

.proposal-use-case .use-case-audience,
.proposal-use-case .use-case-toggle {
  color: #d9f99d;
}

.proposal-use-case .use-case-toggle {
  background: rgba(255, 255, 255, 0.1);
}

.proposal-experience-header,
.proposal-review-layout,
.proposal-evidence-grid {
  display: grid;
  gap: 16px;
}

.proposal-experience-header {
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  margin-bottom: 18px;
}

.proposal-static-badge {
  border-radius: 999px;
  background: #d9f99d;
  color: #244600;
  padding: 6px 10px;
  font-size: 0.78rem;
  font-weight: 780;
}

.proposal-review-layout {
  grid-template-columns: minmax(0, 1fr) 320px;
  align-items: start;
}

.proposal-matrix,
.proposal-sources {
  border: 1px solid var(--proposal-line);
  border-radius: 12px;
  background: var(--proposal-paper);
  color: #000000;
  padding: 18px;
}

.proposal-claim-row {
  border: 1px solid var(--proposal-line);
  border-left-width: 5px;
  border-radius: 10px;
  background: #ffffff;
  color: #000000;
  padding: 15px;
}

.proposal-claim-row + .proposal-claim-row {
  margin-top: 10px;
}

.proposal-claim-row[data-proposal-claim-state="supported"] { border-left-color: var(--proposal-green); }
.proposal-claim-row[data-proposal-claim-state="needs_qualification"] { border-left-color: var(--proposal-amber); }
.proposal-claim-row[data-proposal-claim-state="unsupported"] { border-left-color: var(--proposal-rust); }
.proposal-claim-row[data-proposal-claim-state="evidence_gap"] { border-left-color: var(--proposal-blue); }

.proposal-claim-heading {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 12px;
}

.proposal-claim-row dl,
.proposal-receipt dl {
  display: grid;
  grid-template-columns: 130px minmax(0, 1fr);
  gap: 8px 12px;
  margin: 13px 0 0;
}

.proposal-claim-row dt,
.proposal-receipt dt {
  font-size: 0.8rem;
  font-weight: 760;
}

.proposal-claim-row dd,
.proposal-receipt dd {
  margin: 0;
}
```

- [ ] **Step 4: Add status, summary, sources, and receipt styling**

Add:

```css
.proposal-status {
  display: inline-flex;
  width: fit-content;
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
}

.proposal-status.supported { background: #dff7ed; color: #075d47; }
.proposal-status.qualification { background: #fff1d6; color: #7a3f00; }
.proposal-status.unsupported { background: #fde4df; color: #7f2419; }
.proposal-status.gap { background: #e5ecff; color: #173da8; }

.proposal-summary,
.proposal-receipt {
  border-radius: 12px;
  background: var(--proposal-navy);
  color: #ffffff;
  padding: 18px;
}

.proposal-summary .eyebrow,
.proposal-summary h4,
.proposal-receipt .eyebrow,
.proposal-receipt h4 {
  color: #d9f99d;
}

.proposal-summary ul,
.proposal-sources ul {
  display: grid;
  gap: 9px;
  margin: 14px 0;
  padding: 0;
  list-style: none;
}

.proposal-summary li {
  display: grid;
  gap: 5px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.15);
  padding-bottom: 9px;
}

.proposal-summary p {
  color: #d8e2ee;
}

.proposal-evidence-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 16px;
}

.proposal-sources li {
  display: grid;
  gap: 3px;
  border-left: 3px solid var(--proposal-green);
  padding-left: 10px;
}

.proposal-sources li span {
  color: #000000;
  font-size: 0.82rem;
}

.proposal-receipt dt {
  color: #b9c9d8;
}
```

- [ ] **Step 5: Add exact responsive behavior**

Inside the existing `@media (max-width: 900px)` block, add:

```css
  .proposal-experience-header,
  .proposal-review-layout,
  .proposal-evidence-grid {
    grid-template-columns: 1fr;
  }

  .proposal-static-badge {
    justify-self: start;
  }
```

Inside the existing `@media (max-width: 560px)` block, add:

```css
  .proposal-claim-row dl,
  .proposal-receipt dl {
    grid-template-columns: 1fr;
  }

  .proposal-claim-row dt,
  .proposal-receipt dt {
    margin-top: 6px;
  }
```

At 900 pixels and below the former matrix rows remain semantic articles and
visually read as stacked cards; no horizontal table scrolling is introduced.

- [ ] **Step 6: Run style and syntax verification**

Run:

```powershell
python demo/frontend/validate_static_demo.py
node --check demo/frontend/app.js
git diff --check
```

Expected: Python prints `STATIC_DEMO_FRONTEND_VALIDATION_OK` and `traces=5`;
Node and `git diff --check` exit `0` with no output.

- [ ] **Step 7: Commit the scoped proposal theme**

```powershell
git add -- demo/frontend/styles.css demo/frontend/validate_static_demo.py
git diff --cached --check
git commit -m "feat: style static RFP evidence review"
```

---

### Task 3: Document and Verify the Public Static Experience

**Files:**
- Modify: `demo/frontend/README.md`
- Modify: `demo/README.md`
- Verify: `demo/frontend/index.html`
- Verify: `demo/frontend/styles.css`
- Verify: `demo/frontend/app.js`
- Verify: `demo/frontend/validate_static_demo.py`

**Interfaces:**
- Consumes: the completed proposal matrix and scoped theme from Tasks 1 and 2.
- Produces: accurate public documentation and verified source-tree and Pages-root artifacts.

- [ ] **Step 1: Update frontend documentation**

In `demo/frontend/README.md`, replace:

```markdown
- Generic detail panels for the seven use cases awaiting separate visual themes
```

with:

```markdown
- Dedicated fictional Personal Documents and Business Proposals evidence reviews
- Generic detail panels for the six use cases awaiting separate visual themes
```

After the Personal Documents description, add:

```markdown
`Business Proposals` presents a fictional IT service-platform RFP claim matrix
with supported, qualified, unsupported, and evidence-gap states. Its static
receipt demonstrates proposal review boundaries without approving a submission
or guaranteeing delivery outcomes.
```

- [ ] **Step 2: Update package documentation**

Extend the use-case paragraph in `demo/README.md` with:

```markdown
Business Proposals uses a fictional RFP response to show how approved
past-performance evidence can support, qualify, or fail to support proposal
claims before submission. It remains a static illustration and does not add
proposal upload, live review, submission approval, or delivery guarantees.
```

- [ ] **Step 3: Run public-surface scans**

Run:

```powershell
$publicFiles = @(
  'demo/frontend/index.html',
  'demo/frontend/styles.css',
  'demo/frontend/app.js',
  'demo/frontend/README.md',
  'demo/README.md'
)
rg -n 'G:\\MNEMOS|G:/MNEMOS|C:\\Users|C:/Users|file:///[A-Za-z]|logs[/\\]' $publicFiles
rg -n 'HubSpot|fully solves this workflow today|certifies compliance|guarantees correctness|approves submissions|guarantees delivery' $publicFiles
```

Expected: both scans return no matches. Boundary phrases such as `does not
approve submissions` and `does not guarantee delivery` are permitted; the
positive-claim scan intentionally uses positive third-person forms only.

- [ ] **Step 4: Stage the GitHub Pages artifact**

Run:

```powershell
$site = Join-Path $env:TEMP 'mnemos-business-proposals-pages'
$resolvedTemp = [System.IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
$resolvedSite = [System.IO.Path]::GetFullPath($site)
if (-not $resolvedSite.StartsWith($resolvedTemp, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Unsafe staging path: $resolvedSite"
}
if (Test-Path -LiteralPath $site) {
  Remove-Item -Recurse -Force -LiteralPath $site
}
New-Item -ItemType Directory -Force -Path $site | Out-Null
Copy-Item -Recurse 'demo/frontend/*' $site
Copy-Item 'demo/demo_index.json' (Join-Path $site 'demo_index.json')
Copy-Item -Recurse 'demo/traces' (Join-Path $site 'traces')
```

- [ ] **Step 5: Smoke-test source and Pages layouts**

Serve the repository root on port `8765` and the staged Pages root on port
`8766`. For both layouts verify HTTP `200` for the page, manifest, and every
path in `traces[]`. Assert:

```powershell
$requiredProposalFragments = @(
  'data-proposal-claim-state="supported"',
  'data-proposal-claim-state="needs_qualification"',
  'data-proposal-claim-state="unsupported"',
  'data-proposal-claim-state="evidence_gap"',
  'zero service disruption',
  'Human proposal review required'
)
```

Expected per layout:

```text
page=200 manifest=200 traces=5 proposal=ok
```

- [ ] **Step 6: Verify browser behavior when a browser target is available**

Check both `http://127.0.0.1:8765/demo/frontend/#use-case-proposal` and
`http://127.0.0.1:8766/#use-case-proposal`:

1. The parent **Where MNEMOS helps** tab activates.
2. Business Proposals is the only expanded accordion panel.
3. All four claim states are visible through text and color.
4. The matrix and summary use two columns on desktop and one column at 900
   pixels or below.
5. The five fictional source labels wrap without clipping.
6. Trace Explorer and Research Intake + Receipt still render and switch tabs.
7. No upload, live inference, approval control, or graph UI appears.

If no browser target is available, record the limitation explicitly and rely
on the automated HTML, HTTP, syntax, and validator checks; do not claim visual
browser verification.

- [ ] **Step 7: Run final automated verification**

Run:

```powershell
python demo/frontend/validate_static_demo.py
node --check demo/frontend/app.js
git diff --check
git status --short
```

Expected:

- Python prints `STATIC_DEMO_FRONTEND_VALIDATION_OK` and `traces=5`.
- Node and `git diff --check` exit `0` with no output.
- Only `demo/frontend/README.md` and `demo/README.md` are uncommitted at this
  task boundary; unrelated `logs/` may remain untracked.

- [ ] **Step 8: Commit documentation**

```powershell
git add -- demo/frontend/README.md demo/README.md
git diff --cached --check
git commit -m "docs: document static RFP claim review"
```

- [ ] **Step 9: Confirm final scope**

Run:

```powershell
git status --short
git diff --stat HEAD~3..HEAD
git log -4 --oneline
```

Expected: no tracked changes remain, unrelated `logs/` may remain untracked,
and exactly three implementation commits follow the committed design and plan
history.
