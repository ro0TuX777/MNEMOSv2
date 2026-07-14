# Personal Documents Accordion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the long static use-case page with a single-open accordion and give Personal Documents a fictional, source-backed bill-review experience.

**Architecture:** Keep all content in the existing static HTML/CSS/JavaScript frontend. The HTML owns the eight accessible trigger/panel relationships and fictional bill content; a small DOM controller owns single-open and hash state; the Python validator owns structure, copy-boundary, and arithmetic assertions.

**Tech Stack:** HTML5, CSS custom properties, browser-native JavaScript, Python 3.12 standard library, GitHub Pages.

## Global Constraints

- Personal Documents is expanded by default.
- Use the approved accordion stack and MNEMOS hybrid theme.
- Keep the bill provider-neutral, fictional, public-safe, and static.
- Preserve `Where MNEMOS can help` positioning; do not claim that MNEMOS fully solves the workflow.
- Do not copy One NZ or BroadConnect logos, branded bill content, article copy, screenshots, or page structure.
- Preserve the other seven use cases and their existing copy until separate themes are approved.
- Preserve the existing five precomputed traces and both supported static data layouts.
- Do not add APIs, uploads, live inference, backend routes, graph implementation, runtime behavior changes, or product-authority claims.
- Do not modify trace JSON, retrieval, governance, promotion, context assembly, Engram schema, graph, or authority behavior.
- Do not stage or modify the unrelated untracked `logs/` directory.

---

## File Structure

- Modify `demo/frontend/validate_static_demo.py`: validate accordion relationships, default state, bill content, fictional receipt fields, and arithmetic.
- Modify `demo/frontend/index.html`: replace the overview-card/detail duplication with eight accordion items and the themed Personal Documents bill experience.
- Modify `demo/frontend/app.js`: add single-open, hash-aware accordion behavior without changing trace loading or intake rendering.
- Modify `demo/frontend/styles.css`: replace card-grid styles with accordion styles and add the scoped MNEMOS hybrid bill theme.
- Modify `demo/frontend/README.md`: document accordion behavior and the fictional static bill example.
- Modify `demo/README.md`: record the shorter single-open use-case presentation and its static boundaries.

### Task 1: Validate and Build the Semantic Accordion

**Files:**
- Modify: `demo/frontend/validate_static_demo.py`
- Modify: `demo/frontend/index.html`

**Interfaces:**
- Consumes: existing `USE_CASE_IDS`, boundary copy, Trace Explorer CTA convention, and eight current use-case detail bodies.
- Produces: eight `.use-case-trigger[data-use-case-target]` buttons and eight matching `[data-use-case-panel]` regions with Personal Documents expanded by default.

- [ ] **Step 1: Add failing accordion and bill validation**

Add the decimal import and fictional bill constants near the top of
`validate_static_demo.py`:

```python
from decimal import Decimal

BILL_AMOUNTS = {
    "previous_total": Decimal("89.00"),
    "current_total": Decimal("102.40"),
    "plan_change": Decimal("5.00"),
    "usage_change": Decimal("8.40"),
}

REQUIRED_BILL_FRAGMENTS = (
    "Why did my bill increase this month?",
    "current-month-statement.pdf",
    "prior-month-statement.pdf",
    "rate-change-notice.pdf",
    "+$13.40 total increase",
    "cause of out-of-plan usage",
    "illustrative_static_example",
    "No financial, tax, or legal advice",
)
```

Extend `StaticDemoHTMLParser.__init__()` with:

```python
self.use_case_targets: list[str] = []
self.use_case_controls: dict[str, str] = {}
self.expanded_use_cases: list[str] = []
self.use_case_panels: dict[str, tuple[str, str | None, bool]] = {}
```

Extend `handle_starttag()` with:

```python
if tag == "button" and (target := attributes.get("data-use-case-target")):
    self.use_case_targets.append(target)
    if controls := attributes.get("aria-controls"):
        self.use_case_controls[target] = controls
    if attributes.get("aria-expanded") == "true":
        self.expanded_use_cases.append(target)
if "data-use-case-panel" in attributes and (panel_id := attributes.get("id")):
    self.use_case_panels[panel_id] = (
        attributes.get("role", ""),
        attributes.get("aria-labelledby"),
        "hidden" in attributes,
    )
```

Replace the current `anchor_targets.count(use_case_id)` assertion with:

```python
    if parser.use_case_targets != list(USE_CASE_IDS):
        errors.append(f"use-case trigger order mismatch: {parser.use_case_targets}")

    if parser.expanded_use_cases != ["use-case-bill"]:
        errors.append(f"default expanded use case mismatch: {parser.expanded_use_cases}")

    for use_case_id in USE_CASE_IDS:
        expected_trigger_id = f"{use_case_id}-trigger"
        if parser.use_case_controls.get(use_case_id) != use_case_id:
            errors.append(f"use-case control mismatch: {use_case_id}")
        role, labelled_by, is_hidden = parser.use_case_panels.get(
            use_case_id, ("", None, False)
        )
        if role != "region" or labelled_by != expected_trigger_id:
            errors.append(f"use-case region mismatch: {use_case_id}")
        expected_hidden = use_case_id != "use-case-bill"
        if is_hidden != expected_hidden:
            errors.append(f"use-case default visibility mismatch: {use_case_id}")
```

Add the bill checks before `if errors:`:

```python
    for fragment in REQUIRED_BILL_FRAGMENTS:
        if fragment.lower() not in normalized_html:
            errors.append(f"missing fictional bill fragment: {fragment}")

    calculated_increase = BILL_AMOUNTS["current_total"] - BILL_AMOUNTS["previous_total"]
    explained_increase = BILL_AMOUNTS["plan_change"] + BILL_AMOUNTS["usage_change"]
    if calculated_increase != Decimal("13.40") or calculated_increase != explained_increase:
        errors.append("fictional bill arithmetic mismatch")
```

- [ ] **Step 2: Run validation and verify RED**

Run:

```powershell
python demo/frontend/validate_static_demo.py
```

Expected: exit `1` with `use-case trigger order mismatch`, default-expanded,
region, and fictional-bill fragment errors. Existing trace checks must not
report missing or invalid trace files.

- [ ] **Step 3: Replace the use-case card grid with one accordion**

In `demo/frontend/index.html`, keep `.use-cases-intro`, remove
`.use-case-grid` and `.use-case-details`, and add:

```html
<div class="use-case-accordion" aria-label="MNEMOS use cases">
  <!-- eight use-case-item sections in USE_CASE_IDS order -->
</div>
```

Every item follows this exact relationship pattern:

```html
<section class="use-case-item">
  <button
    id="use-case-compliance-trigger"
    class="use-case-trigger"
    type="button"
    data-use-case-target="use-case-compliance"
    aria-expanded="false"
    aria-controls="use-case-compliance"
  >
    <span><span class="use-case-audience">Compliance / audit</span><strong>Build evidence-backed compliance answers</strong></span>
    <span class="use-case-toggle" aria-hidden="true">+</span>
  </button>
  <article
    id="use-case-compliance"
    class="use-case-panel"
    data-use-case-panel
    role="region"
    aria-labelledby="use-case-compliance-trigger"
    hidden
  >
    <!-- move the existing compliance detail body here unchanged -->
  </article>
</section>
```

Use this exact trigger map and preserve the existing panel copy:

| Panel ID | Trigger ID | Audience | Title | Initial state |
| --- | --- | --- | --- | --- |
| `use-case-bill` | `use-case-bill-trigger` | Personal documents | Understand a bill without losing the source | expanded, panel visible |
| `use-case-compliance` | `use-case-compliance-trigger` | Compliance / audit | Build evidence-backed compliance answers | collapsed, panel hidden |
| `use-case-legal` | `use-case-legal-trigger` | Legal review | Trace contract answers back to clauses | collapsed, panel hidden |
| `use-case-research` | `use-case-research-trigger` | Research | Keep research claims attached to sources | collapsed, panel hidden |
| `use-case-proposal` | `use-case-proposal-trigger` | Business proposals | Avoid unsupported proposal claims | collapsed, panel hidden |
| `use-case-healthcare` | `use-case-healthcare-trigger` | Healthcare documents | Organize health documents with source-backed explanations | collapsed, panel hidden |
| `use-case-education` | `use-case-education-trigger` | Education | Study from sources, not guesses | collapsed, panel hidden |
| `use-case-journalism` | `use-case-journalism-trigger` | Journalism | Trace claims back to documents | collapsed, panel hidden |

The Personal Documents section uses `class="use-case-item bill-use-case is-expanded"`.
Its trigger uses `aria-expanded="true"`; its panel has no `hidden` attribute.

- [ ] **Step 4: Add the fictional Personal Documents panel**

Inside `#use-case-bill`, replace the current generic body with this structure:

```html
<div class="bill-experience-header">
  <div>
    <p class="eyebrow">Personal document understanding — bill</p>
    <h3>Understand a bill without losing the source</h3>
    <p class="question">Why did my bill increase this month?</p>
  </div>
  <span class="bill-static-badge">Illustrative static example</span>
</div>

<div class="bill-review-grid">
  <section class="bill-document" aria-labelledby="bill-comparison-heading">
    <p class="eyebrow">Fictional statement comparison</p>
    <h4 id="bill-comparison-heading">This month versus last month</h4>
    <table class="bill-table">
      <thead><tr><th scope="col">Item</th><th scope="col">Previous</th><th scope="col">Current</th><th scope="col">Change</th></tr></thead>
      <tbody>
        <tr><th scope="row">Monthly plan</th><td>$89.00</td><td>$94.00</td><td class="bill-change">+$5.00</td></tr>
        <tr><th scope="row">Out-of-plan usage</th><td>$0.00</td><td>$8.40</td><td class="bill-change">+$8.40</td></tr>
        <tr class="bill-total"><th scope="row">Total</th><td>$89.00</td><td>$102.40</td><td class="bill-change">+$13.40</td></tr>
      </tbody>
    </table>
  </section>

  <section class="bill-answer" aria-labelledby="bill-answer-heading">
    <p class="eyebrow">Source-backed explanation</p>
    <h4 id="bill-answer-heading">What shaped the answer</h4>
    <p>The bill increased by $13.40. The statement shows a $5.00 monthly plan-rate adjustment and $8.40 in out-of-plan usage.</p>
    <p class="bill-unsupported">The documents establish the charges, but they do not establish why the extra usage occurred.</p>
  </section>
</div>
```

Add the source list and receipt below the comparison:

```html
<div class="bill-evidence-grid">
  <section class="bill-sources" aria-labelledby="bill-sources-heading">
    <p class="eyebrow">Evidence used</p>
    <h4 id="bill-sources-heading">Public-safe fictional documents</h4>
    <ul>
      <li><code>current-month-statement.pdf</code><span>Plan, usage, and total-due lines</span></li>
      <li><code>prior-month-statement.pdf</code><span>Previous plan and total amounts</span></li>
      <li><code>rate-change-notice.pdf</code><span>Documented $5.00 plan adjustment</span></li>
    </ul>
  </section>

  <section class="bill-receipt" aria-labelledby="bill-receipt-heading">
    <p class="eyebrow">Illustrative receipt</p>
    <h4 id="bill-receipt-heading">What is supported—and what is not</h4>
    <dl>
      <dt>Question</dt><dd>Why did my bill increase this month?</dd>
      <dt>Supported result</dt><dd><code>+$13.40 total increase</code></dd>
      <dt>Unsupported result</dt><dd><code>cause of out-of-plan usage</code></dd>
      <dt>Decision state</dt><dd><code>illustrative_static_example</code></dd>
    </dl>
  </section>
</div>
```

Finish with the approved path, boundary, why-it-matters copy, and two CTAs:

```html
<h4>Evidence path</h4>
<ol class="mini-trace"><li>Bill statement</li><li>Charge comparison</li><li>Bounded explanation</li></ol>
<p class="use-case-boundary">MNEMOS does not provide financial, tax, or legal advice.</p>
<p class="use-case-why"><strong>Why this matters:</strong> A visible source trail separates documented charge changes from explanations the documents cannot support.</p>
<div class="use-case-actions">
  <a class="use-case-cta" href="#trace-demo" data-tab-link="trace-demo">Explore the evidence demo</a>
  <a class="use-case-secondary-cta" href="demo_index.json" data-demo-index-link>Inspect demo JSON</a>
</div>
```

- [ ] **Step 5: Preserve generic content for the other seven panels**

Move each existing article body unchanged into its matching accordion region.
Keep each existing `Audience`, `Typical question`, `Documents involved`,
`What MNEMOS shows`, trace, boundary, why-it-matters copy, and
`data-tab-link="trace-demo"` CTA. Do not add new themes or copy.

- [ ] **Step 6: Run validation and verify GREEN**

Run:

```powershell
python demo/frontend/validate_static_demo.py
```

Expected:

```text
STATIC_DEMO_FRONTEND_VALIDATION_OK
traces=5
```

- [ ] **Step 7: Commit semantic content**

```powershell
git add -- demo/frontend/index.html demo/frontend/validate_static_demo.py
git diff --cached --check
git commit -m "feat: add static bill-review accordion content"
```

### Task 2: Add Hash-Aware Single-Open Behavior and Hybrid Theme

**Files:**
- Modify: `demo/frontend/validate_static_demo.py`
- Modify: `demo/frontend/app.js`
- Modify: `demo/frontend/styles.css`

**Interfaces:**
- Consumes: Task 1 trigger attributes, panel attributes, and existing `activateTab(targetId)`.
- Produces: `DEFAULT_USE_CASE_ID`, `activateUseCase(useCaseId, options)`, `syncUseCaseFromHash()`, and responsive accordion/bill styles.

- [ ] **Step 1: Add failing JavaScript behavior assertions**

Extend `required_loader_fragments` with:

```python
        'const DEFAULT_USE_CASE_ID = "use-case-bill"',
        "function activateUseCase(useCaseId, options = {})",
        "function syncUseCaseFromHash()",
        'window.addEventListener("hashchange", syncUseCaseFromHash)',
        "setupUseCases()",
```

- [ ] **Step 2: Run validation and verify RED**

Run:

```powershell
python demo/frontend/validate_static_demo.py
```

Expected: exit `1` listing the new missing JavaScript fragments.

- [ ] **Step 3: Register accordion elements and default state**

Add after `dataLayouts`:

```javascript
const DEFAULT_USE_CASE_ID = "use-case-bill";
```

Add to `elements`:

```javascript
useCaseTriggers: document.querySelectorAll("[data-use-case-target]"),
useCasePanels: document.querySelectorAll("[data-use-case-panel]"),
demoIndexLinks: document.querySelectorAll("[data-demo-index-link]"),
```

- [ ] **Step 4: Implement the minimal accordion controller**

Add after `setupTabs()`:

```javascript
function validUseCaseId(useCaseId) {
  return Array.from(elements.useCasePanels).some((panel) => panel.id === useCaseId);
}

function useCaseIdFromHash() {
  const useCaseId = decodeURIComponent(window.location.hash.slice(1));
  return validUseCaseId(useCaseId) ? useCaseId : null;
}

function activateUseCase(useCaseId, options = {}) {
  const selectedId = validUseCaseId(useCaseId) ? useCaseId : DEFAULT_USE_CASE_ID;
  for (const trigger of elements.useCaseTriggers) {
    const isSelected = trigger.dataset.useCaseTarget === selectedId;
    trigger.setAttribute("aria-expanded", String(isSelected));
    trigger.closest(".use-case-item")?.classList.toggle("is-expanded", isSelected);
    const indicator = trigger.querySelector(".use-case-toggle");
    if (indicator) {
      indicator.textContent = isSelected ? "−" : "+";
    }
  }
  for (const panel of elements.useCasePanels) {
    panel.hidden = panel.id !== selectedId;
  }
  if (options.updateHash) {
    window.history.replaceState(null, "", `#${selectedId}`);
  }
}

function syncUseCaseFromHash() {
  const useCaseId = useCaseIdFromHash();
  if (!useCaseId) {
    activateUseCase(DEFAULT_USE_CASE_ID);
    return false;
  }
  activateTab("use-cases-demo");
  activateUseCase(useCaseId);
  return true;
}

function setupUseCases() {
  for (const trigger of elements.useCaseTriggers) {
    trigger.addEventListener("click", () => {
      activateUseCase(trigger.dataset.useCaseTarget, { updateHash: true });
    });
  }
  window.addEventListener("hashchange", syncUseCaseFromHash);
  if (!syncUseCaseFromHash()) {
    activateUseCase(DEFAULT_USE_CASE_ID);
  }
}
```

Call `setupUseCases()` immediately after `setupTabs()` in `init()`.

- [ ] **Step 5: Keep the secondary manifest CTA repository-relative**

After `elements.inspectDemoJson.href = loaded.layout.indexPath;`, add:

```javascript
for (const link of elements.demoIndexLinks) {
  link.href = loaded.layout.indexPath;
}
```

Do not alter trace fetching, trace selection, Research Intake rendering, or
error handling.

- [ ] **Step 6: Replace card-grid styles with accordion styles**

Remove `.use-case-grid`, `.use-case-card`, and `.use-case-details` rules. Add:

```css
.use-cases-demo,
.use-case-accordion {
  display: grid;
  gap: 14px;
}

.use-case-item {
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
  box-shadow: var(--shadow);
}

.use-case-trigger {
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  border: 0;
  background: var(--panel);
  color: var(--ink);
  padding: 16px 18px;
  text-align: left;
}

.use-case-trigger > span:first-child {
  display: grid;
  gap: 4px;
}

.use-case-trigger:hover,
.use-case-trigger:focus-visible,
.use-case-item.is-expanded > .use-case-trigger {
  background: var(--accent-soft);
}

.use-case-toggle {
  display: grid;
  flex: 0 0 32px;
  width: 32px;
  height: 32px;
  place-items: center;
  border-radius: 50%;
  background: var(--paper);
  color: var(--accent);
  font-size: 1.2rem;
  font-weight: 800;
}

.use-case-panel {
  border-top: 1px solid var(--line-soft);
  padding: 22px;
}

.use-case-panel[hidden] {
  display: none;
}
```

- [ ] **Step 7: Add the scoped MNEMOS hybrid bill theme**

Add:

```css
.bill-use-case {
  --bill-navy: #112337;
  --bill-deep-blue: #00194c;
  --bill-blue: #204ce5;
  --bill-source-green: #0b7f66;
  --bill-source-soft: #d9f99d;
  --bill-paper: #f3f8f6;
}

.bill-use-case > .use-case-trigger {
  background: var(--bill-navy);
  color: #ffffff;
}

.bill-use-case .use-case-audience,
.bill-use-case .use-case-toggle {
  color: var(--bill-source-soft);
}

.bill-use-case .use-case-toggle {
  background: rgba(255, 255, 255, 0.1);
}

.bill-experience-header,
.bill-review-grid,
.bill-evidence-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.bill-experience-header {
  align-items: start;
  margin-bottom: 18px;
}

.bill-static-badge {
  width: fit-content;
  justify-self: end;
  border-radius: 999px;
  background: var(--bill-source-soft);
  color: #244600;
  padding: 6px 10px;
  font-size: 0.78rem;
  font-weight: 780;
}

.bill-document,
.bill-sources {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--bill-paper);
  padding: 18px;
}

.bill-answer,
.bill-receipt {
  border-radius: 12px;
  background: var(--bill-deep-blue);
  color: #ffffff;
  padding: 18px;
}

.bill-answer .eyebrow,
.bill-receipt .eyebrow,
.bill-answer h4,
.bill-receipt h4 {
  color: var(--bill-source-soft);
}

.bill-table {
  width: 100%;
  margin-top: 14px;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.bill-table th,
.bill-table td {
  border-bottom: 1px solid var(--line);
  padding: 9px 6px;
  text-align: right;
}

.bill-table th:first-child {
  text-align: left;
}

.bill-change {
  color: var(--bill-blue);
  font-weight: 800;
}

.bill-total th,
.bill-total td {
  border-top: 2px solid var(--bill-blue);
  font-weight: 800;
}

.bill-unsupported {
  border-left: 3px solid var(--warn);
  padding-left: 10px;
  color: #ffe4a8;
}

.bill-evidence-grid {
  margin-top: 16px;
}

.bill-sources ul {
  display: grid;
  gap: 9px;
  margin: 14px 0 0;
  padding: 0;
  list-style: none;
}

.bill-sources li {
  display: grid;
  gap: 3px;
  border-left: 3px solid var(--bill-source-green);
  padding-left: 10px;
}

.bill-sources li span,
.bill-receipt dt {
  color: var(--muted);
  font-size: 0.82rem;
}

.bill-receipt dl {
  display: grid;
  grid-template-columns: 130px minmax(0, 1fr);
  gap: 9px 12px;
  margin: 14px 0 0;
}

.bill-receipt dd {
  margin: 0;
}

.use-case-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.use-case-secondary-cta {
  display: inline-flex;
  width: fit-content;
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--accent);
  padding: 9px 14px;
  font-weight: 760;
  text-decoration: none;
}
```

At `max-width: 900px`, set `.bill-review-grid`, `.bill-evidence-grid`, and
`.bill-experience-header` to one column and set `.bill-static-badge` to
`justify-self: start`. At `max-width: 560px`, make `.use-case-panel` padding
`16px`, set `.bill-document` to `overflow-x: auto`, set `.bill-table` to
`min-width: 520px`, and make `.bill-receipt dl` one column.

Add a reduced-motion guard even though interaction uses no required animation:

```css
@media (prefers-reduced-motion: reduce) {
  .use-case-trigger,
  .use-case-toggle {
    transition: none;
  }
}
```

- [ ] **Step 8: Run behavior-adjacent checks**

Run:

```powershell
node --check demo/frontend/app.js
python demo/frontend/validate_static_demo.py
```

Expected: Node exits `0`; Python prints
`STATIC_DEMO_FRONTEND_VALIDATION_OK` and `traces=5`.

- [ ] **Step 9: Commit interaction and theme**

```powershell
git add -- demo/frontend/app.js demo/frontend/styles.css demo/frontend/validate_static_demo.py
git diff --cached --check
git commit -m "feat: add accessible use-case accordion behavior"
```

### Task 3: Document and Verify the Complete Static Experience

**Files:**
- Modify: `demo/frontend/README.md`
- Modify: `demo/README.md`
- Verify: `demo/frontend/index.html`
- Verify: `demo/frontend/app.js`
- Verify: `demo/frontend/styles.css`
- Verify: `demo/frontend/validate_static_demo.py`

**Interfaces:**
- Consumes: completed semantic accordion, hash controller, scoped bill theme, and unchanged trace loader.
- Produces: accurate public documentation and a verified source-tree and Pages-root artifact.

- [ ] **Step 1: Update frontend documentation**

Add under the `Where MNEMOS helps` tab description in
`demo/frontend/README.md`:

```markdown
The use-case tab uses a single-open accordion to keep the page compact.
`Personal Documents` opens by default with a fictional bill comparison,
source-backed answer, unsupported-cause boundary, and illustrative receipt.
The example contains no real provider or customer data and performs no live
document processing.
```

Under `What The UI Shows`, replace the old eight-card bullet with:

```markdown
- Eight accessible use-case accordion controls with one expanded panel
- A fictional Personal Documents bill comparison and static evidence receipt
- Generic detail panels for the seven use cases awaiting separate visual themes
```

- [ ] **Step 2: Update package documentation**

Replace the existing use-case paragraph in `demo/README.md` with:

```markdown
The frontend also includes a static `Where MNEMOS helps` accordion. Personal
Documents opens by default with a fictional, provider-neutral bill comparison
and illustrative receipt. The remaining use cases preserve their public-safe
detail content in compact expandable rows. These examples do not add live
workflow, upload, inference, professional authority, or runtime capabilities.
```

- [ ] **Step 3: Run public-surface scans**

Run:

```powershell
$publicFiles = @('demo/frontend/index.html', 'demo/frontend/styles.css', 'demo/frontend/app.js', 'demo/frontend/README.md', 'demo/README.md')
rg -n 'G:\\MNEMOS|G:/MNEMOS|C:\\Users|C:/Users|file:///[A-Za-z]|logs[/\\]' $publicFiles
rg -n 'One NZ|BroadConnect|fully solves this workflow today|certifies compliance|guarantees correctness|provides legal advice|provides medical advice' $publicFiles
```

Expected: both scans return no matches. The reference names and URLs remain in
the design spec only, not the public demo.

- [ ] **Step 4: Build and smoke-test the Pages artifact**

Stage the site in the OS temp directory:

```powershell
$site = Join-Path $env:TEMP 'mnemos-personal-documents-pages'
Remove-Item -Recurse -Force -LiteralPath $site -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $site | Out-Null
Copy-Item -Recurse 'demo/frontend/*' $site
Copy-Item 'demo/demo_index.json' (Join-Path $site 'demo_index.json')
Copy-Item -Recurse 'demo/traces' (Join-Path $site 'traces')
```

Serve the repository root on port `8765` and the staged Pages root on port
`8766`. For each layout, verify HTTP `200` for the page, manifest, and all five
trace paths. Confirm the returned HTML contains all eight
`data-use-case-target` values and the fictional bill fragments.

- [ ] **Step 5: Verify browser behavior and responsive presentation**

Open the source-tree preview at
`http://127.0.0.1:8765/demo/frontend/` and verify:

1. Personal Documents is expanded by default.
2. Selecting each of the other seven rows collapses the previous panel and
   expands exactly one selected panel.
3. The selected hash updates without a page reload.
4. Loading `#use-case-legal` activates the parent tab and legal panel.
5. Loading an invalid hash falls back to Personal Documents without an error.
6. Keyboard Enter and Space activate every trigger.
7. Trace Explorer and Research Intake + Receipt still work.
8. Desktop and mobile widths keep the bill table, receipt, and compact rows
   readable.
9. No live control, upload field, or inference claim appears.

Repeat checks 1–7 against `http://127.0.0.1:8766/`.

- [ ] **Step 6: Run final automated verification**

Run:

```powershell
python demo/frontend/validate_static_demo.py
node --check demo/frontend/app.js
git diff --check
git status --short
```

Expected:

- Python prints `STATIC_DEMO_FRONTEND_VALIDATION_OK` and `traces=5`.
- Node exits `0` with no output.
- `git diff --check` produces no output.
- Only the two intended README files are uncommitted at this task boundary;
  unrelated `logs/` may remain untracked.

- [ ] **Step 7: Commit documentation**

```powershell
git add -- demo/frontend/README.md demo/README.md
git diff --cached --check
git commit -m "docs: document personal documents accordion"
```

- [ ] **Step 8: Confirm final scope**

Run:

```powershell
git status --short
git diff --stat HEAD~3..HEAD
git log -4 --oneline
```

Expected: no tracked changes remain, unrelated `logs/` remains untracked, and
the three implementation commits plus the committed design/plan history are
visible.
