# Where MNEMOS Helps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a static third demo tab that helps broad audiences understand evidence-sensitive MNEMOS use cases while preserving explicit authority boundaries.

**Architecture:** Extend the existing single-page tab interface with semantic HTML use-case cards and detail panels. Reuse the existing tab controller, add only a small in-page Trace Explorer CTA behavior, and extend the Python validator to assert the new static structure without introducing another data source or route.

**Tech Stack:** HTML5, CSS, browser-native JavaScript, Python 3.12 standard library, GitHub Pages.

## Global Constraints

- Use the top-level tab label `Where MNEMOS helps`.
- Lead with `AI memory for work where the source matters.`
- Use `Where MNEMOS can help` positioning, not `MNEMOS fully solves this workflow today`.
- Keep all use-case content static, public-safe, and repository-relative.
- Do not add APIs, uploads, live inference, backend-supported routes, graph implementation, or runtime behavior changes.
- Do not add legal, medical, tax, compliance-certification, production-authority, or guaranteed-correctness claims.
- Preserve the existing five precomputed trace files and their loading behavior.
- Preserve the existing Research Intake + Receipt static experience.
- Do not stage or modify the unrelated untracked `logs/` directory.

---

## File Structure

- Modify `demo/frontend/validate_static_demo.py`: validate tab targets, eight use-case anchors, matching detail panels, Trace Explorer CTAs, and public-boundary copy.
- Modify `demo/frontend/index.html`: add the third tab, eight overview cards, eight full detail panels, and static boundaries.
- Modify `demo/frontend/styles.css`: add approachable responsive layouts, emphasized cards, detail-panel styling, compact three-step traces, and CTA styling.
- Modify `demo/frontend/app.js`: refactor existing tab activation into one helper and connect use-case CTAs to the existing Trace Explorer tab.
- Modify `demo/frontend/README.md`: describe the third tab and its local validation.
- Modify `demo/README.md`: include the use-case tab in the public static-demo package description.

### Task 1: Validate and Add Static Use-Case Content

**Files:**
- Modify: `demo/frontend/validate_static_demo.py`
- Modify: `demo/frontend/index.html`

**Interfaces:**
- Consumes: existing `.tab-button[data-tab-target]` and `.tab-panel[id]` conventions.
- Produces: eight unique use-case IDs, eight matching `href="#<id>"` links, and eight `data-tab-link="trace-demo"` CTAs.

- [ ] **Step 1: Add failing structural validation**

Add the standard-library parser import and approved identifiers near the existing constants:

```python
from html.parser import HTMLParser

USE_CASE_IDS = (
    "use-case-bill",
    "use-case-compliance",
    "use-case-legal",
    "use-case-research",
    "use-case-proposal",
    "use-case-healthcare",
    "use-case-education",
    "use-case-journalism",
)

REQUIRED_USE_CASE_BOUNDARIES = (
    "does not provide financial, tax, or legal advice",
    "does not certify compliance or replace an auditor",
    "does not provide legal advice or replace attorney judgment",
    "does not establish settled truth",
    "does not approve submissions",
    "does not provide medical advice, diagnosis, or treatment guidance",
    "does not replace teacher judgment or academic integrity rules",
    "does not replace editorial judgment",
)
```

Add a parser that records IDs, in-page links, tab targets, and Trace Explorer CTA targets:

```python
class StaticDemoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.anchor_targets: list[str] = []
        self.tab_targets: list[str] = []
        self.tab_links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.ids.add(element_id)
        if tag == "a" and (href := attributes.get("href", "")).startswith("#"):
            self.anchor_targets.append(href[1:])
        if tab_target := attributes.get("data-tab-target"):
            self.tab_targets.append(tab_target)
        if tab_link := attributes.get("data-tab-link"):
            self.tab_links.append(tab_link)
```

Before the final `if errors:` block, parse `index.html` and add exact assertions:

```python
    index_html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    parser = StaticDemoHTMLParser()
    parser.feed(index_html)

    expected_tabs = {"trace-demo", "intake-demo", "use-cases-demo"}
    if set(parser.tab_targets) != expected_tabs:
        errors.append(f"tab targets mismatch: {sorted(parser.tab_targets)}")

    for use_case_id in USE_CASE_IDS:
        if use_case_id not in parser.ids:
            errors.append(f"missing use-case detail panel: {use_case_id}")
        if parser.anchor_targets.count(use_case_id) != 1:
            errors.append(f"use-case anchor count mismatch: {use_case_id}")

    if parser.tab_links.count("trace-demo") != len(USE_CASE_IDS):
        errors.append("each use case must link back to the Trace Explorer")

    normalized_html = " ".join(index_html.lower().split())
    for boundary in REQUIRED_USE_CASE_BOUNDARIES:
        if boundary not in normalized_html:
            errors.append(f"missing use-case boundary: {boundary}")

    if "mnemos fully solves this workflow today" in normalized_html:
        errors.append("unsupported fully-solves claim found")
```

- [ ] **Step 2: Run validation and confirm the new checks fail**

Run:

```powershell
python demo/frontend/validate_static_demo.py
```

Expected: exit code `1`, including `tab targets mismatch` and missing
`use-case-*` detail-panel errors. Existing trace validation must still report
no missing trace files.

- [ ] **Step 3: Add the third tab and tab hero**

In `demo/frontend/index.html`, add this button after Trace Explorer:

```html
<button class="tab-button" type="button" data-tab-target="use-cases-demo">Where MNEMOS helps</button>
```

Add a new `<section id="use-cases-demo" class="tab-panel">` after the existing
Trace Explorer panel and before Research Intake + Receipt. Begin it with:

```html
<section class="use-cases-intro">
  <p class="eyebrow">Where MNEMOS can help</p>
  <h2>AI memory for work where the source matters.</h2>
  <p>
    MNEMOS can help connect answers, decisions, handoffs, and evaluations
    back to the documents and evidence that shaped them.
  </p>
  <div class="boundary-strip" aria-label="Use-case boundaries">
    <span>Illustrative static examples</span>
    <span>Source-backed review</span>
    <span>No professional authority</span>
  </div>
</section>
```

- [ ] **Step 4: Add the eight overview cards**

Use a `<nav class="use-case-grid" aria-label="MNEMOS use cases">`. Add exactly
one link per ID, using `class="use-case-card featured"` for bill, compliance,
and legal, and `class="use-case-card"` for the other five:

```html
<a class="use-case-card featured" href="#use-case-bill">
  <span class="use-case-audience">Personal documents</span>
  <strong>Understand a bill without losing the source</strong>
  <span>See what changed and where each explanation came from.</span>
</a>
```

Use these remaining card labels and titles:

| ID | Audience label | Title |
| --- | --- | --- |
| `use-case-compliance` | Compliance / audit | Build evidence-backed compliance answers |
| `use-case-legal` | Legal review | Trace contract answers back to clauses |
| `use-case-research` | Research | Keep research claims attached to sources |
| `use-case-proposal` | Business proposals | Avoid unsupported proposal claims |
| `use-case-healthcare` | Healthcare documents | Organize health documents with source-backed explanations |
| `use-case-education` | Education | Study from sources, not guesses |
| `use-case-journalism` | Journalism | Trace claims back to documents |

- [ ] **Step 5: Add the eight semantic detail panels**

Create `<div class="use-case-details">` containing eight `<article>` elements.
Each article uses its approved ID, `class="use-case-detail panel"`, an `<h3>`
title, a `<dl class="use-case-facts">`, an ordered three-item
`<ol class="mini-trace">`, a boundary paragraph using
`class="use-case-boundary"`, a `Why this matters` paragraph, and this CTA:

```html
<a class="use-case-cta" href="#trace-demo" data-tab-link="trace-demo">Explore the evidence demo</a>
```

Use this exact content matrix:

| ID | Audience | Typical question | Documents involved | What MNEMOS shows | What MNEMOS does not claim | Example evidence trace | Why this matters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `use-case-bill` | People reviewing household or service bills | Why did my bill increase this month? | Current and prior bills, rate notices, and account statements | Charge changes, the source lines behind them, and unsupported explanations that need review | MNEMOS does not provide financial, tax, or legal advice. | Bill statement → charge comparison → source-backed explanation | A clear source trail makes an unfamiliar charge easier to question or verify. |
| `use-case-compliance` | Compliance, audit, risk, and governance teams | What evidence supports this control statement? | Control statements, policies, procedures, test evidence, and review records | Which artifacts support a statement, what remains missing, and the review boundary | MNEMOS does not certify compliance or replace an auditor. | Control statement → supporting artifact → audit review packet | Evidence remains attached when an answer moves between owners or review stages. |
| `use-case-legal` | Legal operations teams and counsel conducting document review | What does this agreement say about termination notice? | Agreements, amendments, schedules, and referenced policies | The relevant clause, its source location, and questions requiring professional judgment | MNEMOS does not provide legal advice or replace attorney judgment. | Contract clause → retrieval receipt → attorney review question | Reviewers can inspect the clause that shaped an answer before relying on it. |
| `use-case-research` | Researchers, students, librarians, and review teams | Which papers support this claim, and which disagree? | Papers, notes, datasets, and methodology appendices | Supporting and disagreeing sources, claim boundaries, and unresolved gaps | MNEMOS supports research review; it does not establish settled truth. | Research claim → supporting and disagreeing papers → bounded source summary | Claims remain connected to the literature instead of becoming detached summaries. |
| `use-case-proposal` | Proposal, capture, and business-development teams | Which proposal statements are supported by past performance evidence? | Requests for proposal, drafts, approved case studies, and past performance records | Supported statements, evidence gaps, and claims that need qualification | MNEMOS supports proposal review; it does not approve submissions. | Proposal claim → past performance evidence → support or gap flag | Teams can catch unsupported language before it becomes part of a formal submission. |
| `use-case-healthcare` | Patients, caregivers, and administrative teams reviewing their own documents | What changed between these two lab reports or benefit statements? | Lab reports, benefit statements, care summaries, and dated correspondence | Document differences, source locations, and questions to raise with a qualified professional | MNEMOS does not provide medical advice, diagnosis, or treatment guidance. | Dated health documents → change comparison → source-backed review questions | Source-backed comparisons can make complex records easier to discuss with the right professional. |
| `use-case-education` | Learners and educators working from assigned materials | Why is this answer supported by the lesson material? | Lessons, readings, assignment instructions, and rubrics | The passages behind an explanation and where the materials do not support a claim | MNEMOS is a learning aid; it does not replace teacher judgment or academic integrity rules. | Lesson material → cited passage → supported study explanation | Students can inspect why an answer follows from the assigned source. |
| `use-case-journalism` | Journalists, editors, researchers, and fact-checkers | Which records support this statement? | Public records, transcripts, reports, and source notes | Supporting records, conflicting evidence, and claims that remain unverified | MNEMOS supports evidence organization; it does not replace editorial judgment. | Public record → claim comparison → editorial review note | A visible evidence trail helps reviewers distinguish sourced reporting from unsupported assertion. |

- [ ] **Step 6: Run the structural validator**

Run:

```powershell
python demo/frontend/validate_static_demo.py
```

Expected:

```text
STATIC_DEMO_FRONTEND_VALIDATION_OK
traces=5
```

- [ ] **Step 7: Commit the static content and validator**

```powershell
git add -- demo/frontend/index.html demo/frontend/validate_static_demo.py
git diff --cached --check
git commit -m "feat: add static MNEMOS use-case content"
```

### Task 2: Style Use Cases and Connect Trace CTAs

**Files:**
- Modify: `demo/frontend/styles.css`
- Modify: `demo/frontend/app.js`
- Modify: `demo/frontend/validate_static_demo.py`

**Interfaces:**
- Consumes: `[data-tab-target]`, `.tab-panel`, and `[data-tab-link]` attributes from Task 1.
- Produces: `activateTab(targetId)` and responsive, visually emphasized use-case layouts.

- [ ] **Step 1: Add a failing JavaScript fragment check**

Extend `required_loader_fragments` in `validate_static_demo.py` with:

```python
        "function activateTab(targetId)",
        'document.querySelectorAll("[data-tab-link]")',
```

- [ ] **Step 2: Run validation and confirm the helper check fails**

Run:

```powershell
python demo/frontend/validate_static_demo.py
```

Expected: exit code `1` with `frontend loader missing fragment: function activateTab(targetId)`.

- [ ] **Step 3: Refactor tab activation and connect the CTAs**

Replace the duplicated activation body in `setupTabs()` with:

```javascript
function activateTab(targetId) {
  for (const tabButton of elements.tabButtons) {
    tabButton.classList.toggle("active", tabButton.dataset.tabTarget === targetId);
  }
  for (const panel of elements.tabPanels) {
    panel.classList.toggle("active", panel.id === targetId);
  }
}

function setupTabs() {
  for (const button of elements.tabButtons) {
    button.addEventListener("click", () => activateTab(button.dataset.tabTarget));
  }
  for (const link of document.querySelectorAll("[data-tab-link]")) {
    link.addEventListener("click", () => activateTab(link.dataset.tabLink));
  }
}
```

Do not alter trace fetching, trace selection, Research Intake rendering, or
`init()` sequencing.

- [ ] **Step 4: Add the use-case visual system**

Append focused styles to `styles.css` using existing color variables. Include:

```css
.use-cases-demo,
.use-case-details {
  display: grid;
  gap: 22px;
}

.use-cases-intro {
  max-width: 820px;
}

.use-cases-intro h2 {
  font-size: clamp(2rem, 4vw, 3.25rem);
}

.use-cases-intro > p:not(.eyebrow) {
  max-width: 760px;
  color: var(--muted);
  font-size: 1.08rem;
}

.use-case-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.use-case-card {
  display: grid;
  align-content: start;
  gap: 9px;
  min-height: 190px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
  color: var(--ink);
  padding: 18px;
  text-decoration: none;
  box-shadow: var(--shadow);
  transition: border-color 150ms ease, transform 150ms ease;
}

.use-case-card:hover,
.use-case-card:focus-visible {
  border-color: var(--accent);
  transform: translateY(-2px);
}

.use-case-card.featured {
  border-top: 4px solid var(--copper);
  background: linear-gradient(145deg, var(--accent-soft), var(--panel) 48%);
}

.use-case-audience {
  color: var(--accent);
  font-size: 0.76rem;
  font-weight: 780;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.use-case-card strong {
  font-size: 1.08rem;
  line-height: 1.2;
}

.use-case-card span:last-child,
.use-case-facts dd,
.use-case-why {
  color: var(--muted);
}

.use-case-detail {
  scroll-margin-top: 24px;
}

.use-case-detail h3 {
  font-size: clamp(1.4rem, 2.5vw, 2rem);
}

.use-case-facts {
  display: grid;
  grid-template-columns: 190px minmax(0, 1fr);
  gap: 10px 16px;
  margin: 18px 0;
}

.use-case-facts dt {
  color: var(--ink);
  font-weight: 760;
}

.use-case-facts dd {
  margin: 0;
}

.mini-trace {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 28px;
  margin: 12px 0 18px;
  padding: 0;
  list-style: none;
  counter-reset: mini-trace;
}

.mini-trace li {
  position: relative;
  border: 1px solid var(--line);
  border-radius: 9px;
  background: var(--paper);
  padding: 12px;
  font-weight: 700;
}

.mini-trace li:not(:last-child)::after {
  content: "→";
  position: absolute;
  top: 50%;
  right: -20px;
  color: var(--copper);
  transform: translateY(-50%);
}

.use-case-boundary {
  border-left: 4px solid var(--warn);
  background: var(--warn-soft);
  color: var(--warn);
  padding: 12px 14px;
}

.use-case-cta {
  display: inline-flex;
  width: fit-content;
  border-radius: 8px;
  background: var(--accent);
  color: var(--accent-ink);
  padding: 9px 14px;
  font-weight: 760;
  text-decoration: none;
}
```

Extend the `@media (max-width: 900px)` block with `.use-case-grid` set to two
columns. Extend the `@media (max-width: 560px)` block with `.use-case-grid`,
`.use-case-facts`, and `.mini-trace` set to one column, and hide mini-trace
arrows on narrow screens:

```css
.mini-trace li:not(:last-child)::after {
  display: none;
}
```

- [ ] **Step 5: Run syntax and structural checks**

Run:

```powershell
node --check demo/frontend/app.js
python demo/frontend/validate_static_demo.py
```

Expected: Node exits `0`; Python prints
`STATIC_DEMO_FRONTEND_VALIDATION_OK` and `traces=5`.

- [ ] **Step 6: Commit interaction and styling**

```powershell
git add -- demo/frontend/app.js demo/frontend/styles.css demo/frontend/validate_static_demo.py
git diff --cached --check
git commit -m "feat: style MNEMOS use-case tab"
```

### Task 3: Document and Verify the Complete Static Site

**Files:**
- Modify: `demo/frontend/README.md`
- Modify: `demo/README.md`
- Verify: `demo/frontend/index.html`
- Verify: `demo/frontend/app.js`
- Verify: `demo/frontend/styles.css`
- Verify: `demo/frontend/validate_static_demo.py`

**Interfaces:**
- Consumes: completed third-tab HTML, CSS, JavaScript, and validator.
- Produces: accurate preview documentation and a verified Pages-ready static artifact.

- [ ] **Step 1: Update frontend documentation**

Change `demo/frontend/README.md` from “two tabs” to “three tabs” and add:

```markdown
- `Where MNEMOS helps` for eight static, public-safe use cases spanning
  personal documents, enterprise evidence work, and professional review.
```

Under `What The UI Shows`, add:

```markdown
- Eight static use-case cards with full detail panels and authority boundaries
- In-page links from each use case back to the precomputed Trace Explorer
```

- [ ] **Step 2: Update package documentation**

In `demo/README.md`, add this paragraph under `Static Frontend And Pages Hosting`:

```markdown
The frontend also includes a static `Where MNEMOS helps` tab. It presents
eight illustrative use cases with source-oriented evidence paths and explicit
professional-authority boundaries. These pages do not add live workflow or
runtime capabilities.
```

- [ ] **Step 3: Run focused repository scans**

Run:

```powershell
rg -n "G:\\MNEMOS|G:/MNEMOS|C:\\Users|C:/Users|AppData|file:///|logs[/\\]" demo/frontend demo/README.md
rg -n "fully solves this workflow today|certifies compliance|guarantees correctness|provides legal advice|provides medical advice" demo/frontend
```

Expected: both commands return no matches. The approved negative boundary
phrases use `does not provide...`, so the second expression intentionally
searches unqualified positive claims.

- [ ] **Step 4: Build the Pages artifact in a temporary directory**

Run:

```powershell
$site = Join-Path $env:TEMP 'mnemos-use-cases-pages'
Remove-Item -Recurse -Force $site -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $site | Out-Null
Copy-Item -Recurse 'demo/frontend/*' $site
Copy-Item 'demo/demo_index.json' (Join-Path $site 'demo_index.json')
Copy-Item -Recurse 'demo/traces' (Join-Path $site 'traces')
```

Expected: the temporary site contains `index.html`, `app.js`, `styles.css`,
`demo_index.json`, and five files under `traces/`.

- [ ] **Step 5: Preview and verify the source-tree layout**

Start from the repository root:

```powershell
python -m http.server 8765
```

Open `http://127.0.0.1:8765/demo/frontend/`. Verify:

- all three tab buttons activate their matching panels;
- all eight use-case cards scroll to their matching detail panels;
- each detail CTA returns to the Trace Explorer;
- all five trace cards load and remain selectable;
- the three emphasized cards are visually prominent;
- mobile-width layout uses one-column cards and traces;
- all professional-authority boundaries remain visible.

Stop the preview server after verification.

- [ ] **Step 6: Preview and verify the staged Pages layout**

Start from the staged site directory:

```powershell
python -m http.server 8766 --directory $site
```

Open `http://127.0.0.1:8766/` and repeat the tab, anchor, CTA, and five-trace
checks. This proves both supported data layouts still work.

Stop the preview server after verification.

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
- Node exits `0` with no output.
- `git diff --check` produces no output.
- Only the two intended README files are uncommitted at this task boundary;
  `logs/` may remain unrelated and untracked.

- [ ] **Step 8: Commit documentation**

```powershell
git add -- demo/frontend/README.md demo/README.md
git diff --cached --check
git commit -m "docs: document MNEMOS use-case tab"
```

- [ ] **Step 9: Confirm final scope**

Run:

```powershell
git status --short
git log -4 --oneline
```

Expected: no tracked changes remain; unrelated `logs/` remains untracked; the
three implementation commits and this plan/spec history are visible.
