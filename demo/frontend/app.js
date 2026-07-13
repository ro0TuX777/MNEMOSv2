const dataLayouts = [
  {
    name: "pages-root",
    indexPath: "demo_index.json",
    traceBasePath: "",
  },
  {
    name: "source-tree",
    indexPath: "../demo_index.json",
    traceBasePath: "../",
  },
];

const state = {
  index: null,
  traces: new Map(),
  activeTraceId: null,
};

const elements = {
  tabButtons: document.querySelectorAll(".tab-button"),
  tabPanels: document.querySelectorAll(".tab-panel"),
  inspectDemoJson: document.querySelector("#inspectDemoJson"),
  scenarioCards: document.querySelector("#scenarioCards"),
  loadingState: document.querySelector("#loadingState"),
  traceContent: document.querySelector("#traceContent"),
  traceStatus: document.querySelector("#traceStatus"),
  traceTitle: document.querySelector("#traceTitle"),
  traceQuestion: document.querySelector("#traceQuestion"),
  receiptId: document.querySelector("#receiptId"),
  shortAnswer: document.querySelector("#shortAnswer"),
  boundaryPanel: document.querySelector("#boundaryPanel"),
  whyThisMatters: document.querySelector("#whyThisMatters"),
  evidenceList: document.querySelector("#evidenceList"),
  decisionDetails: document.querySelector("#decisionDetails"),
  tracePath: document.querySelector("#tracePath"),
  unsupportedClaims: document.querySelector("#unsupportedClaims"),
  limitations: document.querySelector("#limitations"),
  provenance: document.querySelector("#provenance"),
  scenarioTemplate: document.querySelector("#scenarioTemplate"),
  intakeLoadingState: document.querySelector("#intakeLoadingState"),
  intakeContent: document.querySelector("#intakeContent"),
  intakePdfName: document.querySelector("#intakePdfName"),
  intakeExtractionPath: document.querySelector("#intakeExtractionPath"),
  intakeQuestion: document.querySelector("#intakeQuestion"),
  intakeReceipt: document.querySelector("#intakeReceipt"),
  intakeAnswer: document.querySelector("#intakeAnswer"),
  intakeBoundaries: document.querySelector("#intakeBoundaries"),
  intakeEvidenceList: document.querySelector("#intakeEvidenceList"),
  intakeTracePath: document.querySelector("#intakeTracePath"),
};

function text(value) {
  if (value === null || value === undefined || value === "") {
    return "Not recorded";
  }
  return String(value);
}

function normalizeStatus(value) {
  return text(value).replaceAll("_", " ");
}

function clear(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function appendDefinitionList(container, entries) {
  clear(container);
  for (const [label, value] of entries) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    const codeLike = /id|hash|mode|basis|state/i.test(label);
    if (codeLike) {
      const code = document.createElement("code");
      code.textContent = text(value);
      dd.appendChild(code);
    } else {
      dd.textContent = text(value);
    }
    container.append(dt, dd);
  }
}

function setupTabs() {
  for (const button of elements.tabButtons) {
    button.addEventListener("click", () => {
      const targetId = button.dataset.tabTarget;
      for (const tabButton of elements.tabButtons) {
        tabButton.classList.toggle("active", tabButton === button);
      }
      for (const panel of elements.tabPanels) {
        panel.classList.toggle("active", panel.id === targetId);
      }
    });
  }
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

async function loadIndexWithFallback() {
  const failures = [];
  for (const layout of dataLayouts) {
    try {
      const index = await fetchJson(layout.indexPath);
      return { index, layout };
    } catch (error) {
      failures.push(`${layout.name}: ${error.message}`);
    }
  }
  throw new Error(`Failed to load demo index. ${failures.join(" | ")}`);
}

function tracePathForItem(item) {
  return `${state.dataLayout.traceBasePath}${item.path}`;
}

function statusLabelForTrace(trace) {
  const statuses = new Set((trace.evidence_used || []).map((item) => item.status));
  if (trace.decision_state?.includes("not_retained") || statuses.has("rejected")) {
    return "not retained";
  }
  if (trace.decision_state?.includes("research_only") || statuses.has("research_only")) {
    return "research only";
  }
  if (trace.decision_state?.includes("shadow") || statuses.has("experimental")) {
    return "shadow / experimental";
  }
  return trace.demo_status || "static demo";
}

function renderScenarioCards() {
  clear(elements.scenarioCards);
  for (const item of state.index.traces) {
    const trace = state.traces.get(item.trace_id);
    const fragment = elements.scenarioTemplate.content.cloneNode(true);
    const button = fragment.querySelector(".scenario-card");
    const status = fragment.querySelector(".card-status");
    const selectedLabel = fragment.querySelector(".selected-label");
    const title = fragment.querySelector("strong");
    const question = fragment.querySelector(".card-question");
    const isActive = item.trace_id === state.activeTraceId;

    button.dataset.traceId = item.trace_id;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
    selectedLabel.hidden = !isActive;
    status.textContent = normalizeStatus(statusLabelForTrace(trace));
    title.textContent = item.title;
    question.textContent = item.question;
    button.addEventListener("click", () => selectTrace(item.trace_id));
    elements.scenarioCards.appendChild(fragment);
  }
}

function renderBoundaryPanel(trace) {
  clear(elements.boundaryPanel);
  const heading = document.createElement("h3");
  heading.textContent = "Boundary: what this trace does not authorize";
  const intro = document.createElement("p");
  intro.textContent =
    "Research-only, rejected, or shadow statuses are shown as boundaries, not as product claims.";
  const list = document.createElement("ul");
  list.className = "boundary-list";
  const boundaryItems = trace.demo_panels?.decision_boundary_panel || [];
  for (const item of boundaryItems) {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  }
  elements.boundaryPanel.append(heading, intro, list);
}

function renderEvidence(trace) {
  renderEvidenceInto(elements.evidenceList, trace);
}

function renderEvidenceInto(container, trace) {
  clear(container);
  for (const evidence of trace.evidence_used || []) {
    const article = document.createElement("article");
    article.className = "evidence-item";

    const header = document.createElement("div");
    header.className = "evidence-item-header";
    const title = document.createElement("h4");
    title.textContent = evidence.artifact_title;
    const status = document.createElement("span");
    status.className = "evidence-status";
    status.textContent = normalizeStatus(evidence.status);
    header.append(title, status);

    const summary = document.createElement("p");
    summary.textContent = evidence.public_safe_summary;

    const supported = document.createElement("p");
    supported.textContent = evidence.claim_supported;

    const path = document.createElement("code");
    path.className = "artifact-path";
    path.textContent = evidence.artifact_path;

    article.append(header, summary, supported, path);
    container.appendChild(article);
  }
}

function renderTracePath(trace) {
  renderTracePathInto(elements.tracePath, trace);
}

function renderWhyThisMatters(trace) {
  const answerPanel = trace.demo_panels?.answer_panel;
  const evidenceCount = (trace.evidence_used || []).length;
  const unsupportedCount = (trace.excluded_or_unsupported_claims || []).length;
  const boundaryCount = (trace.demo_panels?.decision_boundary_panel || []).length;
  elements.whyThisMatters.textContent =
    `${text(answerPanel)} This trace links ${evidenceCount} evidence artifact${evidenceCount === 1 ? "" : "s"} to a visible decision state, while keeping ${unsupportedCount} unsupported claim${unsupportedCount === 1 ? "" : "s"} and ${boundaryCount} boundary note${boundaryCount === 1 ? "" : "s"} explicit.`;
}

function renderTracePathInto(container, trace) {
  clear(container);
  for (const step of trace.trace_path || []) {
    const li = document.createElement("li");
    const body = document.createElement("div");
    body.className = "trace-step-body";
    const label = document.createElement("span");
    label.className = "trace-label";
    label.textContent = normalizeStatus(step.label);
    const description = document.createElement("span");
    description.className = "trace-description";
    description.textContent = step.description;
    body.append(label, description);
    li.appendChild(body);
    container.appendChild(li);
  }
}

function renderUnsupportedClaims(trace) {
  clear(elements.unsupportedClaims);
  for (const item of trace.excluded_or_unsupported_claims || []) {
    const article = document.createElement("article");
    article.className = "claim-item";
    const claim = document.createElement("h4");
    claim.textContent = item.claim;
    const reason = document.createElement("p");
    reason.textContent = item.reason;
    article.append(claim, reason);
    elements.unsupportedClaims.appendChild(article);
  }
}

function renderLimitations(trace) {
  clear(elements.limitations);
  for (const item of trace.limitations || []) {
    const li = document.createElement("li");
    li.textContent = item;
    elements.limitations.appendChild(li);
  }
}

function renderTrace(trace) {
  elements.traceStatus.textContent = normalizeStatus(trace.demo_status);
  elements.traceTitle.textContent = trace.title;
  elements.traceQuestion.textContent = trace.question;
  elements.receiptId.textContent = trace.provenance?.receipt_id || "demo-local-id";
  elements.shortAnswer.textContent = trace.short_answer;

  renderBoundaryPanel(trace);
  renderWhyThisMatters(trace);
  renderEvidence(trace);
  renderTracePath(trace);
  renderUnsupportedClaims(trace);
  renderLimitations(trace);

  appendDefinitionList(elements.decisionDetails, [
    ["Decision state", trace.decision_state],
    ["Public safe", trace.public_safe ? "true" : "false"],
    ["Evidence count", (trace.evidence_used || []).length],
    ["Trace steps", (trace.trace_path || []).length],
  ]);

  appendDefinitionList(elements.provenance, [
    ["Receipt ID", trace.provenance?.receipt_id],
    ["Mode", trace.provenance?.mode],
    ["Source basis", trace.provenance?.source_basis],
    ["Content hash", trace.provenance?.content_hash],
    ["Generated at", trace.provenance?.generated_at],
  ]);

  elements.loadingState.hidden = true;
  elements.traceContent.hidden = false;
}

function renderResearchIntake(trace) {
  if (!trace) {
    elements.intakeLoadingState.textContent = "Research Intake trace was not found.";
    elements.intakeLoadingState.hidden = false;
    elements.intakeContent.hidden = true;
    return;
  }

  elements.intakePdfName.textContent = "public-safe-sample-research.pdf";
  elements.intakeExtractionPath.value = "pypdf first; Docling OCR fallback if configured";
  elements.intakeQuestion.value = trace.question;
  elements.intakeAnswer.textContent = trace.short_answer;

  appendDefinitionList(elements.intakeReceipt, [
    ["Receipt ID", trace.provenance?.receipt_id],
    ["Mode", trace.provenance?.mode],
    ["Source basis", trace.provenance?.source_basis],
    ["Decision state", trace.decision_state],
    ["Content hash", trace.provenance?.content_hash],
    ["Generated at", trace.provenance?.generated_at],
  ]);

  clear(elements.intakeBoundaries);
  for (const item of trace.demo_panels?.decision_boundary_panel || []) {
    const li = document.createElement("li");
    li.textContent = item;
    elements.intakeBoundaries.appendChild(li);
  }

  renderEvidenceInto(elements.intakeEvidenceList, trace);
  renderTracePathInto(elements.intakeTracePath, trace);

  elements.intakeLoadingState.hidden = true;
  elements.intakeContent.hidden = false;
}

function selectTrace(traceId) {
  state.activeTraceId = traceId;
  renderScenarioCards();
  renderTrace(state.traces.get(traceId));
}

async function init() {
  try {
    setupTabs();
    const loaded = await loadIndexWithFallback();
    state.index = loaded.index;
    state.dataLayout = loaded.layout;
    elements.inspectDemoJson.href = loaded.layout.indexPath;
    for (const item of state.index.traces || []) {
      const trace = await fetchJson(tracePathForItem(item));
      state.traces.set(item.trace_id, trace);
    }

    if (!state.index.traces?.length) {
      throw new Error("No demo traces found.");
    }

    state.activeTraceId = state.index.traces[0].trace_id;
    renderScenarioCards();
    renderTrace(state.traces.get(state.activeTraceId));
    renderResearchIntake(state.traces.get("research_intake_ocr"));
  } catch (error) {
    elements.loadingState.textContent = error.message;
    elements.loadingState.hidden = false;
    elements.traceContent.hidden = true;
    elements.intakeLoadingState.textContent = error.message;
    elements.intakeLoadingState.hidden = false;
    elements.intakeContent.hidden = true;
  }
}

init();
