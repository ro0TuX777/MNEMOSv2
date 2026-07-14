"""Validate the static MNEMOS demo frontend data contract."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

REQUIRED_TRACE_FIELDS = {
    "trace_id",
    "title",
    "demo_status",
    "public_safe",
    "question",
    "short_answer",
    "decision_state",
    "evidence_used",
    "trace_path",
    "excluded_or_unsupported_claims",
    "provenance",
    "limitations",
    "demo_panels",
}

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

LOCAL_PATH_PATTERNS = [
    re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]"),
    re.compile(r"/Users/"),
    re.compile(r"AppData"),
    re.compile(r"file:///"),
]

SECRET_PATTERNS = [
    re.compile(r"BEGIN (RSA |OPENSSH |EC |DSA |PRIVATE )?KEY"),
    re.compile(r"gho_[A-Za-z0-9_]+"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"password\s*[:=]\s*[^\s,}]+", re.IGNORECASE),
    re.compile(r"api[_-]?key\s*[:=]\s*[^\s,}]+", re.IGNORECASE),
    re.compile(r"credential\s*[:=]\s*[^\s,}]+", re.IGNORECASE),
]


class StaticDemoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.anchor_targets: list[str] = []
        self.tab_targets: list[str] = []
        self.tab_links: list[str] = []
        self.use_case_targets: list[str] = []
        self.use_case_controls: dict[str, str] = {}
        self.expanded_use_cases: list[str] = []
        self.use_case_panels: dict[str, tuple[str, str | None, bool]] = {}
        self.proposal_claim_states: list[str] = []

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
        if proposal_state := attributes.get("data-proposal-claim-state"):
            self.proposal_claim_states.append(proposal_state)
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_text(path: Path, patterns: list[re.Pattern[str]], label: str, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in patterns:
        if pattern.search(text):
            errors.append(f"{path.relative_to(ROOT)}: {label}: {pattern.pattern}")


def main() -> int:
    errors: list[str] = []
    index_path = ROOT / "demo_index.json"
    index = load_json(index_path)

    for item in index.get("traces", []):
        trace_path = ROOT / item["path"]
        if not trace_path.exists():
            errors.append(f"missing trace path: {item['path']}")
            continue

        trace = load_json(trace_path)
        missing = sorted(REQUIRED_TRACE_FIELDS - set(trace))
        if missing:
            errors.append(f"{item['path']}: missing fields {missing}")
        if trace.get("trace_id") != item.get("trace_id"):
            errors.append(f"{item['path']}: trace_id mismatch")
        if trace.get("demo_status") != "precomputed_static_demo":
            errors.append(f"{item['path']}: not a precomputed static demo")
        if trace.get("public_safe") is not True:
            errors.append(f"{item['path']}: public_safe is not true")

        for evidence in trace.get("evidence_used", []):
            artifact_path = Path(evidence.get("artifact_path", ""))
            if artifact_path.is_absolute() or not (ROOT.parent / artifact_path).exists():
                errors.append(f"{item['path']}: invalid evidence path {artifact_path}")

    for path in [ROOT / "README.md", ROOT / "demo_index.json", *ROOT.glob("traces/*.json")]:
        scan_text(path, LOCAL_PATH_PATTERNS, "local path", errors)
        scan_text(path, SECRET_PATTERNS, "secret pattern", errors)

    for path in [FRONTEND / "index.html", FRONTEND / "styles.css", FRONTEND / "app.js"]:
        if not path.exists():
            errors.append(f"missing frontend file: {path.relative_to(ROOT)}")

    app_js = (FRONTEND / "app.js").read_text(encoding="utf-8")
    styles_css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    required_loader_fragments = [
        'indexPath: "demo_index.json"',
        'indexPath: "../demo_index.json"',
        'traceBasePath: ""',
        'traceBasePath: "../"',
        "tracePathForItem",
        "function activateTab(targetId)",
        'document.querySelectorAll("[data-tab-link]")',
        'const DEFAULT_USE_CASE_ID = "use-case-bill"',
        "function activateUseCase(useCaseId, options = {})",
        "function syncUseCaseFromHash()",
        'window.addEventListener("hashchange", syncUseCaseFromHash)',
        "setupUseCases()",
    ]
    for fragment in required_loader_fragments:
        if fragment not in app_js:
            errors.append(f"frontend loader missing fragment: {fragment}")

    bill_text_rules = (
        r"\.bill-document,\s*\.bill-sources\s*\{[^}]*color:\s*#000000;",
        r"\.bill-sources li span\s*\{[^}]*color:\s*#000000;",
    )
    for rule in bill_text_rules:
        if not re.search(rule, styles_css, re.DOTALL):
            errors.append(f"frontend styles missing black bill text rule: {rule}")

    proposal_style_rules = (
        r"\.proposal-use-case\s*\{[^}]*--proposal-navy:\s*#112337;",
        r"\.proposal-review-layout\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s*320px;",
        r"\.proposal-claim-row\s*\{[^}]*color:\s*#000000;",
        r"@media\s*\(max-width:\s*900px\)[\s\S]*\.proposal-experience-header,\s*\.proposal-review-layout,\s*\.proposal-evidence-grid\s*\{[^}]*grid-template-columns:\s*1fr;",
    )
    for rule in proposal_style_rules:
        if not re.search(rule, styles_css, re.DOTALL):
            errors.append(f"frontend styles missing proposal rule: {rule}")

    index_html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    parser = StaticDemoHTMLParser()
    parser.feed(index_html)

    expected_tabs = {"trace-demo", "intake-demo", "use-cases-demo"}
    if set(parser.tab_targets) != expected_tabs:
        errors.append(f"tab targets mismatch: {sorted(parser.tab_targets)}")

    if parser.use_case_targets != list(USE_CASE_IDS):
        errors.append(f"use-case trigger order mismatch: {parser.use_case_targets}")

    if parser.expanded_use_cases != ["use-case-bill"]:
        errors.append(f"default expanded use case mismatch: {parser.expanded_use_cases}")

    for use_case_id in USE_CASE_IDS:
        expected_trigger_id = f"{use_case_id}-trigger"
        if use_case_id not in parser.ids:
            errors.append(f"missing use-case detail panel: {use_case_id}")
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

    if parser.tab_links.count("trace-demo") != len(USE_CASE_IDS):
        errors.append("each use case must link back to the Trace Explorer")

    normalized_html = " ".join(index_html.lower().split())
    for boundary in REQUIRED_USE_CASE_BOUNDARIES:
        if boundary not in normalized_html:
            errors.append(f"missing use-case boundary: {boundary}")

    if "mnemos fully solves this workflow today" in normalized_html:
        errors.append("unsupported fully-solves claim found")

    for fragment in REQUIRED_BILL_FRAGMENTS:
        if fragment.lower() not in normalized_html:
            errors.append(f"missing fictional bill fragment: {fragment}")

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

    calculated_increase = BILL_AMOUNTS["current_total"] - BILL_AMOUNTS["previous_total"]
    explained_increase = BILL_AMOUNTS["plan_change"] + BILL_AMOUNTS["usage_change"]
    if calculated_increase != Decimal("13.40") or calculated_increase != explained_increase:
        errors.append("fictional bill arithmetic mismatch")

    if errors:
        print("\n".join(errors))
        return 1

    print("STATIC_DEMO_FRONTEND_VALIDATION_OK")
    print(f"traces={len(index.get('traces', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
