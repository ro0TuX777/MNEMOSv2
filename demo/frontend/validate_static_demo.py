"""Validate the static MNEMOS demo frontend data contract."""

from __future__ import annotations

import json
import re
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
    required_loader_fragments = [
        'indexPath: "demo_index.json"',
        'indexPath: "../demo_index.json"',
        'traceBasePath: ""',
        'traceBasePath: "../"',
        "tracePathForItem",
    ]
    for fragment in required_loader_fragments:
        if fragment not in app_js:
            errors.append(f"frontend loader missing fragment: {fragment}")

    if errors:
        print("\n".join(errors))
        return 1

    print("STATIC_DEMO_FRONTEND_VALIDATION_OK")
    print(f"traces={len(index.get('traces', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
