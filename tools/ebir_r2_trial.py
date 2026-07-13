"""External EBIR-R2 reviewer-trial kit.

This wrapper prepares separated reviewer/admin bundles, validates reviewer
material, compiles completed responses, and performs admin-only scoring after
responses are frozen. It reuses the existing EBIR-R2 harness and does not alter
MNEMOS runtime behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.compile_ebir_r2_pilot_report import render_report
from tools.render_ebir_r2_pilot_markdown import forbidden_hits, render_forms
from tools.run_ebir_r2_preflight import run_preflight
from tools.score_ebir_r2_gold_report import score


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "ebir_r2_external"
DEFAULT_PROTOCOL = PROJECT_ROOT / "docs" / "ebir_r2_trial_protocol.md"
PILOT_TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "ebir_r2_reviewer_tasks.json"
FULL_TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "ebir_r2_full_reviewer_tasks.json"
DEFAULT_REVIEWERS = PROJECT_ROOT / "configs" / "ebir_r2_reviewers.json"
TEMPLATE_DIR = PROJECT_ROOT / "templates" / "ebir_r2"

ADMIN_ONLY_PATTERNS = (
    "condition_key",
    "raw_evidence",
    "one_pass_reconciliation",
    "ebir_refinement",
    "gold_label",
    "expected_resolved_value",
    "expected_outcome",
    "assignment_manifest",
    "preflight_report",
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_template(name: str, target: Path) -> None:
    target.write_text((TEMPLATE_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")


def _truthset_path(args: argparse.Namespace) -> Path:
    if args.truthset:
        return args.truthset
    if args.pilot:
        return PILOT_TRUTHSET
    return FULL_TRUTHSET


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _public_manifest(forms_dir: Path, preflight_report: Dict[str, Any]) -> Dict[str, Any]:
    forms = sorted(forms_dir.glob("reviewer_R*.md"))
    return {
        "trial": "EBIR-R2 external reviewer trial",
        "reviewer_packet_count": sum(
            len(re.findall(r"^## Packet:", form.read_text(encoding="utf-8"), re.M))
            for form in forms
        ),
        "reviewer_file_count": len(forms),
        "reviewer_files": [f"packets/{form.name}" for form in forms],
        "case_count": preflight_report.get("case_count"),
        "promotion_status": preflight_report.get("promotion_status"),
        "human_value_claim": "blocked_until_independent_blinded_reviewers_complete_scoring",
        "reviewer_admin_boundary": "reviewers_receive_reviewer_bundle_only",
    }


def prepare(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    work_dir = output_dir / "work"
    forms_dir = work_dir / "review_forms"
    reviewer_bundle = output_dir / "reviewer_bundle"
    admin_bundle = output_dir / "admin_bundle"
    packets_dir = reviewer_bundle / "packets"
    responses_dir = output_dir / "responses"

    _reset_dir(work_dir)
    _reset_dir(reviewer_bundle)
    _reset_dir(admin_bundle)
    packets_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    truthset = _truthset_path(args)
    preflight_report = run_preflight(
        truthset_path=truthset,
        reviewers_path=args.reviewers,
        output_dir=work_dir,
        seed=args.seed,
        blind=True,
    )
    if not preflight_report.get("overall_pass"):
        print("[FAIL] preflight gates failed")
        return 1

    render_result = render_forms(
        manifest=work_dir / "assignment_manifest.json",
        packets_dir=work_dir / "reviewer_packets",
        output_dir=forms_dir,
    )
    if not render_result.get("overall_pass"):
        print("[FAIL] reviewer Markdown rendering failed")
        for error in render_result.get("gate_errors", []):
            print(f"  - {error}")
        return 1

    for form in sorted(forms_dir.glob("reviewer_R*.md")):
        shutil.copy2(form, packets_dir / form.name)
    _copy_template("README_FOR_REVIEWERS.md", reviewer_bundle / "README_FOR_REVIEWERS.md")
    _copy_template("RETURN_CHECKLIST.md", reviewer_bundle / "RETURN_CHECKLIST.md")
    _write_json(reviewer_bundle / "manifest_public.json", _public_manifest(forms_dir, preflight_report))

    shutil.copy2(work_dir / "assignment_manifest.json", admin_bundle / "assignment_manifest.json")
    shutil.copy2(work_dir / "preflight_report.json", admin_bundle / "preflight_report.json")
    (admin_bundle / "trial_admin_notes.md").write_text(
        "\n".join(
            [
                "# EBIR-R2 Admin Notes",
                "",
                "Do not distribute this directory to reviewers.",
                "",
                f"Truthset: `{truthset}`",
                f"Reviewer config: `{args.reviewers}`",
                f"Seed: `{args.seed}`",
                "",
                "Keep responses frozen before running admin-only scoring.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    validation = validate(args)
    if validation != 0:
        return validation

    print(f"[OK] reviewer bundle: {reviewer_bundle}")
    print(f"[OK] admin bundle: {admin_bundle}")
    print(f"[OK] responses dir: {responses_dir}")
    return 0


def _scan_reviewer_bundle(reviewer_bundle: Path) -> List[str]:
    errors: List[str] = []
    if not reviewer_bundle.exists():
        return [f"missing reviewer bundle: {reviewer_bundle}"]
    for path in reviewer_bundle.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(reviewer_bundle)
        markdown_hits = forbidden_hits(text) if path.suffix.lower() == ".md" else []
        for hit in markdown_hits:
            errors.append(f"{rel}: forbidden reviewer-facing text `{hit}`")
        lowered = text.lower()
        for pattern in ADMIN_ONLY_PATTERNS:
            if pattern.lower() in lowered:
                errors.append(f"{rel}: admin-only marker `{pattern}` present")
    return errors


def validate(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    reviewer_bundle = output_dir / "reviewer_bundle"
    admin_bundle = output_dir / "admin_bundle"
    errors = _scan_reviewer_bundle(reviewer_bundle)
    packet_files = sorted((reviewer_bundle / "packets").glob("reviewer_R*.md"))
    if not packet_files:
        errors.append("no reviewer packet Markdown files found")
    if not (reviewer_bundle / "README_FOR_REVIEWERS.md").exists():
        errors.append("reviewer README missing")
    if not (reviewer_bundle / "RETURN_CHECKLIST.md").exists():
        errors.append("return checklist missing")
    if not (admin_bundle / "assignment_manifest.json").exists():
        errors.append("admin assignment manifest missing")
    if not (admin_bundle / "preflight_report.json").exists():
        errors.append("admin preflight report missing")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        if args.fail_on_gate:
            return 1
        return 1
    print("[PASS] reviewer/admin bundle separation validated")
    return 0


def compile_responses(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    compiled_dir = output_dir / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)
    result = render_report(
        protocol=args.protocol,
        manifest_path=output_dir / "admin_bundle" / "assignment_manifest.json",
        responses_dir=args.responses_dir or (output_dir / "responses"),
        output=compiled_dir / "ebir_r2_external_report.md",
    )
    if result.get("overall_pass"):
        print(f"[PASS] compiled responses: {result['output']}")
        return 0
    for error in result.get("errors", []):
        print(f"[FAIL] {error}")
    return 1 if args.fail_on_gate else 0


def score_responses(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    scoring_dir = output_dir / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    result = score(
        manifest_path=output_dir / "admin_bundle" / "assignment_manifest.json",
        responses_dir=args.responses_dir or (output_dir / "responses"),
        output_json=scoring_dir / "ebir_r2_gold_scoring.json",
        output_md=scoring_dir / "ebir_r2_gold_scoring.md",
        synthetic_dry_run=args.synthetic_dry_run,
    )
    if result.get("status") == "PASS":
        print(f"[PASS] scoring complete: {scoring_dir}")
        return 0
    for error in result.get("errors", []):
        print(f"[FAIL] {error}")
    return 1 if args.fail_on_gate else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        p.add_argument("--responses-dir", type=Path, default=None)
        p.add_argument("--fail-on-gate", action="store_true")

    prepare_p = sub.add_parser("prepare", help="prepare reviewer/admin bundles")
    add_common(prepare_p)
    prepare_p.add_argument("--truthset", type=Path, default=None)
    group = prepare_p.add_mutually_exclusive_group()
    group.add_argument("--pilot", action="store_true", help="use small instrument-test truthset")
    group.add_argument("--full", action="store_true", help="use full reviewer truthset")
    prepare_p.add_argument("--reviewers", type=Path, default=DEFAULT_REVIEWERS)
    prepare_p.add_argument("--seed", type=int, default=20260619)
    prepare_p.set_defaults(func=prepare)

    validate_p = sub.add_parser("validate", help="validate reviewer/admin separation")
    add_common(validate_p)
    validate_p.set_defaults(func=validate)

    compile_p = sub.add_parser("compile", help="compile completed reviewer responses")
    add_common(compile_p)
    compile_p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    compile_p.set_defaults(func=compile_responses)

    score_p = sub.add_parser("score", help="score frozen responses against gold labels")
    add_common(score_p)
    score_p.add_argument("--synthetic-dry-run", action="store_true")
    score_p.set_defaults(func=score_responses)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

