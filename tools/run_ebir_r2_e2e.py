"""Run the EBIR-R2 evaluation pipeline end to end.

This orchestration tool is evaluation-only. It calls the frozen EBIR-R2
preflight, Markdown renderer, response compiler, and admin-only gold scorer.
It does not change reviewer-facing wording, response schema, compiler behavior,
scoring fields, retrieval, governance, promotion, Context Atlas, A1, Graph Tier,
stores, routes, or production APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.compile_ebir_r2_pilot_report import render_report  # noqa: E402
from tools.render_ebir_r2_pilot_markdown import render_forms  # noqa: E402
from tools.run_ebir_r2_preflight import run_preflight  # noqa: E402
from tools.score_ebir_r2_gold_report import score  # noqa: E402


DEFAULT_TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "ebir_r2_full_reviewer_tasks.json"
DEFAULT_REVIEWERS = PROJECT_ROOT / "configs" / "ebir_r2_reviewers.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "ebir_r2_full"
DEFAULT_PROTOCOL = PROJECT_ROOT / "docs" / "ebir_r2_trial_protocol.md"


def run_e2e(
    *,
    truthset: Path,
    reviewers: Path,
    protocol: Path,
    output_dir: Path,
    seed: int,
    responses_dir: Optional[Path],
    synthetic_dry_run: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    forms_dir = output_dir / "full_review_forms"
    response_dir = responses_dir or (output_dir / "full_responses")
    response_dir.mkdir(parents=True, exist_ok=True)

    preflight = run_preflight(
        truthset_path=truthset,
        reviewers_path=reviewers,
        output_dir=output_dir,
        seed=seed,
        blind=True,
    )

    render = render_forms(
        manifest=output_dir / "assignment_manifest.json",
        packets_dir=output_dir / "reviewer_packets",
        output_dir=forms_dir,
    )

    compile_report = render_report(
        protocol=protocol,
        manifest_path=output_dir / "assignment_manifest.json",
        responses_dir=response_dir,
        output=output_dir / "ebir_r2_full_report.md",
    )

    scoring: Optional[Dict[str, Any]] = None
    if compile_report["overall_pass"]:
        scoring = score(
            manifest_path=output_dir / "assignment_manifest.json",
            responses_dir=response_dir,
            output_json=output_dir / "ebir_r2_full_gold_scoring.json",
            output_md=output_dir / "ebir_r2_full_gold_scoring.md",
            synthetic_dry_run=synthetic_dry_run,
        )

    overall_pass = (
        bool(preflight.get("overall_pass"))
        and bool(render.get("overall_pass"))
        and bool(compile_report.get("overall_pass"))
        and (scoring is not None and scoring.get("status") == "PASS")
    )
    status = {
        "pipeline": "EBIR-R2 end-to-end evaluation",
        "evaluation_only": True,
        "synthetic_dry_run": synthetic_dry_run,
        "truthset": str(truthset),
        "reviewers": str(reviewers),
        "protocol": str(protocol),
        "output_dir": str(output_dir),
        "responses_dir": str(response_dir),
        "forms_dir": str(forms_dir),
        "preflight_pass": bool(preflight.get("overall_pass")),
        "render_pass": bool(render.get("overall_pass")),
        "compile_pass": bool(compile_report.get("overall_pass")),
        "scoring_pass": scoring is not None and scoring.get("status") == "PASS",
        "overall_pass": overall_pass,
        "artifacts": {
            "preflight_report": str(output_dir / "preflight_report.json"),
            "assignment_manifest": str(output_dir / "assignment_manifest.json"),
            "reviewer_forms": str(forms_dir),
            "response_report": str(output_dir / "ebir_r2_full_report.md"),
            "gold_scoring_json": str(output_dir / "ebir_r2_full_gold_scoring.json"),
            "gold_scoring_md": str(output_dir / "ebir_r2_full_gold_scoring.md"),
        },
        "blocked_until_responses": not bool(compile_report.get("overall_pass")),
        "value_claims_blocked": synthetic_dry_run or not overall_pass,
    }
    status_path = output_dir / "ebir_r2_e2e_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truthset", type=Path, default=DEFAULT_TRUTHSET)
    parser.add_argument("--reviewers", type=Path, default=DEFAULT_REVIEWERS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--responses-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    status = run_e2e(
        truthset=args.truthset,
        reviewers=args.reviewers,
        protocol=args.protocol,
        output_dir=args.output_dir,
        seed=args.seed,
        responses_dir=args.responses_dir,
        synthetic_dry_run=args.synthetic_dry_run,
    )
    print(f"[{'PASS' if status['preflight_pass'] else 'FAIL'}] preflight")
    print(f"[{'PASS' if status['render_pass'] else 'FAIL'}] markdown_render")
    print(f"[{'PASS' if status['compile_pass'] else 'FAIL'}] response_compile")
    print(f"[{'PASS' if status['scoring_pass'] else 'SKIP'}] gold_scoring")
    print(f"overall: {'PASS' if status['overall_pass'] else 'FAIL'}")
    print(f"status: {args.output_dir / 'ebir_r2_e2e_status.json'}")
    if args.fail_on_gate and not status["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
