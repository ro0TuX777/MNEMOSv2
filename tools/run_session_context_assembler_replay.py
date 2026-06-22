"""CLI entry point for the session-context-assembler Phase 3 offline A/B/C
replay harness.

Authorized scope: offline replay only against the frozen R0/R1 benchmark
corpora. This script
has no import path into mnemos/, service/, or mnemos_sdk/; it only imports
the isolated prototype.session_context_assembler package and the standard
library. It performs no runtime integration, no agent wiring, no
durable-memory write, no retrieval-ranking change, no authority change, no
governance mutation, and no promotion behavior. See
docs/adr/0007-session-context-assembler-shadow-only.md and
docs/session_context_assembler_phase_3_notes.md.

Usage:
    python tools/run_session_context_assembler_replay.py [--corpus-version r0|r1] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prototype.session_context_assembler.corpus import load_validated_corpus  # noqa: E402
from prototype.session_context_assembler.replay import (  # noqa: E402
    compute_aggregate_gates,
    compute_condition_comparison,
    compute_s1_advancement_gates,
    run_replay,
)

CORPUS_PATH = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r0.json"
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r0.manifest.json"
RESULTS_JSON_PATH = REPO_ROOT / "benchmarks" / "results" / "session_context_assembler_r0_replay.json"
RESULTS_MD_PATH = REPO_ROOT / "benchmarks" / "results" / "session_context_assembler_r0_replay.md"

DISCLAIMER_LABELS = (
    "OFFLINE_REPLAY_ONLY",
    "NO_HUMAN_VALUE_CLAIM",
    "NO_RUNTIME_INTEGRATION",
    "NO_PRODUCTION_READINESS_CLAIM",
)

KNOWN_LIMITATIONS = (
    "required_source_recall for sca_r0_urd_001 and sca_r0_urd_002 is 0.0 "
    "for ALL THREE conditions, including the unconstrained full-history "
    "condition A. This is an extractor/corpus ceiling, not a selection "
    "failure: those two cases' required source IDs are referenced only in "
    "case metadata and are never literally embedded in turn text, so no "
    "regex-based extractor can recover them regardless of how much history "
    "is included. See docs/session_context_assembler_phase_1_notes.md "
    "('ID extraction is a stand-in'). A future corpus revision (r1) adding "
    "an explicit source-artifact pool separate from turn text would remove "
    "this ceiling.",
    "contradiction_awareness_result for sca_r0_urd_003 is 'omitted' for all "
    "three conditions for the same reason: its required source ID "
    "(SRC-SCA-explain-format-discussion) is never inline-extractable, so "
    "the all-required-ids-recovered rule cannot return 'mixed' even though "
    "the decision ID (DEC-SCA-113) is recoverable. Only the "
    "contradiction_aware_followup family (caf_001-003) has fully "
    "inline-extractable required IDs in r0, so it is the only family where "
    "contradiction_awareness_result currently differentiates conditions.",
)


def _render_comparison_table(comparison) -> List[str]:
    lines = [
        "| Condition | Cases | Mean prior-decision recall | Decision-artifact retention | Mean source recall "
        "| Contradiction awareness | Irrelevant-history selection rate | Mean prompt token reduction | Mean token estimate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for condition, stats in comparison.items():
        lines.append(
            f"| {condition} | {stats['case_count']} | "
            f"{_fmt(stats['mean_required_prior_decision_recall'])} | "
            f"{_fmt(stats['mean_decision_artifact_retention'])} | "
            f"{_fmt(stats['mean_required_source_recall'])} | "
            f"{_fmt(stats['contradiction_awareness_rate'])} | "
            f"{_fmt(stats['irrelevant_history_selection_rate'])} | "
            f"{_fmt(stats['mean_prompt_token_reduction'])} | "
            f"{_fmt(stats['mean_token_estimate'])} |"
        )
    return lines


def render_markdown(
    gate_summary, comparison, stress_comparison, stress_budget, manifest,
    seed, generated_at, corpus_version="r0", known_limitations=KNOWN_LIMITATIONS,
    advancement=None,
) -> str:
    lines = []
    title_suffix = " Phase 4R S1" if advancement is not None else ""
    lines.append(
        f"# Session Context Assembler — {corpus_version.upper()}{title_suffix} Replay Report"
    )
    lines.append("")

    lines.append(" ".join(f"`{label}`" for label in DISCLAIMER_LABELS))
    lines.append("")

    if advancement is not None:
        lines.append("## Phase 4R S1 advancement requirements")
        lines.append("")
        lines.append("| Requirement | Value | Required | Passed |")
        lines.append("|---|---|---|---|")
        for name, gate in advancement["gates"].items():
            lines.append(
                f"| {name} | {_fmt(gate['value'])} | {gate['required']} | "
                f"{'PASS' if gate['passed'] else 'FAIL'} |"
            )
        lines.append("")
        outcome = (
            "PASS" if advancement["all_advancement_requirements_passed"]
            else "FAIL"
        )
        lines.append(f"**Phase 4R S1 outcome: {outcome}**")
        lines.append("")
    condition_word = "four" if advancement is not None else "three"
    lines.append(
        f"This report measures replay behavior of {condition_word} context-construction "
        f"conditions against the frozen `session_context_assembler_{corpus_version}` "
        "corpus. It is an offline prototype evaluation. It does not "
        "authorize consumer runtime integration, production use, or any claim that "
        "human value or production readiness is established — see "
        "[ADR 0007](../../docs/adr/0007-session-context-assembler-shadow-only.md)."
    )
    lines.append("")
    lines.append(f"- Generated: {generated_at}")
    lines.append(f"- Seed: {seed}")
    lines.append(f"- Corpus manifest file_sha256: `{manifest['file_sha256']}`")
    lines.append(f"- Cases replayed: {manifest['case_count']}")
    lines.append("")

    lines.append(
        "## Required safety/provenance gates "
        f"({gate_summary['evaluated_condition']})"
    )
    lines.append("")
    lines.append("| Gate | Value | Required | Passed |")
    lines.append("|---|---|---|---|")
    for name, gate in gate_summary["gates"].items():
        lines.append(f"| {name} | {gate['value']} | {gate['required']} | {'PASS' if gate['passed'] else 'FAIL'} |")
    lines.append("")
    overall = "PASS" if gate_summary["all_required_gates_passed"] else "FAIL"
    lines.append(f"**All required safety/provenance gates: {overall}**")
    lines.append("")
    if advancement is None:
        lines.append(
            "A gate PASS authorizes review of Phase 4 gates only. It does not "
            "authorize consumer runtime integration or production use."
        )
    else:
        lines.append(
            "A Phase 4R PASS authorizes Phase 5 human-review design only. It "
            "does not authorize consumer runtime integration or production use."
        )
    lines.append("")

    lines.append("## Condition comparison (descriptive, not gated)")
    lines.append("")
    if corpus_version == "r0":
        lines.append(
            "B and C use each case's non-binding R0 budget; A is unconstrained. "
            "The primary R0 comparison therefore selects full history throughout."
        )
    else:
        lines.append(
            "B, C0, and C1 use each case's identical binding R1 budget (50% of full history, "
            "with 15-token and largest-atomic-episode floors); A is unconstrained."
        )
    lines.append("")
    lines.extend(_render_comparison_table(comparison))
    lines.append("")
    lines.append(
        "### Harness self-check: fixed stress budget "
        f"(token_budget_override={stress_budget}, not a corpus-budget replay)"
    )
    lines.append("")
    lines.append(
        "This secondary pass forces B and C below their natural full-history "
        "size to confirm the truncation/selection machinery actually "
        "differentiates conditions when a budget binds. It exists only to "
        "show the harness is not silently inert — it is not a corpus-budget "
        "replay, is not gated, and is not a quality claim."
    )
    lines.append("")
    lines.extend(_render_comparison_table(stress_comparison))
    lines.append("")
    lines.append(
        "`NO_HUMAN_VALUE_CLAIM`: the above is replay-measured recall/token "
        "accounting against this corpus's own fixture labels, not a human "
        "or model judgment of answer quality. No Phase 5 human review has run."
    )
    lines.append("")

    lines.append("## Observations (not conclusions)")
    lines.append("")
    b = comparison["B_sliding_window"]["mean_decision_artifact_retention"]
    observed_condition = (
        "C1_selector_s1_mandatory_preservation"
        if advancement is not None else "C_governed_episode_selected"
    )
    c = comparison[observed_condition]["mean_decision_artifact_retention"]
    relation = "higher than" if c > b else "lower than" if c < b else "equal to"
    lines.append(
        f"Under the corpus-budget replay, {observed_condition} decision-artifact retention ({_fmt(c)}) "
        f"is {relation} B ({_fmt(b)})."
    )
    lines.append("")

    lines.append("## Known measurement limitations in this run")
    lines.append("")
    for item in known_limitations:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Per-case results")
    lines.append("")
    result_name = (
        "session_context_assembler_r1_s1_replay.json"
        if advancement is not None
        else f"session_context_assembler_{corpus_version}_replay.json"
    )
    lines.append(
        "Full per-case, per-condition records are in "
        f"[{result_name}]({result_name})."
    )
    lines.append("")

    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus-version", choices=("r0", "r1"), default="r0")
    parser.add_argument(
        "--selector-s1", action="store_true",
        help="Include Phase 4R selector S1 as condition C1 (R1 only).",
    )
    parser.add_argument(
        "--stress-budget",
        type=int,
        default=20,
        help=(
            "Fixed token budget for the harness self-check stress pass "
            "(default 20). r0's configured expected_context_budget values "
            "don't bind against this corpus's short fixture turns, so the "
            "primary replay alone can't show whether B and C diverge under "
            "real pressure; this secondary pass is a harness diagnostic, "
            "not a corpus-budget replay or a quality claim."
        ),
    )
    args = parser.parse_args()
    if args.selector_s1 and args.corpus_version != "r1":
        parser.error("--selector-s1 is authorized only with --corpus-version r1")

    if args.corpus_version == "r0":
        corpus_path, manifest_path = CORPUS_PATH, MANIFEST_PATH
        results_json_path, results_md_path = RESULTS_JSON_PATH, RESULTS_MD_PATH
        known_limitations = KNOWN_LIMITATIONS
    else:
        corpus_path = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r1.json"
        manifest_path = REPO_ROOT / "benchmarks" / "truthsets" / "session_context_assembler_r1.manifest.json"
        results_json_path = REPO_ROOT / "benchmarks" / "results" / "session_context_assembler_r1_replay.json"
        results_md_path = REPO_ROOT / "benchmarks" / "results" / "session_context_assembler_r1_replay.md"
        known_limitations = (
            "R1 repairs structural measurability but does not tune the Phase 1 "
            "selector. Low retention is an evaluation result, not a corpus-generation failure.",
        )
        if args.selector_s1:
            results_json_path = REPO_ROOT / "benchmarks" / "results" / "session_context_assembler_r1_s1_replay.json"
            results_md_path = REPO_ROOT / "benchmarks" / "results" / "session_context_assembler_r1_s1_replay.md"
            known_limitations = (
                "S1 classifies contradiction candidates from deterministic runtime text "
                "signals and structured source links; no model-level semantic judgment is used.",
                "Five cases emit a conservative budget-insufficient abstention because an "
                "additional runtime-visible mandatory candidate cannot fit. The R1-scored "
                "artifact is nevertheless retained in each case; no omission is silent.",
            )

    corpus_data = load_validated_corpus(corpus_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    records = run_replay(
        corpus_data, manifest["file_sha256"], seed=args.seed,
        include_s1=args.selector_s1,
    )
    evaluated_condition = (
        "C1_selector_s1_mandatory_preservation"
        if args.selector_s1 else "C_governed_episode_selected"
    )
    gate_summary = compute_aggregate_gates(records, condition=evaluated_condition)
    comparison = compute_condition_comparison(records)
    advancement = compute_s1_advancement_gates(records) if args.selector_s1 else None

    stress_records = run_replay(
        corpus_data, manifest["file_sha256"], seed=args.seed,
        token_budget_override=args.stress_budget, include_s1=args.selector_s1,
    )
    stress_comparison = compute_condition_comparison(stress_records)

    generated_at = datetime.now(timezone.utc).isoformat()

    results_doc = {
        "schema": f"session_context_assembler_{args.corpus_version}_replay_v1",
        "disclaimer_labels": list(DISCLAIMER_LABELS),
        "generated_at": generated_at,
        "seed": args.seed,
        "corpus_manifest_hash": manifest["file_sha256"],
        "case_count": manifest["case_count"],
        "gate_summary": gate_summary,
        "s1_advancement": advancement,
        "condition_comparison": comparison,
        "records": records,
        "harness_self_check_stress_pass": {
            "description": (
                "Secondary diagnostic pass using a fixed token_budget_override "
                "instead of each case's expected_context_budget, to confirm "
                "the truncation/selection machinery differentiates B and C "
                "when a budget is tight enough to bind. Not a corpus-budget "
                "replay, not gated, not a quality claim."
            ),
            "token_budget_override": args.stress_budget,
            "condition_comparison": stress_comparison,
            "records": stress_records,
        },
    }

    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.write_text(json.dumps(results_doc, indent=2, sort_keys=True), encoding="utf-8")
    results_md_path.write_text(
        render_markdown(
            gate_summary, comparison, stress_comparison, args.stress_budget,
            manifest, args.seed, generated_at, args.corpus_version,
            known_limitations, advancement
        ),
        encoding="utf-8",
    )

    condition_count = 4 if args.selector_s1 else 3
    print(f"Replayed {manifest['case_count']} cases x {condition_count} conditions (seed={args.seed}).")
    print(
        "All required safety/provenance gates passed: "
        f"{gate_summary['all_required_gates_passed']}"
    )
    print(f"Wrote {results_json_path}")
    print(f"Wrote {results_md_path}")
    return 0 if gate_summary["all_required_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
