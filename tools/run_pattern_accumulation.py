"""
run_pattern_accumulation.py — Phase 18 offline PatternEngramCandidate accumulation runner.

Reads CognitiveCycleRecord dicts from a JSON input file (collected from
search responses with cognitive_cycle=true), evaluates each cycle, extracts
PatternEngramCandidates for good/bad quality cycles, consolidates them into
the PatternCandidateStore, and emits a JSON report.

Usage
-----
    # Dry run (no store writes):
    python tools/run_pattern_accumulation.py --input cycles.json --dry-run

    # Live run with store persistence:
    python tools/run_pattern_accumulation.py \\
        --input cycles.json \\
        --store pattern_candidates.json \\
        --output benchmarks/results/pattern_accumulation_<timestamp>.json

    # Limit cycles processed:
    python tools/run_pattern_accumulation.py --input cycles.json --limit 20

    # Run gate check (exit 1 if gate fails):
    python tools/run_pattern_accumulation.py --input cycles.json --fail-on-gate

Input format
------------
The input JSON file must contain either:
  - A list of cycle dicts:  [{cycle_id: ..., trigger_type: ..., ...}, ...]
  - A dict with a "cognitive_cycles" key wrapping the list
  - A dict with a "cycles" key wrapping the list

Each cycle dict must be a CognitiveCycleRecord.to_dict() output.

Gate criteria
-------------
Gate passes if:
  - cycles_evaluated >= 10
  - candidates_new + candidates_merged >= 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Allow running from repo root without installing the package
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mnemos.cognitive.cycle import CognitiveCycleRecord
from mnemos.cognitive.cycle_evaluator import (
    CycleEvaluator,
    QUALITY_GOOD,
    QUALITY_NEUTRAL,
)
from mnemos.cognitive.pattern_learner import (
    SituationAbstractor,
    PatternLearner,
    PatternConsolidator,
)
from mnemos.cognitive.pattern_store import PatternCandidateStore


# ── Gate thresholds ──────────────────────────────────────────────────────────

_GATE_MIN_EVALUATED = 10
_GATE_MIN_CANDIDATES = 5


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_cycles(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Load cycle dicts from a JSON file.  Supports list and wrapped-dict formats."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, list):
        cycles = raw
    elif isinstance(raw, dict):
        cycles = (
            raw.get("cognitive_cycles")
            or raw.get("cycles")
            or []
        )
    else:
        raise ValueError(f"Unexpected JSON format in {path!r}: expected list or dict")

    if limit is not None and limit > 0:
        cycles = cycles[:limit]
    return cycles


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ── Main accumulation logic ──────────────────────────────────────────────────


def run_accumulation(
    *,
    input_path: str,
    store_path: Optional[str] = None,
    output_path: Optional[str] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    fail_on_gate: bool = False,
    klow: int = 7,
    khigh: int = 13,
) -> Dict[str, Any]:
    """
    Run the full offline accumulation pipeline.

    Returns the report dict.  Writes the store (if not dry_run and store_path
    is set) and the output JSON report.
    """
    run_id = str(uuid.uuid4())
    ran_at = datetime.now(timezone.utc).isoformat()

    # Load cycles
    raw_cycles = _load_cycles(input_path, limit)
    print(f"[accumulation] Loaded {len(raw_cycles)} cycle records from {input_path!r}")

    # Reconstruct CognitiveCycleRecord objects
    cycles: List[CognitiveCycleRecord] = []
    parse_errors = 0
    for i, raw in enumerate(raw_cycles):
        try:
            cycles.append(CognitiveCycleRecord.from_dict(raw))
        except Exception as exc:
            parse_errors += 1
            print(f"[accumulation] Warning: skipping cycle[{i}] — parse error: {exc}")

    print(f"[accumulation] Reconstructed {len(cycles)} cycles ({parse_errors} parse errors)")

    # Initialise components
    evaluator = CycleEvaluator(klow=klow, khigh=khigh)
    abstractor = SituationAbstractor()
    learner = PatternLearner()
    consolidator = PatternConsolidator()

    # Load existing store (or start fresh)
    store = PatternCandidateStore(persist_path=store_path)
    if store_path and not dry_run:
        store.load()
        print(f"[accumulation] Store loaded: {store.count()} existing candidates")

    # Track situations for consolidation
    existing_situations = []
    existing_pool = store.list_all()
    # Pre-compute situation tokens for existing candidates using their applies_when text
    from mnemos.cognitive.pattern_learner import SituationSummary
    for existing_cand in existing_pool:
        # Reconstruct a minimal SituationSummary from applies_when for consolidation
        synth_sit = SituationSummary(
            trigger_type="",
            route_class="",
            governance_mode="",
            forecast_active=False,
            candidate_pressure="",
            weak_dimensions=[],
            strong_dimensions=[],
            quality_label="",
            situation_text=existing_cand.applies_when,
        )
        existing_situations.append(synth_sit)

    # Counters
    n_good = n_bad = n_neutral = 0
    n_new = n_merged = n_contradicted = n_skipped = 0

    for cycle in cycles:
        evaluation = evaluator.evaluate(cycle)

        if evaluation.quality_label == QUALITY_NEUTRAL:
            n_neutral += 1
            continue
        if evaluation.quality_label == QUALITY_GOOD:
            n_good += 1
        else:
            n_bad += 1

        try:
            situation = abstractor.abstract(cycle, evaluation)
            candidate = learner.extract(cycle, evaluation, situation)
        except Exception as exc:
            print(f"[accumulation] Warning: extraction failed for cycle {cycle.cycle_id}: {exc}")
            n_skipped += 1
            continue

        result_candidate, action = consolidator.consolidate(
            candidate,
            existing_pool,
            new_situation=situation,
            existing_situations=existing_situations,
        )

        if action == "new":
            n_new += 1
            if not dry_run:
                store.add(result_candidate)
            existing_pool.append(result_candidate)
            existing_situations.append(situation)
            print(
                f"[accumulation]   NEW candidate: {result_candidate.candidate_id[:8]}... "
                f"type={result_candidate.pattern_type} quality={evaluation.quality_label}"
            )

        elif action == "merged":
            n_merged += 1
            if not dry_run:
                try:
                    store.update(result_candidate.candidate_id, result_candidate)
                except KeyError:
                    # Merged into a candidate from this run (not yet in store)
                    pass
            # Update pool entry
            for idx, p in enumerate(existing_pool):
                if p.candidate_id == result_candidate.candidate_id:
                    existing_pool[idx] = result_candidate
                    break
            print(
                f"[accumulation]   MERGED into: {result_candidate.candidate_id[:8]}... "
                f"({len(result_candidate.supporting_cycle_ids)} cycles)"
            )

        elif action == "contradicted":
            n_contradicted += 1
            if not dry_run:
                store.add(result_candidate)
            existing_pool.append(result_candidate)
            existing_situations.append(situation)
            print(
                f"[accumulation]   CONTRADICTED: {result_candidate.candidate_id[:8]}... "
                f"contradicts {result_candidate.contradicting_engram_ids}"
            )

    # Persist store
    if not dry_run and store_path:
        store.save()
        print(f"[accumulation] Store saved to {store_path!r} ({store.count()} total candidates)")

    # Build report
    gate_passed = (
        len(cycles) >= _GATE_MIN_EVALUATED
        and (n_new + n_merged) >= _GATE_MIN_CANDIDATES
    )

    report: Dict[str, Any] = {
        "run_id": run_id,
        "ran_at": ran_at,
        "input_path": input_path,
        "dry_run": dry_run,
        "limit": limit,
        "cycles_loaded": len(raw_cycles),
        "parse_errors": parse_errors,
        "cycles_evaluated": len(cycles),
        "good_cycles": n_good,
        "bad_cycles": n_bad,
        "neutral_cycles": n_neutral,
        "extraction_skipped": n_skipped,
        "candidates_new": n_new,
        "candidates_merged": n_merged,
        "candidates_contradicted": n_contradicted,
        "candidates_total": store.count() if not dry_run else (n_new + n_merged + n_contradicted),
        "store_path": store_path,
        "gate": {
            "min_cycles_required": _GATE_MIN_EVALUATED,
            "min_candidates_required": _GATE_MIN_CANDIDATES,
            "cycles_evaluated": len(cycles),
            "candidates_actionable": n_new + n_merged,
            "passed": gate_passed,
        },
    }

    # Write output report
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"[accumulation] Report written to {output_path!r}")

    # Gate result
    gate_emoji = "✓" if gate_passed else "✗"
    print(
        f"[accumulation] Gate {gate_emoji}: "
        f"{len(cycles)} cycles evaluated, "
        f"{n_new + n_merged} actionable candidates "
        f"(new={n_new} merged={n_merged} contradicted={n_contradicted})"
    )

    if fail_on_gate and not gate_passed:
        print("[accumulation] Gate FAILED — exiting with code 1", file=sys.stderr)
        sys.exit(1)

    return report


# ── CLI entry point ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline PatternEngramCandidate accumulation runner (Phase 18)"
    )
    p.add_argument(
        "--input", required=True,
        help="Path to JSON file containing CognitiveCycleRecord dicts",
    )
    p.add_argument(
        "--store",
        default=None,
        help="Path to PatternCandidateStore JSON file (created if absent)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path for the JSON accumulation report (auto-named if omitted)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate and extract but do not write to the store",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of cycles to process",
    )
    p.add_argument(
        "--fail-on-gate", action="store_true",
        help="Exit with code 1 if the phase gate does not pass",
    )
    p.add_argument("--klow", type=int, default=7, help="Evaluator bad-quality threshold")
    p.add_argument("--khigh", type=int, default=13, help="Evaluator good-quality threshold")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    output = args.output
    if output is None and not args.dry_run:
        ts = _timestamp()
        output = os.path.join(
            "benchmarks", "results", f"pattern_accumulation_{ts}.json"
        )

    run_accumulation(
        input_path=args.input,
        store_path=args.store,
        output_path=output,
        dry_run=args.dry_run,
        limit=args.limit,
        fail_on_gate=args.fail_on_gate,
        klow=args.klow,
        khigh=args.khigh,
    )
