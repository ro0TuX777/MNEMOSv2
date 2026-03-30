#!/usr/bin/env python3
"""Run Wave 4 hygiene control-loop pass and emit reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.engram.model import Engram
from mnemos.governance.governor import Governor


def _load_engrams(path: Path) -> List[Engram]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        rows = payload.get("engrams", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("Input JSON must be a list or an object with an 'engrams' list")
    return [Engram.from_dict(row) for row in rows]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _evaluate_gate(payload: Dict[str, Any]) -> bool:
    report = payload["report"]
    threshold = payload["thresholds"]
    return (
        report["decay"]["stale_promoted"] >= threshold["min_stale_promoted"]
        and report["prune"]["promoted"] >= threshold["min_prune_promoted"]
        and report["sweep"]["contradictions_found"] >= threshold["min_contradictions_found"]
    )


def _render_report(payload: Dict[str, Any]) -> str:
    r = payload["report"]
    t = payload["thresholds"]
    gate = "PASS" if payload["pass"] else "HOLD"
    lines = [
        "# Wave 4 Hygiene Control-Loop Report",
        "",
        f"*Generated: {payload['timestamp']}*",
        "",
        "## Run",
        "",
        f"- Mode: `{payload['mode']}`",
        f"- Input engrams: `{payload['input_count']}`",
        f"- Gate result: `{gate}`",
        "",
        "## Metrics",
        "",
        "| Area | Metric | Value | Threshold |",
        "|---|---|---:|---:|",
        f"| Decay | scanned | {r['decay']['scanned']} | n/a |",
        f"| Decay | decayed | {r['decay']['decayed']} | n/a |",
        f"| Decay | stale_promoted | {r['decay']['stale_promoted']} | >= {t['min_stale_promoted']} |",
        f"| Prune | scanned | {r['prune']['scanned']} | n/a |",
        f"| Prune | promoted | {r['prune']['promoted']} | >= {t['min_prune_promoted']} |",
        f"| Sweep | clusters_scanned | {r['sweep']['clusters_scanned']} | n/a |",
        f"| Sweep | contradictions_found | {r['sweep']['contradictions_found']} | >= {t['min_contradictions_found']} |",
        f"| Sweep | winners_set | {r['sweep']['winners_set']} | n/a |",
        f"| Sweep | losers_set | {r['sweep']['losers_set']} | n/a |",
        f"| Aggregate | total_mutations | {r['total_mutations']} | n/a |",
        "",
    ]
    if payload.get("mutated_output_path"):
        lines.extend(["## Output", "", f"- Mutated snapshot: `{payload['mutated_output_path']}`", ""])
    lines.extend(
        [
            "## Promotion Recommendation",
            "",
            "- pass: keep hygiene runner enabled for scheduled dry-run evidence collection.",
            "- hold: review seed/threshold alignment and hygiene runner logic before promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Wave 4 hygiene control-loop benchmark pass")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "benchmarks" / "truthsets" / "wave4_hygiene_seed.json"),
        help="Input engram JSON file (list or {'engrams': [...]})",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "apply"],
        default="dry-run",
        help="dry-run computes without mutation; apply mutates in-memory and can emit snapshot",
    )
    parser.add_argument(
        "--now-iso",
        default="2026-03-30T00:00:00Z",
        help="ISO timestamp used by decay calculations",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "benchmarks" / "outputs"),
        help="Base output directory for raw/summaries artifacts",
    )
    parser.add_argument("--min-stale-promoted", type=int, default=1)
    parser.add_argument("--min-prune-promoted", type=int, default=1)
    parser.add_argument("--min-contradictions-found", type=int, default=1)
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Return non-zero exit code when thresholds are not met",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    engrams = _load_engrams(input_path)

    governor = Governor()
    report = governor.run_hygiene(
        engrams,
        now_iso=args.now_iso,
        dry_run=(args.mode == "dry-run"),
    )
    report_dict = {
        "decay": asdict(report.decay),
        "prune": asdict(report.prune),
        "sweep": asdict(report.sweep),
        "total_mutations": report.total_mutations,
    }

    thresholds = {
        "min_stale_promoted": args.min_stale_promoted,
        "min_prune_promoted": args.min_prune_promoted,
        "min_contradictions_found": args.min_contradictions_found,
    }
    payload = {
        "track": "wave4_hygiene_control_loop",
        "timestamp": args.now_iso,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": args.mode,
        "input_path": str(input_path),
        "input_count": len(engrams),
        "thresholds": thresholds,
        "report": report_dict,
        "governor_stats": governor.stats(),
    }
    payload["pass"] = _evaluate_gate(payload)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.output_dir)
    raw_path = out_base / "raw" / f"wave4_hygiene_{ts}_raw.json"
    report_path = out_base / "summaries" / f"wave4_hygiene_{ts}_report.md"

    mutated_output_path = ""
    if args.mode == "apply":
        mutated_output = out_base / "raw" / f"wave4_hygiene_{ts}_mutated_engrams.json"
        mutated_rows = [e.to_dict(include_governance=True, include_lineage=True) for e in engrams]
        _write_json(mutated_output, {"engrams": mutated_rows})
        mutated_output_path = str(mutated_output)
        payload["mutated_output_path"] = mutated_output_path

    _write_json(raw_path, payload)
    _write_text(report_path, _render_report(payload))

    print("Wave 4 hygiene run complete")
    print(f"  Raw:    {raw_path}")
    print(f"  Report: {report_path}")
    if mutated_output_path:
        print(f"  Mutated snapshot: {mutated_output_path}")
    print(f"  Gate:   {'PASS' if payload['pass'] else 'HOLD'}")

    if args.fail_on_gate and not payload["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
