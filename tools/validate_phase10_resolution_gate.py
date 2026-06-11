"""Validate Phase 10 Resolution Engram priority through the live service."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "knowledge_reconciliation_v1.json"
RAW_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "summaries"


def _resolution_id(entity_key: str, attribute_key: str) -> str:
    conflict_group_id = f"conflict:{entity_key}:{attribute_key}"
    return f"resolution_{conflict_group_id.replace(':', '_')}"


def _is_resolution(result: Dict[str, Any]) -> bool:
    return bool(
        ((result.get("engram") or {}).get("metadata") or {}).get(
            "is_resolution_engram", False
        )
    )


def _governance_modifier(result: Dict[str, Any]) -> float:
    governance = result.get("governance") or {}
    modifiers = governance.get("modifiers") or {}
    return float(modifiers.get("contradiction", 0.0) or 0.0)


def _score_by_id(results: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for result in results:
        engram_id = ((result.get("engram") or {}).get("id")) or ""
        if engram_id:
            out[str(engram_id)] = float(result.get("governed_score", result.get("score", 0.0)) or 0.0)
    return out


def evaluate(*, truthset: Path, base_url: str, top_k: int) -> Dict[str, Any]:
    payload = json.loads(truthset.read_text(encoding="utf-8"))
    rows = []

    for case in payload["collisions"]:
        parent_ids = [item["id"] for item in case["engrams"]]
        expected_resolution_id = _resolution_id(case["entity_key"], case["attribute_key"])
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/mnemos/search",
            json={
                "query": case["query"],
                "top_k": top_k,
                "filters": {"metadata.truthset_case_id": case["id"]},
                "retrieval_mode": "semantic",
                "governance": "advisory",
                "explain": True,
                "explain_governance": True,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        top = results[0] if results else {}
        top_id = ((top.get("engram") or {}).get("id")) or None
        top_is_resolution = _is_resolution(top)
        top_hit_pass = top_is_resolution and top_id == expected_resolution_id

        suppressed = (
            ((data.get("meta") or {}).get("governance_explain") or {}).get(
                "suppressed_candidates"
            )
            or []
        )
        suppressed_ids = {str(row.get("engram_id")) for row in suppressed}
        scores = _score_by_id(results)
        resolution_score = scores.get(expected_resolution_id, 0.0)
        lower_scored_parents = [
            pid for pid in parent_ids if pid in scores and scores[pid] < resolution_score
        ]
        suppressed_parents = [pid for pid in parent_ids if pid in suppressed_ids]
        missing_parents = [
            pid for pid in parent_ids if pid not in scores and pid not in suppressed_ids
        ]
        parent_suppression_pass = all(
            pid in suppressed_parents or pid in lower_scored_parents
            for pid in parent_ids
        )

        modifier = _governance_modifier(top)
        modifier_pass = top_hit_pass and abs(modifier - 1.25) <= 1e-6

        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected_resolution_id": expected_resolution_id,
                "top_id": top_id,
                "top_is_resolution": top_is_resolution,
                "top_hit_pass": top_hit_pass,
                "parent_ids": parent_ids,
                "suppressed_parents": suppressed_parents,
                "lower_scored_parents": lower_scored_parents,
                "missing_parents": missing_parents,
                "parent_suppression_pass": parent_suppression_pass,
                "resolution_contradiction_modifier": modifier,
                "modifier_pass": modifier_pass,
                "result_ids": [
                    ((result.get("engram") or {}).get("id")) for result in results
                ],
            }
        )

    gates = {
        "top_hit": all(row["top_hit_pass"] for row in rows),
        "parent_suppression": all(row["parent_suppression_pass"] for row in rows),
        "modifier_audit": all(row["modifier_pass"] for row in rows),
    }
    return {
        "truthset": str(truthset.resolve().relative_to(PROJECT_ROOT)),
        "base_url": base_url,
        "rows": rows,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def write_artifacts(result: Dict[str, Any]) -> tuple[Path, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"phase10_resolution_gate_{timestamp}_raw.json"
    summary_path = SUMMARY_DIR / f"phase10_resolution_gate_{timestamp}_summary.md"
    raw_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    rows = result["rows"]
    lines = [
        "# Phase 10 Resolution Gate Summary",
        "",
        f"- Base URL: `{result['base_url']}`",
        f"- Truthset: `{result['truthset']}`",
        f"- Gate: **{'PASS' if result['overall_pass'] else 'FAIL'}**",
        "",
        "## Gates",
        "",
        f"- Top hit: `{'PASS' if result['gates']['top_hit'] else 'FAIL'}`",
        f"- Parent suppression: `{'PASS' if result['gates']['parent_suppression'] else 'FAIL'}`",
        f"- Modifier audit: `{'PASS' if result['gates']['modifier_audit'] else 'FAIL'}`",
        "",
        "## Cases",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['id']}` top=`{row['top_id']}` "
            f"modifier=`{row['resolution_contradiction_modifier']}` "
            f"missing_parents=`{','.join(row['missing_parents'])}`"
        )
    lines.extend(["", f"- Raw: `{raw_path.name}`", ""])
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return raw_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truthset", type=Path, default=TRUTHSET)
    parser.add_argument("--base-url", default="http://localhost:8700")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args()

    result = evaluate(truthset=args.truthset, base_url=args.base_url, top_k=args.top_k)
    for row in result["rows"]:
        print(
            f"{row['id']}: top={row['top_id']} "
            f"modifier={row['resolution_contradiction_modifier']:.2f} "
            f"missing_parents={len(row['missing_parents'])}"
        )
    print(f"top_hit: {'PASS' if result['gates']['top_hit'] else 'FAIL'}")
    print(
        "parent_suppression: "
        f"{'PASS' if result['gates']['parent_suppression'] else 'FAIL'}"
    )
    print(f"modifier_audit: {'PASS' if result['gates']['modifier_audit'] else 'FAIL'}")
    print(f"overall: {'PASS' if result['overall_pass'] else 'FAIL'}")
    if not args.no_artifacts:
        raw, summary = write_artifacts(result)
        print(f"raw: {raw}")
        print(f"summary: {summary}")
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
