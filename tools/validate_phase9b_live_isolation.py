"""Phase 9b live-runtime isolation validation.

Proves the __exclude_summaries__ sentinel is active in the deployed service
container by exercising the public API (not host-side tier logic):

1. Default-path searches (CLASS_A truthset) must return zero summary engrams.
2. Positive control: an explicit metadata filter must still reach the summary
   layer, proving exclusion is sentinel-driven rather than corpus accident.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "query_complexity_v1.json"
RAW_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "raw"


def _is_summary(engram: dict) -> bool:
    return bool((engram.get("metadata") or {}).get("is_summary_engram", False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8700")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args()

    payload = json.loads(TRUTHSET.read_text(encoding="utf-8"))
    class_a = [q for q in payload["queries"] if q["label"] == "CLASS_A"]

    rows = []
    for item in class_a:
        resp = requests.post(
            f"{args.base_url}/v1/mnemos/search",
            json={"query": item["query"], "top_k": args.top_k},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        leaks = [
            r["engram"]["id"]
            for r in results
            if _is_summary(r.get("engram") or {})
        ]
        rows.append(
            {
                "id": item["id"],
                "results": len(results),
                "summary_leaks": len(leaks),
                "leaked_ids": leaks,
            }
        )
        print(f"{item['id']}: results={len(results)} leaks={len(leaks)}")

    isolation_pass = bool(rows) and all(
        r["summary_leaks"] == 0 and r["results"] > 0 for r in rows
    )

    # Positive control — the summary layer must remain reachable through an
    # explicit metadata filter; otherwise "zero leaks" would also be true of
    # a corpus that simply lost its summaries.
    control = requests.post(
        f"{args.base_url}/v1/mnemos/search",
        json={
            "query": "overall corpus posture summary",
            "top_k": 5,
            "filters": {"metadata.is_summary_engram": True},
        },
        timeout=120,
    )
    control.raise_for_status()
    control_results = (control.json().get("results")) or []
    control_summaries = sum(
        1 for r in control_results if _is_summary(r.get("engram") or {})
    )
    control_pass = control_summaries > 0
    print(f"positive control: {control_summaries}/{len(control_results)} summary hits")

    overall = isolation_pass and control_pass
    print(f"isolation: {'PASS' if isolation_pass else 'FAIL'}")
    print(f"positive_control: {'PASS' if control_pass else 'FAIL'}")
    print(f"overall: {'PASS' if overall else 'FAIL'}")

    if not args.no_artifacts:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        out = RAW_DIR / f"phase9b_live_isolation_{time.strftime('%Y%m%d_%H%M%S')}_raw.json"
        out.write_text(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "query_count": len(rows),
                    "rows": rows,
                    "positive_control_summary_hits": control_summaries,
                    "gates": {
                        "isolation": isolation_pass,
                        "positive_control": control_pass,
                    },
                    "overall_pass": overall,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"raw: {out}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
