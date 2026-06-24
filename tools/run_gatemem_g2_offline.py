"""Run the G2 adapter over verified external G1 clean projections only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.gatemem_g1.io import require_external_output_path  # noqa: E402
from prototype.gatemem_g2 import (  # noqa: E402
    G2AdapterConfig,
    OfflineGovernedAdapter,
    load_clean_projections_jsonl,
    run_offline_adapter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--minimum-score", type=float, default=0.08)
    parser.add_argument("--max-disclosed-records", type=int, default=4)
    parser.add_argument("--max-answer-characters", type=int, default=2200)
    args = parser.parse_args()

    projections = load_clean_projections_jsonl(args.projections)
    adapter = OfflineGovernedAdapter(
        G2AdapterConfig(
            top_k=args.top_k,
            minimum_score=args.minimum_score,
            max_disclosed_records=args.max_disclosed_records,
            max_answer_characters=args.max_answer_characters,
        )
    )
    summary = run_offline_adapter(
        projections,
        adapter,
        predictions_path=args.predictions,
        diagnostics_path=args.diagnostics,
    )
    summary_path = require_external_output_path(args.run_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("No runtime route, hosted judge, deletion claim, or submission was used.")


if __name__ == "__main__":
    main()

