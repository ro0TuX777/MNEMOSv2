"""Normalize external G1 shadow observations into external predictions.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.gatemem_g1 import (  # noqa: E402
    shadow_observation_from_dict,
    write_predictions_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observations = []
    with args.observations.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Observation row {line_number} is not an object.")
            observations.append(shadow_observation_from_dict(value))

    count = write_predictions_jsonl(observations, args.output)
    print(f"Wrote {count} GateMem-compatible G1 predictions to {args.output.resolve()}")
    print("No scoring, hosted judge, submission, or deletion capability claim was performed.")


if __name__ == "__main__":
    main()

