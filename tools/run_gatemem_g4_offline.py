"""Generate and run the authorized G4 local synthetic development lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.gatemem_g4 import generate_and_run  # noqa: E402

DEFAULT_CORPUS = Path(r"G:\MNEMOS-research\gatemem_g4_development_corpus")
DEFAULT_OUTPUT = Path(r"G:\MNEMOS-research\gatemem_g4_reference_run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run = generate_and_run(args.corpus_root, args.output_root)
    print(json.dumps(run["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
