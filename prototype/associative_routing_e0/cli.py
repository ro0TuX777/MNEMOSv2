"""
Local, read-only CLI for the Associative Routing View E0.

Usage:
    python -m prototype.associative_routing_e0.cli query "Why is GateMem work paused?"
    python -m prototype.associative_routing_e0.cli verify
    python -m prototype.associative_routing_e0.cli manifest
"""

from __future__ import annotations

import argparse
import json
import sys

from .projection import build_projection
from .router import AssociativeRouter
from .verify import verify_projection


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="associative_routing_e0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Run a read-only associative routing query.")
    query_parser.add_argument("text")

    subparsers.add_parser("verify", help="Run the E0 projection verification tool.")
    subparsers.add_parser("manifest", help="Print the projection snapshot manifest.")

    args = parser.parse_args(argv)

    if args.command == "query":
        router = AssociativeRouter.from_fixtures()
        response = router.route(args.text)
        print(json.dumps(response.to_dict(), indent=2))
        return 0

    if args.command == "verify":
        result = verify_projection()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "pass" else 1

    if args.command == "manifest":
        projection = build_projection()
        print(json.dumps(projection.manifest, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
