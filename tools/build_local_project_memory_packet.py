#!/usr/bin/env python3
"""Build a read-only, explicitly scoped local project memory packet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.local_project_memory_r0.errors import ErrorCode, ProjectMemoryError  # noqa: E402
from prototype.local_project_memory_r0.models import ScopeSpec  # noqa: E402
from prototype.local_project_memory_r0.packet import build_packet, write_packet  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--scope-root", action="append", default=[])
    parser.add_argument("--scope-file", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--max-total-bytes", type=int, default=10_485_760)
    parser.add_argument("--max-file-bytes", type=int, default=1_048_576)
    return parser.parse_args(argv)


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        project_root = args.project_root.resolve()
        output = args.output.resolve(strict=False)
        if _inside(project_root, output):
            raise ProjectMemoryError(
                ErrorCode.OUTPUT_INSIDE_PROJECT,
                "packet output must be outside the target project",
            )
        if output.exists():
            raise ProjectMemoryError(
                ErrorCode.OUTPUT_ALREADY_EXISTS,
                "packet output already exists",
            )
        scope = ScopeSpec(
            roots=tuple(args.scope_root),
            files=tuple(args.scope_file),
            excludes=tuple(args.exclude),
            max_files=args.max_files,
            max_total_bytes=args.max_total_bytes,
            max_file_bytes=args.max_file_bytes,
        )
        packet = build_packet(project_root, args.repo_id, scope)
        write_packet(output, packet)
        print(f"Packet: {output}")
        print(f"Repository: {packet.snapshot.repo_id}")
        print(f"Snapshot: {packet.snapshot.snapshot_id}")
        print(f"Files: {len(packet.snapshot.files)}")
        print(f"Artifacts: {len(packet.artifacts)}")
        print(f"Usable: {str(packet.usable).lower()}")
        return 0 if packet.usable else 3
    except ProjectMemoryError as exc:
        print(f"{exc.code.value}: {exc.args[0].split(': ', 1)[-1]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
