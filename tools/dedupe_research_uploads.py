#!/usr/bin/env python3
"""Report and optionally remove content-duplicate files in an upload directory.

The research intake UI historically numbered re-uploads (``name-1.pdf``,
``name-2.pdf``, ...) instead of recognising identical content, so a directory
could accumulate many byte-for-byte copies of the same document. Each copy was
extracted, chunked, and indexed on its own, inflating the index and crowding
retrieval with duplicate passages.

``_save_uploads`` now dedupes new uploads at the boundary. This tool cleans up
the copies that were created before that fix.

By default it only reports. Pass ``--apply`` to delete redundant copies,
keeping one canonical file per distinct content hash.

Scope: this touches the filesystem only. Engrams already indexed from the
removed copies still live in MNEMOS; purging those is a separate, index-level
step (delete by engram id, or re-index from the deduped file set).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(paths: list[Path]) -> Path:
    """Prefer the shortest, then lexicographically first name.

    ``constitution.pdf`` wins over ``constitution-1.pdf``; a bare arXiv id
    wins over its numbered re-uploads.
    """
    return sorted(paths, key=lambda p: (len(p.name), p.name))[0]


def find_duplicate_groups(upload_dir: Path) -> dict[str, list[Path]]:
    by_size: dict[int, list[Path]] = {}
    for path in sorted(upload_dir.iterdir()):
        if path.is_file():
            by_size.setdefault(path.stat().st_size, []).append(path)

    groups: dict[str, list[Path]] = {}
    for candidates in by_size.values():
        if len(candidates) < 2:
            continue  # a unique size cannot be a duplicate
        for path in candidates:
            groups.setdefault(_digest(path), []).append(path)
    return {digest: paths for digest, paths in groups.items() if len(paths) > 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upload_dir", type=Path, help="Directory of uploaded files")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete redundant copies. Without this flag the tool only reports.",
    )
    args = parser.parse_args(argv)

    upload_dir: Path = args.upload_dir
    if not upload_dir.is_dir():
        print(f"not a directory: {upload_dir}", file=sys.stderr)
        return 2

    groups = find_duplicate_groups(upload_dir)
    total_files = sum(1 for p in upload_dir.iterdir() if p.is_file())
    redundant = [p for paths in groups.values() for p in paths if p != _canonical(paths)]
    reclaimable = sum(p.stat().st_size for p in redundant)

    print(f"directory: {upload_dir}")
    print(f"files: {total_files}  duplicate_groups: {len(groups)}  redundant_copies: {len(redundant)}")
    print(f"reclaimable_bytes: {reclaimable:,}")

    for digest, paths in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        keep = _canonical(paths)
        drop = [p for p in paths if p != keep]
        print(f"\n  {digest[:12]}  keep {keep.name}")
        for path in drop:
            print(f"      {'removed' if args.apply else 'would remove'}: {path.name}")
            if args.apply:
                path.unlink()

    if not args.apply and redundant:
        print("\ndry run — re-run with --apply to delete the redundant copies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
