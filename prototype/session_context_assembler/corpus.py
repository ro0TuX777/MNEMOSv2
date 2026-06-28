"""Read-only loading and hash validation for frozen assembler corpora.

This module only ever opens files in read mode. It must never be extended
to write, modify, or delete corpus or manifest files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple


class ManifestMismatchError(Exception):
    """Raised when the corpus file or a case no longer matches its recorded hash."""


def compute_case_hash(case: dict) -> str:
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode("utf-8")).hexdigest()


def compute_file_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def load_corpus(corpus_path: str | Path) -> Tuple[dict, bytes]:
    raw = Path(corpus_path).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return data, raw


def load_manifest(manifest_path: str | Path) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def validate_manifest(corpus_data: dict, corpus_raw: bytes, manifest: dict) -> None:
    """Raise ManifestMismatchError if the corpus no longer matches its frozen manifest."""
    file_hash = compute_file_hash(corpus_raw)
    if file_hash != manifest["file_sha256"]:
        raise ManifestMismatchError(
            f"corpus file hash {file_hash} does not match manifest file_sha256 "
            f"{manifest['file_sha256']}"
        )
    for case in corpus_data["cases"]:
        case_hash = compute_case_hash(case)
        expected = manifest["case_hashes"].get(case["id"])
        if expected != case_hash:
            raise ManifestMismatchError(
                f"case {case['id']} hash mismatch: computed {case_hash}, "
                f"manifest recorded {expected}"
            )


def load_validated_corpus(corpus_path: str | Path, manifest_path: str | Path) -> dict:
    """Load the corpus and manifest and raise if they have diverged."""
    data, raw = load_corpus(corpus_path)
    manifest = load_manifest(manifest_path)
    validate_manifest(data, raw, manifest)
    return data
