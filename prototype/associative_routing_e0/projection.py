"""
Deterministic projection builder for the Associative Routing View E0.

Builds an in-memory, versioned, reproducible projection (Cue registry + Tag
registry + content index, each stamped with a snapshot hash) from the raw
fixture corpus. The projection is a derived read-only artifact: rebuilding
from the same inputs must always produce the same hashes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional

from .models import ContentRef, Cue, Tag
from .registry import Corpus, load_corpus

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Projection:
    content_index: Dict[str, ContentRef]
    cues: Dict[str, Cue]
    tags: Dict[str, Tag]
    snapshot: str
    manifest: dict


def build_projection(fixtures_dir: Path = FIXTURES_DIR, corpus: Optional[Corpus] = None) -> Projection:
    """Build a deterministic projection snapshot from the fixture corpus.

    ``corpus`` may be supplied to avoid re-reading disk (e.g. by the
    verification tool, which already loaded and validated it).
    """

    if corpus is None:
        corpus = load_corpus(fixtures_dir)

    source_input_hashes = {
        "source_index.json": _sha256_hex(corpus.raw_source_index_bytes),
        "cue_registry.json": _sha256_hex(corpus.raw_cue_registry_bytes),
        "tag_registry.json": _sha256_hex(corpus.raw_tag_registry_bytes),
    }

    cue_registry_hash = _sha256_hex(
        _canonical_json([corpus.cues[cid].to_dict() for cid in sorted(corpus.cues)])
    )
    tag_registry_hash = _sha256_hex(
        _canonical_json([corpus.tags[tid].to_dict() for tid in sorted(corpus.tags)])
    )
    content_index_hash = _sha256_hex(
        _canonical_json([corpus.content_index[c].to_dict() for c in sorted(corpus.content_index)])
    )

    projection_output_hash = _sha256_hex(
        _canonical_json(
            {
                "content_index_hash": content_index_hash,
                "cue_registry_hash": cue_registry_hash,
                "tag_registry_hash": tag_registry_hash,
            }
        )
    )
    snapshot = f"sha256:{projection_output_hash}"

    stamped_cues = {cid: replace(cue, projection_snapshot=snapshot) for cid, cue in corpus.cues.items()}
    stamped_tags = {tid: replace(tag, projection_snapshot=snapshot) for tid, tag in corpus.tags.items()}

    manifest = {
        "schema_version": "1",
        "projection_snapshot": snapshot,
        "source_input_hashes": source_input_hashes,
        "cue_registry_hash": cue_registry_hash,
        "tag_registry_hash": tag_registry_hash,
        "content_index_hash": content_index_hash,
        "projection_output_hash": projection_output_hash,
        "counts": {
            "content": len(corpus.content_index),
            "cues": len(corpus.cues),
            "tags": len(corpus.tags),
        },
    }

    return Projection(
        content_index=corpus.content_index,
        cues=stamped_cues,
        tags=stamped_tags,
        snapshot=snapshot,
        manifest=manifest,
    )
