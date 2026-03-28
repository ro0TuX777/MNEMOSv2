"""
Validate a benchmark truthset JSON file.

Usage:
  python tools/validate_truthset.py --file benchmarks/outputs/truthset_v1_template.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ALLOWED_REGIMES = {"semantic", "light_filter", "heavy_filter"}
ALLOWED_INTENTS = {"factoid", "procedural", "comparative", "definition"}
ALLOWED_LABELS = {0, 1, 2}


def _is_non_empty_string(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _add_error(errors: List[str], msg: str) -> None:
    errors.append(msg)


def validate_truthset(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if not isinstance(data, dict):
        return ["Top-level JSON must be an object."]

    # Top-level required fields
    for field in ("version", "created_at", "corpus", "label_scale", "queries"):
        if field not in data:
            _add_error(errors, f"Missing top-level field: '{field}'")

    if "version" in data and not _is_non_empty_string(data["version"]):
        _add_error(errors, "Field 'version' must be a non-empty string.")

    if "created_at" in data and not _is_non_empty_string(data["created_at"]):
        _add_error(errors, "Field 'created_at' must be a non-empty string.")

    corpus = data.get("corpus")
    if not isinstance(corpus, dict):
        _add_error(errors, "Field 'corpus' must be an object.")
    else:
        for field in ("type", "source_dir"):
            if field not in corpus:
                _add_error(errors, f"Missing corpus field: '{field}'")
            elif not _is_non_empty_string(corpus[field]):
                _add_error(errors, f"Corpus field '{field}' must be a non-empty string.")

    label_scale = data.get("label_scale")
    if not isinstance(label_scale, dict):
        _add_error(errors, "Field 'label_scale' must be an object.")
    else:
        for key in ("0", "1", "2"):
            if key not in label_scale:
                _add_error(errors, f"label_scale must define key '{key}'.")
            elif not _is_non_empty_string(label_scale[key]):
                _add_error(errors, f"label_scale['{key}'] must be a non-empty string.")

    queries = data.get("queries")
    if not isinstance(queries, list):
        _add_error(errors, "Field 'queries' must be an array.")
        return errors

    query_ids_seen = set()
    for i, q in enumerate(queries):
        prefix = f"queries[{i}]"

        if not isinstance(q, dict):
            _add_error(errors, f"{prefix} must be an object.")
            continue

        for field in ("query_id", "query_text", "regime", "intent_tag", "filters", "labels"):
            if field not in q:
                _add_error(errors, f"{prefix} missing field '{field}'.")

        query_id = q.get("query_id")
        if not _is_non_empty_string(query_id):
            _add_error(errors, f"{prefix}.query_id must be a non-empty string.")
        elif query_id in query_ids_seen:
            _add_error(errors, f"Duplicate query_id found: '{query_id}'.")
        else:
            query_ids_seen.add(query_id)

        if not _is_non_empty_string(q.get("query_text", "")):
            _add_error(errors, f"{prefix}.query_text must be a non-empty string.")

        regime = q.get("regime")
        if regime not in ALLOWED_REGIMES:
            _add_error(
                errors,
                f"{prefix}.regime must be one of {sorted(ALLOWED_REGIMES)}.",
            )

        intent = q.get("intent_tag")
        if intent not in ALLOWED_INTENTS:
            _add_error(
                errors,
                f"{prefix}.intent_tag must be one of {sorted(ALLOWED_INTENTS)}.",
            )

        filters = q.get("filters")
        if not isinstance(filters, dict):
            _add_error(errors, f"{prefix}.filters must be an object.")

        labels = q.get("labels")
        if not isinstance(labels, list):
            _add_error(errors, f"{prefix}.labels must be an array.")
            continue

        chunk_ids_seen = set()
        for j, lab in enumerate(labels):
            lp = f"{prefix}.labels[{j}]"
            if not isinstance(lab, dict):
                _add_error(errors, f"{lp} must be an object.")
                continue

            for field in ("chunk_id", "relevance"):
                if field not in lab:
                    _add_error(errors, f"{lp} missing field '{field}'.")

            chunk_id = lab.get("chunk_id")
            if not _is_non_empty_string(chunk_id):
                _add_error(errors, f"{lp}.chunk_id must be a non-empty string.")
            elif chunk_id in chunk_ids_seen:
                _add_error(errors, f"{lp}.chunk_id '{chunk_id}' is duplicated within query.")
            else:
                chunk_ids_seen.add(chunk_id)

            relevance = lab.get("relevance")
            if relevance not in ALLOWED_LABELS:
                _add_error(errors, f"{lp}.relevance must be one of {sorted(ALLOWED_LABELS)}.")

            if "notes" in lab and not isinstance(lab["notes"], str):
                _add_error(errors, f"{lp}.notes must be a string when provided.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark truthset JSON.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to truthset JSON file.",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return 2

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] Invalid JSON: {e}")
        return 2

    errors = validate_truthset(data)
    if errors:
        print(f"[FAIL] Validation errors: {len(errors)}")
        for e in errors:
            print(f"  - {e}")
        return 1

    n_queries = len(data.get("queries", []))
    print(f"[OK] Truthset valid: {path} ({n_queries} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
