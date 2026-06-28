"""External-only JSONL output helpers for GateMem G1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import CleanInputProjection

MNEMOS_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def require_external_output_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    try:
        path.relative_to(MNEMOS_REPOSITORY_ROOT)
    except ValueError:
        return path
    raise ValueError("GateMem-derived G1 outputs must remain outside the MNEMOS repository.")


def _write_rows(rows: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    path = require_external_output_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(materialized)


def write_projections_jsonl(
    projections: Iterable[CleanInputProjection], output_path: str | Path
) -> int:
    seen: set[str] = set()
    rows = []
    for projection in projections:
        if projection.checkpoint_id in seen:
            raise ValueError(f"Duplicate checkpoint_id: {projection.checkpoint_id}")
        seen.add(projection.checkpoint_id)
        rows.append(projection.to_dict())
    return _write_rows(rows, output_path)


def write_json_rows_external(rows: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    return _write_rows(rows, output_path)

