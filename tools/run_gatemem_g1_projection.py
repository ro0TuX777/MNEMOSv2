"""Build GateMem G1 clean projections without importing or editing GateMem.

The output path must be external to the MNEMOS repository. This tool performs
projection only; it does not call retrieval, score predictions, or run a judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.gatemem_g1 import project_clean_input, write_projections_jsonl


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _load_permitted_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        checkpoint_id = str(row.get("checkpoint_id") or "")
        metadata = row.get("permitted_metadata")
        if not checkpoint_id or not isinstance(metadata, dict):
            raise ValueError(
                "Every permitted-metadata row requires checkpoint_id and permitted_metadata."
            )
        if checkpoint_id in result:
            raise ValueError(f"Duplicate permitted metadata: {checkpoint_id}")
        result[checkpoint_id] = metadata
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permitted-metadata", type=Path)
    args = parser.parse_args()

    episodes = _load_jsonl(args.episodes)
    checkpoints = _load_jsonl(args.checkpoints)
    metadata_by_checkpoint = _load_permitted_metadata(args.permitted_metadata)
    episode_by_id = {str(item.get("episode_id") or ""): item for item in episodes}
    if "" in episode_by_id or len(episode_by_id) != len(episodes):
        raise ValueError("Episode IDs must be non-empty and unique.")

    projections = []
    for checkpoint in checkpoints:
        episode_id = str(checkpoint.get("episode_id") or "")
        episode = episode_by_id.get(episode_id)
        if episode is None:
            raise ValueError(f"No episode found for checkpoint episode_id={episode_id!r}")
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        projections.append(
            project_clean_input(
                episode,
                checkpoint,
                permitted_metadata=metadata_by_checkpoint.get(checkpoint_id, {}),
            )
        )

    count = write_projections_jsonl(projections, args.output)
    print(f"Wrote {count} clean projections to {args.output.resolve()}")
    print("No retrieval, scoring, hosted judge, or runtime integration was performed.")


if __name__ == "__main__":
    main()
