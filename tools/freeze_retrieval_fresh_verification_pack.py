"""Freeze the fresh verification pack with reproducibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.mnemos_seed_manifest import DEFAULT_MANIFEST_PATH, load_seed_manifest
from tools.snapshot_retrieval_reproducibility import collect_mcp_snapshot, collect_service_snapshot


PACK_PATH = ROOT / "docs" / "experiments" / "retrieval_hygiene_r0_fresh_verification_pack.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _docker_image_id(image: str) -> str:
    return subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_freeze_manifest(base_url: str) -> dict[str, Any]:
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    manifest = load_seed_manifest(DEFAULT_MANIFEST_PATH)
    service_snapshot = collect_service_snapshot(base_url, 60.0)
    mcp_snapshot = collect_mcp_snapshot(base_url)
    retrieval_files = [
        "service/app.py",
        "mnemos/retrieval/retrieval_router.py",
    ]
    freeze = {
        "verification_pack_id": pack["verification_pack_id"],
        "pack_path": str(PACK_PATH.relative_to(ROOT)),
        "pack_sha256": _sha256(PACK_PATH),
        "seed_manifest_path": str(DEFAULT_MANIFEST_PATH.relative_to(ROOT)),
        "seed_snapshot_id": manifest.get("seed_snapshot_id", "unknown"),
        "collection_snapshot": service_snapshot.get("collection_snapshot", "unknown"),
        "git_revision": _git_head(),
        "service_build_revision": {
            "docker_image": "mnemos-mnemos:latest",
            "docker_image_id": _docker_image_id("mnemos-mnemos:latest"),
            "retrieval_code_sha256": {
                rel: _sha256(ROOT / rel) for rel in retrieval_files
            },
        },
        "mcp_build_revision": {
            "server_path": "mcp_servers/mnemos/server.py",
            "server_sha256": _sha256(ROOT / "mcp_servers" / "mnemos" / "server.py"),
            "snapshot": mcp_snapshot,
        },
        "runner_revision": {
            "freeze_runner_sha256": _sha256(Path(__file__)),
            "verification_runner_path": "tools/run_retrieval_fresh_verification.py",
            "verification_runner_sha256": _sha256(ROOT / "tools" / "run_retrieval_fresh_verification.py"),
            "result_scaffold_path": "tools/scaffold_retrieval_fresh_verification_result.py",
            "result_scaffold_sha256": _sha256(ROOT / "tools" / "scaffold_retrieval_fresh_verification_result.py"),
        },
        "expected_neighborhood_definitions": pack.get("neighborhoods", {}),
        "abstention_expectations": {
            item["query_id"]: item["expected_behavior"]
            for item in pack.get("queries", [])
            if "abstain" in str(item.get("expected_behavior", ""))
            or "must_not_abstain" in str(item.get("category", ""))
        },
        "service_snapshot": service_snapshot,
        "disclosure": pack.get("disclosure", {}),
    }
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8700")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = build_freeze_manifest(args.base_url.rstrip("/"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
