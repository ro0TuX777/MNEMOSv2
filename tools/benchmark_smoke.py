"""Generate a compact reproducible benchmark smoke artifact.

This script is the public, low-friction benchmark entry point. By default it
runs the installer benchmark track because that path does not require Docker or
live vector stores. Operators with the benchmark backend stack running can use
``--track retrieval`` for a small retrieval smoke run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics.system_metrics import capture_environment, snapshot_to_dict
from benchmarks.run_profile_benchmarks import run_suite


PUBLIC_PACKAGES = [
    "flask",
    "gunicorn",
    "requests",
    "qdrant-client",
    "pgvector",
    "sentence-transformers",
    "torch",
    "einops",
    "psycopg",
    "psycopg-pool",
    "numpy",
    "scipy",
    "python-dotenv",
]


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    matches = list(directory.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _run_text(cmd: List[str], timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (result.stdout or result.stderr).strip()
    except Exception:
        return ""


def _package_versions(packages: Iterable[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_metadata() -> Dict[str, str]:
    return {
        "commit": _run_text(["git", "rev-parse", "HEAD"]),
        "branch": _run_text(["git", "branch", "--show-current"]),
        "dirty": "true" if _run_text(["git", "status", "--short"]) else "false",
    }


def _extract_notes(results: Dict[str, Any], track: str) -> List[str]:
    notes: List[str] = []
    if track == "installer":
        installer = results.get("installer", {})
        for profile, data in installer.get("profiles", {}).items():
            summary = data.get("summary", {})
            notes.append(
                f"{profile}: installer median {summary.get('installer_median_s')}s, "
                f"validation pass {summary.get('validation_pass_rate')}"
            )
    elif track == "retrieval":
        retrieval = results.get("retrieval", {})
        for tier, regimes in retrieval.get("search", {}).items():
            for regime, data in regimes.items():
                if data.get("status") != "success":
                    notes.append(f"{tier}/{regime}: {data.get('status')}")
                    continue
                notes.append(
                    f"{tier}/{regime}: recall@10={data.get('recall_at_10')}, "
                    f"mrr@10={data.get('mrr_at_10')}, "
                    f"p50={data.get('latency_p50_ms')}ms"
                )
    else:
        notes.append(f"Track {track} completed; inspect raw artifact for details.")
    return notes


def _passed(results: Dict[str, Any], track: str) -> bool:
    if track == "installer":
        installer = results.get("installer", {})
        if installer.get("status") != "success":
            return False
        for data in installer.get("profiles", {}).values():
            summary = data.get("summary", {})
            if summary.get("installer_errors", 1) != 0:
                return False
            if summary.get("manual_errors", 1) != 0:
                return False
            if summary.get("validation_pass_rate", 0) < 1.0:
                return False
        return True

    if track == "retrieval":
        retrieval = results.get("retrieval", {})
        if not retrieval:
            return False
        statuses: List[str] = []
        for group in ("ingest", "search"):
            section = retrieval.get(group, {})
            if group == "search":
                for regimes in section.values():
                    statuses.extend(row.get("status", "") for row in regimes.values())
            else:
                statuses.extend(row.get("status", "") for row in section.values())
        return bool(statuses) and all(status == "success" for status in statuses)

    return bool(results.get(track))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# MNEMOS Benchmark Smoke Report",
        "",
        f"Generated: `{payload['timestamp']}`",
        f"Track: `{payload['track']}`",
        f"Result: `{'PASS' if payload['passed'] else 'FAIL'}`",
        "",
        "## Artifacts",
        "",
        f"- Raw benchmark artifact: `{payload.get('raw_artifact') or 'not found'}`",
        f"- Benchmark report: `{payload.get('report_artifact') or 'not found'}`",
        f"- Corpus SHA-256: `{payload.get('corpus_sha256') or 'not found'}`",
        f"- Query-set SHA-256: `{payload.get('queries_sha256') or 'not found'}`",
        "",
        "## Environment",
        "",
        "| Property | Value |",
        "|---|---|",
    ]
    for key, value in payload["environment"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Package Versions", "", "| Package | Version |", "|---|---|"])
    for package, version in payload["package_versions"].items():
        lines.append(f"| {package} | {version} |")

    lines.extend(["", "## Notes", ""])
    for note in payload.get("notes", []):
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Installer smoke validates reproducible profile generation, not retrieval quality.",
            "- Retrieval smoke validates a small configured benchmark run, not final product-quality claims.",
            "- Use curated or real-world labeled benchmarks before making quality comparisons.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public MNEMOS benchmark smoke pack.")
    parser.add_argument(
        "--track",
        choices=["installer", "retrieval"],
        default="installer",
        help="Smoke track to run. retrieval requires benchmark vector backends.",
    )
    parser.add_argument("--corpus-size", type=int, default=200, help="Synthetic corpus size.")
    parser.add_argument("--runs", type=int, default=1, help="Repeated runs for benchmark medians.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic corpus seed.")
    parser.add_argument("--gpu", default="cpu", help="GPU device for benchmark code.")
    parser.add_argument(
        "--output-dir",
        default="benchmarks/outputs/smoke",
        help="Directory for compact smoke artifacts.",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    results = run_suite(
        tracks=[args.track],
        corpus_size=args.corpus_size,
        corpus_type="synthetic",
        n_runs=args.runs,
        gpu_device=args.gpu,
        seed=args.seed,
    )

    datasets_dir = PROJECT_ROOT / "benchmarks" / "outputs" / "datasets"
    raw_artifact = _latest_file(PROJECT_ROOT / "benchmarks" / "outputs" / "raw", "*_profile_benchmarks.json")
    report_artifact = _latest_file(PROJECT_ROOT / "benchmarks" / "outputs" / "summaries", "*_report.md")

    env = capture_environment()
    payload: Dict[str, Any] = {
        "timestamp": env.timestamp,
        "smoke_id": timestamp,
        "track": args.track,
        "passed": _passed(results, args.track),
        "command": " ".join(sys.argv),
        "git": _git_metadata(),
        "environment": snapshot_to_dict(env),
        "platform": platform.platform(),
        "package_versions": _package_versions(PUBLIC_PACKAGES),
        "corpus_path": str(datasets_dir / "corpus.json"),
        "queries_path": str(datasets_dir / "queries.json"),
        "corpus_sha256": _sha256(datasets_dir / "corpus.json"),
        "queries_sha256": _sha256(datasets_dir / "queries.json"),
        "raw_artifact": str(raw_artifact) if raw_artifact else None,
        "report_artifact": str(report_artifact) if report_artifact else None,
        "results": results,
        "notes": _extract_notes(results, args.track),
    }

    output_dir = PROJECT_ROOT / args.output_dir
    json_path = output_dir / f"{timestamp}_benchmark_smoke.json"
    markdown_path = output_dir / f"{timestamp}_benchmark_smoke.md"
    _write_json(json_path, payload)
    _write_markdown(markdown_path, payload)

    print(f"\nSmoke JSON: {json_path}")
    print(f"Smoke report: {markdown_path}")
    print(f"Smoke result: {'PASS' if payload['passed'] else 'FAIL'}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

