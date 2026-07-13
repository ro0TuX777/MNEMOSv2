"""Verify the frozen G4 reference-contract baseline without executing policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "benchmarks" / "results" / "gatemem_g4_frozen_reference_manifest.json"
)
DEFAULT_GATE = ROOT / "benchmarks" / "results" / "gatemem_g4_gate.json"


def verify_frozen_reference(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    gate_path: str | Path = DEFAULT_GATE,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual_files: dict[str, str] = {}
    for relative, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen G4 reference drift: {relative}")
        actual_files[relative] = actual
    canonical = json.dumps(
        {
            "files": actual_files,
            "corpus_composite_sha256": manifest["corpus_composite_sha256"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    composite = hashlib.sha256(canonical).hexdigest()
    if composite != manifest["composite_sha256"]:
        raise ValueError("Frozen G4 reference composite mismatch")

    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    expected = manifest["expected_evidence"]
    checks = {
        "gate_passed": gate["all_passed"] is expected["all_gates_passed"],
        "gate_count": len(gate["gates"]) == expected["gate_count"],
        "case_count": gate["counts"]["cases"] == expected["case_count"],
        "exact_matches": gate["counts"]["exact_matches"] == expected["exact_matches"],
        "no_mismatches": gate["counts"]["mismatches"] == [],
        "implementation_composite": gate["implementation_fingerprint"]["composite_sha256"] == composite,
        "corpus_composite": gate["implementation_fingerprint"]["corpus_composite_sha256"] == manifest["corpus_composite_sha256"],
        "reference_claim": gate["classification"] == manifest["classification"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Frozen G4 evidence mismatch: {failed}")
    return {"manifest": manifest, "checks": checks, "verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    args = parser.parse_args()
    result = verify_frozen_reference(args.manifest, args.gate)
    print(result["manifest"]["status"])
    print(result["manifest"]["composite_sha256"])
    print(f"Checks passed: {len(result['checks'])}/{len(result['checks'])}")


if __name__ == "__main__":
    main()
