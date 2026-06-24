"""Generate a deterministic SPDX 2.3 Python dependency SBOM and hygiene report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
DEFAULT_SBOM = ROOT / "docs" / "sbom" / "mnemos-python.spdx.json"
DEFAULT_HYGIENE = ROOT / "docs" / "sbom" / "dependency-hygiene.json"


def _name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def declared_dependencies(path: Path = REQUIREMENTS) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)(.*)$", value)
        if not match:
            raise ValueError(f"unsupported requirement: {value}")
        rows.append({"name": match.group(1), "normalized_name": _name(match.group(1)), "specifier": match.group(2) or "*", "raw": value})
    return rows


def build_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    requirements_sha = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    declared = declared_dependencies()
    installed = {_name(dist.metadata["Name"]): dist.version for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        created = subprocess.check_output(["git", "show", "-s", "--format=%cI", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "UNKNOWN"
        created = "1970-01-01T00:00:00Z"
    packages = []
    missing = []
    unpinned = []
    for index, dep in enumerate(declared, 1):
        version = installed.get(dep["normalized_name"])
        if version is None:
            missing.append(dep["name"])
        if not re.fullmatch(r"==[^,;]+", dep["specifier"]):
            unpinned.append(dep["raw"])
        packages.append({
            "SPDXID": f"SPDXRef-Package-{index}",
            "name": dep["name"],
            "versionInfo": version or dep["specifier"],
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "externalRefs": [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl", "referenceLocator": f"pkg:pypi/{dep['normalized_name']}" + (f"@{version}" if version else "")}],
            "comment": f"Declared requirement: {dep['raw']}",
        })
    namespace_id = uuid.uuid5(uuid.NAMESPACE_URL, f"mnemos:{commit}:{requirements_sha}")
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "MNEMOS Python dependency SBOM",
        "documentNamespace": f"https://mnemos.invalid/spdx/{namespace_id}",
        "creationInfo": {"created": created, "creators": ["Tool: tools/generate_release_sbom.py"], "comment": "Generated from requirements.txt and the current resolved Python environment."},
        "documentDescribes": [package["SPDXID"] for package in packages],
        "packages": packages,
        "annotations": [{"annotationType": "OTHER", "annotator": "Tool: tools/generate_release_sbom.py", "comment": f"git_commit={commit}; requirements_sha256={requirements_sha}"}],
    }
    hygiene = {
        "schema": "mnemos-dependency-hygiene-v1",
        "git_commit": commit,
        "requirements_sha256": requirements_sha,
        "declared_dependency_count": len(declared),
        "resolved_dependency_count": len(declared) - len(missing),
        "missing_from_environment": sorted(missing),
        "non_exact_requirements": sorted(unpinned),
        "hash_pinned_requirements": 0,
        "vulnerability_audit": "NOT_RUN_NO_APPROVED_SCANNER_INSTALLED",
        "release_ready": not missing and not unpinned,
        "limitations": ["Source/Python dependency SBOM only; container and OS packages are not covered.", "No vulnerability claim is made without an approved scanner.", "requirements.txt is not a hash-pinned lockfile."],
    }
    return sbom, hygiene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbom", type=Path, default=DEFAULT_SBOM)
    parser.add_argument("--hygiene", type=Path, default=DEFAULT_HYGIENE)
    args = parser.parse_args()
    sbom, hygiene = build_artifacts()
    args.sbom.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.hygiene.write_text(json.dumps(hygiene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.sbom}")
    print(f"Wrote {args.hygiene}")


if __name__ == "__main__":
    main()
