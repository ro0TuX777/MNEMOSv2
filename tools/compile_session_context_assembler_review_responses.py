"""Validate and compile completed Phase 5 human-review responses.

This tool does not generate responses, contact reviewers, or call a model. It
verifies the frozen coordinator manifest and packet hashes before compiling
pseudonymous human submissions. Packet generation remains a separate tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

RESPONSE_SCHEMA = "sca_phase5_review_response_v1"
OUTPUT_SCHEMA = "sca_phase5_compiled_responses_v1"
PSEUDONYM = re.compile(r"^REV-[A-Z0-9][A-Z0-9-]{2,31}$")

ENUMS = {
    "prior_decision_preserved": {"yes", "no", "uncertain", "not_applicable"},
    "contradiction_representation": {
        "correct", "incorrect", "unclear", "not_applicable"
    },
    "necessary_material_omitted": {"yes", "no", "uncertain"},
    "task_effect": {"easier", "harder", "no_different"},
}
LIKERT_FIELDS = (
    "source_references_understandable",
    "source_references_sufficient",
    "synthetic_context_distinguishable",
    "abstention_clear_and_cautious",
)
IDENTITY_KEYS = {"name", "email", "phone", "employee_id", "user_id"}
TOP_LEVEL_FIELDS = {
    "schema", "study_id", "reviewer_pseudonym", "task_code", "package_reviews"
}
REVIEW_FIELDS = {
    "condition_code", "prior_decision_preserved",
    "source_references_understandable", "source_references_sufficient",
    "contradiction_representation", "necessary_material_omitted",
    "synthetic_context_distinguishable", "abstention_clear_and_cautious",
    "task_effect", "short_rationale",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_manifest(path: Path) -> Dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("frozen") or not manifest.get("design_only_review_not_run"):
        raise ValueError("manifest is not the frozen Phase 5 design manifest")
    root = path.parent
    for entry in manifest["packets"]:
        data = (root / entry["path"]).read_bytes()
        if _sha256(data) != entry["sha256"]:
            raise ValueError(f"packet hash mismatch: {entry['path']}")
    return manifest


def _packet_index(manifest_path: Path, manifest: Dict) -> Dict[str, Dict]:
    root = manifest_path.parent
    return {
        entry["task_code"]: json.loads(
            (root / entry["path"]).read_text(encoding="utf-8")
        )
        for entry in manifest["packets"]
    }


def _condition_keys(manifest: Dict) -> Dict[str, Dict[str, str]]:
    return {
        item["task_code"]: item["condition_key"]
        for item in manifest["coordinator_only_condition_key"]
    }


def _validate_likert(field: str, value) -> None:
    if value == "not_applicable":
        return
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
        raise ValueError(f"{field} must be an integer 1..5 or not_applicable")


def validate_response(response: Dict, packet: Dict) -> None:
    leaked_identity = IDENTITY_KEYS & set(response)
    if leaked_identity:
        raise ValueError(f"direct identity fields are forbidden: {sorted(leaked_identity)}")
    unknown = set(response) - TOP_LEVEL_FIELDS
    if unknown:
        raise ValueError(f"unknown top-level response fields: {sorted(unknown)}")
    if response.get("schema") != RESPONSE_SCHEMA:
        raise ValueError("unexpected response schema")
    if response.get("study_id") != packet["study_id"]:
        raise ValueError("study_id does not match packet")
    if response.get("task_code") != packet["task_code"]:
        raise ValueError("task_code does not match packet")
    pseudonym = response.get("reviewer_pseudonym", "")
    if not PSEUDONYM.fullmatch(pseudonym):
        raise ValueError("reviewer_pseudonym must match REV-[A-Z0-9-]")

    reviews = response.get("package_reviews")
    if not isinstance(reviews, list) or len(reviews) != 3:
        raise ValueError("exactly three package_reviews are required")
    expected_codes = {item["condition_code"] for item in packet["packages"]}
    observed_codes = {item.get("condition_code") for item in reviews}
    if observed_codes != expected_codes:
        raise ValueError("package review codes do not match the masked packet")

    package_by_code = {item["condition_code"]: item for item in packet["packages"]}
    for review in reviews:
        unknown_review = set(review) - REVIEW_FIELDS
        if unknown_review:
            raise ValueError(f"unknown package review fields: {sorted(unknown_review)}")
        for field, allowed in ENUMS.items():
            if review.get(field) not in allowed:
                raise ValueError(f"invalid {field}: {review.get(field)!r}")
        for field in LIKERT_FIELDS:
            _validate_likert(field, review.get(field))
        rationale = review.get("short_rationale")
        if not isinstance(rationale, str) or not 10 <= len(rationale.strip()) <= 1000:
            raise ValueError("short_rationale must contain 10..1000 characters")
        has_warning = "warning" in package_by_code[review["condition_code"]]
        abstention_rating = review["abstention_clear_and_cautious"]
        if has_warning and abstention_rating == "not_applicable":
            raise ValueError("abstention rating is required when a warning is visible")
        if not has_warning and abstention_rating != "not_applicable":
            raise ValueError("abstention rating must be not_applicable without a warning")


def _mean(values: Iterable[int]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def compile_responses(manifest_path: Path, response_paths: List[Path]) -> Dict:
    manifest = _load_manifest(manifest_path)
    packets = _packet_index(manifest_path, manifest)
    condition_keys = _condition_keys(manifest)
    seen = set()
    by_condition = defaultdict(list)
    qualitative = []

    for path in sorted(response_paths):
        response = json.loads(path.read_text(encoding="utf-8"))
        task_code = response.get("task_code")
        if task_code not in packets:
            raise ValueError(f"unknown task_code in {path}: {task_code}")
        validate_response(response, packets[task_code])
        submission_key = (response["reviewer_pseudonym"], task_code)
        if submission_key in seen:
            raise ValueError(f"duplicate reviewer/task submission: {submission_key}")
        seen.add(submission_key)
        for review in response["package_reviews"]:
            condition = condition_keys[task_code][review["condition_code"]]
            by_condition[condition].append(review)
            qualitative.append(
                {
                    "reviewer_pseudonym": response["reviewer_pseudonym"],
                    "task_code": task_code,
                    "condition": condition,
                    "short_rationale": review["short_rationale"].strip(),
                }
            )

    aggregates = {}
    for condition, reviews in sorted(by_condition.items()):
        aggregates[condition] = {
            "review_count": len(reviews),
            "prior_decision_preserved_counts": dict(
                sorted(_counts(r["prior_decision_preserved"] for r in reviews).items())
            ),
            "contradiction_representation_counts": dict(
                sorted(_counts(r["contradiction_representation"] for r in reviews).items())
            ),
            "necessary_material_omitted_counts": dict(
                sorted(_counts(r["necessary_material_omitted"] for r in reviews).items())
            ),
            "task_effect_counts": dict(
                sorted(_counts(r["task_effect"] for r in reviews).items())
            ),
        }
        for field in LIKERT_FIELDS:
            aggregates[condition][f"mean_{field}"] = _mean(
                r[field] for r in reviews if isinstance(r[field], int)
            )

    return {
        "schema": OUTPUT_SCHEMA,
        "study_id": manifest["study_id"],
        "source_manifest_sha256": _sha256(manifest_path.read_bytes()),
        "human_response_file_count": len(response_paths),
        "unique_reviewer_task_submissions": len(seen),
        "aggregates_by_unblinded_condition": aggregates,
        "qualitative_responses": qualitative,
        "interpretation_boundary": (
            "Compilation only. Human-value conclusions require the approved "
            "protocol and cannot authorize runtime or production integration."
        ),
    }


def _counts(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--responses-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    response_paths = sorted(args.responses_dir.glob("*.json"))
    if not response_paths:
        parser.error("no human response JSON files found; no synthetic responses are generated")
    compiled = compile_responses(args.manifest, response_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(compiled, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Compiled {len(response_paths)} human response files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
