"""
Verification tool for the Associative Routing View E0 projection.

Confirms, independent of the registry loader's own validation:

* all Tag targets exist;
* all Tags have source support;
* no orphan Cues or Tags;
* no prohibited authority fields;
* snapshot hashes match a freshly rebuilt projection;
* the projection can be rebuilt deterministically (two independent builds
  from disk produce byte-identical hashes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .models import ALLOWED_CUE_FIELDS, ALLOWED_TAG_FIELDS
from .projection import FIXTURES_DIR, build_projection
from .registry import RegistryValidationError, load_corpus

#: Fields that would constitute authority/trust/governance leakage into the
#: E0 retrieval-metadata layer. Even though ALLOWED_*_FIELDS already
#: enumerates a closed allowlist (so anything else is rejected at load time),
#: this explicit denylist is checked independently as a second, defense-in-
#: depth gate against authority fields creeping into the schema itself.
_PROHIBITED_AUTHORITY_FIELDS = {
    "trust_score",
    "utility_score",
    "freshness_score",
    "authoritative",
    "authority",
    "promote",
    "promoted",
    "promotion_status",
    "disclosure_decision",
    "disclosure",
    "governance_state",
    "access_decision",
    "retention_decision",
    "deletion_decision",
    "confidence",
    "is_fact",
}


def verify_no_prohibited_fields(raw_records: list, kind: str) -> list:
    violations = []
    for record in raw_records:
        hit = _PROHIBITED_AUTHORITY_FIELDS & set(record.keys())
        if hit:
            violations.append({"kind": kind, "id": record.get(f"{kind}_id"), "fields": sorted(hit)})
    return violations


def verify_projection(fixtures_dir: Path = FIXTURES_DIR) -> Dict[str, Any]:
    raw_cue_registry = json.loads((fixtures_dir / "cue_registry.json").read_text(encoding="utf-8"))
    raw_tag_registry = json.loads((fixtures_dir / "tag_registry.json").read_text(encoding="utf-8"))

    checks: Dict[str, Any] = {}
    errors: list = []

    # 1. Load + the registry loader's own structural validation (allowlisted
    #    fields, resolvable targets, resolvable source records, no orphan
    #    cues). Any RegistryValidationError fails the whole verification.
    try:
        corpus = load_corpus(fixtures_dir)
        checks["registry_loads_and_validates"] = True
    except RegistryValidationError as exc:
        checks["registry_loads_and_validates"] = False
        errors.append(str(exc))
        return {"status": "fail", "checks": checks, "errors": errors}

    # 2. All Tag targets exist (re-derived independently of the loader).
    missing_targets = [
        t.tag_id for t in corpus.tags.values() if t.to_content_id not in corpus.content_index
    ]
    checks["all_tag_targets_resolve"] = not missing_targets
    if missing_targets:
        errors.append(f"Tags with unresolvable to_content_id: {missing_targets}")

    # 3. All Tags have source support.
    unsupported_tags = [t.tag_id for t in corpus.tags.values() if not t.source_record_ids]
    checks["all_tags_source_linked"] = not unsupported_tags
    if unsupported_tags:
        errors.append(f"Tags without source_record_ids: {unsupported_tags}")

    # 4. No orphan cues (already enforced by loader) or orphan tags (a tag
    #    whose from_cue_id does not resolve — also enforced by loader, but
    #    re-checked here independently).
    orphan_tags = [t.tag_id for t in corpus.tags.values() if t.from_cue_id not in corpus.cues]
    checks["no_orphan_tags"] = not orphan_tags
    if orphan_tags:
        errors.append(f"Tags with unresolvable from_cue_id: {orphan_tags}")

    orphan_cues = [
        cue_id for cue_id in corpus.cues if not any(t.from_cue_id == cue_id for t in corpus.tags.values())
    ]
    checks["no_orphan_cues"] = not orphan_cues
    if orphan_cues:
        errors.append(f"Orphan cues with no outgoing tag: {orphan_cues}")

    # 5. No prohibited authority fields, checked against raw JSON (not the
    #    parsed dataclasses, so a field the loader silently dropped would
    #    still be caught).
    cue_field_violations = [
        {"id": c.get("cue_id"), "fields": sorted(set(c.keys()) - ALLOWED_CUE_FIELDS)}
        for c in raw_cue_registry.get("cues", [])
        if set(c.keys()) - ALLOWED_CUE_FIELDS
    ]
    tag_field_violations = [
        {"id": t.get("tag_id"), "fields": sorted(set(t.keys()) - ALLOWED_TAG_FIELDS)}
        for t in raw_tag_registry.get("tags", [])
        if set(t.keys()) - ALLOWED_TAG_FIELDS
    ]
    authority_violations = verify_no_prohibited_fields(
        raw_cue_registry.get("cues", []), "cue"
    ) + verify_no_prohibited_fields(raw_tag_registry.get("tags", []), "tag")
    checks["no_authority_fields_present"] = (
        not cue_field_violations and not tag_field_violations and not authority_violations
    )
    if cue_field_violations:
        errors.append(f"Cue records with disallowed fields: {cue_field_violations}")
    if tag_field_violations:
        errors.append(f"Tag records with disallowed fields: {tag_field_violations}")
    if authority_violations:
        errors.append(f"Authority-field violations: {authority_violations}")

    # 6/7. Deterministic rebuild: two independent builds from disk must
    #      produce byte-identical hashes.
    projection_a = build_projection(fixtures_dir)
    projection_b = build_projection(fixtures_dir)
    checks["projection_rebuilds_deterministically"] = (
        projection_a.manifest == projection_b.manifest
    )
    if not checks["projection_rebuilds_deterministically"]:
        errors.append("Two independent projection builds produced different manifests.")

    checks["snapshot_hash_matches_manifest"] = projection_a.snapshot == (
        f"sha256:{projection_a.manifest['projection_output_hash']}"
    )
    if not checks["snapshot_hash_matches_manifest"]:
        errors.append("projection.snapshot does not match manifest.projection_output_hash.")

    status = "pass" if all(checks.values()) else "fail"
    return {
        "status": status,
        "checks": checks,
        "errors": errors,
        "manifest": projection_a.manifest,
    }


def main() -> None:
    result = verify_projection()
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
