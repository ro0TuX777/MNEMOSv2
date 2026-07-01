from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_r1_corpus_manifest_is_non_empty_frozen_and_hash_verified() -> None:
    manifest = _load_json("docs/evidence_admission_and_budgeting_r1_corpus_manifest.json")

    assert manifest["manifest_id"].startswith("evidence_admission_and_budgeting_r1_corpus_")
    assert manifest["status"] == "FROZEN_BEFORE_R1_ENFORCEMENT"
    assert manifest["r0_baseline_commit"] == "bef472112a751436c7af35cf472e13ccfa3a2329"
    assert manifest["corpus_curator"] == "Codex implementation team"
    assert manifest["collection"]["name"] == "evidence_admission_r1_frozen_corpus"
    assert manifest["chunking"]["strategy"] == "word_window"
    assert manifest["embedding_profile"]["retrieval_mode"] == "semantic"

    sources = manifest["sources"]
    units = manifest["retrieval_units"]
    assert len(sources) >= 40
    assert len(units) >= 100

    families = {source["family"] for source in sources}
    assert {
        "gatemem_governance_status",
        "retrieval_hygiene_associative_evidence_admission",
        "unrelated_mnemos_documentation",
    }.issubset(families)

    roles = {source["role"] for source in sources}
    assert {
        "current_state_record",
        "superseded_record",
        "dependency_blocker_record",
        "negative_control_material",
        "duplicate_or_near_duplicate_condition",
    }.issubset(roles)

    source_by_path = {source["path"]: source for source in sources}
    for path, source in source_by_path.items():
        raw = (ROOT / path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source["sha256"]

    valid_paths = set(source_by_path)
    unit_ids = set()
    for unit in units:
        assert unit["source_path"] in valid_paths
        assert unit["unit_id"] not in unit_ids
        assert unit["text_sha256"]
        assert unit["word_count"] > 0
        unit_ids.add(unit["unit_id"])


def test_r1_formal_pack_template_has_schema_but_no_scored_queries() -> None:
    template = _load_json("docs/experiments/evidence_admission_and_budgeting_r1_formal_pack_template.json")

    assert template["pack_id"] == "evidence_admission_and_budgeting_r1_formal_pack_template_v1"
    assert template["pack_type"] == "formal_pack_template"
    assert template["author_boundary"]["required_author"] == "independent_non_implementation_author"
    assert template["queries"] == []
    assert template["minimum_query_count"] == 50
    assert template["fresh_verification_pack_minimum_query_count"] == 20
    assert "BOUNDED_SEMANTIC_RETRIEVAL" in template["allowed_enforced_route_labels"]
    assert "HYBRID_RETRIEVAL" in template["forbidden_enforced_route_labels"]

    required = set(template["query_schema"]["required_fields"])
    assert {
        "query_id",
        "query",
        "accepted_evidence_neighborhood",
        "minimum_required_source_lineage",
        "lower_cost_route_allowed",
        "normal_retrieval_fallback_required",
        "abstention_expected",
    }.issubset(required)
    assert template["schema_examples"]
    assert all(example["scored"] is False for example in template["schema_examples"])


def test_r1_development_pack_is_clearly_non_formal() -> None:
    pack = _load_json("docs/experiments/evidence_admission_and_budgeting_r1_development_pack.json")

    assert pack["pack_id"] == "evidence_admission_and_budgeting_r1_development_pack_v1"
    assert pack["pack_type"] == "development_diagnostic"
    assert pack["evidence_status"] == [
        "DEVELOPMENT_ONLY",
        "NOT_FORMAL_EVIDENCE",
        "NOT_FRESH_VERIFICATION",
        "NOT_INDEPENDENTLY_AUTHORED",
    ]
    assert len(pack["queries"]) >= 8
    assert all(query["scored_for_formal_claim"] is False for query in pack["queries"])


def test_r1_design_and_preregistration_lock_pre_enforcement_boundaries() -> None:
    design = (ROOT / "docs/evidence_admission_and_budgeting_r1_design_note.md").read_text(encoding="utf-8")
    prereg = (ROOT / "docs/evidence_admission_and_budgeting_r1_preregistration.md").read_text(encoding="utf-8")

    assert "NO_DEFAULT_ENFORCEMENT" in design
    assert "NO_HYBRID_ROUTE_ENFORCEMENT_IN_R1" in design
    assert "NORMAL_RETRIEVAL_FALLBACK" in design
    assert "before enforcement implementation or policy tuning" in prereg
    assert "no more than 2 percentage-point drop" in prereg
    assert "independent_non_implementation_author" in prereg


def test_r1_freeze_tool_default_sources_match_manifest() -> None:
    from tools.freeze_evidence_admission_r1_corpus import DEFAULT_SOURCE_SPECS

    manifest = _load_json("docs/evidence_admission_and_budgeting_r1_corpus_manifest.json")
    manifest_paths = [source["path"] for source in manifest["sources"]]
    default_paths = [spec["path"] for spec in DEFAULT_SOURCE_SPECS]

    assert manifest_paths == default_paths
