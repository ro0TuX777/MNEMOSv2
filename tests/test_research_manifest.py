from __future__ import annotations

import json

from tools.research_manifest import (
    MANIFEST_FILENAME,
    NEW_DOCUMENT,
    NEW_VERSION,
    REUSED_EXACT,
    find_active_by_identity,
    load_manifest,
    normalize_identity_key,
    plan_and_store,
    record_decision,
    save_manifest,
)


def test_identity_key_is_project_scoped_and_case_insensitive():
    assert normalize_identity_key("MNEMOS", "Report.PDF") == "mnemos::report.pdf"
    # Same filename in different projects must not collide.
    assert normalize_identity_key("proj-a", "report.pdf") != normalize_identity_key(
        "proj-b", "report.pdf"
    )


def test_plan_and_store_new_document_then_exact_reuse(tmp_path):
    manifest = load_manifest(tmp_path)

    first = plan_and_store(
        tmp_path, manifest, filename="constitution.pdf", data=b"BYTES-A", project="p"
    )
    assert first.action == NEW_DOCUMENT
    assert first.stored_path == tmp_path / "constitution.pdf"
    record_decision(manifest, first, engram_ids=["research::a0", "research::a1"])

    # Identical bytes, any name -> reuse the stored file, no new path.
    again = plan_and_store(
        tmp_path, manifest, filename="constitution.pdf", data=b"BYTES-A", project="p"
    )
    assert again.action == REUSED_EXACT
    assert again.stored_path == tmp_path / "constitution.pdf"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != MANIFEST_FILENAME) == [
        "constitution.pdf"
    ]


def test_plan_and_store_new_version_overwrites_and_supersedes(tmp_path):
    manifest = load_manifest(tmp_path)

    v1 = plan_and_store(tmp_path, manifest, filename="report.pdf", data=b"V1", project="p")
    record_decision(manifest, v1, engram_ids=["research::v1c0", "research::v1c1"])

    # Same identity, different bytes -> new version overwrites the canonical file.
    v2 = plan_and_store(tmp_path, manifest, filename="report.pdf", data=b"V2-longer", project="p")
    assert v2.action == NEW_VERSION
    assert v2.stored_path == tmp_path / "report.pdf"
    assert v2.superseded is not None
    assert (tmp_path / "report.pdf").read_bytes() == b"V2-longer"

    record, stale = record_decision(manifest, v2, engram_ids=["research::v2c0", "research::v2c1"])
    assert record["version"] == 2
    # Old engrams are stale because none carry over.
    assert set(stale) == {"research::v1c0", "research::v1c1"}

    active = find_active_by_identity(manifest, v2.identity_key)
    assert active is not None and active["sha256"] == v2.sha256
    superseded = [r for r in manifest["records"] if r.get("superseded_by")]
    assert len(superseded) == 1 and superseded[0]["superseded_by"] == v2.sha256


def test_new_version_keeps_overlapping_engrams(tmp_path):
    manifest = load_manifest(tmp_path)
    v1 = plan_and_store(tmp_path, manifest, filename="doc.pdf", data=b"one", project="p")
    record_decision(manifest, v1, engram_ids=["shared", "old-only"])
    v2 = plan_and_store(tmp_path, manifest, filename="doc.pdf", data=b"two", project="p")
    _, stale = record_decision(manifest, v2, engram_ids=["shared", "new-only"])
    # Only the non-overlapping old id is retired.
    assert stale == ["old-only"]


def test_same_name_different_project_are_distinct_documents(tmp_path):
    manifest = load_manifest(tmp_path)
    a = plan_and_store(tmp_path, manifest, filename="notes.pdf", data=b"A", project="proj-a")
    record_decision(manifest, a, engram_ids=["a"])
    b = plan_and_store(tmp_path, manifest, filename="notes.pdf", data=b"B", project="proj-b")
    record_decision(manifest, b, engram_ids=["b"])

    assert a.action == NEW_DOCUMENT and b.action == NEW_DOCUMENT
    # Different project scoping -> b is numbered, not treated as a's version.
    assert b.stored_path == tmp_path / "notes-1.pdf"
    assert all(not r.get("superseded_by") for r in manifest["records"])


def test_manifest_round_trip(tmp_path):
    manifest = load_manifest(tmp_path)
    d = plan_and_store(tmp_path, manifest, filename="x.pdf", data=b"x", project="p")
    record_decision(manifest, d, engram_ids=["research::x0"])
    save_manifest(tmp_path, manifest)

    on_disk = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert on_disk["records"][0]["engram_ids"] == ["research::x0"]
    assert load_manifest(tmp_path)["records"][0]["identity_key"] == "p::x.pdf"
