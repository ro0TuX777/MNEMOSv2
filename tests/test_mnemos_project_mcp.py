from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mcp_servers.mnemos_project import server
from prototype.local_project_memory_r0.models import ScopeSpec
from prototype.local_project_memory_r0.packet import build_packet, write_packet


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def configured_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    source = repo / "pkg/logic.py"
    source.parent.mkdir()
    source.write_bytes(
        b"SNAPSHOT_POLICY = 'reject stale source'\n\ndef verify_snapshot():\n    return True\n"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    packet = build_packet(
        repo,
        "fixture",
        ScopeSpec(roots=("pkg",), files=(), excludes=()),
    )
    packet_path = tmp_path / "packet.md"
    write_packet(packet_path, packet)
    monkeypatch.setenv("MNEMOS_PROJECT_PACKET", str(packet_path))
    monkeypatch.setenv("MNEMOS_PROJECT_ROOT", str(repo))
    monkeypatch.setenv("MNEMOS_PROJECT_REPO_ID", "fixture")
    monkeypatch.setattr(server, "_STATE", None)
    return repo, source, packet_path, packet


def test_server_declares_only_read_only_tools() -> None:
    assert server.READ_ONLY_TOOL_NAMES == {
        "project_memory_health",
        "get_project_identity",
        "search_project_memory",
        "get_project_artifact",
        "verify_project_snapshot",
    }
    text = Path(server.__file__).read_text(encoding="utf-8")
    for forbidden in ("write_observation", "record_decision", "subprocess", "MNEMOS_BASE_URL"):
        assert forbidden not in text


def test_search_returns_hashes_spans_and_boundary(configured_server) -> None:
    response = server.search_project_memory("snapshot verification", top_k=3)
    assert response["status"] == "ok"
    assert response["snapshot_id"].startswith("sha256:")
    assert response["results"]
    result = response["results"][0]
    assert result["provenance_span"]["start_line"] >= 1
    assert result["file_hash"].startswith("sha256:")
    assert result["content_hash"].startswith("sha256:")
    assert "human approval" in response["authority_boundary"].lower()


def test_changed_file_forces_abstention(configured_server) -> None:
    _, source, _, _ = configured_server
    source.write_bytes(b"SNAPSHOT_POLICY = 'changed'\n")
    response = server.search_project_memory("anything", top_k=3)
    assert response["status"] == "abstained"
    assert response["reason_code"] == "SNAPSHOT_MISMATCH"
    assert response["results"] == []


def test_identity_and_exact_artifact_are_source_backed(configured_server) -> None:
    _, _, _, packet = configured_server
    identity = server.get_project_identity()
    assert identity["status"] == "ok"
    assert identity["repo_id"] == "fixture"
    artifact = server.get_project_artifact(packet.artifacts[0].artifact_id)
    assert artifact["status"] == "ok"
    assert artifact["artifact"]["artifact_id"] == packet.artifacts[0].artifact_id


def test_invalid_filters_and_top_k_return_structured_errors(configured_server) -> None:
    invalid_json = server.search_project_memory("snapshot", artifact_types_json="not-json")
    assert invalid_json["status"] == "invalid_request"
    invalid_top_k = server.search_project_memory("snapshot", top_k=21)
    assert invalid_top_k["status"] == "invalid_request"


def test_repo_id_mismatch_fails_closed(configured_server, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOS_PROJECT_REPO_ID", "other")
    monkeypatch.setattr(server, "_STATE", None)
    response = server.project_memory_health()
    assert response["status"] == "unavailable"
    assert response["reason_code"] == "REPO_ID_MISMATCH"


def test_tampered_packet_fails_closed(configured_server, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, packet_path, _ = configured_server
    packet_path.write_text(
        packet_path.read_text(encoding="utf-8").replace("return True", "return False", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "_STATE", None)
    response = server.project_memory_health()
    assert response["status"] == "unavailable"
    assert response["reason_code"] == "PACKET_INTEGRITY_INVALID"


def test_missing_configuration_is_non_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MNEMOS_PROJECT_PACKET", "MNEMOS_PROJECT_ROOT", "MNEMOS_PROJECT_REPO_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server, "_STATE", None)
    response = server.project_memory_health()
    assert response["status"] == "unavailable"
    assert response["reason_code"] == "PROJECT_ROOT_REQUIRED"
    assert "error" not in json.dumps(response).lower()
