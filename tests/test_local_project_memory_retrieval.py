from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from prototype.local_project_memory_r0.errors import ErrorCode, ProjectMemoryError
from prototype.local_project_memory_r0.models import ScopeSpec
from prototype.local_project_memory_r0.packet import build_packet
from prototype.local_project_memory_r0.retrieval import ProjectMemoryIndex


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def packet(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    files = {
        "pkg/worker.py": (
            "STALE_POLICY = 'reject'\n\n"
            "class Worker:\n"
            "    def run(self):\n"
            "        return 'active snapshot'\n"
        ),
        "tests/test_worker.py": (
            "from pkg.worker import Worker\n\n"
            "def test_worker_run():\n"
            "    assert Worker().run()\n"
        ),
        "docs/decision.md": (
            "# Snapshot Decision\n\n"
            "Status: Accepted\n\n"
            "Reject stale source hashes before returning project evidence.\n"
        ),
    }
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    value = build_packet(
        repo,
        "fixture",
        ScopeSpec(roots=("pkg", "tests", "docs"), files=(), excludes=()),
    )
    return repo, value


def test_exact_symbol_ranks_first(packet) -> None:
    _, value = packet
    hits = ProjectMemoryIndex(value).search("pkg.worker.Worker.run", top_k=5)
    assert hits[0].artifact.qualified_name == "pkg.worker.Worker.run"
    assert "exact_qualified_name" in hits[0].match_reasons
    assert hits[0].score_components["exact_qualified_name"] == 100


def test_concept_terms_return_source_backed_logic(packet) -> None:
    _, value = packet
    hits = ProjectMemoryIndex(value).search("reject stale snapshot source hashes", top_k=8)
    assert hits
    assert hits[0].artifact.file_path == "docs/decision.md"
    assert all(hit.artifact.content for hit in hits)
    assert all(hit.artifact.span.start_line >= 1 for hit in hits)
    assert all(hit.artifact.file_hash.startswith("sha256:") for hit in hits)


def test_filters_are_eligibility_gates(packet) -> None:
    _, value = packet
    hits = ProjectMemoryIndex(value).search(
        "snapshot",
        path_prefix="docs/",
        artifact_types=("markdown_section",),
    )
    assert hits
    assert all(hit.artifact.file_path.startswith("docs/") for hit in hits)
    assert all(hit.artifact.artifact_type == "markdown_section" for hit in hits)


def test_exact_path_receives_path_boost(packet) -> None:
    _, value = packet
    hits = ProjectMemoryIndex(value).search("tests/test_worker.py", top_k=3)
    assert hits[0].artifact.file_path == "tests/test_worker.py"
    assert hits[0].score_components["exact_path"] == 90


def test_cross_scope_artifact_is_never_indexed(packet) -> None:
    _, value = packet
    injected = replace(value.artifacts[0], file_path="outside/secret.py")
    tampered = replace(value, artifacts=(*value.artifacts, injected))
    with pytest.raises(ProjectMemoryError) as exc:
        ProjectMemoryIndex(tampered)
    assert exc.value.code is ErrorCode.CROSS_SCOPE_EVIDENCE


def test_unusable_packet_cannot_be_indexed(packet) -> None:
    _, value = packet
    with pytest.raises(ProjectMemoryError) as exc:
        ProjectMemoryIndex(replace(value, usable=False))
    assert exc.value.code is ErrorCode.STRUCTURED_PARSE_INCOMPLETE


def test_search_is_deterministic_and_top_k_is_bounded(packet) -> None:
    _, value = packet
    index = ProjectMemoryIndex(value)
    first = index.search("Worker snapshot", top_k=3)
    second = index.search("Worker snapshot", top_k=3)
    assert [hit.artifact.artifact_id for hit in first] == [hit.artifact.artifact_id for hit in second]
    with pytest.raises(ValueError):
        index.search("Worker", top_k=0)
    with pytest.raises(ValueError):
        index.search("Worker", top_k=21)


def test_get_returns_exact_artifact(packet) -> None:
    _, value = packet
    index = ProjectMemoryIndex(value)
    expected = value.artifacts[0]
    assert index.get(expected.artifact_id) == expected
    with pytest.raises(KeyError):
        index.get("missing")
