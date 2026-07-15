from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from prototype.local_project_memory_r0.errors import ErrorCode, ProjectMemoryError
from prototype.local_project_memory_r0.models import ScopeSpec
from prototype.local_project_memory_r0.packet import (
    BEGIN_SENTINEL,
    build_packet,
    load_packet,
    write_packet,
)
from tools.build_local_project_memory_packet import main


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo


def _tree_hashes(repo: Path) -> dict[str, str]:
    return {
        path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in repo.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }


def test_cli_requires_explicit_scope_and_never_defaults_to_repo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path, {"selected/a.py": "VALUE = 1\n"})
    output = tmp_path / "packet.md"
    result = main([
        "--project-root", str(repo),
        "--repo-id", "fixture",
        "--output", str(output),
    ])
    assert result == 2
    assert "SCOPE_REQUIRED" in capsys.readouterr().err
    assert not output.exists()


def test_output_inside_project_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path, {"selected/a.py": "VALUE = 1\n"})
    result = main([
        "--project-root", str(repo),
        "--repo-id", "fixture",
        "--scope-root", "selected",
        "--output", str(repo / "packet.md"),
    ])
    assert result == 2
    assert "OUTPUT_INSIDE_PROJECT" in capsys.readouterr().err


def test_packet_round_trip_preserves_integrity(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"selected/a.py": "VALUE = 1\n", "docs/a.md": "# A\n"})
    packet = build_packet(
        repo,
        "fixture",
        ScopeSpec(roots=("selected",), files=("docs/a.md",), excludes=()),
    )
    path = tmp_path / "packet.md"
    write_packet(path, packet)
    loaded = load_packet(path)
    assert loaded.snapshot.snapshot_id == packet.snapshot.snapshot_id
    assert loaded.packet_sha256 == packet.packet_sha256
    assert loaded.artifacts == packet.artifacts
    assert BEGIN_SENTINEL in path.read_text(encoding="utf-8")


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"selected/a.py": "VALUE = 1\n"})
    packet = build_packet(
        repo,
        "fixture",
        ScopeSpec(roots=("selected",), files=(), excludes=()),
    )
    path = tmp_path / "packet.md"
    path.write_text("operator data", encoding="utf-8")
    with pytest.raises(ProjectMemoryError) as exc:
        write_packet(path, packet)
    assert exc.value.code is ErrorCode.OUTPUT_ALREADY_EXISTS
    assert path.read_text(encoding="utf-8") == "operator data"


def test_cli_build_does_not_mutate_target_tree(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"selected/a.py": "VALUE = 1\n"})
    before = _tree_hashes(repo)
    output = tmp_path / "packet.md"
    result = main([
        "--project-root", str(repo),
        "--repo-id", "fixture",
        "--scope-root", "selected",
        "--output", str(output),
    ])
    assert result == 0
    assert output.exists()
    assert _tree_hashes(repo) == before


def test_parse_failure_writes_incomplete_packet_and_returns_three(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"selected/broken.py": "def broken(:\n"})
    output = tmp_path / "packet.md"
    result = main([
        "--project-root", str(repo),
        "--repo-id", "fixture",
        "--scope-root", "selected",
        "--output", str(output),
    ])
    assert result == 3
    packet = load_packet(output)
    assert packet.usable is False
    assert packet.failures[0]["reason_code"] == "STRUCTURED_PARSE_INCOMPLETE"


def test_tampered_packet_is_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"selected/a.py": "VALUE = 1\n"})
    packet = build_packet(
        repo,
        "fixture",
        ScopeSpec(roots=("selected",), files=(), excludes=()),
    )
    path = tmp_path / "packet.md"
    write_packet(path, packet)
    text = path.read_text(encoding="utf-8").replace("VALUE = 1", "VALUE = 9", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProjectMemoryError) as exc:
        load_packet(path)
    assert exc.value.code is ErrorCode.PACKET_INTEGRITY_INVALID


def test_packet_contains_boundaries_exclusions_and_approval(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path,
        {"selected/a.py": "VALUE = 1\n", "selected/data.json": "{}\n"},
    )
    packet = build_packet(
        repo,
        "fixture",
        ScopeSpec(roots=("selected",), files=(), excludes=()),
    )
    path = tmp_path / "packet.md"
    write_packet(path, packet)
    text = path.read_text(encoding="utf-8")
    assert "unsupported_language" in text
    assert "Human approval checkpoints" in text
    assert "before lint" in text
    assert "before code mutation" in text
    assert str(repo.resolve()) not in text
