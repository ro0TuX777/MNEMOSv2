from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from prototype.local_project_memory_r0.errors import ErrorCode, ProjectMemoryError
from prototype.local_project_memory_r0.models import ScopeSpec
from prototype.local_project_memory_r0.snapshot import build_snapshot, verify_snapshot


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo


def _tree_hashes(repo: Path) -> dict[str, str]:
    return {
        path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in repo.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }


def test_scope_admits_only_tracked_python_and_markdown(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        {
            "mnemos/a.py": "VALUE = 1\n",
            "docs/a.md": "# A\n",
            "mnemos/data.json": "{}\n",
            "outside.py": "VALUE = 2\n",
        },
    )
    scope = ScopeSpec(roots=("mnemos",), files=("docs/a.md",), excludes=())
    manifest = build_snapshot(repo, "fixture", scope)
    assert [item.path for item in manifest.files] == ["docs/a.md", "mnemos/a.py"]
    assert manifest.files[0].language == "markdown"
    assert manifest.files[1].language == "python"


def test_explicit_untracked_file_fails(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"mnemos/a.py": "VALUE = 1\n"})
    (repo / "mnemos/untracked.py").write_text("TOKEN = 'not admitted'\n", encoding="utf-8")
    scope = ScopeSpec(roots=(), files=("mnemos/untracked.py",), excludes=())
    with pytest.raises(ProjectMemoryError) as exc:
        build_snapshot(repo, "fixture", scope)
    assert exc.value.code is ErrorCode.SCOPE_FILE_NOT_ADMITTED


def test_empty_filtered_scope_fails(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"selected/data.json": "{}\n"})
    with pytest.raises(ProjectMemoryError) as exc:
        build_snapshot(
            repo,
            "fixture",
            ScopeSpec(roots=("selected",), files=(), excludes=()),
        )
    assert exc.value.code is ErrorCode.EMPTY_ADMITTED_SCOPE


def test_snapshot_reads_without_mutating_project(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"selected/a.py": "VALUE = 1\n"})
    before = _tree_hashes(repo)
    build_snapshot(repo, "fixture", ScopeSpec(roots=("selected",), files=(), excludes=()))
    assert _tree_hashes(repo) == before


def test_dirty_bytes_change_snapshot_not_base_commit(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"selected/a.py": "VALUE = 1\n"})
    scope = ScopeSpec(roots=("selected",), files=(), excludes=())
    clean = build_snapshot(repo, "fixture", scope)
    (repo / "selected/a.py").write_text("VALUE = 2\n", encoding="utf-8")
    dirty = build_snapshot(repo, "fixture", scope)
    assert dirty.commit_hash == clean.commit_hash
    assert dirty.snapshot_id != clean.snapshot_id
    assert dirty.working_tree_state == "dirty"
    assert dirty.dirty_paths == ("selected/a.py",)


def test_changes_outside_scope_do_not_stale_snapshot(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        {"selected/a.py": "VALUE = 1\n", "logs/report.md": "initial\n"},
    )
    scope = ScopeSpec(roots=("selected",), files=(), excludes=())
    before = build_snapshot(repo, "fixture", scope)
    (repo / "logs/report.md").write_text("changed\n", encoding="utf-8")
    after = build_snapshot(repo, "fixture", scope)
    assert after.snapshot_id == before.snapshot_id
    assert verify_snapshot(repo, before).fresh is True


def test_operator_exclusion_and_secret_descendant_are_reported(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        {
            "selected/a.py": "VALUE = 1\n",
            "selected/generated.py": "VALUE = 2\n",
            "selected/.env.py": "SECRET = 'x'\n",
        },
    )
    scope = ScopeSpec(
        roots=("selected",),
        files=(),
        excludes=("selected/generated.py",),
    )
    manifest = build_snapshot(repo, "fixture", scope)
    assert [item.path for item in manifest.files] == ["selected/a.py"]
    reasons = {item["reason"] for item in manifest.exclusions}
    assert {"operator_exclusion", "secret_path"}.issubset(reasons)
    assert all(".env" not in str(item) for item in manifest.exclusions)


def test_explicit_secret_file_fails_before_hashing(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"selected/.env.py": "SECRET = 'x'\n"})
    with pytest.raises(ProjectMemoryError) as exc:
        build_snapshot(
            repo,
            "fixture",
            ScopeSpec(roots=(), files=("selected/.env.py",), excludes=()),
        )
    assert exc.value.code is ErrorCode.SECRET_PATH_REJECTED


def test_resource_ceilings_fail_closed(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        {"selected/a.py": "A = 1\n", "selected/b.py": "B = 2\n"},
    )
    with pytest.raises(ProjectMemoryError) as exc:
        build_snapshot(
            repo,
            "fixture",
            ScopeSpec(roots=("selected",), files=(), excludes=(), max_files=1),
        )
    assert exc.value.code is ErrorCode.RESOURCE_LIMIT_EXCEEDED


def test_live_verification_reports_changed_admitted_file(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"selected/a.py": "VALUE = 1\n"})
    manifest = build_snapshot(
        repo,
        "fixture",
        ScopeSpec(roots=("selected",), files=(), excludes=()),
    )
    (repo / "selected/a.py").write_text("VALUE = 9\n", encoding="utf-8")
    verification = verify_snapshot(repo, manifest)
    assert verification.fresh is False
    assert verification.reason_code == ErrorCode.SNAPSHOT_MISMATCH.value
    assert verification.mismatches[0]["path"] == "selected/a.py"
