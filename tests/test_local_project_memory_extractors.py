from __future__ import annotations

import builtins
import subprocess
from pathlib import Path

import pytest

from prototype.local_project_memory_r0.errors import ErrorCode, ProjectMemoryError
from prototype.local_project_memory_r0.markdown_extractor import extract_markdown
from prototype.local_project_memory_r0.models import ScopeSpec, SourceSpan
from prototype.local_project_memory_r0.python_extractor import extract_python
from prototype.local_project_memory_r0.snapshot import build_snapshot


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _fixture(tmp_path: Path, relative: str, source: str):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source.encode("utf-8"))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    manifest = build_snapshot(
        repo,
        "fixture",
        ScopeSpec(roots=(), files=(relative,), excludes=()),
    )
    return repo, manifest, manifest.files[0]


def test_python_symbols_preserve_decorators_and_exact_spans(tmp_path: Path) -> None:
    source = "@router.get('/health')\ndef health():\n    return {'status': 'ok'}\n"
    root, manifest, snapshot_file = _fixture(tmp_path, "service/app.py", source)
    artifacts = extract_python(root, manifest, snapshot_file)
    handler = next(item for item in artifacts if item.qualified_name == "service.app.health")
    assert handler.span == SourceSpan(1, 3)
    assert handler.content == source
    assert handler.metadata["route_path"] == "/health"
    assert handler.metadata["route_detection"] == "heuristic"
    assert handler.metadata["http_methods"] == ["GET"]
    assert handler.file_hash == snapshot_file.file_hash
    assert handler.content_hash.startswith("sha256:")


def test_python_emits_classes_methods_imports_constants_and_tests(tmp_path: Path) -> None:
    source = (
        "import os\n"
        "LIMIT = 5\n\n"
        "class Worker:\n"
        "    def run(self):\n"
        "        return LIMIT\n\n"
        "def test_worker():\n"
        "    assert Worker().run() == 5\n"
    )
    root, manifest, snapshot_file = _fixture(tmp_path, "pkg/worker.py", source)
    artifacts = extract_python(root, manifest, snapshot_file)
    kinds = {(item.qualified_name, item.metadata.get("symbol_kind")) for item in artifacts}
    assert ("pkg.worker.Worker", "class") in kinds
    assert ("pkg.worker.Worker.run", "method") in kinds
    assert ("pkg.worker.test_worker", "test_function") in kinds
    assert any(item.artifact_type == "python_import" for item in artifacts)
    assert any(item.artifact_type == "python_config_constant" and item.qualified_name.endswith("LIMIT") for item in artifacts)


def test_python_does_not_import_or_execute_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, manifest, snapshot_file = _fixture(
        tmp_path,
        "selected/a.py",
        "raise RuntimeError('must not execute')\n",
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "selected.a":
            raise AssertionError("target module imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = extract_python(root, manifest, snapshot_file)
    assert result[0].artifact_type == "python_module"


def test_python_syntax_error_is_structurally_incomplete(tmp_path: Path) -> None:
    root, manifest, snapshot_file = _fixture(tmp_path, "selected/broken.py", "def broken(:\n")
    with pytest.raises(ProjectMemoryError) as exc:
        extract_python(root, manifest, snapshot_file)
    assert exc.value.code is ErrorCode.STRUCTURED_PARSE_INCOMPLETE


def test_changed_file_is_rejected_before_extraction(tmp_path: Path) -> None:
    root, manifest, snapshot_file = _fixture(tmp_path, "selected/a.py", "VALUE = 1\n")
    (root / "selected/a.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ProjectMemoryError) as exc:
        extract_python(root, manifest, snapshot_file)
    assert exc.value.code is ErrorCode.PACKET_INTEGRITY_INVALID


def test_markdown_heading_sections_have_exact_lines(tmp_path: Path) -> None:
    source = "# Decision\n\nStatus: Accepted\n\n## Boundary\nRead only.\n"
    root, manifest, snapshot_file = _fixture(tmp_path, "docs/decision.md", source)
    artifacts = extract_markdown(root, manifest, snapshot_file)
    boundary = next(
        item
        for item in artifacts
        if item.metadata.get("heading_path") == ["Decision", "Boundary"]
    )
    assert boundary.span == SourceSpan(5, 6)
    assert boundary.content == "## Boundary\nRead only.\n"
    document = next(item for item in artifacts if item.artifact_type == "markdown_decision")
    assert document.metadata["status"]["normalized"] == "accepted"


def test_markdown_setext_and_supersession_are_explicit(tmp_path: Path) -> None:
    source = "ADR 0001\n========\n\nSupersedes: docs/old.md\n"
    root, manifest, snapshot_file = _fixture(tmp_path, "docs/adr/0001.md", source)
    artifacts = extract_markdown(root, manifest, snapshot_file)
    document = next(item for item in artifacts if item.metadata["heading_level"] == 0)
    section = next(item for item in artifacts if item.metadata["heading_level"] == 1)
    assert document.artifact_type == "markdown_adr"
    assert section.metadata["heading_path"] == ["ADR 0001"]
    assert document.metadata["supersedes"] == ["docs/old.md"]
