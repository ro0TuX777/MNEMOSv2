from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from mcp_servers.mnemos_project.server import READ_ONLY_TOOL_NAMES
from prototype.local_project_memory_r0.models import ScopeSpec
from prototype.local_project_memory_r0.packet import build_packet, load_packet, write_packet
from prototype.local_project_memory_r0.retrieval import ProjectMemoryIndex
from prototype.local_project_memory_r0.snapshot import verify_snapshot


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {
    "project_memory_health",
    "get_project_identity",
    "search_project_memory",
    "get_project_artifact",
    "verify_project_snapshot",
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _tree_hashes(repo: Path) -> dict[str, str]:
    return {
        path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in repo.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }


def test_project_memory_lane_has_no_runtime_or_storage_coupling() -> None:
    files = [
        *sorted((ROOT / "prototype" / "local_project_memory_r0").glob("*.py")),
        ROOT / "tools" / "build_local_project_memory_packet.py",
        ROOT / "mcp_servers" / "mnemos_project" / "server.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    forbidden = (
        "mnemos_sdk",
        "MNEMOS_BASE_URL",
        "MNEMOS_QDRANT_COLLECTION",
        "qdrant_client",
        "psycopg",
        "/v1/mnemos/index",
        "subprocess.Popen",
    )
    assert not any(value in text for value in forbidden)


def test_builder_and_sidecar_expose_no_mutation_flags_or_tools() -> None:
    result = subprocess.run(
        ["python", "tools/build_local_project_memory_packet.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = result.stdout
    forbidden_flags = ("--lint", "--fix", "--write", "--collection", "--format")
    assert all(flag not in help_text for flag in forbidden_flags)
    assert READ_ONLY_TOOL_NAMES == EXPECTED_TOOLS
    assert not any(word in " ".join(READ_ONLY_TOOL_NAMES) for word in ("index", "observe", "write", "mutate"))


def test_build_search_and_verify_do_not_mutate_target_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "src" / "logic.py").write_text(
        "def calculate_total(values):\n    return sum(values)\n",
        encoding="utf-8",
    )
    (repo / "docs" / "logic.md").write_text(
        "# Calculation\n\nThe total is computed from all selected values.\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")

    before = _tree_hashes(repo)
    packet_path = tmp_path / "packet.md"
    packet = build_packet(
        repo,
        "fixture",
        ScopeSpec(roots=("src",), files=("docs/logic.md",), excludes=()),
    )
    write_packet(packet_path, packet)
    loaded = load_packet(packet_path)
    hits = ProjectMemoryIndex(loaded).search("how calculate total", top_k=3)
    verification = verify_snapshot(repo, loaded.snapshot)

    assert hits
    assert verification.fresh is True
    assert _tree_hashes(repo) == before


def test_protected_runtime_files_have_no_feature_branch_diff() -> None:
    protected = (
        "service/app.py",
        "mnemos/config.py",
        "mnemos/retrieval",
        "docker-compose.yml",
        "mcp_servers/mnemos/server.py",
    )
    result = subprocess.run(
        ["git", "diff", "--", *protected],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""
