from tools.seed_mnemos_repo_context import DEFAULT_FILES


def test_default_seed_files_are_present():
    assert len(DEFAULT_FILES) == 4
    assert "docs/benchmarks/gatemem_program_status.md" in DEFAULT_FILES
    assert "benchmarks/results/gatemem_g4_frozen_reference_manifest.md" in DEFAULT_FILES
