from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_focused_workflow_verifies_frozen_g4_before_regression_tests():
    text = (ROOT / ".github/workflows/focused-research-gates.yml").read_text(encoding="utf-8")
    verifier = "python tools/verify_gatemem_g4_frozen.py"
    tests = "python -m pytest -q tests/test_gatemem_g4.py"
    assert verifier in text and tests in text
    assert text.index(verifier) < text.index(tests)
    assert '"prototype/gatemem_g4/**"' in text
    assert '"benchmarks/results/gatemem_g4_frozen_reference_manifest.json"' in text


def test_focused_workflow_generates_r2_artifacts_before_assembler_tests():
    text = (ROOT / ".github/workflows/focused-research-gates.yml").read_text(encoding="utf-8")
    generator = "python tools/run_session_context_assembler_r2_verification.py"
    tests = "tests/test_session_context_assembler_phase_5a.py"
    assert generator in text and tests in text
    assert text.index(generator) < text.index(tests)


def test_release_sbom_workflow_uploads_evidence_and_enforces_hygiene():
    text = (ROOT / ".github/workflows/release-sbom.yml").read_text(encoding="utf-8")
    assert "python tools/generate_release_sbom.py" in text
    assert "actions/upload-artifact@v4" in text
    assert "d['release_ready']" in text
