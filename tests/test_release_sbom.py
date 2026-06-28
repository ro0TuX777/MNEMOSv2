from tools.generate_release_sbom import build_artifacts, declared_dependencies


def test_spdx_sbom_covers_every_declared_dependency():
    sbom, hygiene = build_artifacts()
    declared = declared_dependencies()
    assert sbom["spdxVersion"] == "SPDX-2.3"
    assert len(sbom["packages"]) == len(declared) == hygiene["declared_dependency_count"]
    assert len(sbom["documentDescribes"]) == len(declared)


def test_hygiene_report_refuses_release_ready_claim_for_unlocked_requirements():
    _, hygiene = build_artifacts()
    assert hygiene["non_exact_requirements"]
    assert hygiene["hash_pinned_requirements"] == 0
    assert hygiene["release_ready"] is False
    assert hygiene["vulnerability_audit"].startswith("NOT_RUN")
