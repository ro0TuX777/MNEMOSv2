"""Freeze and aggregate guards for GateMem G2A cross-domain replay."""

from __future__ import annotations

import json
from pathlib import Path

from tools.compile_gatemem_g2a_cross_domain import DOMAINS, verify_frozen_baseline


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def test_frozen_g1_g2_core_matches_published_manifest():
    manifest = verify_frozen_baseline(RESULTS / "gatemem_g2_baseline_manifest.json")
    assert manifest["composite_sha256"] == (
        "4211bc91e7dbe53f588a8ecb00a04e6e44d2fb775ca1e888d058680acb5ad209"
    )


def test_cross_domain_report_covers_all_domains_without_forgetting_claim():
    report = json.loads(
        (RESULTS / "gatemem_g2a_cross_domain_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "GATEMEM_G2A_CROSS_DOMAIN_BASELINE_REPLAY_COMPLETE"
    assert set(report["per_domain"]) == set(DOMAINS)
    assert report["counts"]["checkpoints"] == 2218
    assert report["provenance_integrity"]["rate"] == 1.0
    assert report["deletion_case_refusal"]["active_forgetting_score"] == "NOT_SCORED"
    assert report["deletion_case_refusal"]["deletion_capability_claim"] is False


def test_each_domain_report_is_hash_pinned_and_marks_active_forgetting_unscored():
    for domain in DOMAINS:
        report = json.loads(
            (RESULTS / f"gatemem_g2a_{domain}_report.json").read_text(encoding="utf-8")
        )
        assert report["status"] == "GATEMEM_G2A_DOMAIN_REPLAY_COMPLETE"
        assert report["frozen_baseline_composite_sha256"] == (
            "4211bc91e7dbe53f588a8ecb00a04e6e44d2fb775ca1e888d058680acb5ad209"
        )
        assert report["provenance_integrity"]["rate"] == 1.0
        assert report["deletion_case_refusal"]["active_forgetting_score"] == "NOT_SCORED"

