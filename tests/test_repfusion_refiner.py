"""Tests for the EBIR shadow refinement lane."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys

from mnemos.audit.forensic_ledger import ForensicLedger
from mnemos.engram.model import Engram
from mnemos.governance.hygiene.repfusion_refiner import RepFusionRefiner
from mnemos.governance.models.memory_state import GovernanceMeta


def _iso_days_ago(days: float) -> str:
    now = datetime.datetime(2026, 6, 11, tzinfo=datetime.timezone.utc)
    ref = now - datetime.timedelta(days=days)
    return ref.isoformat()


def _engram(
    eid: str,
    value: str,
    content: str,
    *,
    authority: float = 0.7,
    trust: float = 0.8,
) -> Engram:
    return Engram(
        id=eid,
        content=content,
        source=f"fixture://{eid}",
        created_at=_iso_days_ago(0),
        metadata={"evidence_span": content, "artifact_id": f"artifact:{eid}"},
        governance=GovernanceMeta(
            entity_key="project:x",
            attribute_key="status",
            normalized_value=value,
            source_type="fixture",
            source_id=eid,
            source_authority=authority,
            trust_score=trust,
            utility_score=0.8,
        ),
    )


def test_refiner_is_shadow_only_and_preserves_parent_engrams():
    old = _engram(
        "old",
        "cancelled_2024",
        "Project X was cancelled in 2024 after field trials.",
        authority=0.6,
    )
    new = _engram(
        "new",
        "extended_2026",
        "Project X was extended to 2026 for continuity analysis.",
        authority=0.8,
    )
    before = [engram.to_dict(include_governance=True) for engram in [old, new]]

    report = RepFusionRefiner(max_passes=3).run([old, new])

    assert report.shadow_only is True
    assert len(report.records) == 1
    record = report.records[0]
    assert record.auto_promoted is False
    assert record.final_candidate.promotable is False
    assert record.packet.parents[0].lineage["artifact_id"]
    assert record.passes[0].packet_hash == RepFusionRefiner.packet_hash(record.packet)
    assert [engram.to_dict(include_governance=True) for engram in [old, new]] == before


def test_structured_challenge_adds_temporal_uncertainty_without_hidden_reasoning(tmp_path):
    ledger = ForensicLedger(db_path=str(tmp_path / "audit.db"))
    old = _engram(
        "old",
        "cancelled_2024",
        "Project X was cancelled in 2024 after field trials.",
        authority=0.6,
    )
    new = _engram(
        "new",
        "extended_2026",
        "Project X was extended to 2026 for continuity analysis.",
        authority=0.8,
    )

    report = RepFusionRefiner(max_passes=3, ledger=ledger).run([old, new])

    pass_record = report.records[0].passes[0]
    assert pass_record.critique.unresolved_temporal_ambiguity
    assert "uncertainty_notes" in pass_record.revision_delta.changed_fields
    assert any(
        "Temporal ambiguity" in note
        for note in pass_record.candidate_after.uncertainty_notes
    )

    txs = ledger.get_recent_transactions(component="governance.repfusion_refiner")
    assert txs
    for tx in txs:
        metadata = json.loads(tx["metadata"])
        assert "chain_of_thought" not in metadata
        assert "critique" in metadata or "final_candidate" in metadata


def test_security_sensitive_case_abstains():
    unclassified = _engram(
        "a",
        "unclassified_only",
        "Tenant Aurora is restricted to unclassified operational support data.",
        authority=0.6,
    )
    secret = _engram(
        "b",
        "secret_allowed",
        "Tenant Aurora may process secret mission telemetry inside the shared enclave.",
        authority=0.8,
    )

    report = RepFusionRefiner(max_passes=3).run([unclassified, secret])

    candidate = report.records[0].final_candidate
    assert candidate.status == "unresolved"
    assert candidate.resolved_value is None
    assert any("operator review" in note.lower() for note in candidate.operator_review_notes)


def test_ebir_benchmark_runs_and_writes_artifact(tmp_path):
    output = tmp_path / "ebir.json"

    proc = subprocess.run(
        [
            sys.executable,
            "tools/run_ebir_refinement_benchmark.py",
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["phase"] == "separate_refinement_lane_not_phase_16"
    assert payload["promotion_status"] == "blocked_from_authoritative_resolution_promotion"
    assert payload["overall_pass"] is True
    assert payload["aggregate"]["case_count"] == 10
    assert payload["aggregate"]["ebir_pass_count"] == 10
    assert payload["aggregate"]["ebir_regression_count"] == 0
    assert payload["aggregate"]["ebir_safety_violation_count"] == 0
    assert payload["gates"]["all_safety_assertions"] is True
    assert payload["gates"]["non_regression_against_one_pass"] is True
    for row in payload["rows"]:
        assert row["classification"]["ebir_pass"] is True
        assert row["classification"]["ebir_safety_violation"] is False
        assert row["ebir"]["shadow_only"] is True
        assert row["ebir"]["auto_promoted"] is False
        assert row["ebir"]["promotable"] is False
        assert row["ebir"]["assertions"]["packet_hash_equality"] is True
        assert row["ebir"]["assertions"]["zero_parent_evidence_mutation"] is True
