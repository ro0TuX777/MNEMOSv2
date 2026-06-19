"""Tests for the EBIR-R2 isolated preflight harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_ebir_r2_preflight_command_builds_blinded_packets(tmp_path):
    output_dir = tmp_path / "ebir_r2"

    proc = subprocess.run(
        [
            sys.executable,
            "tools/run_ebir_r2_preflight.py",
            "--truthset",
            "benchmarks/truthsets/ebir_r2_reviewer_tasks.json",
            "--reviewers",
            "configs/ebir_r2_reviewers.json",
            "--seed",
            "20260619",
            "--blind",
            "--output-dir",
            str(output_dir),
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((output_dir / "preflight_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "assignment_manifest.json").read_text(encoding="utf-8"))

    assert report["phase"] == "preflight_packet_generation_only"
    assert report["promotion_status"] == "blocked_from_authoritative_resolution_promotion"
    assert report["overall_pass"] is True
    assert report["case_count"] == 3
    assert report["packet_count"] == 9
    assert all(gate["pass"] for gate in report["gates"].values())
    assert len(manifest["assignments"]) == 9

    packet_paths = sorted((output_dir / "reviewer_packets").glob("*.json"))
    assert len(packet_paths) == 9
    packets = [json.loads(path.read_text(encoding="utf-8")) for path in packet_paths]
    packet_key_sets = {tuple(sorted(packet.keys())) for packet in packets}
    candidate_key_sets = {tuple(sorted(packet["candidate"].keys())) for packet in packets}

    assert len(packet_key_sets) == 1
    assert len(candidate_key_sets) == 1
    for packet in packets:
        packet_text = json.dumps(packet, sort_keys=True)
        question_ids = {question["id"] for question in packet["reviewer_questions"]}
        assert "synthesized_recommendation_impression" in question_ids
        assert "synthesized_recommendation_impression_confidence" in question_ids
        assert "gold_label" not in packet_text
        assert "expected_resolved_value" not in packet_text
        assert "raw_evidence" not in packet_text
        assert "one_pass_reconciliation" not in packet_text
        assert "ebir_refinement" not in packet_text
        assert "EBIR" not in packet_text
        assert "ebir" not in packet_text.lower()
        assert "passes" not in packet
        assert "condition_key" not in packet
        assert "promotion_status" not in packet


def test_ebir_r2_parent_evidence_identity_uses_visible_evidence_only(tmp_path):
    output_dir = tmp_path / "ebir_r2"
    subprocess.run(
        [
            sys.executable,
            "tools/run_ebir_r2_preflight.py",
            "--blind",
            "--output-dir",
            str(output_dir),
            "--fail-on-gate",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((output_dir / "assignment_manifest.json").read_text(encoding="utf-8"))
    packet_by_id = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (output_dir / "reviewer_packets").glob("*.json")
    }
    evidence_by_case: dict[str, set[str]] = {}
    packet_by_case: dict[str, set[str]] = {}
    candidate_by_case: dict[str, set[str]] = {}
    for assignment in manifest["assignments"]:
        packet = packet_by_id[assignment["packet_id"]]
        case_id = assignment["case_id"]
        evidence_by_case.setdefault(case_id, set()).add(
            json.dumps(packet["parent_evidence"], sort_keys=True)
        )
        packet_by_case.setdefault(case_id, set()).add(
            json.dumps(packet, sort_keys=True)
        )
        candidate_by_case.setdefault(case_id, set()).add(
            json.dumps(packet["candidate"], sort_keys=True)
        )

    for case_id in evidence_by_case:
        assert len(evidence_by_case[case_id]) == 1
        assert len(packet_by_case[case_id]) == 3
        assert len(candidate_by_case[case_id]) > 1


def test_ebir_r2_assignment_manifest_is_balanced_and_case_unique(tmp_path):
    output_dir = tmp_path / "ebir_r2"
    subprocess.run(
        [
            sys.executable,
            "tools/run_ebir_r2_preflight.py",
            "--blind",
            "--output-dir",
            str(output_dir),
            "--fail-on-gate",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((output_dir / "assignment_manifest.json").read_text(encoding="utf-8"))
    assignments = manifest["assignments"]
    reviewer_case_pairs = {
        (assignment["reviewer_id"], assignment["case_id"]) for assignment in assignments
    }
    assert len(reviewer_case_pairs) == len(assignments)

    counts: dict[str, dict[str, int]] = {}
    for assignment in assignments:
        reviewer = assignment["reviewer_id"]
        counts.setdefault(
            reviewer,
            {"raw_evidence": 0, "one_pass_reconciliation": 0, "ebir_refinement": 0},
        )
        counts[reviewer][assignment["condition_key"]] += 1

    assert counts
    for reviewer_counts in counts.values():
        assert reviewer_counts == {
            "raw_evidence": 1,
            "one_pass_reconciliation": 1,
            "ebir_refinement": 1,
        }


def test_ebir_r2_preflight_requires_blinding(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "tools/run_ebir_r2_preflight.py",
            "--output-dir",
            str(tmp_path / "ebir_r2"),
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "requires --blind" in (proc.stdout + proc.stderr)


def test_ebir_r2_full_truthset_preflight_passes(tmp_path):
    output_dir = tmp_path / "ebir_r2_full"

    proc = subprocess.run(
        [
            sys.executable,
            "tools/run_ebir_r2_preflight.py",
            "--truthset",
            "benchmarks/truthsets/ebir_r2_full_reviewer_tasks.json",
            "--reviewers",
            "configs/ebir_r2_reviewers.json",
            "--seed",
            "20260619",
            "--blind",
            "--output-dir",
            str(output_dir),
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((output_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert report["truthset_version"] == "ebir-r2-full-reviewer-tasks-v1"
    assert report["case_count"] == 9
    assert report["packet_count"] == 27
    assert report["overall_pass"] is True
    assert all(gate["pass"] for gate in report["gates"].values())
