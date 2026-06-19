"""Tests for EBIR-R2 pilot Markdown rendering and report compilation."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


FORBIDDEN_REVIEWER_TEXT = (
    "raw_evidence",
    "one_pass_reconciliation",
    "ebir_refinement",
    "ebir",
    "repfusion",
    "gold_label",
    "expected_resolved_value",
    "fixture://",
    "packet_hash",
    "critique",
    "revision_delta",
    "promotion_status",
    "auto_promoted",
    "promotable",
    "reviewer_slot_01",
    "reviewer_slot_02",
    "reviewer_slot_03",
)


def _run_preflight(output_dir: Path) -> None:
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


def _run_renderer(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "tools/render_ebir_r2_pilot_markdown.py",
            "--manifest",
            str(output_dir / "assignment_manifest.json"),
            "--packets-dir",
            str(output_dir / "reviewer_packets"),
            "--output-dir",
            str(output_dir / "pilot_review_forms"),
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def _complete_form(form_text: str) -> str:
    completed = form_text
    completed = completed.replace("[free-text response]", "Reviewed response.")
    completed = completed.replace(
        "[List evidence IDs and brief explanation]",
        "E01 and E02 were compared against the proposed handling.",
    )
    completed = completed.replace("- [ ] Escalate / abstain", "- [x] Escalate / abstain")
    completed = completed.replace("- [ ] No", "- [x] No")
    completed = completed.replace("- [ ] 2 - Correct", "- [x] 2 - Correct")
    completed = completed.replace("- [ ] 4", "- [x] 4")
    return completed


def _packet_sections(text: str) -> dict[str, str]:
    packet_ids = re.findall(r"^## Packet: (r2pkt_[a-f0-9]+)\s*$", text, re.M)
    sections: dict[str, str] = {}
    for index, packet_id in enumerate(packet_ids):
        start = text.index(f"## Packet: {packet_id}")
        if index + 1 < len(packet_ids):
            end = text.index(f"## Packet: {packet_ids[index + 1]}")
        else:
            end = len(text)
        sections[packet_id] = text[start:end]
    return sections


def _parent_evidence_section(section: str) -> str:
    return section.split("### Parent Evidence", 1)[1].split("### Assessment Material", 1)[0].strip()


def _assessment_section(section: str) -> str:
    return section.split("### Assessment Material", 1)[1].split("### Reviewer Response", 1)[0].strip()


def test_pilot_markdown_forms_are_masked_and_assignment_scoped(tmp_path):
    output_dir = tmp_path / "ebir_r2"
    _run_preflight(output_dir)

    proc = _run_renderer(output_dir)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    forms = sorted((output_dir / "pilot_review_forms").glob("reviewer_R*.md"))
    assert [path.name for path in forms] == [
        "reviewer_R01.md",
        "reviewer_R02.md",
        "reviewer_R03.md",
    ]

    manifest = json.loads((output_dir / "assignment_manifest.json").read_text(encoding="utf-8"))
    reviewer_ids = sorted({row["reviewer_id"] for row in manifest["assignments"]})
    pseudo = {reviewer_id: f"R{index + 1:02d}" for index, reviewer_id in enumerate(reviewer_ids)}
    assigned_by_pseudo: dict[str, set[str]] = {value: set() for value in pseudo.values()}
    for row in manifest["assignments"]:
        assigned_by_pseudo[pseudo[row["reviewer_id"]]].add(row["packet_id"])

    for form in forms:
        text = form.read_text(encoding="utf-8")
        lowered = text.lower()
        pseudo_id = form.stem.replace("reviewer_", "")
        assert f"Reviewer ID: {pseudo_id}" in text
        assert set(_packet_sections(text)) == assigned_by_pseudo[pseudo_id]
        for forbidden in FORBIDDEN_REVIEWER_TEXT:
            assert forbidden not in lowered
        assert not re.search(r"\br2_[a-z0-9_]+\b", lowered)
        assert not re.search(r"\bsource_[a-f0-9]{16}\b", lowered)


def test_parent_evidence_identity_survives_markdown_rendering(tmp_path):
    output_dir = tmp_path / "ebir_r2"
    _run_preflight(output_dir)
    assert _run_renderer(output_dir).returncode == 0

    manifest = json.loads((output_dir / "assignment_manifest.json").read_text(encoding="utf-8"))
    all_sections: dict[str, str] = {}
    for form in (output_dir / "pilot_review_forms").glob("reviewer_R*.md"):
        all_sections.update(_packet_sections(form.read_text(encoding="utf-8")))

    evidence_by_case: dict[str, set[str]] = {}
    assessment_by_case: dict[str, set[str]] = {}
    for row in manifest["assignments"]:
        section = all_sections[row["packet_id"]]
        evidence_by_case.setdefault(row["case_id"], set()).add(
            _parent_evidence_section(section)
        )
        assessment_by_case.setdefault(row["case_id"], set()).add(
            _assessment_section(section)
        )

    for case_id in evidence_by_case:
        assert len(evidence_by_case[case_id]) == 1
        assert len(assessment_by_case[case_id]) > 1


def test_pilot_report_compiler_accepts_valid_pseudonymous_responses(tmp_path):
    output_dir = tmp_path / "ebir_r2"
    _run_preflight(output_dir)
    assert _run_renderer(output_dir).returncode == 0

    responses = output_dir / "pilot_responses"
    responses.mkdir()
    for form in (output_dir / "pilot_review_forms").glob("reviewer_R*.md"):
        pseudo = form.stem.replace("reviewer_", "")
        (responses / f"reviewer_{pseudo}_completed.md").write_text(
            _complete_form(form.read_text(encoding="utf-8")),
            encoding="utf-8",
        )

    report = output_dir / "ebir_r2_pilot_report.md"
    proc = subprocess.run(
        [
            sys.executable,
            "tools/compile_ebir_r2_pilot_report.py",
            "--protocol",
            "docs/ebir_r2_trial_protocol.md",
            "--manifest",
            str(output_dir / "assignment_manifest.json"),
            "--responses-dir",
            str(responses),
            "--output",
            str(report),
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_text = report.read_text(encoding="utf-8")
    assert "Status: PILOT_INSTRUMENT_TEST_ONLY" in report_text
    assert "Packets Reviewed: 9" in report_text
    assert "ADMIN-ONLY - UNBLIND AFTER ALL RESPONSES FROZEN" in report_text
    assert "reviewer_slot_01" not in report_text
    assert "reviewer_slot_02" not in report_text
    assert "reviewer_slot_03" not in report_text


def test_pilot_report_compiler_rejects_invalid_and_unblinded_response(tmp_path):
    output_dir = tmp_path / "ebir_r2"
    _run_preflight(output_dir)
    assert _run_renderer(output_dir).returncode == 0

    responses = output_dir / "pilot_responses"
    responses.mkdir()
    first_form = output_dir / "pilot_review_forms" / "reviewer_R01.md"
    invalid = first_form.read_text(encoding="utf-8").replace(
        "Reviewer ID: R01",
        "Reviewer ID: reviewer_slot_01",
    )
    (responses / "reviewer_R01_completed.md").write_text(invalid, encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "tools/compile_ebir_r2_pilot_report.py",
            "--protocol",
            "docs/ebir_r2_trial_protocol.md",
            "--manifest",
            str(output_dir / "assignment_manifest.json"),
            "--responses-dir",
            str(responses),
            "--output",
            str(output_dir / "ebir_r2_pilot_report.md"),
            "--fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "missing response file" in combined
    assert "real reviewer identity" in combined or "reviewer ID mismatch" in combined
