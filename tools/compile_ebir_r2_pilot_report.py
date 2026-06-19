"""Compile EBIR-R2 pilot Markdown responses into an admin pilot report.

The compiler validates pseudonymous Markdown responses and writes a report even
when responses are missing or invalid. With --fail-on-gate it exits non-zero
until the pilot response set is complete and valid.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


CHECKED_RE = re.compile(r"- \[[xX]\] (.+)")
PACKET_RE = re.compile(r"^## Packet: (r2pkt_[a-f0-9]+)\s*$", re.M)
PSEUDO_RE = re.compile(r"^Reviewer ID:\s*(R\d{2})\s*$", re.M)
REAL_ID_RE = re.compile(r"\breviewer_[a-z0-9_]+\b", re.I)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pseudonym_map(manifest: Dict[str, Any]) -> Dict[str, str]:
    reviewer_ids = sorted({row["reviewer_id"] for row in manifest["assignments"]})
    return {reviewer_id: f"R{index + 1:02d}" for index, reviewer_id in enumerate(reviewer_ids)}


def assigned_packets(manifest: Dict[str, Any]) -> Dict[str, List[str]]:
    mapping = pseudonym_map(manifest)
    packets: Dict[str, List[str]] = defaultdict(list)
    for row in manifest["assignments"]:
        packets[mapping[row["reviewer_id"]]].append(row["packet_id"])
    return {reviewer: sorted(ids) for reviewer, ids in packets.items()}


def section_for_packet(text: str, packet_id: str) -> str:
    start_match = re.search(rf"^## Packet: {re.escape(packet_id)}\s*$", text, re.M)
    if not start_match:
        return ""
    next_match = re.search(r"^## Packet: r2pkt_[a-f0-9]+\s*$", text[start_match.end():], re.M)
    if not next_match:
        return text[start_match.end():]
    return text[start_match.end(): start_match.end() + next_match.start()]


def selected_in_question(section: str, question_number: int) -> List[str]:
    start = re.search(rf"\*\*{question_number}\. .*?\*\*", section)
    if not start:
        return []
    next_q = re.search(r"\*\*\d+\. .*?\*\*", section[start.end():])
    chunk = section[start.end():] if not next_q else section[start.end(): start.end() + next_q.start()]
    return [match.group(1).strip() for match in CHECKED_RE.finditer(chunk)]


def first_free_text(section: str, question_number: int) -> str:
    start = re.search(rf"\*\*{question_number}\. .*?\*\*", section)
    if not start:
        return ""
    next_q = re.search(r"\*\*\d+\. .*?\*\*", section[start.end():])
    chunk = section[start.end():] if not next_q else section[start.end(): start.end() + next_q.start()]
    lines = [
        line.strip()
        for line in chunk.splitlines()
        if line.strip() and not line.strip().startswith("- [")
    ]
    return "\n".join(lines).strip()


def valid_one(value: List[str], allowed: set[str]) -> bool:
    return len(value) == 1 and value[0] in allowed


def parse_response_file(path: Path, expected_reviewer: str, assigned: List[str]) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    errors: List[str] = []
    reviewer_match = PSEUDO_RE.search(text)
    reviewer_id = reviewer_match.group(1) if reviewer_match else None
    if reviewer_id != expected_reviewer:
        errors.append(f"reviewer ID mismatch: expected {expected_reviewer}, found {reviewer_id}")
    real_identity_hits = sorted(set(REAL_ID_RE.findall(text)))
    if real_identity_hits:
        errors.append(f"real reviewer identity fields present: {real_identity_hits}")

    packet_ids = PACKET_RE.findall(text)
    if len(packet_ids) != len(set(packet_ids)):
        errors.append("duplicate packet responses found")
    missing = sorted(set(assigned) - set(packet_ids))
    extra = sorted(set(packet_ids) - set(assigned))
    if missing:
        errors.append(f"missing assigned packets: {missing}")
    if extra:
        errors.append(f"unexpected packet responses: {extra}")

    packet_results: List[Dict[str, Any]] = []
    for packet_id in assigned:
        section = section_for_packet(text, packet_id)
        if not section:
            continue
        handling = selected_in_question(section, 2)
        unsupported = selected_in_question(section, 5)
        quality = selected_in_question(section, 6)
        confidence = selected_in_question(section, 7)
        synthesized = selected_in_question(section, 8)
        synthesized_confidence = selected_in_question(section, 9)

        if not valid_one(handling, {"Resolve", "Partially resolve", "Escalate / abstain"}):
            errors.append(f"{packet_id}: recommended handling must have exactly one valid selection")
        if not valid_one(unsupported, {"Yes", "No", "Unsure"}):
            errors.append(f"{packet_id}: unsupported-claim answer must have exactly one valid selection")
        if len(quality) != 1 or not quality[0][0:1] in {"0", "1", "2"}:
            errors.append(f"{packet_id}: quality score must have exactly one valid selection")
        if not valid_one(confidence, {"1", "2", "3", "4", "5"}):
            errors.append(f"{packet_id}: confidence score must have exactly one valid selection")
        if not valid_one(synthesized, {"Yes", "No", "Unsure"}):
            errors.append(f"{packet_id}: synthesized-recommendation impression must have exactly one valid selection")
        if not valid_one(synthesized_confidence, {"1", "2", "3", "4", "5"}):
            errors.append(f"{packet_id}: synthesized impression confidence must have exactly one valid selection")
        for q_num in (1, 3, 4, 10):
            answer = first_free_text(section, q_num)
            if not answer or answer.startswith("["):
                errors.append(f"{packet_id}: question {q_num} free-text response is incomplete")

        packet_results.append(
            {
                "packet_id": packet_id,
                "handling": handling[0] if handling else None,
                "unsupported_claim": unsupported[0] if unsupported else None,
                "quality": quality[0][0:1] if quality else None,
                "confidence": confidence[0] if confidence else None,
                "synthesized_impression": synthesized[0] if synthesized else None,
                "synthesized_impression_confidence": (
                    synthesized_confidence[0] if synthesized_confidence else None
                ),
                "notes": first_free_text(section, 10),
            }
        )

    return {
        "path": str(path),
        "reviewer_id": reviewer_id,
        "errors": errors,
        "packet_results": packet_results,
        "text": text,
    }


def protocol_version(protocol: Path) -> str:
    text = protocol.read_text(encoding="utf-8")
    status = re.search(r"^Status:\s*\*\*(.*?)\*\*", text, re.M | re.S)
    if not status:
        return "unknown"
    return " ".join(status.group(1).split())


def truthset_version(manifest: Dict[str, Any]) -> str:
    truthset_ref = manifest.get("truthset")
    if not truthset_ref:
        return "unknown"
    truthset_path = Path(truthset_ref)
    if not truthset_path.exists():
        return str(truthset_ref)
    payload = load_json(truthset_path)
    return str(payload.get("version", truthset_ref))


def render_report(
    *,
    protocol: Path,
    manifest_path: Path,
    responses_dir: Path,
    output: Path,
) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    assigned = assigned_packets(manifest)
    response_results: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for pseudo_id, packets in sorted(assigned.items()):
        path = responses_dir / f"reviewer_{pseudo_id}_completed.md"
        if not path.exists():
            errors.append(f"missing response file for {pseudo_id}: {path}")
            response_results[pseudo_id] = {
                "path": str(path),
                "reviewer_id": None,
                "errors": [f"missing response file for {pseudo_id}"],
                "packet_results": [],
                "text": "",
            }
            continue
        parsed = parse_response_file(path, pseudo_id, packets)
        response_results[pseudo_id] = parsed
        errors.extend(f"{pseudo_id}: {error}" for error in parsed["errors"])

    completed = sum(len(result["packet_results"]) for result in response_results.values())
    assigned_count = sum(len(packets) for packets in assigned.values())
    impression_counter = Counter(
        packet["synthesized_impression"]
        for result in response_results.values()
        for packet in result["packet_results"]
        if packet["synthesized_impression"]
    )
    impression_conf_counter = Counter(
        packet["synthesized_impression_confidence"]
        for result in response_results.values()
        for packet in result["packet_results"]
        if packet["synthesized_impression_confidence"]
    )

    decision = "PROCEED_TO_FULL_R2" if not errors else "REVISE_PROTOCOL_ONCE"
    lines = [
        "# EBIR-R2 Pilot Review Report",
        "",
        "Status: PILOT_INSTRUMENT_TEST_ONLY",
        f"Protocol Version: {protocol_version(protocol)}",
        f"Truthset Version: {truthset_version(manifest)}",
        f"Assignment Seed: {manifest.get('seed')}",
        f"Reviewers: {len(assigned)} pseudonymous reviewers",
        f"Packets Reviewed: {completed}",
        "",
        "## Preflight Status",
        "",
        "- Gate result: see `eval_results/ebir_r2/preflight_report.json`",
        "- Test result: pending current CI/test run attachment",
        f"- Packet count: {assigned_count}",
        "- Assignment-balance result: deterministic balanced assignment from manifest",
        "- No-mutation assertion: preflight no-write gate must remain PASS",
        "",
        "## Response Completeness",
        "",
        f"- Packets assigned: {assigned_count}",
        f"- Packets completed: {completed}",
        f"- Missing or invalid responses: {len(errors)}",
    ]
    if errors:
        lines.extend(["", "Validation errors:"])
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- Validation errors: none")

    lines.extend(
        [
            "",
            "## Reviewer-Facing Usability Findings",
            "",
            "- Recurring confusion: pending pilot response review",
            "- Unclear evidence identifiers: pending pilot response review",
            "- Formatting issues: pending pilot response review",
            "- Ambiguity in resolution versus escalation: pending pilot response review",
            "- Missing response fields: see validation errors above",
            "",
            "## Masking-Integrity Findings",
            "",
            f"- Synthesized-recommendation impression distribution: {dict(impression_counter)}",
            f"- Confidence in that impression: {dict(impression_conf_counter)}",
            "- Qualitative packet-type inference notes: pending pilot response review",
            "- No causal claim is made that any condition performed better.",
            "",
            "## Pseudonymous Response Appendix",
            "",
        ]
    )
    for pseudo_id, result in sorted(response_results.items()):
        lines.extend([f"### {pseudo_id}", ""])
        if not result["packet_results"]:
            lines.append("- No completed packet responses available.")
        for packet in result["packet_results"]:
            lines.extend(
                [
                    f"#### Packet {packet['packet_id']}",
                    "",
                    f"- Handling: {packet['handling']}",
                    f"- Unsupported claim: {packet['unsupported_claim']}",
                    f"- Quality: {packet['quality']}",
                    f"- Confidence: {packet['confidence']}",
                    f"- Synthesized recommendation impression: {packet['synthesized_impression']}",
                    f"- Impression confidence: {packet['synthesized_impression_confidence']}",
                    f"- Usability notes: {packet['notes'] or 'none'}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Restricted Post-Freeze Section",
            "",
            "`ADMIN-ONLY - UNBLIND AFTER ALL RESPONSES FROZEN`",
            "",
            "Case-condition mapping is restricted to protocol defect, masking failure, and scoring ambiguity analysis. This pilot sample must not be used to claim EBIR value.",
            "",
        ]
    )
    for row in manifest["assignments"]:
        lines.append(
            f"- Packet {row['packet_id']}: case `{row['case_id']}`, condition `{row['condition_key']}`, reviewer pseudonym `{pseudonym_map(manifest)[row['reviewer_id']]}`"
        )

    lines.extend(
        [
            "",
            "## Pilot Decision",
            "",
            decision,
            "",
            "## Change-Control Record",
            "",
            "Any protocol revision must occur once only. Freeze this pilot report before modifying truthsets, packet normalization, reviewer wording, or rubric text.",
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return {
        "overall_pass": not errors,
        "errors": errors,
        "output": str(output),
        "packets_assigned": assigned_count,
        "packets_completed": completed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    result = render_report(
        protocol=args.protocol,
        manifest_path=args.manifest,
        responses_dir=args.responses_dir,
        output=args.output,
    )
    if result["errors"]:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
    else:
        print("[PASS] all assigned responses complete and valid")
    print(f"packets: {result['packets_completed']}/{result['packets_assigned']}")
    print(f"report: {result['output']}")
    print(f"overall: {'PASS' if result['overall_pass'] else 'FAIL'}")
    if args.fail_on_gate and not result["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
