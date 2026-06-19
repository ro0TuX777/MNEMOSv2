"""Render EBIR-R2 pilot reviewer Markdown forms.

This tool consumes already-blinded preflight reviewer packets and creates one
human-friendly Markdown form per pseudonymous reviewer. It does not call MNEMOS
runtime services and does not alter retrieval, governance, promotion, memory,
Context Atlas, A1, Graph Tier, stores, routes, or production APIs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


FORBIDDEN_REVIEWER_PATTERNS = (
    "raw_evidence",
    "one_pass_reconciliation",
    "ebir_refinement",
    "ebir",
    "repfusion",
    "gold_label",
    "expected_resolved_value",
    "expected_outcome",
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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pseudonym_map(manifest: Dict[str, Any]) -> Dict[str, str]:
    reviewer_ids = sorted({row["reviewer_id"] for row in manifest["assignments"]})
    return {reviewer_id: f"R{index + 1:02d}" for index, reviewer_id in enumerate(reviewer_ids)}


def evidence_map(packet: Dict[str, Any]) -> Dict[str, str]:
    return {
        evidence["evidence_id"]: f"E{index + 1:02d}"
        for index, evidence in enumerate(packet["parent_evidence"])
    }


def replace_evidence_ids(text: Any, mapping: Dict[str, str]) -> str:
    rendered = "" if text is None else str(text)
    for original, blinded in mapping.items():
        rendered = rendered.replace(original, blinded)
    return rendered


def neutral_candidate(packet: Dict[str, Any], mapping: Dict[str, str]) -> str:
    candidate = packet["candidate"]
    if not candidate.get("provided"):
        return "No proposed resolution is provided. Review the parent evidence directly."

    lines = [
        f"- Proposed status: {replace_evidence_ids(candidate.get('status'), mapping)}",
        f"- Proposed value: {replace_evidence_ids(candidate.get('resolved_value'), mapping) or 'none'}",
        f"- Summary: {replace_evidence_ids(candidate.get('summary'), mapping)}",
        f"- Confidence: {candidate.get('confidence')}",
    ]
    notes = candidate.get("uncertainty_notes") or []
    if notes:
        lines.append("- Uncertainty notes:")
        lines.extend(f"  - {replace_evidence_ids(note, mapping)}" for note in notes)

    support_map = candidate.get("parent_support_map") or {}
    if support_map:
        lines.append("- Parent support map:")
        for parent_id, supports in support_map.items():
            blinded_parent = mapping.get(parent_id, "E??")
            rendered_supports = ", ".join(replace_evidence_ids(value, mapping) for value in supports)
            lines.append(f"  - {blinded_parent}: {rendered_supports}")

    claims = candidate.get("claim_support") or []
    if claims:
        lines.append("- Claim support:")
        for claim in claims:
            claim_text = replace_evidence_ids(claim.get("claim"), mapping)
            support_ids = [
                mapping.get(parent_id, "E??")
                for parent_id in claim.get("supporting_parent_ids", [])
            ]
            lines.append(f"  - {claim_text} ({', '.join(support_ids)})")
    return "\n".join(lines)


def parent_evidence(packet: Dict[str, Any], mapping: Dict[str, str]) -> str:
    lines: List[str] = []
    for evidence in packet["parent_evidence"]:
        blinded_id = mapping[evidence["evidence_id"]]
        lines.extend(
            [
                f"#### Evidence {blinded_id}",
                "",
                f"- Date: {evidence.get('created_at')}",
                f"- Source authority: {evidence.get('source_authority')}",
                f"- Trust score: {evidence.get('trust_score')}",
                "",
                replace_evidence_ids(evidence.get("content"), mapping),
                "",
            ]
        )
    return "\n".join(lines).strip()


def packet_form(packet: Dict[str, Any]) -> str:
    mapping = evidence_map(packet)
    return f"""---

## Packet: {packet["packet_id"]}

### Task
{packet["task"]["review_task"]}

### Parent Evidence
{parent_evidence(packet, mapping)}

### Assessment Material
{neutral_candidate(packet, mapping)}

### Reviewer Response

**1. Best-supported resolution**
[free-text response]

**2. Recommended handling**
- [ ] Resolve
- [ ] Partially resolve
- [ ] Escalate / abstain

**3. Decisive evidence**
[List evidence IDs and brief explanation]

**4. Remaining uncertainty**
[free-text response]

**5. Did the material contain an unsupported claim?**
- [ ] Yes
- [ ] No
- [ ] Unsure

If yes or unsure, identify it:
[free-text response]

**6. Evidence-supported decision quality**
- [ ] 0 - Incorrect, unsupported, misleading, or unsafe
- [ ] 1 - Partially correct, incomplete, unclear, or weakly supported
- [ ] 2 - Correct, evidence-supported, appropriately cautious, and reviewable

**7. Confidence**
- [ ] 1
- [ ] 2
- [ ] 3
- [ ] 4
- [ ] 5

**8. Did this packet appear to include a synthesized recommendation?**
- [ ] Yes
- [ ] No
- [ ] Unsure

**9. Confidence in that impression**
- [ ] 1
- [ ] 2
- [ ] 3
- [ ] 4
- [ ] 5

**10. Notes on clarity, ambiguity, or packet usability**
[free-text response]
"""


def forbidden_hits(text: str) -> List[str]:
    lowered = text.lower()
    hits = [pattern for pattern in FORBIDDEN_REVIEWER_PATTERNS if pattern in lowered]
    hits.extend(sorted(set(re.findall(r"\br2_[a-z0-9_]+\b", lowered))))
    hits.extend(sorted(set(re.findall(r"\bsource_[a-f0-9]{16}\b", lowered))))
    return hits


def render_forms(manifest: Path, packets_dir: Path, output_dir: Path) -> Dict[str, Any]:
    manifest_payload = load_json(manifest)
    reviewer_to_pseudo = pseudonym_map(manifest_payload)
    packets = {
        path.stem: load_json(path)
        for path in packets_dir.glob("r2pkt_*.json")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("reviewer_R*.md"):
        stale.unlink()

    reviewer_rows: Dict[str, List[Dict[str, Any]]] = {
        pseudo: [] for pseudo in reviewer_to_pseudo.values()
    }
    for row in manifest_payload["assignments"]:
        reviewer_rows[reviewer_to_pseudo[row["reviewer_id"]]].append(row)

    written: List[str] = []
    gate_errors: List[str] = []
    for pseudo_id in sorted(reviewer_rows):
        rows = sorted(reviewer_rows[pseudo_id], key=lambda row: row["packet_id"])
        body = [
            "# Evidence Review Pilot",
            "",
            f"Reviewer ID: {pseudo_id}",
            "",
            "Please assess each packet independently using only the material provided.",
            "Do not infer or search for any external study condition.",
            "",
        ]
        for row in rows:
            packet = packets.get(row["packet_id"])
            if packet is None:
                gate_errors.append(f"missing packet JSON for {row['packet_id']}")
                continue
            body.append(packet_form(packet))
        rendered = "\n".join(body).rstrip() + "\n"
        hits = forbidden_hits(rendered)
        if hits:
            gate_errors.append(f"{pseudo_id} contains forbidden reviewer-facing text: {hits}")
        path = output_dir / f"reviewer_{pseudo_id}.md"
        path.write_text(rendered, encoding="utf-8")
        written.append(str(path))

    return {
        "forms": written,
        "reviewer_count": len(reviewer_rows),
        "gate_errors": gate_errors,
        "overall_pass": not gate_errors and len(written) == len(reviewer_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--packets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    result = render_forms(args.manifest, args.packets_dir, args.output_dir)
    for form in result["forms"]:
        print(f"[FORM] {form}")
    if result["gate_errors"]:
        for error in result["gate_errors"]:
            print(f"[FAIL] {error}")
    print(f"overall: {'PASS' if result['overall_pass'] else 'FAIL'}")
    if args.fail_on_gate and not result["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
