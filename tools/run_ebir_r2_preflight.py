"""Run EBIR-R2 isolated reviewer-trial preflight.

This harness builds blinded raw-evidence, one-pass reconciliation, and EBIR
refinement packets from a frozen synthetic/sanitized truthset. It validates
trial structure only: no reviewer scoring UI, no production routes, no
retrieval-path changes, and no writes to MNEMOS runtime state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUTHSET = PROJECT_ROOT / "benchmarks" / "truthsets" / "ebir_r2_reviewer_tasks.json"
DEFAULT_REVIEWERS = PROJECT_ROOT / "configs" / "ebir_r2_reviewers.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "ebir_r2"

sys.path.insert(0, str(PROJECT_ROOT))

from mnemos.engram.model import Engram
from mnemos.governance.hygiene.reconciliation_runner import ReconciliationRunner
from mnemos.governance.hygiene.repfusion_refiner import (
    CandidateResolution,
    RepFusionRefiner,
)
from mnemos.governance.models.memory_state import GovernanceMeta


CONDITIONS = ("raw_evidence", "one_pass_reconciliation", "ebir_refinement")
BLOCKED_PROMOTION_STATUS = "blocked_from_authoritative_resolution_promotion"
REVIEWER_PACKET_KEYS = (
    "packet_id",
    "reviewer_id",
    "case_blind_id",
    "condition_blind_id",
    "task",
    "parent_evidence",
    "candidate",
    "reviewer_questions",
    "scoring_rubric",
)
CANDIDATE_KEYS = (
    "provided",
    "status",
    "resolved_value",
    "summary",
    "confidence",
    "uncertainty_notes",
    "parent_support_map",
    "claim_support",
)
FORBIDDEN_REVIEWER_KEYS = {
    "gold_label",
    "expected_outcome",
    "expected_resolution_state",
    "expected_resolved_value",
    "expected_abstention",
    "condition",
    "condition_key",
    "condition_label",
    "ebir",
    "passes",
    "critique",
    "revision_delta",
    "packet_hash",
    "promotion_status",
    "auto_promoted",
    "promotable",
}
FORBIDDEN_REVIEWER_STRINGS = (
    "raw_evidence",
    "one_pass_reconciliation",
    "ebir_refinement",
    "EBIR",
    "RepFusion",
    "shadow_only",
    "auto_promoted",
    "promotable",
    "gold_label",
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _blind_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def _case_to_engrams(case: Dict[str, Any]) -> List[Engram]:
    engrams: List[Engram] = []
    for item in case["parent_evidence_packet"]:
        metadata = {
            "truthset_case_id": case["id"],
            "evidence_span": item["content"],
            "temporal_validity": case["temporal_validity"],
            "artifact_id": item["source"],
        }
        engrams.append(
            Engram(
                id=item["id"],
                content=item["content"],
                source=item["source"],
                created_at=item.get("created_at", "2026-06-19T00:00:00Z"),
                metadata=metadata,
                governance=GovernanceMeta(
                    entity_key=case["entity_key"],
                    attribute_key=case["attribute_key"],
                    normalized_value=item["normalized_value"],
                    source_type="fixture",
                    source_id=item["source"],
                    source_authority=float(item["source_authority"]),
                    trust_score=float(item["trust_score"]),
                    utility_score=0.8,
                ),
            )
        )
    return engrams


def _snapshot_engrams(engrams: List[Engram]) -> str:
    payload = [
        engram.to_dict(include_governance=True, include_lineage=True)
        for engram in engrams
    ]
    return _stable_hash(payload)


def _baseline_candidate(engram: Engram) -> CandidateResolution:
    parent_ids = list(engram.metadata.get("parent_ids") or [])
    content = engram.content.lower()
    status = "unresolved" if "unresolved conflict" in content else "one_pass"
    resolved_value: Optional[str] = None
    if status != "unresolved" and engram.governance:
        resolved_value = engram.governance.normalized_value
    return CandidateResolution(
        status=status,
        resolved_value=resolved_value,
        summary=engram.content,
        confidence=0.55 if status == "unresolved" else engram.confidence,
        uncertainty_notes=(
            ["One-pass baseline retained the conflict."]
            if status == "unresolved"
            else []
        ),
        parent_support_map={parent_id: ["mentioned"] for parent_id in parent_ids},
        operator_review_notes=[],
        promotable=True,
    )


def _candidate_to_reviewer(candidate: Optional[CandidateResolution]) -> Dict[str, Any]:
    if candidate is None:
        return {
            "provided": False,
            "status": None,
            "resolved_value": None,
            "summary": None,
            "confidence": None,
            "uncertainty_notes": [],
            "parent_support_map": {},
            "claim_support": [],
        }
    candidate_dict = candidate.to_dict()
    notes = [
        note
        for note in candidate_dict.get("uncertainty_notes", [])
        if "shadow" not in note.lower()
    ]
    return {
        "provided": True,
        "status": candidate_dict.get("status"),
        "resolved_value": candidate_dict.get("resolved_value"),
        "summary": _sanitize_text(str(candidate_dict.get("summary", ""))),
        "confidence": candidate_dict.get("confidence"),
        "uncertainty_notes": [_sanitize_text(str(note)) for note in notes],
        "parent_support_map": candidate_dict.get("parent_support_map", {}),
        "claim_support": candidate_dict.get("claim_support", []),
    }


def _sanitize_text(text: str) -> str:
    replacements = {
        "Shadow candidate only; ": "",
        "shadow candidate only; ": "",
        "parent engrams remain authoritative inputs": "parent evidence remains authoritative",
        "EBIR": "refinement",
        "RepFusion": "refinement",
    }
    sanitized = text
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)
    return sanitized


def _normalize_parent_evidence(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "evidence_id": item["id"],
            "content": item["content"],
            "source": _blind_id("source", case["id"], item["id"]),
            "created_at": item.get("created_at"),
            "source_authority": item.get("source_authority"),
            "trust_score": item.get("trust_score"),
        }
        for item in case["parent_evidence_packet"]
    ]


def _build_condition_materials(case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    engrams = _case_to_engrams(case)
    before_hash = _snapshot_engrams(engrams)
    side_effects = {
        "engram_snapshot_before": before_hash,
        "engram_snapshot_after": None,
        "reconciliation_writes": 0,
        "retrieval_writes": 0,
        "governance_writes": 0,
        "ranking_writes": 0,
        "promotion_writes": 0,
        "memory_writes": 0,
    }

    runner = ReconciliationRunner()
    baseline_start = time.perf_counter()
    baseline_report = runner.run(engrams, dry_run=True, indexer=None)
    baseline_latency_ms = (time.perf_counter() - baseline_start) * 1000
    side_effects["reconciliation_writes"] += baseline_report.resolution_engram_writes
    if not baseline_report.records:
        raise RuntimeError(f"no one-pass record produced for {case['id']}")
    one_pass = _baseline_candidate(baseline_report.records[0].resolution_engram)

    refiner = RepFusionRefiner(max_passes=3)
    ebir_start = time.perf_counter()
    ebir_report = refiner.run(engrams)
    ebir_latency_ms = (time.perf_counter() - ebir_start) * 1000
    if not ebir_report.records:
        raise RuntimeError(f"no EBIR record produced for {case['id']}")
    ebir_record = ebir_report.records[0]
    ebir_candidate = ebir_record.final_candidate
    after_hash = _snapshot_engrams(engrams)
    side_effects["engram_snapshot_after"] = after_hash

    materials = {
        "raw_evidence": {
            "candidate": None,
            "latency_ms": 0.0,
            "shadow_only": True,
            "auto_promoted": False,
            "promotable": False,
        },
        "one_pass_reconciliation": {
            "candidate": one_pass,
            "latency_ms": baseline_latency_ms,
            "shadow_only": True,
            "auto_promoted": False,
            "promotable": False,
        },
        "ebir_refinement": {
            "candidate": ebir_candidate,
            "latency_ms": ebir_latency_ms,
            "shadow_only": ebir_record.shadow_only is True,
            "auto_promoted": ebir_record.auto_promoted is True,
            "promotable": ebir_candidate.promotable is True,
        },
    }
    return materials, side_effects


def _assign_packets(
    cases: List[Dict[str, Any]],
    reviewers: List[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    if len(reviewers) < len(CONDITIONS):
        raise ValueError("EBIR-R2 preflight requires at least three reviewers")
    reviewer_ids = [reviewer["reviewer_id"] for reviewer in reviewers]
    rng = random.Random(seed)
    condition_offset = rng.randrange(len(CONDITIONS))
    assignments: List[Dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for condition_index, condition in enumerate(CONDITIONS):
            rotated_condition_index = (condition_index + condition_offset) % len(CONDITIONS)
            condition = CONDITIONS[rotated_condition_index]
            reviewer_id = reviewer_ids[
                (case_index + rotated_condition_index) % len(reviewer_ids)
            ]
            packet_id = _blind_id("r2pkt", seed, case["id"], condition, reviewer_id)
            assignments.append(
                {
                    "packet_id": packet_id,
                    "case_id": case["id"],
                    "case_blind_id": _blind_id("case", seed, case["id"]),
                    "condition_key": condition,
                    "condition_blind_id": _blind_id("cond", seed, case["id"], condition),
                    "reviewer_id": reviewer_id,
                }
            )
    return assignments


def _build_reviewer_packet(
    *,
    assignment: Dict[str, Any],
    case: Dict[str, Any],
    material: Dict[str, Any],
    truthset: Dict[str, Any],
) -> Dict[str, Any]:
    packet = {
        "packet_id": assignment["packet_id"],
        "reviewer_id": assignment["reviewer_id"],
        "case_blind_id": assignment["case_blind_id"],
        "condition_blind_id": assignment["condition_blind_id"],
        "task": {
            "review_task": case["review_task"],
            "entity_key": case["entity_key"],
            "attribute_key": case["attribute_key"],
        },
        "parent_evidence": _normalize_parent_evidence(case),
        "candidate": _candidate_to_reviewer(material["candidate"]),
        "reviewer_questions": truthset["reviewer_questions"],
        "scoring_rubric": truthset["scoring_rubric"],
    }
    return packet


def _find_forbidden_key(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in FORBIDDEN_REVIEWER_KEYS:
                return key
            found = _find_forbidden_key(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_forbidden_key(item)
            if found:
                return found
    return None


def _contains_forbidden_string(payload: Any) -> Optional[str]:
    text = json.dumps(payload, sort_keys=True).lower()
    for forbidden in FORBIDDEN_REVIEWER_STRINGS:
        if forbidden.lower() in text:
            return forbidden
    return None


def _schema_signature(packet: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "packet_keys": sorted(packet.keys()),
        "task_keys": sorted(packet["task"].keys()),
        "parent_evidence_keys": sorted(packet["parent_evidence"][0].keys())
        if packet["parent_evidence"]
        else [],
        "candidate_keys": sorted(packet["candidate"].keys()),
        "question_keys": sorted(packet["reviewer_questions"][0].keys())
        if packet["reviewer_questions"]
        else [],
    }


def _condition_counts_by_reviewer(assignments: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for assignment in assignments:
        reviewer = assignment["reviewer_id"]
        counts.setdefault(reviewer, {condition: 0 for condition in CONDITIONS})
        counts[reviewer][assignment["condition_key"]] += 1
    return counts


def _balanced(assignments: List[Dict[str, Any]], reviewers: List[Dict[str, Any]]) -> bool:
    counts = _condition_counts_by_reviewer(assignments)
    for reviewer in reviewers:
        counts.setdefault(
            reviewer["reviewer_id"],
            {condition: 0 for condition in CONDITIONS},
        )
    for condition in CONDITIONS:
        values = [counts[reviewer["reviewer_id"]][condition] for reviewer in reviewers]
        if max(values) - min(values) > 1:
            return False
    total_values = [
        sum(counts[reviewer["reviewer_id"]].values()) for reviewer in reviewers
    ]
    return max(total_values) - min(total_values) <= 1


def _evaluate_gates(
    *,
    truthset: Dict[str, Any],
    reviewers: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
    reviewer_packets: List[Dict[str, Any]],
    admin_rows: List[Dict[str, Any]],
    side_effects: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    case_ids = [case["id"] for case in truthset["cases"]]
    condition_by_case: Dict[str, set[str]] = {case_id: set() for case_id in case_ids}
    reviewer_case_pairs: set[tuple[str, str]] = set()
    duplicate_reviewer_case = False
    parent_hashes: Dict[str, set[str]] = {case_id: set() for case_id in case_ids}
    shadow_only_ok = True
    for row in admin_rows:
        condition_by_case[row["case_id"]].add(row["condition_key"])
        pair = (row["reviewer_id"], row["case_id"])
        if pair in reviewer_case_pairs:
            duplicate_reviewer_case = True
        reviewer_case_pairs.add(pair)
        parent_hashes[row["case_id"]].add(row["parent_evidence_hash"])
        if row["condition_key"] == "ebir_refinement":
            shadow_only_ok = shadow_only_ok and row["shadow_only"] and not row["auto_promoted"] and not row["promotable"]

    forbidden_keys = [_find_forbidden_key(packet) for packet in reviewer_packets]
    forbidden_strings = [_contains_forbidden_string(packet) for packet in reviewer_packets]
    signatures = [_schema_signature(packet) for packet in reviewer_packets]
    first_signature = signatures[0] if signatures else {}
    normalized_schema = bool(signatures) and all(
        signature == first_signature for signature in signatures
    )

    no_writes = (
        side_effects["engram_snapshot_before"] == side_effects["engram_snapshot_after"]
        and side_effects["reconciliation_writes"] == 0
        and side_effects["retrieval_writes"] == 0
        and side_effects["governance_writes"] == 0
        and side_effects["ranking_writes"] == 0
        and side_effects["promotion_writes"] == 0
        and side_effects["memory_writes"] == 0
    )

    gates = {
        "all_cases_have_raw_one_pass_ebir_variants": {
            "pass": all(condition_by_case[case_id] == set(CONDITIONS) for case_id in case_ids),
            "detail": {case_id: sorted(condition_by_case[case_id]) for case_id in case_ids},
        },
        "parent_evidence_identical_across_conditions": {
            "pass": all(len(parent_hashes[case_id]) == 1 for case_id in case_ids),
            "detail": {case_id: sorted(parent_hashes[case_id]) for case_id in case_ids},
        },
        "gold_labels_absent_from_reviewer_packets": {
            "pass": all(found is None for found in forbidden_keys),
            "detail": [found for found in forbidden_keys if found is not None],
        },
        "condition_labels_and_ebir_internals_removed": {
            "pass": all(found is None for found in forbidden_strings),
            "detail": [found for found in forbidden_strings if found is not None],
        },
        "no_reviewer_receives_same_underlying_case_twice": {
            "pass": not duplicate_reviewer_case,
            "detail": "unique reviewer/case pairs" if not duplicate_reviewer_case else "duplicate reviewer/case pair found",
        },
        "assignment_balanced_across_conditions": {
            "pass": _balanced(assignments, reviewers),
            "detail": _condition_counts_by_reviewer(assignments),
        },
        "packet_schema_normalized_across_conditions": {
            "pass": normalized_schema,
            "detail": first_signature,
        },
        "ebir_remains_shadow_only": {
            "pass": shadow_only_ok,
            "detail": [
                {
                    "case_id": row["case_id"],
                    "shadow_only": row["shadow_only"],
                    "auto_promoted": row["auto_promoted"],
                    "promotable": row["promotable"],
                }
                for row in admin_rows
                if row["condition_key"] == "ebir_refinement"
            ],
        },
        "no_retrieval_governance_ranking_promotion_or_memory_writes": {
            "pass": no_writes,
            "detail": side_effects,
        },
    }
    return gates


def run_preflight(
    *,
    truthset_path: Path,
    reviewers_path: Path,
    output_dir: Path,
    seed: int,
    blind: bool,
) -> Dict[str, Any]:
    if not blind:
        raise ValueError("EBIR-R2 preflight requires --blind")
    truthset = _load_json(truthset_path)
    reviewer_config = _load_json(reviewers_path)
    reviewers = reviewer_config["reviewers"]
    cases = truthset["cases"]
    assignments = _assign_packets(cases, reviewers, seed)

    materials_by_case: Dict[str, Dict[str, Any]] = {}
    side_effect_summary = {
        "engram_snapshot_before": "",
        "engram_snapshot_after": "",
        "reconciliation_writes": 0,
        "retrieval_writes": 0,
        "governance_writes": 0,
        "ranking_writes": 0,
        "promotion_writes": 0,
        "memory_writes": 0,
    }
    case_side_effects: Dict[str, Any] = {}
    for case in cases:
        materials, side_effects = _build_condition_materials(case)
        materials_by_case[case["id"]] = materials
        case_side_effects[case["id"]] = side_effects

    side_effect_summary["engram_snapshot_before"] = _stable_hash(
        {
            case_id: effects["engram_snapshot_before"]
            for case_id, effects in case_side_effects.items()
        }
    )
    side_effect_summary["engram_snapshot_after"] = _stable_hash(
        {
            case_id: effects["engram_snapshot_after"]
            for case_id, effects in case_side_effects.items()
        }
    )

    reviewer_packets: List[Dict[str, Any]] = []
    admin_rows: List[Dict[str, Any]] = []
    case_by_id = {case["id"]: case for case in cases}
    for assignment in assignments:
        case = case_by_id[assignment["case_id"]]
        material = materials_by_case[case["id"]][assignment["condition_key"]]
        reviewer_packet = _build_reviewer_packet(
            assignment=assignment,
            case=case,
            material=material,
            truthset=truthset,
        )
        reviewer_packets.append(reviewer_packet)
        parent_hash = _stable_hash(reviewer_packet["parent_evidence"])
        admin_rows.append(
            {
                "packet_id": assignment["packet_id"],
                "case_id": assignment["case_id"],
                "case_blind_id": assignment["case_blind_id"],
                "condition_key": assignment["condition_key"],
                "condition_blind_id": assignment["condition_blind_id"],
                "reviewer_id": assignment["reviewer_id"],
                "parent_evidence_hash": parent_hash,
                "candidate_provided": reviewer_packet["candidate"]["provided"],
                "shadow_only": material["shadow_only"],
                "auto_promoted": material["auto_promoted"],
                "promotable": material["promotable"],
                "latency_ms": material["latency_ms"],
            }
        )

    gates = _evaluate_gates(
        truthset=truthset,
        reviewers=reviewers,
        assignments=assignments,
        reviewer_packets=reviewer_packets,
        admin_rows=admin_rows,
        side_effects=side_effect_summary,
    )
    overall_pass = all(gate["pass"] for gate in gates.values())
    manifest = {
        "trial": "EBIR-R2 Shadow Burn-In And Human Review Value Trial",
        "phase": "preflight_packet_generation_only",
        "truthset": str(truthset_path),
        "reviewers": str(reviewers_path),
        "seed": seed,
        "blind": blind,
        "promotion_status": BLOCKED_PROMOTION_STATUS,
        "conditions": list(CONDITIONS),
        "assignments": admin_rows,
    }
    report = {
        "trial": manifest["trial"],
        "phase": manifest["phase"],
        "promotion_status": BLOCKED_PROMOTION_STATUS,
        "truthset_version": truthset["version"],
        "reviewer_config_version": reviewer_config["version"],
        "case_count": len(cases),
        "packet_count": len(reviewer_packets),
        "reviewer_count": len(reviewers),
        "gates": gates,
        "overall_pass": overall_pass,
        "outputs": {
            "manifest": str(output_dir / "assignment_manifest.json"),
            "packets_dir": str(output_dir / "reviewer_packets"),
            "report": str(output_dir / "preflight_report.json"),
        },
    }

    packets_dir = output_dir / "reviewer_packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    for stale_packet in packets_dir.glob("r2pkt_*.json"):
        stale_packet.unlink()
    for packet in reviewer_packets:
        packet_path = packets_dir / f"{packet['packet_id']}.json"
        packet_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assignment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truthset", type=Path, default=DEFAULT_TRUTHSET)
    parser.add_argument("--reviewers", type=Path, default=DEFAULT_REVIEWERS)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    report = run_preflight(
        truthset_path=args.truthset,
        reviewers_path=args.reviewers,
        output_dir=args.output_dir,
        seed=args.seed,
        blind=args.blind,
    )
    for gate_name, gate in report["gates"].items():
        print(f"[{'PASS' if gate['pass'] else 'FAIL'}] {gate_name}")
    print(f"overall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"output: {args.output_dir}")
    if args.fail_on_gate and not report["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
