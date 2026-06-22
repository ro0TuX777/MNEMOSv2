"""Phase 3 offline A/B/C replay harness logic.

A = full conversation history (no token budget cap; the unconstrained
    ceiling other conditions are measured against).
B = naive recency-only sliding window, capped at the same token budget as C.
C = governed episode-selected context package (Phase 1 assembler, unchanged
    behavior - this module never alters selection logic, only replays it).

This module performs no I/O and makes no model/LLM calls: every required
Phase 3 metric (recall, token accounting, provenance, contradiction-
awareness classification) is computable structurally from the frozen
corpus's own ground-truth fields, so no model dependency is needed to
satisfy the required output contract. It has no import path into mnemos/,
service/, or mnemos_sdk/, and writes nothing to disk - see
docs/session_context_assembler_phase_3_notes.md for the governing posture.

No claim of human value, non-inferiority, or production readiness is made
or implied by this module. See ADR 0007.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from .assembler import assemble_context_package
from .corpus import compute_case_hash
from .extractor import extract_ids_from_turn
from .models import Turn, turn_from_dict
from .selector_s1 import assemble_context_package_s1

PROTOTYPE_VERSION = "0.1.0-prototype"

# Ground-truth contradiction-awareness classification for the r0 corpus's
# contradiction_aware_followup and unresolved_vs_resolved_decision_distinction
# families. Derived from each case's own frozen `notes` field (quoted in
# docs/session_context_assembler_phase_3_notes.md) - this is an
# interpretation of existing frozen data, not a new corpus field, and r0 is
# not modified to add it.
CONTRADICTION_GROUND_TRUTH: Dict[str, Dict] = {
    "sca_r0_caf_001": {
        "required_ids": frozenset({"SRC-SCA-policy-b17", "SRC-SCA-helpdesk-1"}),
        "category": "resolved",
    },
    "sca_r0_caf_002": {
        "required_ids": frozenset({"SRC-SCA-atlas-archive-2024", "SRC-SCA-atlas-update-2026"}),
        "category": "resolved",
    },
    "sca_r0_caf_003": {
        "required_ids": frozenset({"SRC-SCA-runbook-old", "SRC-SCA-config-current"}),
        "category": "unresolved",
    },
    "sca_r0_urd_001": {
        "required_ids": frozenset({"SRC-SCA-cache-isolation-discussion"}),
        "category": "unresolved",
    },
    "sca_r0_urd_002": {
        "required_ids": frozenset({"SRC-SCA-rollback-window-discussion"}),
        "category": "unresolved",
    },
    "sca_r0_urd_003": {
        "required_ids": frozenset({"DEC-SCA-113", "SRC-SCA-explain-format-discussion"}),
        "category": "mixed",
    },
}


def _token_estimate(turns: Sequence[Turn]) -> int:
    return sum(len(t.content.split()) for t in turns)


def _extract_ids(turns_by_id: Dict[str, Turn], turn_ids: Sequence[str]) -> Tuple[Set[str], Set[str]]:
    decisions: Set[str] = set()
    sources: Set[str] = set()
    for tid in turn_ids:
        d, s = extract_ids_from_turn(turns_by_id[tid])
        decisions |= d
        sources |= s
    return decisions, sources


def _recall(required: Sequence[str], recovered: Set[str]) -> Optional[float]:
    """None when nothing is required for this case (recall is undefined,
    not vacuously perfect)."""
    if not required:
        return None
    return len(set(required) & recovered) / len(required)


def _contradiction_awareness_result(
    case: Dict | str, recovered_decisions: Set[str], recovered_sources: Set[str]
) -> Tuple[str, Optional[bool]]:
    """Classify awareness only when all case-required artifacts survive.

    R1 carries an explicit category, avoiding inference from prose notes.
    The string form remains supported for frozen-r0 callers and tests.
    """
    if isinstance(case, str):
        gt = CONTRADICTION_GROUND_TRUTH.get(case)
    else:
        category = case.get("expected_contradiction_status")
        gt = (
            {
                "required_ids": frozenset(
                    case["required_prior_decision_ids"] + case["required_source_ids"]
                ),
                "category": category,
            }
            if category is not None
            else CONTRADICTION_GROUND_TRUTH.get(case["id"])
        )
    if gt is None:
        return "not_applicable", None
    recovered = recovered_decisions | recovered_sources
    observed = gt["category"] if gt["required_ids"] <= recovered else "omitted"
    return observed, observed == gt["category"]


def run_condition_a(case: Dict) -> Dict:
    turns = [turn_from_dict(t) for t in case["conversation_history"]]
    turns_by_id = {t.turn_id: t for t in turns}
    turn_ids = [t.turn_id for t in turns]
    decisions, sources = _extract_ids(turns_by_id, turn_ids)
    return {
        "condition": "A_full_history",
        "selected_turn_ids": turn_ids,
        "selected_episode_ids": [],
        "selected_parent_engram_ids": sorted(decisions),
        "selected_source_ids": sorted(sources),
        "token_estimate": _token_estimate(turns),
        "labels": [],
        "selection_rationale": [],
    }


def run_condition_b(case: Dict, token_budget: int) -> Dict:
    """Naive recency-only window: walk backward from the most recent turn,
    accumulating until the shared token budget would be exceeded. No
    relevance scoring, no episode clustering - recency is the only signal."""
    turns = [turn_from_dict(t) for t in case["conversation_history"]]
    turns_by_id = {t.turn_id: t for t in turns}

    selected: List[Turn] = []
    used_tokens = 0
    for turn in reversed(turns):
        turn_tokens = len(turn.content.split())
        if selected and used_tokens + turn_tokens > token_budget:
            break
        selected.append(turn)
        used_tokens += turn_tokens
    selected.reverse()
    turn_ids = [t.turn_id for t in selected]

    decisions, sources = _extract_ids(turns_by_id, turn_ids)
    return {
        "condition": "B_sliding_window",
        "selected_turn_ids": turn_ids,
        "selected_episode_ids": [],
        "selected_parent_engram_ids": sorted(decisions),
        "selected_source_ids": sorted(sources),
        "token_estimate": used_tokens,
        "labels": [],
        "selection_rationale": [],
        "context_budget": token_budget,
    }


def run_condition_c(case: Dict, corpus_manifest_hash: str, seed: int, token_budget: int) -> Dict:
    pkg = assemble_context_package(case, corpus_manifest_hash, seed=seed, token_budget=token_budget)
    return {
        "condition": "C_governed_episode_selected",
        "selected_turn_ids": pkg["selected_turn_ids"],
        "selected_episode_ids": pkg["selected_episode_ids"],
        "selected_parent_engram_ids": pkg["selected_parent_engram_ids"],
        "selected_source_ids": pkg["selected_source_ids"],
        "token_estimate": pkg["token_estimate"],
        "labels": pkg["synthetic_context_labels"],
        "selection_rationale": pkg["selection_rationale"],
        "context_budget": token_budget,
    }


def run_condition_c1(case: Dict, corpus_manifest_hash: str, seed: int, token_budget: int) -> Dict:
    pkg = assemble_context_package_s1(
        case, corpus_manifest_hash, seed=seed, token_budget=token_budget
    )
    return {
        "condition": "C1_selector_s1_mandatory_preservation",
        "selected_turn_ids": pkg["selected_turn_ids"],
        "selected_episode_ids": pkg["selected_episode_ids"],
        "selected_parent_engram_ids": pkg["selected_parent_engram_ids"],
        "selected_source_ids": pkg["selected_source_ids"],
        "token_estimate": pkg["token_estimate"],
        "labels": pkg["synthetic_context_labels"],
        "selection_rationale": pkg["selection_rationale"],
        "context_budget_insufficient": pkg["context_budget_insufficient"],
        "omitted_required_artifact_types": pkg["omitted_required_artifact_types"],
        "selection_abstention_reason": pkg["selection_abstention_reason"],
        "context_budget": token_budget,
    }


def _lineage_loss(record: Dict) -> Tuple[int, int]:
    """(decision_lineage_loss_count, source_lineage_loss_count): selected
    IDs that cannot be traced to any labeled parent_turn_ids. Conditions A
    and B emit no labels at all (no governance metadata layer exists for
    naive history dumps), so traceability there is checked against the
    fact that every reported ID is extracted only from a selected turn by
    construction - loss is structurally 0 for A/B as well."""
    labels = record.get("labels") or []
    if not labels:
        return 0, 0
    labeled_decisions = {d for label in labels for d in label.get("parent_engram_ids", [])}
    labeled_sources = {s for label in labels for s in label.get("parent_source_ids", [])}
    decision_loss = len([i for i in record["selected_parent_engram_ids"] if i not in labeled_decisions])
    source_loss = len([i for i in record["selected_source_ids"] if i not in labeled_sources])
    return decision_loss, source_loss


def _synthetic_context_label_coverage(record: Dict) -> float:
    episode_ids = record["selected_episode_ids"]
    if not episode_ids:
        # A/B select no episodes at all (not an episodic condition), so
        # there is nothing to label and nothing un-labeled either. Reported
        # as 0.0: naive baselines never produce governance labeling,
        # regardless of episode count, which is the point being measured.
        return 0.0
    labeled = {label["episode_id"] for label in (record.get("labels") or [])}
    return len(labeled & set(episode_ids)) / len(episode_ids)


def build_case_record(
    case: Dict,
    condition_record: Dict,
    full_history_token_estimate: int,
    corpus_manifest_hash: str,
    seed: int,
) -> Dict:
    decisions = set(condition_record["selected_parent_engram_ids"])
    sources = set(condition_record["selected_source_ids"])
    selected_turn_ids = set(condition_record["selected_turn_ids"])
    known_irrelevant = set(case["known_irrelevant_history_turn_ids"])

    contradiction_result, contradiction_matches = _contradiction_awareness_result(
        case, decisions, sources
    )
    decision_loss, source_loss = _lineage_loss(condition_record)
    token_estimate = condition_record["token_estimate"]

    return {
        "session_id": case["session_id"],
        "task_id": case["task_id"],
        "case_id": case["id"],
        "case_family": case["case_family"],
        "condition": condition_record["condition"],
        "prototype_version": PROTOTYPE_VERSION,
        "seed": seed,
        "corpus_manifest_hash": corpus_manifest_hash,
        "case_hash": compute_case_hash(case),
        "selected_turn_ids": condition_record["selected_turn_ids"],
        "selected_episode_ids": condition_record["selected_episode_ids"],
        "selected_parent_engram_ids": condition_record["selected_parent_engram_ids"],
        "selected_source_ids": condition_record["selected_source_ids"],
        "token_estimate": token_estimate,
        "required_prior_decision_recall": _recall(case["required_prior_decision_ids"], decisions),
        "decision_artifact_retention": _recall(case["required_prior_decision_ids"], decisions),
        "required_source_recall": _recall(case["required_source_ids"], sources),
        "contradiction_awareness_result": contradiction_result,
        "contradiction_awareness_matches_expected": contradiction_matches,
        "known_irrelevant_history_selected": len(selected_turn_ids & known_irrelevant),
        "known_irrelevant_history_available": len(known_irrelevant),
        "provenance_loss_count": decision_loss + source_loss,
        "decision_lineage_loss_count": decision_loss,
        "source_lineage_loss_count": source_loss,
        "synthetic_context_label_coverage": _synthetic_context_label_coverage(condition_record),
        "synthetic_context_labels": condition_record.get("labels") or [],
        "selection_rationale": condition_record.get("selection_rationale") or [],
        "context_budget_insufficient": condition_record.get(
            "context_budget_insufficient", False
        ),
        "omitted_required_artifact_types": condition_record.get(
            "omitted_required_artifact_types", []
        ),
        "selection_abstention_reason": condition_record.get(
            "selection_abstention_reason"
        ),
        "context_budget": condition_record.get("context_budget"),
        "prompt_token_reduction": (
            1.0 - (token_estimate / full_history_token_estimate) if full_history_token_estimate else 0.0
        ),
    }


def run_replay(
    corpus_data: Dict,
    corpus_manifest_hash: str,
    seed: int = 0,
    token_budget_override: Optional[int] = None,
    include_s1: bool = False,
) -> List[Dict]:
    """Run conditions A, B, C against every case in the corpus. Pure
    function: no file I/O, no model calls, no mutation of input data.

    By default, B and C are each capped at the case's own
    `expected_context_budget` (the primary, required Phase 3 replay). Pass
    `token_budget_override` to cap B and C at a fixed budget instead - used
    only for the harness self-check stress pass, which exists to confirm
    the truncation/selection machinery actually differentiates conditions
    when given a budget tight enough to bind (see
    docs/session_context_assembler_phase_3_notes.md for why r0's configured
    budgets do not bind against this corpus's short fixture turns)."""
    records: List[Dict] = []
    for case in corpus_data["cases"]:
        token_budget = (
            token_budget_override if token_budget_override is not None else case["expected_context_budget"]
        )

        a = run_condition_a(case)
        full_history_token_estimate = a["token_estimate"]
        records.append(build_case_record(case, a, full_history_token_estimate, corpus_manifest_hash, seed))

        b = run_condition_b(case, token_budget)
        records.append(build_case_record(case, b, full_history_token_estimate, corpus_manifest_hash, seed))

        c = run_condition_c(case, corpus_manifest_hash, seed, token_budget)
        records.append(build_case_record(case, c, full_history_token_estimate, corpus_manifest_hash, seed))

        if include_s1:
            c1 = run_condition_c1(case, corpus_manifest_hash, seed, token_budget)
            records.append(
                build_case_record(
                    case, c1, full_history_token_estimate, corpus_manifest_hash, seed
                )
            )

    return records


def compute_aggregate_gates(
    records: List[Dict], condition: str = "C_governed_episode_selected"
) -> Dict:
    """Phase 3 required gates, scoped to condition C (the governed
    mechanism under evaluation). unauthorized_memory_write_count and
    unauthorized_governance_mutation_count are 0 by construction - this
    module has no import path to mnemos/service/mnemos_sdk and performs no
    I/O, which test_session_context_assembler_replay.py verifies both
    structurally (AST import scan) and behaviorally (patched write/delete
    calls raise if invoked during a full run)."""
    c_records = [r for r in records if r["condition"] == condition]

    total_sources = sum(len(r["selected_source_ids"]) for r in c_records)
    total_decisions = sum(len(r["selected_parent_engram_ids"]) for r in c_records)
    source_loss = sum(r["source_lineage_loss_count"] for r in c_records)
    decision_loss = sum(r["decision_lineage_loss_count"] for r in c_records)
    provenance_loss_count = sum(r["provenance_loss_count"] for r in c_records)
    label_coverage = (
        sum(r["synthetic_context_label_coverage"] for r in c_records) / len(c_records) if c_records else None
    )

    gates = {
        "source_id_preservation_rate": {
            "value": 1.0 if total_sources == 0 else round(1.0 - (source_loss / total_sources), 6),
            "required": 1.0,
        },
        "parent_engram_lineage_preservation_rate": {
            "value": 1.0 if total_decisions == 0 else round(1.0 - (decision_loss / total_decisions), 6),
            "required": 1.0,
        },
        "provenance_loss_count": {"value": provenance_loss_count, "required": 0},
        "synthetic_context_label_coverage": {"value": label_coverage, "required": 1.0},
        "unauthorized_memory_write_count": {"value": 0, "required": 0},
        "unauthorized_governance_mutation_count": {"value": 0, "required": 0},
    }
    for gate in gates.values():
        gate["passed"] = gate["value"] == gate["required"]

    return {
        "gates": gates,
        "all_required_gates_passed": all(g["passed"] for g in gates.values()),
        "condition_c_case_count": len(c_records),
        "evaluated_condition": condition,
    }


def compute_condition_comparison(records: List[Dict]) -> Dict:
    """Descriptive (non-gated) comparison stats across A/B/C. These are
    replay observations, not human-value, non-inferiority, or production-
    readiness claims - see docs/session_context_assembler_phase_3_notes.md."""
    by_condition: Dict[str, List[Dict]] = {}
    for r in records:
        by_condition.setdefault(r["condition"], []).append(r)

    comparison = {}
    for condition, rows in by_condition.items():
        decision_recalls = [r["required_prior_decision_recall"] for r in rows if r["required_prior_decision_recall"] is not None]
        decision_retentions = [r["decision_artifact_retention"] for r in rows if r["decision_artifact_retention"] is not None]
        source_recalls = [r["required_source_recall"] for r in rows if r["required_source_recall"] is not None]
        irrelevant_selected = sum(r["known_irrelevant_history_selected"] for r in rows)
        irrelevant_available = sum(r["known_irrelevant_history_available"] for r in rows)
        token_reductions = [r["prompt_token_reduction"] for r in rows]
        contradiction_rows = [
            r for r in rows
            if r["contradiction_awareness_matches_expected"] is not None
        ]

        comparison[condition] = {
            "case_count": len(rows),
            "mean_required_prior_decision_recall": (
                sum(decision_recalls) / len(decision_recalls) if decision_recalls else None
            ),
            "mean_decision_artifact_retention": (
                sum(decision_retentions) / len(decision_retentions)
                if decision_retentions
                else None
            ),
            "mean_required_source_recall": sum(source_recalls) / len(source_recalls) if source_recalls else None,
            "irrelevant_history_selection_rate": (
                irrelevant_selected / irrelevant_available if irrelevant_available else None
            ),
            "mean_prompt_token_reduction": sum(token_reductions) / len(token_reductions) if token_reductions else None,
            "mean_token_estimate": sum(r["token_estimate"] for r in rows) / len(rows) if rows else None,
            "contradiction_awareness_rate": (
                sum(
                    1 for r in contradiction_rows
                    if r["contradiction_awareness_matches_expected"]
                ) / len(contradiction_rows)
                if contradiction_rows
                else None
            ),
        }
    return comparison


def compute_s1_advancement_gates(records: List[Dict]) -> Dict:
    """Evaluate the Phase 4R utility and safety requirements for C1."""
    comparison = compute_condition_comparison(records)
    baseline = comparison["B_sliding_window"]
    s1 = comparison["C1_selector_s1_mandatory_preservation"]
    safety = compute_aggregate_gates(
        records, condition="C1_selector_s1_mandatory_preservation"
    )
    c1_rows = [
        row for row in records
        if row["condition"] == "C1_selector_s1_mandatory_preservation"
    ]
    silent_omissions = sum(
        1 for row in c1_rows
        if (
            row["decision_artifact_retention"] not in (None, 1.0)
            or row["required_source_recall"] not in (None, 1.0)
            or row["contradiction_awareness_matches_expected"] is False
        )
        and not row["context_budget_insufficient"]
    )
    gates = {
        "decision_artifact_retention": {
            "value": s1["mean_decision_artifact_retention"],
            "required": f">= B ({baseline['mean_decision_artifact_retention']:.6f})",
            "passed": s1["mean_decision_artifact_retention"]
            >= baseline["mean_decision_artifact_retention"],
        },
        "required_source_recall": {
            "value": s1["mean_required_source_recall"],
            "required": f">= B ({baseline['mean_required_source_recall']:.6f})",
            "passed": s1["mean_required_source_recall"]
            >= baseline["mean_required_source_recall"],
        },
        "contradiction_awareness": {
            "value": s1["contradiction_awareness_rate"],
            "required": f">= B ({baseline['contradiction_awareness_rate']:.6f})",
            "passed": s1["contradiction_awareness_rate"]
            >= baseline["contradiction_awareness_rate"],
        },
        "prompt_token_reduction": {
            "value": s1["mean_prompt_token_reduction"],
            "required": ">= 0.25",
            "passed": s1["mean_prompt_token_reduction"] >= 0.25,
        },
        "provenance_loss_count": safety["gates"]["provenance_loss_count"],
        "synthetic_context_label_coverage": safety["gates"][
            "synthetic_context_label_coverage"
        ],
        "budget_compliance": {
            "value": sum(
                1 for row in c1_rows
                if row["token_estimate"] <= row["context_budget"]
            ),
            "required": len(c1_rows),
            "passed": all(
                row["token_estimate"] <= row["context_budget"] for row in c1_rows
            ),
        },
        "silent_required_artifact_omission_count": {
            "value": silent_omissions,
            "required": 0,
            "passed": silent_omissions == 0,
        },
    }
    return {
        "gates": gates,
        "safety_gates": safety,
        "all_advancement_requirements_passed": (
            all(gate["passed"] for gate in gates.values())
            and safety["all_required_gates_passed"]
        ),
    }
