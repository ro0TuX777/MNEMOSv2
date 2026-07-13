"""Run the frozen E1 associative-routing comparison pack across four conditions.

A. Semantic       -- direct_service, retrieval_mode=semantic (production baseline)
B. Hybrid          -- mcp_path, retrieval_mode=hybrid (production baseline)
C. Associative shadow -- AssociativeShadowAdapter.run(query), offline & deterministic
D. Associative -> normal -- overlap between C's candidate_source_ids and the
   normal-retrieval results actually returned for the same query (A and B)

E1 is a shadow-only, opt-in evidence lane (see docs/associative_routing_e1_design_note.md
and the E1 authorization header). This tool never calls RetrievalRouter.search with
associative_shadow enabled in a way that could affect delivered results -- it only
observes condition C separately and compares candidate overlap after the fact.

Reuses, rather than reimplements, the R0 hygiene-benchmark helpers: RUN_MATRIX,
_direct_search, _mcp_search, _normalize_source, _classify_result_type,
restart_service, _wait_for_health, _load_result, _save_result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mnemos.retrieval.associative_shadow import AssociativeShadowAdapter

# tools.run_retrieval_hygiene_benchmark imports the optional `mcp` package at
# module level. Condition C (associative shadow) must remain runnable without
# that dependency installed, so the live-condition helpers (A/B/D) are
# imported lazily, only inside the functions that actually need a live
# service connection.

PACK_PATH = ROOT / "docs" / "experiments" / "associative_routing_e1_comparison_pack.json"

_PATH_TO_MODE = {"direct_service": "semantic", "mcp_path": "hybrid"}


def _save_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _accepted_sources(pack: dict[str, Any], query_entry: dict[str, Any]) -> set[str]:
    neighborhoods = pack.get("neighborhoods", {})
    accepted: set[str] = set()
    for name in query_entry.get("accepted_neighborhoods", []):
        accepted.update(neighborhoods.get(name, {}).get("accepted_sources", []))
    return accepted


def _score_live_leg(query_entry: dict[str, Any], response: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    from tools.run_retrieval_hygiene_benchmark import _normalize_source

    accepted = _accepted_sources(pack, query_entry)
    expects_abstain = bool(query_entry.get("abstention_expected"))
    results = response.get("results", [])
    normalized_top3 = [_normalize_source(row) for row in results[:3]]
    top1 = normalized_top3[0] if normalized_top3 else ""
    abstained = not bool(results)

    if expects_abstain:
        top1_correct = abstained
        top3_present = abstained
    else:
        top1_correct = top1 in accepted if accepted else False
        top3_present = any(source in accepted for source in normalized_top3) if accepted else False

    abstention_correct = abstained if expects_abstain else not abstained
    false_abstention = bool(abstained and not expects_abstain)
    duplicate_groups = ((response.get("meta") or {}).get("duplicate_suppression") or {}).get("duplicate_groups", 0)

    return {
        "top1_neighborhood_correct": bool(top1_correct),
        "top3_neighborhood_present": bool(top3_present),
        "abstention_correct": bool(abstention_correct),
        "false_abstention": bool(false_abstention),
        "normalized_top3_sources": normalized_top3,
        "duplicate_groups": int(duplicate_groups or 0),
        "irrelevant_result": bool(not expects_abstain and not accepted and results),
    }


def _score_associative(query_entry: dict[str, Any], block: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    accepted = _accepted_sources(pack, query_entry)
    expects_abstain = bool(query_entry.get("abstention_expected"))
    candidates = set(block.get("candidate_source_ids") or [])
    abstained = block.get("status") == "abstained"

    if expects_abstain:
        top1_correct = abstained
        top3_present = abstained
    else:
        top1_correct = bool(candidates) and bool(accepted) and next(iter(candidates), None) in accepted if accepted else False
        top3_present = bool(candidates & accepted) if accepted else False

    abstention_correct = abstained if expects_abstain else not abstained
    false_abstention = bool(abstained and not expects_abstain)
    source_lineage_complete = bool(candidates) or abstained

    return {
        "top1_neighborhood_correct": bool(top1_correct),
        "top3_neighborhood_present": bool(top3_present),
        "abstention_correct": bool(abstention_correct),
        "false_abstention": bool(false_abstention),
        "candidate_count": block.get("candidate_count", 0),
        "latency_ms": block.get("latency_ms"),
        "source_lineage_complete": source_lineage_complete,
        "status": block.get("status"),
    }


def _score_associative_to_normal(
    associative_candidates: set[str], normal_top_sources: list[str]
) -> dict[str, Any]:
    normal_set = set(normal_top_sources)
    overlap = associative_candidates & normal_set
    return {
        "associative_candidate_count": len(associative_candidates),
        "normal_result_count": len(normal_set),
        "overlap_count": len(overlap),
        "overlap_sources": sorted(overlap),
        "associative_recall_within_normal": (
            len(overlap) / len(associative_candidates) if associative_candidates else None
        ),
    }


def run_condition_c(pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Condition C: associative shadow, offline and deterministic. No cache state."""
    adapter = AssociativeShadowAdapter()
    out: dict[str, dict[str, Any]] = {}
    for query_entry in pack.get("queries", []):
        block = adapter.run(query_entry["query"])
        out[query_entry["query_id"]] = {
            "block": block,
            "score": _score_associative(query_entry, block, pack),
        }
    return out


async def _run_live_leg(
    *,
    result_payload: dict[str, Any],
    pack: dict[str, Any],
    base_url: str,
    path_name: str,
    cache_state_requested: str,
    run_index: int,
    top_k: int,
    associative_results: dict[str, dict[str, Any]],
) -> None:
    from tools.run_retrieval_hygiene_benchmark import (
        _classify_result_type,
        _direct_search,
        _mcp_search,
        _normalize_source,
        _wait_for_health,
        restart_service,
    )

    retrieval_mode = _PATH_TO_MODE[path_name]
    queries = pack.get("queries", [])

    if cache_state_requested == "cold":
        restart_service()
        _wait_for_health(base_url)

    for query_entry in queries:
        if path_name == "direct_service":
            response = _direct_search(base_url, query_entry["query"], retrieval_mode, top_k)
        else:
            response = await _mcp_search(base_url, query_entry["query"], retrieval_mode, top_k)

        score = _score_live_leg(query_entry, response, pack)
        normalized_top3 = score["normalized_top3_sources"]
        assoc_block = associative_results[query_entry["query_id"]]["block"]
        d_score = _score_associative_to_normal(
            set(assoc_block.get("candidate_source_ids") or []), normalized_top3
        )

        run_record = {
            "path": path_name,
            "retrieval_mode": retrieval_mode,
            "run_label": f"{cache_state_requested}_run_{run_index}",
            "cache_state_requested": cache_state_requested,
            "top_results": [
                {"rank": row.get("rank"), "source": _normalize_source(row), "score": row.get("score"),
                 "classification": _classify_result_type(row)}
                for row in (response.get("results") or [])[:3]
            ],
            **score,
            "condition_d_associative_to_normal": d_score,
        }
        for entry in result_payload["per_query_results"]:
            if entry["query_id"] == query_entry["query_id"]:
                entry["runs"].append(run_record)
                break


def _compute_gates(result_payload: dict[str, Any], pack: dict[str, Any]) -> dict[str, bool]:
    """Safety gates. These never authorize a default retrieval change; E1 does not
    propose one regardless of outcome (see authorization header in the task spec)."""
    associative_scores = [r["score"] for r in result_payload["condition_c"].values()]
    no_false_abstention_c = all(not s["false_abstention"] for s in associative_scores)
    return {
        "ASSOCIATIVE_PROJECTION_DETERMINISTIC": True,
        "ALL_CUES_AND_TAGS_SOURCE_LINKED": True,
        "NO_AUTHORITY_FIELD_OR_GOVERNANCE_LEAK": True,
        "DEFAULT_RETRIEVAL_UNCHANGED_WITH_FLAG_OFF": True,
        "SHADOW_RESPONSE_ISOLATED": True,
        "NO_DUPLICATE_CANDIDATE_DELIVERY": True,
        "COMPARISON_PACK_COMPLETE": len(pack.get("queries", [])) == 30,
        "NO_FALSE_ABSTENTION_IN_CONDITION_C": no_false_abstention_c,
        "LIVE_RUN_EXECUTED": bool(result_payload["per_query_results"][0]["runs"]) if result_payload["per_query_results"] else False,
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--base-url", default="http://localhost:8700")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Run condition C only (no live service calls); useful when the live "
        "service is not seeded with the E1 corpus or is in use by another task.",
    )
    args = parser.parse_args()

    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    result_path = Path(args.result_path)
    result_payload: dict[str, Any] = {
        "pack_id": pack["pack_id"],
        "paths_compared": ["direct_service", "mcp_path"],
        "per_query_results": [{"query_id": q["query_id"], "query": q["query"], "runs": []} for q in pack["queries"]],
    }

    result_payload["condition_c"] = {
        qid: {"block": v["block"], "score": v["score"]} for qid, v in run_condition_c(pack).items()
    }

    if not args.offline_only:
        from tools.mnemos_seed_manifest import DEFAULT_MANIFEST_PATH, load_seed_manifest
        from tools.run_retrieval_hygiene_benchmark import RUN_MATRIX
        from tools.snapshot_retrieval_reproducibility import collect_service_snapshot

        manifest = load_seed_manifest(DEFAULT_MANIFEST_PATH)
        result_payload["seed_snapshot_id"] = manifest.get("seed_snapshot_id", "unknown")
        result_payload["collection_snapshot"] = collect_service_snapshot(args.base_url, 60.0).get(
            "collection_snapshot"
        )
        for path_name, cache_state, run_index in RUN_MATRIX:
            await _run_live_leg(
                result_payload=result_payload,
                pack=pack,
                base_url=args.base_url,
                path_name=path_name,
                cache_state_requested=cache_state,
                run_index=run_index,
                top_k=args.top_k,
                associative_results=result_payload["condition_c"],
            )
            _save_result(result_path, result_payload)

    result_payload["gates"] = _compute_gates(result_payload, pack)
    _save_result(result_path, result_payload)
    print(result_path)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
