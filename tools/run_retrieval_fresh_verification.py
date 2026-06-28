"""Run the fresh verification pack across direct and MCP paths."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.mnemos_seed_manifest import DEFAULT_MANIFEST_PATH, load_seed_manifest
from tools.run_retrieval_hygiene_benchmark import (
    RUN_MATRIX,
    _direct_search,
    _mcp_search,
    _normalize_source,
    _classify_result_type,
    _load_result,
    _save_result,
    restart_service,
    _wait_for_health,
)
from tools.snapshot_retrieval_reproducibility import collect_service_snapshot


PACK_PATH = ROOT / "docs" / "experiments" / "retrieval_hygiene_r0_fresh_verification_pack.json"

BEHAVIOR_TO_NEIGHBORHOODS = {
    "abstain_non_gatemem": [],
    "retrieve_gatemem_frozen_baseline": ["gatemem_frozen_baseline"],
    "retrieve_gatemem_claim_boundary_or_frozen_baseline": [
        "gatemem_claim_boundary",
        "gatemem_frozen_baseline",
    ],
    "retrieve_gatemem_pause_blocker": ["gatemem_pause_blocker"],
    "retrieve_gatemem_g5_handoff": ["gatemem_g5_handoff"],
    "retrieve_gatemem_claim_boundary": ["gatemem_claim_boundary"],
}


def _accepted_sources(pack: dict[str, Any], behavior: str) -> set[str]:
    neighborhoods = pack.get("neighborhoods", {})
    accepted: set[str] = set()
    for neighborhood_name in BEHAVIOR_TO_NEIGHBORHOODS.get(behavior, []):
        accepted.update(neighborhoods.get(neighborhood_name, {}).get("accepted_sources", []))
    return accepted


def _score_query(query_entry: dict[str, Any], response: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    expected_behavior = str(query_entry["expected_behavior"])
    accepted = _accepted_sources(pack, expected_behavior)
    results = response.get("results", [])
    normalized_top3 = [_normalize_source(row) for row in results[:3]]
    top1 = normalized_top3[0] if normalized_top3 else ""
    abstained = not bool(results)
    expects_abstain = expected_behavior == "abstain_non_gatemem"
    must_not_abstain = query_entry.get("category") == "relevant_low_score_query_must_not_abstain"

    if expects_abstain:
        top1_correct = abstained
        top3_present = abstained
    else:
        top1_correct = top1 in accepted if accepted else False
        top3_present = any(source in accepted for source in normalized_top3) if accepted else False

    abstention_correct = abstained if expects_abstain else not abstained
    false_abstention = bool(abstained and must_not_abstain)
    duplicate_groups = ((response.get("meta") or {}).get("duplicate_suppression") or {}).get("duplicate_groups", 0)
    suppressed = ((response.get("meta") or {}).get("duplicate_suppression") or {}).get("suppressed_count", 0)
    fingerprint = (response.get("meta") or {}).get("retrieval_fingerprint") or {}
    fingerprint_truthful = bool(
        fingerprint.get("retrieval_profile") and fingerprint.get("configured_retrieval_profile")
    )

    return {
        "top1_behavior_correct": bool(top1_correct),
        "top3_behavior_present": bool(top3_present),
        "abstention_correct": bool(abstention_correct),
        "false_abstention": bool(false_abstention),
        "normalized_top3_sources": normalized_top3,
        "duplicate_groups": int(duplicate_groups or 0),
        "suppressed_candidates": int(suppressed or 0),
        "seeded_summary_count_top3": sum(
            1 for row in results[:3] if _classify_result_type(row) == "seeded_summary"
        ),
        "full_document_count_top3": sum(
            1 for row in results[:3] if _classify_result_type(row) == "full_document"
        ),
        "fingerprint_truthful": fingerprint_truthful,
    }


async def _run_single_matrix_leg(
    *,
    result_payload: dict[str, Any],
    pack: dict[str, Any],
    base_url: str,
    path_name: str,
    cache_state_requested: str,
    run_index: int,
    retrieval_mode: str,
    top_k: int,
) -> None:
    queries = pack.get("queries", [])

    if cache_state_requested == "cold":
        restart_service()
        _wait_for_health(base_url)

    for query_index, query_entry in enumerate(queries):
        if path_name == "direct_service":
            response = _direct_search(base_url, query_entry["query"], retrieval_mode, top_k)
        else:
            response = await _mcp_search(base_url, query_entry["query"], retrieval_mode, top_k)

        score = _score_query(query_entry, response, pack)
        meta = response.get("meta") or {}
        observed_cache_state = (
            "pre_cognitive_cache_hit"
            if meta.get("retrieval_mode") == "pre_cognitive_cache"
            else "no_cache_hit_observed"
        )
        run_record = {
            "path": path_name,
            "run_label": f"{cache_state_requested}_run_{run_index}",
            "cache_state_requested": cache_state_requested,
            "cache_state_observed": observed_cache_state,
            "seed_snapshot_id": result_payload["seed_snapshot_id"],
            "retrieval_fingerprint": meta.get("retrieval_fingerprint", {}),
            "top_results": [
                {
                    "rank": row.get("rank"),
                    "source": _normalize_source(row),
                    "raw_source": (row.get("engram", {}) or {}).get("source"),
                    "score": row.get("score"),
                    "classification": _classify_result_type(row),
                }
                for row in (response.get("results") or [])[:3]
            ],
            "top1_behavior_correct": score["top1_behavior_correct"],
            "top3_behavior_present": score["top3_behavior_present"],
            "abstention_correct": score["abstention_correct"],
            "false_abstention": score["false_abstention"],
            "fingerprint_truthful": score["fingerprint_truthful"],
            "duplicate_groups": score["duplicate_groups"],
            "suppressed_candidates": score["suppressed_candidates"],
            "seeded_summary_count_top3": score["seeded_summary_count_top3"],
            "full_document_count_top3": score["full_document_count_top3"],
            "latency_class": (
                "cold_start"
                if cache_state_requested == "cold" and query_index == 0
                else "steady_state"
            ),
            "errors": [],
            "abstained": not bool(response.get("results")),
            "raw_meta": {
                "retrieval_mode": meta.get("retrieval_mode"),
                "fusion_policy": meta.get("fusion_policy"),
                "duplicate_suppression": meta.get("duplicate_suppression"),
                "abstention_guard": meta.get("abstention_guard"),
            },
        }
        for entry in result_payload["per_query_results"]:
            if entry["query_id"] == query_entry["query_id"]:
                entry["runs"].append(run_record)
                break


def _compute_summary(result_payload: dict[str, Any]) -> None:
    per_path: dict[str, dict[str, Any]] = {}
    agreements = []
    cold_warm_pairs = []
    for path_name in result_payload["paths_compared"]:
        runs = []
        for query in result_payload["per_query_results"]:
            for run in query["runs"]:
                if run["path"] == path_name:
                    runs.append(run)
        if runs:
            top1 = sum(1 for run in runs if run["top1_behavior_correct"]) / len(runs)
            top3 = sum(1 for run in runs if run["top3_behavior_present"]) / len(runs)
            abstention_accuracy = sum(1 for run in runs if run["abstention_correct"]) / len(runs)
            false_abstention_rate = sum(1 for run in runs if run["false_abstention"]) / len(runs)
            dup_rate = sum(1 for run in runs if run["duplicate_groups"] > 0) / len(runs)
        else:
            top1 = top3 = abstention_accuracy = false_abstention_rate = dup_rate = 0.0
        per_path[path_name] = {
            "top1_accuracy": round(top1, 4),
            "top3_recall": round(top3, 4),
            "abstention_accuracy": round(abstention_accuracy, 4),
            "false_abstention_rate": round(false_abstention_rate, 4),
            "duplicate_result_rate": round(dup_rate, 4),
        }

    query_runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for query in result_payload["per_query_results"]:
        query_runs[query["query_id"]] = {}
        for run in query["runs"]:
            query_runs[query["query_id"]].setdefault(run["path"], []).append(run)

    for _, path_runs in query_runs.items():
        direct_runs = path_runs.get("direct_service", [])
        mcp_runs = path_runs.get("mcp_path", [])
        for direct, mcp in zip(direct_runs, mcp_runs):
            direct_has = bool(direct["top_results"])
            mcp_has = bool(mcp["top_results"])
            agreements.append(
                int(
                    (not direct_has and not mcp_has)
                    or (
                        direct_has
                        and mcp_has
                        and direct["top_results"][0]["source"] == mcp["top_results"][0]["source"]
                    )
                )
            )
        for path_name in ("direct_service", "mcp_path"):
            grouped = {}
            for run in path_runs.get(path_name, []):
                grouped.setdefault(run["cache_state_requested"], []).append(run)
            if grouped.get("cold") and grouped.get("warm"):
                for cold, warm in zip(grouped["cold"], grouped["warm"]):
                    cold_warm_pairs.append(
                        int(
                            cold["top_results"][:3] == warm["top_results"][:3]
                            and cold["abstained"] == warm["abstained"]
                            and cold["top1_behavior_correct"] == warm["top1_behavior_correct"]
                        )
                    )

    result_payload["summary"] = {
        "direct_service": per_path.get("direct_service", {}),
        "mcp_path": per_path.get("mcp_path", {}),
        "direct_mcp_top1_agreement": round(sum(agreements) / len(agreements), 4) if agreements else 0.0,
        "cold_warm_consistency": round(sum(cold_warm_pairs) / len(cold_warm_pairs), 4) if cold_warm_pairs else 0.0,
        "notes": result_payload.get("summary", {}).get("notes", []),
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--base-url", default="http://localhost:8700")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mcp-retrieval-mode", default="hybrid")
    parser.add_argument("--direct-retrieval-mode", default="semantic")
    args = parser.parse_args()

    result_path = Path(args.result_path)
    result_payload = _load_result(result_path)
    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    manifest = load_seed_manifest(DEFAULT_MANIFEST_PATH)
    result_payload["seed_snapshot_id"] = manifest.get("seed_snapshot_id", "unknown")
    service_snapshot = collect_service_snapshot(args.base_url, 60.0)
    result_payload["collection_snapshot"] = service_snapshot.get(
        "collection_snapshot",
        result_payload.get("collection_snapshot"),
    )
    result_payload["verification_pack"] = pack
    result_payload["seed_manifest"] = manifest
    _save_result(result_path, result_payload)

    for path_name, cache_state, run_index in RUN_MATRIX:
        retrieval_mode = args.direct_retrieval_mode if path_name == "direct_service" else args.mcp_retrieval_mode
        await _run_single_matrix_leg(
            result_payload=result_payload,
            pack=pack,
            base_url=args.base_url,
            path_name=path_name,
            cache_state_requested=cache_state,
            run_index=run_index,
            retrieval_mode=retrieval_mode,
            top_k=args.top_k,
        )
        _save_result(result_path, result_payload)

    _compute_summary(result_payload)
    _save_result(result_path, result_payload)
    print(result_path)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
