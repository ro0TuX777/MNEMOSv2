"""Run Evidence Admission and Budgeting R0 comparison packs.

Two execution modes are intentionally non-interchangeable:

* direct_runtime: development and diagnostic mode. It may invoke repository
  runtime objects directly and is not sufficient for a deployed-service or
  HTTP-path claim.
* http_service: integration and formal evaluation mode. It calls a configured
  MNEMOS HTTP endpoint and fails closed when service revision identity cannot
  be established.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mnemos.retrieval.evidence_admission.config import EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV
from mnemos.retrieval.evidence_admission.telemetry import redact_for_telemetry

EXECUTION_MODES = ("direct_runtime", "http_service")
REQUEST_FLAG_STATES = ("absent", "false", "true")
GLOBAL_GATE_STATES = ("disabled", "enabled", "external")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, required=True)
    parser.add_argument("--pack-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--service-base-url")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--request-flag-state", choices=REQUEST_FLAG_STATES, default="true")
    parser.add_argument("--global-gate-state", choices=GLOBAL_GATE_STATES, default="external")
    return parser.parse_args(argv)


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _get_json(url: str, timeout_s: float) -> Dict[str, Any]:
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def _post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return data


def _revision_identity_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    revision = payload.get("service_revision")
    if not isinstance(revision, dict):
        return None
    git_revision = str(revision.get("git_revision") or "").strip()
    image_id = str(revision.get("image_id") or "").strip()
    if git_revision:
        suffix = ":dirty" if revision.get("git_dirty") else ""
        return f"git:{git_revision}{suffix}"
    if image_id:
        return f"image:{image_id}"
    return None


def establish_http_service_identity(base_url: str, timeout_s: float) -> Dict[str, Any]:
    base = base_url.rstrip("/")
    observations: Dict[str, Any] = {}
    try:
        observations["health"] = _get_json(f"{base}/health", timeout_s)
        observations["capabilities"] = _get_json(f"{base}/v1/mnemos/capabilities", timeout_s)
        observations["stats"] = _get_json(f"{base}/v1/mnemos/stats", timeout_s)
    except Exception as exc:
        return {
            "verified": False,
            "identity": None,
            "status_code": "SERVICE_REVISION_UNVERIFIED",
            "formal_claim_permitted": False,
            "error": str(exc),
            "observations": observations,
        }

    for key in ("health", "capabilities", "stats"):
        identity = _revision_identity_from_payload(observations.get(key) or {})
        if identity:
            return {
                "verified": True,
                "identity": identity,
                "status_code": "SERVICE_REVISION_VERIFIED",
                "formal_claim_permitted": True,
                "observations": observations,
            }

    return {
        "verified": False,
        "identity": None,
        "status_code": "SERVICE_REVISION_UNVERIFIED",
        "formal_claim_permitted": False,
        "observations": observations,
    }


def collection_snapshot_for_direct_runtime() -> str:
    try:
        from service.app import _ensure_runtime, _runtime

        err = _ensure_runtime()
        if err:
            return "direct_runtime:unavailable"
        return _runtime._collection_snapshot()
    except Exception:
        return "direct_runtime:unknown"


def collection_snapshot_from_http_identity(identity: Dict[str, Any]) -> str:
    retrieval = (
        (((identity.get("observations") or {}).get("stats") or {}).get("stats") or {})
        .get("retrieval", {})
    )
    qdrant = (retrieval.get("tiers") or {}).get("qdrant") or {}
    collection = str(qdrant.get("collection") or "unknown")
    document_count_value = qdrant.get("document_count")
    document_count = "unknown" if document_count_value is None else str(document_count_value)
    if collection == "unknown" and document_count == "unknown":
        return "http_service:unknown"
    return f"{collection}:{document_count}"


def _request_payload(
    query: str,
    top_k: int,
    request_flag_state: str,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "retrieval_mode": "semantic",
        "fusion_policy": "balanced",
        "explain": False,
    }
    if request_flag_state == "true":
        payload["evidence_admission_shadow"] = True
    elif request_flag_state == "false":
        payload["evidence_admission_shadow"] = False
    if filters:
        payload["filters"] = filters
    return payload


def _normalize_top_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in response.get("results") or []:
        engram = row.get("engram") or {}
        metadata = engram.get("metadata") or {}
        rows.append(
            {
                "rank": row.get("rank"),
                "score": row.get("score"),
                "engram_id": engram.get("id"),
                "source": metadata.get("source_uri") or engram.get("source"),
            }
        )
    return rows


def _split_shadow_block(shadow: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not shadow:
        return None, None
    pre_keys = {
        "status",
        "recommended_route",
        "candidate_budget",
        "context_token_budget",
        "expansion_budget",
        "latency_budget_ms",
        "stop_condition",
        "reason_codes",
        "input_snapshot",
        "latency_ms",
        "non_authoritative",
    }
    post_keys = {"sufficiency", "sufficiency_reason_codes", "non_authoritative"}
    return (
        {key: shadow.get(key) for key in pre_keys if key in shadow},
        {key: shadow.get(key) for key in post_keys if key in shadow},
    )


def build_query_record(query_id: str, query: str, response: Dict[str, Any]) -> Dict[str, Any]:
    meta = response.get("meta") or {}
    shadow = meta.get("evidence_admission_shadow") or {}
    pre, post = _split_shadow_block(shadow)
    query_redacted = redact_for_telemetry({"query": query})["query"]
    return {
        "query_id": query_id,
        "query": query_redacted,
        "normal_retrieval": {
            "result_count": len(response.get("results") or []),
            "top_results": _normalize_top_results(response)[:5],
            "meta": {
                "retrieval_mode": meta.get("retrieval_mode"),
                "fusion_policy": meta.get("fusion_policy"),
                "retrieval_fingerprint": meta.get("retrieval_fingerprint"),
                "abstention_guard": meta.get("abstention_guard"),
                "governance_summary": meta.get("governance_summary"),
            },
        },
        "pre_retrieval_recommendation": pre,
        "post_retrieval_sufficiency": post,
        "raw_shadow_block_redacted": redact_for_telemetry(dict(shadow)),
    }


def run_single_direct_runtime_query(
    query: str,
    top_k: int,
    request_flag_state: str,
    global_gate_state: str,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from service.app import _ensure_runtime, _runtime

    old_gate = os.environ.get(EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV)
    if global_gate_state in {"enabled", "disabled"}:
        os.environ[EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV] = "true" if global_gate_state == "enabled" else "false"
    try:
        err = _ensure_runtime()
        if err:
            return {"error": err}
        response = _runtime.search_documents(
            query=query,
            top_k=top_k,
            tiers=None,
            filters=filters,
            retrieval_mode="semantic",
            fusion_policy="balanced",
            explain=False,
            evidence_admission_shadow=(request_flag_state == "true"),
        )
        if request_flag_state == "absent":
            response.get("meta", {}).pop("evidence_admission_shadow", None)
        return build_query_record("direct", query, response)
    finally:
        if global_gate_state in {"enabled", "disabled"}:
            if old_gate is None:
                os.environ.pop(EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV, None)
            else:
                os.environ[EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV] = old_gate


def run_single_http_query(
    base_url: str,
    query_id: str,
    query: str,
    top_k: int,
    request_flag_state: str,
    timeout_s: float,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = _post_json(
        f"{base_url.rstrip()}/v1/mnemos/search",
        _request_payload(query, top_k, request_flag_state, filters),
        timeout_s,
    )
    return build_query_record(query_id, query, response)


def _write_result(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    pack_path = Path(args.pack_path)
    result_path = Path(args.result_path)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    runner_commit = git_head()
    service_base_url = args.service_base_url.rstrip("/") if args.service_base_url else None
    identity = None
    formal_claim_permitted = False

    if args.execution_mode == "http_service":
        if not service_base_url:
            raise SystemExit("--service-base-url is required for http_service mode")
        identity = establish_http_service_identity(service_base_url, args.timeout_s)
        service_identity = identity["identity"] if identity.get("verified") else "SERVICE_REVISION_UNVERIFIED"
        formal_claim_permitted = bool(identity.get("formal_claim_permitted"))
        collection_snapshot = collection_snapshot_from_http_identity(identity)
    else:
        service_identity = f"direct_runtime:{runner_commit}"
        collection_snapshot = collection_snapshot_for_direct_runtime()

    payload: Dict[str, Any] = {
        "pack_id": pack.get("pack_id") or pack.get("verification_pack_id") or pack.get("benchmark_id"),
        "pack_type": pack.get("pack_type", "unspecified"),
        "run_manifest": {
            "execution_mode": args.execution_mode,
            "service_base_url": service_base_url,
            "service_revision_or_image_identity": service_identity,
            "runner_commit": runner_commit,
            "collection_or_corpus_snapshot": collection_snapshot,
            "request_flag_state": args.request_flag_state,
            "global_gate_state": args.global_gate_state,
        },
        "mode_boundaries": {
            "direct_runtime": "development and diagnostic mode; not sufficient for deployed-service or HTTP-path claims",
            "http_service": "integration and formal evaluation mode; required for service-level shadow integration claims",
        },
        "formal_claim_permitted": formal_claim_permitted if args.execution_mode == "http_service" else False,
        "claim_status": (
            "FORMAL_HTTP_SERVICE_EVALUATION_COMPLETE"
            if args.execution_mode == "http_service" and formal_claim_permitted
            else "DIRECT_RUNTIME_ONLY_EVIDENCE"
            if args.execution_mode == "direct_runtime"
            else "HTTP_SERVICE_EVALUATION_BLOCKED"
        ),
        "http_service_identity": identity,
        "per_query_results": [],
        "aggregation_policy": "Results from direct_runtime and http_service runs must not be aggregated into one metric.",
    }

    if args.execution_mode == "http_service" and not formal_claim_permitted:
        _write_result(result_path, payload)
        return 2

    for query_entry in pack.get("queries", []):
        query_id = str(query_entry["query_id"])
        query = str(query_entry["query"])
        request_flag_state = str(query_entry.get("request_flag_state") or args.request_flag_state)
        global_gate_state = (
            str(query_entry.get("global_gate_state") or "external")
            if args.global_gate_state == "external"
            else args.global_gate_state
        )
        filters = query_entry.get("filters") if isinstance(query_entry.get("filters"), dict) else None
        if request_flag_state not in REQUEST_FLAG_STATES:
            raise SystemExit(f"Invalid request_flag_state for {query_id}: {request_flag_state}")
        if global_gate_state not in GLOBAL_GATE_STATES:
            raise SystemExit(f"Invalid global_gate_state for {query_id}: {global_gate_state}")
        if args.execution_mode == "direct_runtime":
            record = run_single_direct_runtime_query(
                query,
                args.top_k,
                request_flag_state,
                global_gate_state,
                filters,
            )
            record["query_id"] = query_id
        else:
            record = run_single_http_query(
                service_base_url or "",
                query_id,
                query,
                args.top_k,
                request_flag_state,
                args.timeout_s,
                filters,
            )
        record["request_flag_state"] = request_flag_state
        record["global_gate_state"] = global_gate_state
        payload["per_query_results"].append(record)

    _write_result(result_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
