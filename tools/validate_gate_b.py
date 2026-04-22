"""
Validate Gate B reference-fidelity input files.

Usage:
  python tools/validate_gate_b.py
  python tools/validate_gate_b.py --queries benchmarks/truthsets/gate_b_sanity_queries.json --pool benchmarks/truthsets/gate_b_candidate_pool.json --labels benchmarks/truthsets/gate_b_labels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


ALLOWED_REGIME = {"semantic"}
ALLOWED_LABELS = {0, 1, 2}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_non_empty_string(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _validate_queries(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        return ["queries file must contain non-empty 'queries' array."]

    seen: Set[str] = set()
    for i, q in enumerate(queries):
        p = f"queries[{i}]"
        if not isinstance(q, dict):
            errors.append(f"{p} must be an object.")
            continue
        qid = q.get("query_id")
        if not _is_non_empty_string(qid):
            errors.append(f"{p}.query_id must be a non-empty string.")
        elif qid in seen:
            errors.append(f"Duplicate query_id in queries file: '{qid}'.")
        else:
            seen.add(qid)
        if not _is_non_empty_string(q.get("query_text")):
            errors.append(f"{p}.query_text must be a non-empty string.")
        if q.get("regime") not in ALLOWED_REGIME:
            errors.append(f"{p}.regime must be 'semantic'.")
    return errors


def _validate_pool(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    pools = data.get("pools")
    if not isinstance(pools, list) or not pools:
        return ["candidate pool file must contain non-empty 'pools' array."]

    seen: Set[str] = set()
    for i, p in enumerate(pools):
        prefix = f"pools[{i}]"
        if not isinstance(p, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        qid = p.get("query_id")
        if not _is_non_empty_string(qid):
            errors.append(f"{prefix}.query_id must be a non-empty string.")
        elif qid in seen:
            errors.append(f"Duplicate query_id in candidate pools: '{qid}'.")
        else:
            seen.add(qid)
        ids = p.get("candidate_ids")
        if not isinstance(ids, list) or not ids:
            errors.append(f"{prefix}.candidate_ids must be a non-empty array.")
            continue
        dedup = set()
        for j, did in enumerate(ids):
            if not _is_non_empty_string(did):
                errors.append(f"{prefix}.candidate_ids[{j}] must be a non-empty string.")
            elif did in dedup:
                errors.append(f"{prefix}.candidate_ids has duplicate doc_id '{did}'.")
            else:
                dedup.add(did)
    return errors


def _validate_labels(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    labels = data.get("labels")
    if not isinstance(labels, list) or not labels:
        return ["labels file must contain non-empty 'labels' array."]

    seen: Set[str] = set()
    for i, row in enumerate(labels):
        prefix = f"labels[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        qid = row.get("query_id")
        if not _is_non_empty_string(qid):
            errors.append(f"{prefix}.query_id must be a non-empty string.")
        elif qid in seen:
            errors.append(f"Duplicate query_id in labels: '{qid}'.")
        else:
            seen.add(qid)
        graded = row.get("graded_labels")
        if not isinstance(graded, list) or not graded:
            errors.append(f"{prefix}.graded_labels must be a non-empty array.")
            continue
        doc_seen: Set[str] = set()
        for j, gl in enumerate(graded):
            gp = f"{prefix}.graded_labels[{j}]"
            if not isinstance(gl, dict):
                errors.append(f"{gp} must be an object.")
                continue
            did = gl.get("doc_id")
            if not _is_non_empty_string(did):
                errors.append(f"{gp}.doc_id must be a non-empty string.")
            elif did in doc_seen:
                errors.append(f"{prefix}.graded_labels has duplicate doc_id '{did}'.")
            else:
                doc_seen.add(did)
            rel = gl.get("relevance")
            if rel not in ALLOWED_LABELS:
                errors.append(f"{gp}.relevance must be one of {sorted(ALLOWED_LABELS)}.")
    return errors


def _query_ids_from_list(items: List[Dict[str, Any]], key: str) -> Set[str]:
    out: Set[str] = set()
    for item in items:
        if isinstance(item, dict) and _is_non_empty_string(item.get(key)):
            out.add(str(item[key]))
    return out


def validate_cross_file(queries: Dict[str, Any], pool: Dict[str, Any], labels: Dict[str, Any]) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    q_ids = _query_ids_from_list(queries.get("queries", []), "query_id")
    p_ids = _query_ids_from_list(pool.get("pools", []), "query_id")
    l_ids = _query_ids_from_list(labels.get("labels", []), "query_id")

    if q_ids != p_ids:
        missing_pool = sorted(q_ids - p_ids)
        extra_pool = sorted(p_ids - q_ids)
        if missing_pool:
            errors.append(f"Candidate pool missing query_ids: {missing_pool}")
        if extra_pool:
            errors.append(f"Candidate pool has unknown query_ids: {extra_pool}")

    if q_ids != l_ids:
        missing_labels = sorted(q_ids - l_ids)
        extra_labels = sorted(l_ids - q_ids)
        if missing_labels:
            errors.append(f"Labels missing query_ids: {missing_labels}")
        if extra_labels:
            errors.append(f"Labels has unknown query_ids: {extra_labels}")

    # Ensure label doc IDs are in candidate pools.
    pool_by_qid = {
        p["query_id"]: set(p.get("candidate_ids", []))
        for p in pool.get("pools", [])
        if isinstance(p, dict) and _is_non_empty_string(p.get("query_id"))
    }
    for row in labels.get("labels", []):
        if not isinstance(row, dict):
            continue
        qid = row.get("query_id")
        if not _is_non_empty_string(qid):
            continue
        candidates = pool_by_qid.get(qid, set())
        for gl in row.get("graded_labels", []):
            if not isinstance(gl, dict):
                continue
            did = gl.get("doc_id")
            if _is_non_empty_string(did) and did not in candidates:
                errors.append(
                    f"Label doc_id '{did}' for query '{qid}' is not in candidate pool."
                )

    # Content-strength checks for Gate B usefulness.
    relevance_one_seen = False
    for row in labels.get("labels", []):
        if not isinstance(row, dict):
            continue
        qid = row.get("query_id")
        if not _is_non_empty_string(qid):
            continue
        graded = row.get("graded_labels", [])
        positive = 0
        for gl in graded:
            if not isinstance(gl, dict):
                continue
            rel = gl.get("relevance")
            if rel == 1:
                relevance_one_seen = True
            if rel in (1, 2):
                positive += 1
        if positive == 0:
            errors.append(
                f"Query '{qid}' has no positive labels (relevance > 0). Gate B set is analytically weak."
            )
        elif positive < 2:
            warnings.append(
                f"Query '{qid}' has only {positive} positive label(s); consider >=2 for stronger diagnostics."
            )

    if not relevance_one_seen:
        warnings.append(
            "No relevance=1 labels found across the set; nDCG signal may be weak (effectively binary labels)."
        )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Gate B reference-fidelity input files.")
    parser.add_argument(
        "--queries",
        type=str,
        default="benchmarks/truthsets/gate_b_sanity_queries.json",
        help="Path to Gate B sanity queries JSON.",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="benchmarks/truthsets/gate_b_candidate_pool.json",
        help="Path to Gate B candidate pool JSON.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="benchmarks/truthsets/gate_b_labels.json",
        help="Path to Gate B labels JSON.",
    )
    args = parser.parse_args()

    paths = [Path(args.queries), Path(args.pool), Path(args.labels)]
    for p in paths:
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            return 2

    try:
        queries = _load_json(Path(args.queries))
        pool = _load_json(Path(args.pool))
        labels = _load_json(Path(args.labels))
    except Exception as e:
        print(f"[ERROR] Invalid JSON: {e}")
        return 2

    errors: List[str] = []
    errors.extend(_validate_queries(queries))
    errors.extend(_validate_pool(pool))
    errors.extend(_validate_labels(labels))
    cross_errors, warnings = validate_cross_file(queries, pool, labels)
    errors.extend(cross_errors)

    if errors:
        print(f"[FAIL] Gate B validation errors: {len(errors)}")
        for e in errors:
            print(f"  - {e}")
        if warnings:
            print(f"[WARN] Additional quality warnings: {len(warnings)}")
            for w in warnings:
                print(f"  - {w}")
        return 1

    n_queries = len(queries.get("queries", []))
    print(
        f"[OK] Gate B inputs valid: {args.queries}, {args.pool}, {args.labels} "
        f"({n_queries} queries)"
    )
    if warnings:
        print(f"[WARN] Gate B quality warnings: {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
