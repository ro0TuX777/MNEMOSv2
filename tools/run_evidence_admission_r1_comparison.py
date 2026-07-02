"""Evidence Admission and Budgeting R1 — four-condition DIRECT-RUNTIME DIAGNOSTIC.

This runner exercises the committed R1 enforcement code path end to end against
a freshly seeded copy of the frozen R1 corpus, over the four preregistered
conditions:

    1. normal        - normal retrieval baseline (no shadow, no enforce)
    2. r0_shadow      - R0 recommendation shadow only (shadow flag + shadow gate)
    3. r1_enforce     - R1 enforcement enabled (enforce flag + R1 global gate on)
    4. r1_gate_off    - R1 requested, global gate disabled (enforce flag, gate off)

IMPORTANT — this is a DIRECT-RUNTIME DIAGNOSTIC, not a formal evaluation:

* It calls repository runtime objects directly (service.app.MnemosRuntime), so
  per the R0/R1 mode boundaries it is development/diagnostic evidence only and
  MUST NOT be aggregated with any http_service formal evidence.
* It seeds the frozen corpus content faithfully (identical sources, identical
  word_window(120/20) chunking -> 684 units) BUT the runtime's natively
  supported embedder is nomic-ai/nomic-embed-text-v1.5, NOT the frozen
  manifest's declared BAAI/bge-base-en-v1.5 (the QdrantTier defaults non-nomic
  models to a 384-dim collection, which bge-base's 768-dim vectors cannot use).
  The embedder actually used is recorded in the run manifest. Because the
  embedder differs from the frozen profile, retrieval scores/orderings are NOT
  comparable to a formal bge-base run.

It does NOT modify R1 policy, thresholds, route mappings, corpus contents, the
source manifest, enforcement code, or development fixtures. It only reads the
frozen corpus + the committed formal pack, seeds a diagnostic collection, runs
the four conditions, and scores/records results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "docs" / "evidence_admission_and_budgeting_r1_corpus_manifest.json"
PACK = ROOT / "docs" / "experiments" / "evidence_admission_and_budgeting_r1_formal_pack.json"
RECEIPT = ROOT / "docs" / "evidence_admission_and_budgeting_r1_formal_pack_freeze_receipt.md"

FORBIDDEN_ROUTES = {
    "HYBRID_RETRIEVAL", "ASSOCIATIVE_EXPANSION_ELIGIBLE", "graph_hybrid_experimental",
    "derived_facts", "summary_inclusion", "governance_override",
}
DRIVER_ROLES = {"current_state_record", "dependency_blocker_record", "duplicate_or_near_duplicate_condition"}


# ── git / hashing helpers ────────────────────────────────────────────────
def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:
        return "unknown"


def git_last_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(path)],
                                       cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── faithful frozen chunking ─────────────────────────────────────────────
def chunk_text(text: str, max_words: int, overlap: int) -> List[str]:
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    if not words or words == [""]:
        return []
    out: List[str] = []
    step = max_words - overlap
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i + max_words]))
        if i + max_words >= len(words):
            break
        i += step
    return out


def build_seed_documents(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = manifest["chunking"]
    docs: List[Dict[str, Any]] = []
    for s in manifest["sources"]:
        path = s["path"]
        text = (ROOT / path).read_text(encoding="utf-8")
        stem = Path(path).name
        # ID must be keyed by the FULL source path, not the basename:
        # several corpus sources share a basename (e.g. four README.md files),
        # and basename-keyed IDs silently overwrite one another on upsert
        # (684 units -> 673 points). Seeding must be collision-free.
        path_key = path.replace("/", "__")
        for idx, chunk in enumerate(chunk_text(text, ch["max_words"], ch["overlap_words"])):
            meta = {
                "source_path": path,
                "filename": stem,
                "family": s["family"],
                "role": s["role"],
                "chunk_index": idx,
                "source_uri": path,
            }
            docs.append({
                "id": f"r1seed::{path_key}::chunk{idx:03d}",
                "content": chunk,
                "source": path,
                "metadata": meta,
            })
    return docs


# ── retrieval helpers ────────────────────────────────────────────────────
def top_sources(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for row in response.get("results") or []:
        eng = row.get("engram") or {}
        meta = eng.get("metadata") or {}
        rows.append({
            "rank": row.get("rank"),
            "score": row.get("score"),
            "source_path": meta.get("source_path"),
            "family": meta.get("family"),
            "role": meta.get("role"),
        })
    return rows


def run_condition_query(runtime, query: str, top_k: int, *, shadow: bool, enforce: bool) -> Dict[str, Any]:
    resp = runtime.search_documents(
        query=query, top_k=top_k, tiers=None, filters=None,
        retrieval_mode="semantic", fusion_policy="balanced", explain=False,
        evidence_admission_shadow=shadow, evidence_admission_enforce=enforce,
    )
    meta = resp.get("meta") or {}
    return {
        "retrieval_mode": meta.get("retrieval_mode"),
        "result_count": len(resp.get("results") or []),
        "top": top_sources(resp)[:top_k],
        "abstention_guard": meta.get("abstention_guard"),
        "evidence_admission_shadow": meta.get("evidence_admission_shadow"),
        "evidence_admission_r1_enforcement": meta.get("evidence_admission_r1_enforcement"),
        "cache": meta.get("cache") or meta.get("pre_cognitive"),
    }


# ── scoring ──────────────────────────────────────────────────────────────
def score_query(pack_q: Dict[str, Any], cond_result: Dict[str, Any]) -> Dict[str, Any]:
    neigh = pack_q["accepted_evidence_neighborhood"]
    drivers = {e["source_path"] for e in neigh if e["acceptance_note"].startswith("Admissible")}
    excluded = {e["source_path"] for e in neigh if not e["acceptance_note"].startswith("Admissible")}
    min_lin = pack_q["minimum_required_source_lineage"]
    want_family = min_lin["must_include_family"]
    want_role = min_lin["must_include_role"]

    retrieved = [r["source_path"] for r in cond_result["top"] if r.get("source_path")]
    retrieved_set = set(retrieved)
    served_abstain = (cond_result["retrieval_mode"] == "abstained") or (cond_result["result_count"] == 0)

    covered = bool(retrieved_set & drivers)
    lineage_ok = any(
        r.get("family") == want_family and r.get("role") == want_role and r.get("role") in DRIVER_ROLES
        for r in cond_result["top"]
    )
    excluded_leak_top1 = bool(retrieved) and retrieved[0] in excluded and retrieved[0] not in drivers

    enf = cond_result.get("evidence_admission_r1_enforcement") or {}
    final_route = enf.get("final_route_served")
    return {
        "abstention_expected": pack_q["abstention_expected"],
        "served_abstain": served_abstain,
        "accepted_evidence_covered": covered if not pack_q["abstention_expected"] else None,
        "lineage_satisfied": lineage_ok if not pack_q["abstention_expected"] else None,
        "excluded_source_led_top1": excluded_leak_top1,
        "final_route_served": final_route,
        "fallback_triggered": enf.get("fallback_triggered"),
        "forbidden_route_served": bool(final_route in FORBIDDEN_ROUTES),
        "top1_source": retrieved[0] if retrieved else None,
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", default="evidence_admission_r1_frozen_corpus_diag")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--embedding-model", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--result-path", default=str(ROOT / "benchmarks" / "results" /
                                              "evidence_admission_r1_direct_runtime_diagnostic_run_001.json"))
    p.add_argument("--reseed", action="store_true", help="drop and re-seed the diagnostic collection")
    return p.parse_args(argv)


CONDITIONS = [
    # name,          shadow, enforce, r1_gate_env, shadow_gate_env, request_flag_state, global_gate_state
    ("normal",        False, False, None,   None,   "absent",  "r1_disabled_shadow_disabled"),
    ("r0_shadow",     True,  False, None,   "true", "shadow_true", "shadow_enabled_r1_disabled"),
    ("r1_enforce",    False, True,  "true", None,   "enforce_true", "r1_enabled"),
    ("r1_gate_off",   False, True,  "false", None,  "enforce_true", "r1_disabled"),
]


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    pack_queries = {q["query_id"]: q for q in pack["queries"]}

    os.environ["MNEMOS_QDRANT_COLLECTION"] = args.collection
    os.environ["MNEMOS_QDRANT_URL"] = args.qdrant_url
    os.environ["MNEMOS_EMBEDDING_MODEL"] = args.embedding_model
    os.environ["MNEMOS_TIERS"] = "qdrant"
    os.environ.setdefault("MNEMOS_LOG_LEVEL", "ERROR")

    from mnemos.retrieval.evidence_admission.config import EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV
    from mnemos.retrieval.evidence_admission import R1_ENFORCEMENT_ENABLE_ENV
    from service.app import MnemosRuntime

    if args.reseed:
        try:
            import requests
            requests.delete(f"{args.qdrant_url}/collections/{args.collection}", timeout=10)
        except Exception:
            pass

    runtime = MnemosRuntime()
    runtime.initialize()

    # seed (idempotent enough for a diagnostic: reseed flag recreates)
    docs = build_seed_documents(manifest)
    index_res = runtime.index_documents(docs, {"index_lexical": False})
    indexed = index_res.get("result", {}).get("tiers", {})

    snapshot = f"{args.collection}:{len(docs)}"
    runner_commit = git_head()

    per_condition: Dict[str, Any] = {}
    raw_by_query: Dict[str, Dict[str, Any]] = {qid: {"query": q["query"], "intent_family": q["intent_family"],
                                                     "conditions": {}} for qid, q in pack_queries.items()}

    def set_gate(env_name: str, value: Optional[str]):
        if value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = value

    for name, shadow, enforce, r1_gate, shadow_gate, req_flag, gate_state in CONDITIONS:
        set_gate(R1_ENFORCEMENT_ENABLE_ENV, r1_gate)
        set_gate(EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV, shadow_gate)
        scores = []
        t0 = time.time()
        for qid, q in pack_queries.items():
            res = run_condition_query(runtime, q["query"], args.top_k, shadow=shadow, enforce=enforce)
            sc = score_query(q, res)
            raw_by_query[qid]["conditions"][name] = {
                "retrieval_mode": res["retrieval_mode"], "result_count": res["result_count"],
                "top": res["top"][:5], "final_route_served": sc["final_route_served"],
                "fallback_triggered": sc["fallback_triggered"], "served_abstain": sc["served_abstain"],
            }
            scores.append((qid, sc))
        elapsed = round(time.time() - t0, 2)

        nonabst = [(qid, s) for qid, s in scores if not s["abstention_expected"]]
        abst = [(qid, s) for qid, s in scores if s["abstention_expected"]]
        covered = sum(1 for _, s in nonabst if s["accepted_evidence_covered"])
        lineage = sum(1 for _, s in nonabst if s["lineage_satisfied"])
        per_condition[name] = {
            "run_manifest": {
                "condition": name,
                "formal_pack_hash": sha256_file(PACK),
                "freeze_receipt_commit": git_last_commit(RECEIPT),
                "service_revision": f"direct_runtime:{runner_commit}",
                "runner_revision": runner_commit,
                "corpus_snapshot": snapshot,
                "embedding_model_used": args.embedding_model,
                "frozen_manifest_declared_embedding": manifest["embedding_profile"]["embedding_model_name"],
                "request_flag_state": req_flag,
                "global_gate_state": gate_state,
                "cache_state": "cold_first_pass",
                "top_k": args.top_k,
            },
            "metrics": {
                "non_abstention_queries": len(nonabst),
                "accepted_evidence_coverage": covered,
                "accepted_evidence_coverage_rate": round(covered / len(nonabst), 4) if nonabst else None,
                "lineage_satisfied_count": lineage,
                "abstention_queries": len(abst),
                "abstention_served_count": sum(1 for _, s in abst if s["served_abstain"]),
                "forbidden_route_served_count": sum(1 for _, s in scores if s["forbidden_route_served"]),
                "excluded_source_led_top1_count": sum(1 for _, s in scores if s["excluded_source_led_top1"]),
                "fallback_triggered_count": sum(1 for _, s in scores if s["fallback_triggered"]),
                "wall_time_s": elapsed,
            },
        }

    # kill-switch identity: condition 4 must equal condition 1 top-k ordering
    identical = 0
    for qid in pack_queries:
        c1 = raw_by_query[qid]["conditions"]["normal"]["top"]
        c4 = raw_by_query[qid]["conditions"]["r1_gate_off"]["top"]
        if [r.get("source_path") for r in c1] == [r.get("source_path") for r in c4]:
            identical += 1

    # non-inferiority (diagnostic): enforced vs normal coverage
    cov_norm = per_condition["normal"]["metrics"]["accepted_evidence_coverage_rate"]
    cov_enf = per_condition["r1_enforce"]["metrics"]["accepted_evidence_coverage_rate"]
    delta_pp = None
    if cov_norm is not None and cov_enf is not None:
        delta_pp = round((cov_norm - cov_enf) * 100, 2)

    payload = {
        "run_id": "evidence_admission_r1_direct_runtime_diagnostic_run_001",
        "claim_status": "DIRECT_RUNTIME_ONLY_EVIDENCE",
        "formal_claim_permitted": False,
        "evidence_class": "direct_runtime_diagnostic",
        "evidence_labels": [
            "DIRECT_RUNTIME_ONLY_EVIDENCE",
            "FORMAL_CLAIM_PERMITTED=false",
            "NOMIC_EMBEDDER_DIAGNOSTIC_ONLY",
            "NOT_AGGREGATABLE_WITH_FORMAL_HTTP_RESULTS",
        ],
        "runtime_configuration_facts": {
            # Recorded so a formal-run failure can be attributed correctly:
            # "R1 policy unsafe" vs "frozen runtime never provided the
            # declared cue/cache mechanisms the pack was designed to exercise".
            "cue_registry_state": "empty_not_populated (service _build_admission_request_context passes empty cue list)",
            "tag_registry_state": "empty_not_populated (service _build_admission_request_context passes empty tag list)",
            "cache_fixture_state": "no_cache_fixtures_seeded (pre-cognitive cache cold; CACHE_ONLY never recommendable)",
            "consequence": "R0 route recommendations collapse to semantic; CUE_ONLY_LOOKUP and CACHE_ONLY enforcement paths are structurally unexercisable in this configuration.",
        },
        "aggregation_policy": "MUST NOT be aggregated with any http_service formal evidence (R0 or R1).",
        "not_formal_reasons": [
            "direct_runtime execution mode (not the http_service formal path)",
            f"embedder used ({args.embedding_model}) differs from frozen-manifest-declared "
            f"{manifest['embedding_profile']['embedding_model_name']}",
            "no verifiable service_revision identity established",
        ],
        "pack_id": pack["pack_id"],
        "formal_pack_hash": sha256_file(PACK),
        "freeze_receipt_commit": git_last_commit(RECEIPT),
        "corpus_manifest_id": manifest["manifest_id"],
        "corpus_snapshot": snapshot,
        "seed_index_tiers": indexed,
        "conditions": per_condition,
        "cross_condition": {
            "kill_switch_identity_rate": round(identical / len(pack_queries), 4),
            "kill_switch_identical_queries": identical,
            "total_queries": len(pack_queries),
            "diagnostic_noninferiority_delta_pp_normal_minus_enforced": delta_pp,
            "preregistered_noninferiority_margin_pp": 2.0,
            "note": "delta is DIAGNOSTIC ONLY; the preregistered non-inferiority test applies to the http_service formal run, not this direct-runtime diagnostic.",
        },
        "per_query": raw_by_query,
    }

    out = Path(args.result_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", out)
    for name in per_condition:
        m = per_condition[name]["metrics"]
        print(f"  {name:12s} coverage={m['accepted_evidence_coverage']}/{m['non_abstention_queries']} "
              f"({m['accepted_evidence_coverage_rate']}) abstain_served={m['abstention_served_count']}/{m['abstention_queries']} "
              f"forbidden={m['forbidden_route_served_count']} fallback={m['fallback_triggered_count']}")
    print(f"  kill_switch_identity={identical}/{len(pack_queries)}  noninf_delta_pp(diag)={delta_pp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
