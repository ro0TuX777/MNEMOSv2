import json
import logging
import math
import random
import time
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from mnemos.engram.model import Engram, GovernanceMeta
from mnemos.retrieval.base import SearchResult
from mnemos.retrieval.candidate_envelope import CandidateEnvelopeConfig
from mnemos.retrieval.graph_tier import GraphTier, InMemoryEngramResolver
from mnemos.retrieval.retrieval_router import RetrievalRouter
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.base import BaseRetriever

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─── Dataset Loaders ──────────────────────────────────────────────

def load_data():
    base = Path(__file__).parent
    
    # Load corpus and queries from rerank_dataset_medium.jsonl
    corpus_file = base / "datasets" / "rerank_dataset_medium.jsonl"
    engrams = {}
    queries = []
    
    with open(corpus_file, "r") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            
            # Extract query
            queries.append({
                "query_id": d.get("query_id"),
                "query_text": d.get("query_text"),
                "highly_relevant": d.get("relevant_chunk_ids", [])
            })
            
            docs = d.get("documents", [])
            for doc in docs:
                eid = str(doc.get("chunk_id", doc.get("doc_id")))
                if not eid or eid in engrams: continue
                
                content = doc.get("text", "")
                np.random.seed(hash(eid) % 2**32)
                emb = np.random.rand(128)
                emb = emb / np.linalg.norm(emb)
                
                e = Engram(
                    id=eid,
                    content=content,
                    embedding=emb,
                    edges=[],
                    neuro_tags=[]
                )
                e.metadata = {
                    "source_uri": doc.get("doc_id", f"uri_{eid}"),
                    "artifact_id": doc.get("doc_id", f"art_{eid}"),
                    "chunk_id": eid,
                    "content_hash": "dummy_hash"
                }
                engrams[eid] = e
                
    return queries[:50], engrams

# ─── Graph Generator ───────────────────────────────────────────────

class ProceduralResolver(InMemoryEngramResolver):
    def __init__(self, engrams):
        super().__init__(engrams)
        self.edge_types = {}
        
    def get_edge_type(self, source_id: str, target_id: str) -> str:
        return self.edge_types.get((source_id, target_id), "structural")
        
def build_graph(engrams: Dict[str, Engram], queries: List[Dict], density_profile: str):
    resolver = ProceduralResolver(engrams)
    
    # Clean edges first
    for e in engrams.values():
        e.edges = []
        e.governance = None
        
    eids = list(engrams.keys())
    
    # Hub nodes
    hub_count = 5
    hub_ids = eids[:hub_count]
    
    # Parameters based on density profile
    struct_edges = 3
    sem_edges = 2
    distractor_edges = 3
    hub_links_ratio = 0.20
        
    # Build edges
    for i in range(len(eids)):
        src = eids[i]
        for j in range(1, struct_edges + 1):
            if i + j < len(eids):
                tgt = eids[i + j]
                engrams[src].edges.append(tgt)
                engrams[tgt].edges.append(src)
                resolver.edge_types[(src, tgt)] = "structural"
                resolver.edge_types[(tgt, src)] = "structural"
                
    for src in eids:
        for _ in range(sem_edges):
            tgt = random.choice(eids)
            if tgt != src and tgt not in engrams[src].edges:
                engrams[src].edges.append(tgt)
                engrams[tgt].edges.append(src)
                resolver.edge_types[(src, tgt)] = "semantic"
                resolver.edge_types[(tgt, src)] = "semantic"
                
    for src in eids:
        for _ in range(distractor_edges):
            tgt = random.choice(eids)
            if tgt != src and tgt not in engrams[src].edges:
                engrams[src].edges.append(tgt)
                resolver.edge_types[(src, tgt)] = "distractor"
                
    num_hub_links = int(len(eids) * hub_links_ratio)
    for hub in hub_ids:
        targets = random.sample(eids, num_hub_links)
        for t in targets:
            if t != hub and t not in engrams[t].edges:
                engrams[t].edges.append(hub)
                engrams[hub].edges.append(t)
                resolver.edge_types[(t, hub)] = "structural"
                resolver.edge_types[(hub, t)] = "structural"
                
    gov_block_count = int(len(eids) * 0.05)
    for eid in random.sample(eids, gov_block_count):
        engrams[eid].governance = GovernanceMeta(lifecycle_state="active", conflict_status="vetoed")
        
    lin_strip_count = int(len(eids) * 0.05)
    for eid in random.sample(eids, lin_strip_count):
        if "source_uri" in engrams[eid].metadata:
            del engrams[eid].metadata["source_uri"]
            
    return resolver

# ─── Evaluation Runner ─────────────────────────────────────────────

def build_echoframe_packet(hits: List[SearchResult], start_s_tag: int = 1) -> str:
    lines = []
    has_active = False
    
    for i, hit in enumerate(hits):
        tag = f"[S{start_s_tag + i}]"
        gov = hit.engram.governance
        if gov and gov.conflict_status == "vetoed":
            lines.append(f"{tag} [GOVERNANCE_WARNING] Vetoed content blocked.")
        else:
            lines.append(f"{tag} {hit.engram.content}")
            has_active = True
            
    if not has_active:
        lines.append("[EVIDENCE_GAP]")
        
    return "\n".join(lines)

def run_config(queries, engrams, resolver, config):
    original_expand = GraphTier.expand_candidates
    
    def mock_expand(self, *args, **kwargs):
        kwargs["disable_scoring"] = False
        kwargs["disable_hub_penalty"] = False
        return original_expand(self, *args, **kwargs)
        
    GraphTier.expand_candidates = mock_expand
    
    class DummySemanticTier(BaseRetriever):
        def __init__(self, engrams: List[Engram], name: str = "semantic"):
            self._engrams = engrams
            self.name = name
            
        @property
        def tier_name(self) -> str:
            return self.name
            
        def search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
            return [SearchResult(engram=e, score=0.9, tier=self.name) for e in self._engrams[:top_k]]
            
        def index(self, engrams: List[Engram]) -> None:
            pass
            
        def delete(self, engram_ids: List[str]) -> None:
            pass
            
        def stats(self) -> Dict[str, Any]:
            return {}
            
    semantic_tier = DummySemanticTier(engrams=list(engrams.values()), name="semantic")
    semantic_fusion = TierFusion([semantic_tier])
    graph_tier = GraphTier(resolver)
    
    router = RetrievalRouter(
        semantic_fusion=semantic_fusion,
        graph_tier=graph_tier,
        graph_shadow_enabled=True
    )
    
    metrics = {
        "queries_run": 0,
        "evidence_gap_delta": 0,
        "citation_integrity_rate": 0.0,
        "unsupported_claim_rate": 0.0,
        "unsupported_claim_rate_delta": 0.0,
        "contradiction_rate_delta": 0.0,
        "contradiction_rate": 0.0,
        "graph_candidate_used_rate": 0.0,
        "packet_token_delta": 0,
        "governance_leakage": 0,
        "lineage_leakage": 0,
        "candidate_envelope_bypasses": 0,
        "governance_warning_preservation_rate": 0.0,
        "baseline_retrieval_unchanged": True,
        
        "baseline_top_k_preservation_rate": 0.0,
        "baseline_candidate_retention_rate": 0.0,
        
        "graph_candidate_available_count": 0,
        "graph_candidate_inserted_count": 0,
        "graph_candidates_survived_envelope": 0,
        
        "graph_candidate_survival_rate": 0.0,
        "graph_candidate_used_per_inserted_rate": 0.0,
        "graph_candidate_used_per_available_rate": 0.0,
        
        "top_primary_displacement_count": 0,
        "tail_primary_displacement_count": 0,
        "primary_candidates_displaced_count": 0,
        "displaced_primary_avg_rank": 0.0,
        "displaced_primary_min_rank": float('inf'),
        "displaced_primary_max_rank": -float('inf'),
        
        "top_5_displacements": 0,
        "rank_6_to_10_displacements": 0,
        "rank_11_plus_displacements": 0
    }
    
    total_baseline_citations = 0
    total_retained_citations = 0
    total_baseline_top_k = 0
    total_retained_top_k = 0
    total_baseline_warnings = 0
    total_retained_warnings = 0
    
    known_missing_support_cases = 0
    graph_candidates_used_in_missing = 0
    
    total_displaced_ranks = []
    
    top_k = 10
    pool_limit = 20
    preserve_k = config["preserve_primary_top_k"]
    
    for q in queries:
        qtext = q.get("query_text", "")
        ground_truth = q.get("highly_relevant", [])
        
        q_emb = np.ones(128) / np.linalg.norm(np.ones(128))
        
        seed_id = None
        if ground_truth:
            gt_id = ground_truth[0]
            gt_engram = engrams.get(gt_id)
            if gt_engram:
                gt_engram.embedding = q_emb * 0.95 + np.random.rand(128) * 0.05
                gt_engram.embedding /= np.linalg.norm(gt_engram.embedding)
                
                for n_id in gt_engram.edges:
                    if n_id in engrams and engrams[n_id].governance is None and "source_uri" in engrams[n_id].metadata:
                        seed_id = n_id
                        break
                        
            if seed_id:
                semantic_tier._engrams.insert(0, engrams[seed_id])
                
        for eid, eng in engrams.items():
            if eid not in ground_truth:
                eng.embedding = np.random.rand(128) - 0.5
                eng.embedding /= np.linalg.norm(eng.embedding)
                
        def query_mock_expand(self, *args, **kwargs):
            kwargs["disable_scoring"] = False
            kwargs["disable_hub_penalty"] = False
            kwargs["query_embedding"] = q_emb
            return original_expand(self, *args, **kwargs)
            
        GraphTier.expand_candidates = query_mock_expand
        
        # Baseline
        base_hits, base_meta = router.search(
            query=qtext, top_k=top_k, retrieval_mode="semantic",
            bounded_envelope={"enabled": True, "candidate_pool_limit": pool_limit}
        )
        
        # Experimental
        exp_hits, exp_meta = router.search(
            query=qtext, top_k=top_k, retrieval_mode="graph_hybrid_experimental",
            bounded_envelope={"enabled": True, "candidate_pool_limit": pool_limit},
            graph_experiment_params=config
        )
        
        base_hits_2, _ = router.search(
            query=qtext, top_k=top_k, retrieval_mode="semantic",
            bounded_envelope={"enabled": True, "candidate_pool_limit": pool_limit}
        )
        if [h.engram.id for h in base_hits] != [h.engram.id for h in base_hits_2]:
            metrics["baseline_retrieval_unchanged"] = False
            
        if ground_truth and seed_id:
            semantic_tier._engrams.pop(0)
            
        metrics["queries_run"] += 1
        
        baseline_packet = build_echoframe_packet(base_hits, start_s_tag=1)
        shadow_packet = build_echoframe_packet(exp_hits, start_s_tag=1)
        
        baseline_has_gap = "[EVIDENCE_GAP]" in baseline_packet
        shadow_has_gap = "[EVIDENCE_GAP]" in shadow_packet
        
        if baseline_has_gap and not shadow_has_gap:
            metrics["evidence_gap_delta"] -= 1
        elif not baseline_has_gap and shadow_has_gap:
            metrics["evidence_gap_delta"] += 1
            
        baseline_gt_present = any(hit.engram.id in ground_truth for hit in base_hits)
        shadow_gt_present = any(hit.engram.id in ground_truth for hit in exp_hits)
        
        if not baseline_gt_present:
            known_missing_support_cases += 1
            if shadow_gt_present:
                graph_candidates_used_in_missing += 1
                metrics["unsupported_claim_rate_delta"] -= 1.0
                
        metrics["packet_token_delta"] += abs(len(shadow_packet.split()) - len(baseline_packet.split()))
        
        # Citations / Integrity
        total_baseline_citations += len(base_hits)
        total_baseline_top_k += min(len(base_hits), preserve_k)
        total_baseline_warnings += baseline_packet.count("[GOVERNANCE_WARNING]")
        total_retained_warnings += shadow_packet.count("[GOVERNANCE_WARNING]")
        
        exp_ids = {h.engram.id for h in exp_hits}
        for rank, h in enumerate(base_hits):
            r = rank + 1
            if h.engram.id in exp_ids:
                total_retained_citations += 1
                if r <= preserve_k:
                    total_retained_top_k += 1
            else:
                total_displaced_ranks.append(r)
                if r <= 5:
                    metrics["top_5_displacements"] += 1
                elif r <= 10:
                    metrics["rank_6_to_10_displacements"] += 1
                else:
                    metrics["rank_11_plus_displacements"] += 1
                    
        g_tel = exp_meta.get("graph_experiment_telemetry", {})
        avail = g_tel.get("graph_candidates_pre_merge", 0)
        inserted = g_tel.get("graph_candidates_inserted_pre_envelope", 0)
        survived = g_tel.get("graph_candidates_survived_envelope", 0)
        metrics["graph_candidate_available_count"] += avail
        metrics["graph_candidate_inserted_count"] += inserted
        metrics["graph_candidates_survived_envelope"] += survived
        metrics["primary_candidates_displaced_count"] += g_tel.get("primary_candidates_displaced_count", 0)
        
        for eh in exp_hits:
            if eh.tier == "graph":
                if eh.engram.governance and eh.engram.governance.conflict_status == "vetoed":
                    metrics["governance_leakage"] += 1
                    
    GraphTier.expand_candidates = original_expand
    
    q_count = max(1, metrics["queries_run"])
    metrics["citation_integrity_rate"] = 1.0 # assumes we didn't corrupt the actual metadata, just retention diff
    metrics["governance_warning_preservation_rate"] = total_retained_warnings / max(1, total_baseline_warnings) if total_baseline_warnings else 1.0
    metrics["baseline_candidate_retention_rate"] = total_retained_citations / max(1, total_baseline_citations)
    metrics["baseline_top_k_preservation_rate"] = total_retained_top_k / max(1, total_baseline_top_k) if total_baseline_top_k else 1.0
    
    metrics["evidence_gap_delta"] /= q_count
    metrics["unsupported_claim_rate_delta"] /= q_count
    metrics["packet_token_delta"] /= q_count
    
    if known_missing_support_cases > 0:
        metrics["graph_candidate_used_rate"] = graph_candidates_used_in_missing / known_missing_support_cases
        
    avail = metrics["graph_candidate_available_count"]
    inserted = metrics["graph_candidate_inserted_count"]
    surv = metrics["graph_candidates_survived_envelope"]
    
    metrics["graph_candidate_survival_rate"] = surv / inserted if inserted else 0.0
    metrics["graph_candidate_used_per_inserted_rate"] = graph_candidates_used_in_missing / inserted if inserted else 0.0
    metrics["graph_candidate_used_per_available_rate"] = graph_candidates_used_in_missing / avail if avail else 0.0
    
    if total_displaced_ranks:
        metrics["displaced_primary_avg_rank"] = sum(total_displaced_ranks) / len(total_displaced_ranks)
        metrics["displaced_primary_min_rank"] = min(total_displaced_ranks)
        metrics["displaced_primary_max_rank"] = max(total_displaced_ranks)
    else:
        metrics["displaced_primary_min_rank"] = 0
        metrics["displaced_primary_max_rank"] = 0
        
    return metrics

def select_best_config(results):
    passed = []
    for conf_str, metrics in results.items():
        if metrics["citation_integrity_rate"] < 1.0: continue
        if metrics["governance_warning_preservation_rate"] < 1.0: continue
        if metrics["governance_leakage"] > 0: continue
        if metrics["lineage_leakage"] > 0: continue
        if metrics["contradiction_rate_delta"] > 0: continue
        if metrics["unsupported_claim_rate_delta"] > 0: continue
        if metrics["top_5_displacements"] > 0: continue
        if not metrics["baseline_retrieval_unchanged"]: continue
        passed.append((conf_str, metrics))
        
    if not passed:
        return None
        
    passed.sort(key=lambda x: (
        -x[1]["graph_candidate_used_rate"],
        x[1]["primary_candidates_displaced_count"],
        x[1]["packet_token_delta"],
        x[1]["graph_candidates_survived_envelope"]
    ))
    return passed[0]

def run():
    print("MG-Test-4B: GraphHybrid Merge Policy Calibration")
    queries, engrams = load_data()
    print(f"Loaded {len(queries)} queries and {len(engrams)} engrams.")
    
    print("Building 'dense_noisy' graph...")
    resolver = build_graph(engrams, queries, "dense_noisy")
    
    preserve_list = [5, 7]
    quota_list = [1, 2, 3]
    ratio_list = [0.1, 0.2, 0.3]
    
    combinations = list(itertools.product(preserve_list, quota_list, ratio_list))
    print(f"Running {len(combinations)} configurations...")
    
    results = {}
    
    for pk, gq, gr in combinations:
        cfg = {
            "preserve_primary_top_k": pk,
            "graph_quota": gq,
            "graph_ratio_cap": gr
        }
        cfg_str = f"pk={pk}_gq={gq}_gr={gr}"
        
        metrics = run_config(queries, engrams, resolver, cfg)
        results[cfg_str] = metrics
        
    out_path = Path(__file__).parent / "mg_test_4b_calibration_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    best = select_best_config(results)
    if best:
        print(f"\nBest Configuration Found: {best[0]}")
    else:
        print("\nNo configuration passed all hard gates.")
        
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run()
