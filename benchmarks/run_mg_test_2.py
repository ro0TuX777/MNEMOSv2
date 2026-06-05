import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnemos.engram.model import Engram
from mnemos.governance.models.memory_state import GovernanceMeta
from mnemos.retrieval.base import BaseRetriever, SearchResult
from mnemos.retrieval.fusion import TierFusion
from mnemos.retrieval.graph_tier import GraphTier, InMemoryEngramResolver
from mnemos.retrieval.retrieval_router import RetrievalRouter

# ─── Mocks & Harness ──────────────────────────────────────────────

class MockSemanticRetriever(BaseRetriever):
    def __init__(self, name: str, mock_results: dict):
        self.name = name
        self._mock_results = mock_results  # map query -> list of SearchResult

    def search(self, query: str, top_k: int = 10, **kwargs):
        # We will directly match query_text from our loop
        return self._mock_results.get(query, [])

    def get_by_ids(self, ids: List[str]):
        return []
    def delete(self, ids: List[str]):
        pass
    def index(self, documents: List[dict]):
        pass
    def stats(self):
        return {}
    @property
    def tier_name(self):
        return self.name

# ─── Dataset Generation ────────────────────────────────────────────

def create_engram(eid, content, edges=None, gov=None, lineage=None, tags=None):
    e = Engram(id=eid, content=content, edges=edges or [], neuro_tags=tags or [])
    if gov:
        e.governance = gov
    # Set lineage metadata
    meta = {}
    if lineage:
        meta.update(lineage)
    e.metadata = meta
    return e

def load_queries():
    q_file = Path(__file__).parent / "truthsets" / "gate_b_sanity_queries.json"
    with open(q_file, "r") as f:
        data = json.load(f)
    return data["queries"]

def generate_synthetic_graph():
    """
    Generate seeds and graph candidates matching the required edge cases.
    We return (engram_dict, mock_semantic_results, utility_labels).
    """
    queries = load_queries()
    
    engram_dict = {}
    mock_semantic_results = {}
    utility_labels = {} # candidate_id -> "useful" or "useless"
    
    # Hub node: highly connected, shows up across multiple queries
    hub_node = create_engram(
        "hub_node_01", "Generic enterprise architecture hub",
        lineage={"artifact_id": "art_hub", "source_uri": "uri_hub", "chunk_id": "c_hub"},
        tags=["useless"]
    )
    engram_dict[hub_node.id] = hub_node
    utility_labels[hub_node.id] = "useless"
    
    for i, q in enumerate(queries):
        qtext = q["query_text"]
        
        # 2 seeds per query
        seed1_id = f"seed_{i}_1"
        seed2_id = f"seed_{i}_2"
        
        # Edges we will build
        edges1 = []
        edges2 = []
        
        # 1. Useful candidate (recovers missing support)
        useful_id = f"useful_{i}"
        e_useful = create_engram(
            useful_id, f"Support fact answering {qtext}",
            lineage={"artifact_id": f"art_{i}", "source_uri": f"uri_{i}", "chunk_id": "c1"},
            tags=["useful"]
        )
        engram_dict[useful_id] = e_useful
        edges1.append(useful_id)
        utility_labels[useful_id] = "useful"
        
        # 2. Useless candidate (distracting)
        useless_id = f"useless_{i}"
        e_useless = create_engram(
            useless_id, f"Irrelevant details somewhat related to {qtext}",
            lineage={"artifact_id": f"art_x_{i}", "source_uri": f"uri_x_{i}", "chunk_id": "c2"},
            tags=["useless"]
        )
        engram_dict[useless_id] = e_useless
        edges2.append(useless_id)
        utility_labels[useless_id] = "useless"
        
        # 3. Governance Blocked (e.g. vetoed)
        gov_id = f"gov_blocked_{i}"
        e_gov = create_engram(
            gov_id, "Bad node",
            gov=GovernanceMeta(conflict_status="vetoed"),
            lineage={"artifact_id": "art", "source_uri": "uri", "chunk_id": "c"}
        )
        engram_dict[gov_id] = e_gov
        edges1.append(gov_id)
        
        # 4. Incomplete Lineage (missing source_uri explicitly)
        lin_id = f"lin_incomplete_{i}"
        e_lin = create_engram(
            lin_id, "Missing source_uri",
            lineage={"artifact_id": "art", "source_uri": "", "chunk_id": "c"}
        )
        engram_dict[lin_id] = e_lin
        edges2.append(lin_id)
        
        # 5. Hub node saturation (all odd queries link to hub)
        if i % 2 != 0:
            edges1.append(hub_node.id)
            
        # Create seeds
        seed1 = create_engram(seed1_id, f"Seed 1 for {qtext}", edges=edges1, lineage={"artifact_id":"art","source_uri":"uri","chunk_id":"c"})
        seed2 = create_engram(seed2_id, f"Seed 2 for {qtext}", edges=edges2, lineage={"artifact_id":"art","source_uri":"uri","chunk_id":"c"})
        engram_dict[seed1_id] = seed1
        engram_dict[seed2_id] = seed2
        
        mock_semantic_results[qtext] = [
            SearchResult(engram=seed1, score=0.9),
            SearchResult(engram=seed2, score=0.85)
        ]
        
    return engram_dict, mock_semantic_results, utility_labels

# ─── Runner ────────────────────────────────────────────────────────

def main():
    print("🔬 MG-Test-2A Synthetic Evaluation Harness")
    
    engram_dict, mock_results, utility_labels = generate_synthetic_graph()
    queries = load_queries()
    
    retriever = MockSemanticRetriever("semantic", mock_results)
    fusion = TierFusion([retriever])
    graph_tier = GraphTier(engram_resolver=InMemoryEngramResolver(engram_dict))
    
    router = RetrievalRouter(
        semantic_fusion=fusion,
        graph_tier=graph_tier,
        graph_shadow_enabled=True
    )
    
    metrics = {
        "queries_run": 0,
        "total_seed_candidates": 0,
        "total_graph_candidates": 0,
        "total_unique_graph_candidates": 0,
        "total_useful_candidates": 0,
        "total_useless_candidates": 0,
        "total_gov_filtered": 0,
        "total_lineage_filtered": 0,
        "latencies_ms": [],
        "hub_frequency": {},
        "per_query_records": []
    }
    
    for q in queries:
        qtext = q["query_text"]
        
        t0 = time.perf_counter()
        hits, meta = router.search(query=qtext, top_k=10)
        t_elapsed = (time.perf_counter() - t0) * 1000
        
        telemetry = meta.get("graph_shadow_telemetry", {})
        
        metrics["queries_run"] += 1
        metrics["latencies_ms"].append(t_elapsed)
        metrics["total_seed_candidates"] += telemetry.get("seed_candidate_count", 0)
        metrics["total_graph_candidates"] += telemetry.get("graph_candidate_count", 0)
        metrics["total_unique_graph_candidates"] += telemetry.get("graph_unique_candidate_count", 0)
        metrics["total_gov_filtered"] += telemetry.get("graph_governance_filtered_count", 0)
        metrics["total_lineage_filtered"] += telemetry.get("graph_lineage_filtered_count", 0)
        
        useful = 0
        useless = 0
        for cand in telemetry.get("candidates", []):
            cid = cand["candidate_id"]
            if utility_labels.get(cid) == "useful":
                useful += 1
            elif utility_labels.get(cid) == "useless":
                useless += 1
                
            metrics["hub_frequency"][cid] = metrics["hub_frequency"].get(cid, 0) + 1
                
        metrics["total_useful_candidates"] += useful
        metrics["total_useless_candidates"] += useless
        
        metrics["per_query_records"].append({
            "query_id": q["query_id"],
            "query_text": qtext,
            "latency_ms": round(t_elapsed, 2),
            "telemetry": telemetry,
            "utility_useful": useful,
            "utility_useless": useless
        })

    # Stats
    latencies = np.array(metrics["latencies_ms"])
    metrics["latency_p50"] = round(np.percentile(latencies, 50), 2)
    metrics["latency_p95"] = round(np.percentile(latencies, 95), 2)
    metrics["latency_p99"] = round(np.percentile(latencies, 99), 2)
    
    # Hub analysis
    hubs = {k: v for k, v in metrics["hub_frequency"].items() if v > 1}
    metrics["hub_nodes_detected"] = len(hubs)
    metrics["hub_occurrences"] = hubs
    
    # Save
    out_path = Path(__file__).parent / "mg_test_2_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
        
    print(f"✅ Evaluation complete. Metrics saved to {out_path}")
    
if __name__ == "__main__":
    main()
