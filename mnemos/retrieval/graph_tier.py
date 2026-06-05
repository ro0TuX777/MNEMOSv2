"""
GraphTier for Phase MG-Test-1 Shadow Telemetry

Traverses Engram.edges from seed candidates to discover multi-hop context.
Currently uses an InMemoryEngramResolver for isolated shadow execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from mnemos.engram.model import Engram
from mnemos.retrieval.base import SearchResult

logger = logging.getLogger(__name__)

class EngramResolver(Protocol):
    def get_by_id(self, engram_id: str) -> Optional[Engram]:
        ...
    def get_degree(self, engram_id: str) -> int:
        ...

class InMemoryEngramResolver:
    def __init__(self, engrams: Dict[str, Engram] = None):
        self._store = engrams or {}
        
    def get_by_id(self, engram_id: str) -> Optional[Engram]:
        return self._store.get(engram_id)
        
    def get_degree(self, engram_id: str) -> int:
        eng = self.get_by_id(engram_id)
        return len(eng.edges) if eng else 0

@dataclass
class GraphCandidate:
    engram: Engram
    seed_id: str
    edge_path: List[str]
    edge_depth: int
    edge_type: str = "related_to"
    graph_score: Optional[float] = None
    lineage_complete: bool = False
    governance_state: str = "active"
    retrieval_reason: str = "linked_neighbor_from_seed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.engram.id,
            "seed_id": self.seed_id,
            "edge_path": self.edge_path,
            "edge_depth": self.edge_depth,
            "edge_type": self.edge_type,
            "graph_score": self.graph_score,
            "lineage_complete": self.lineage_complete,
            "governance_state": self.governance_state,
            "retrieval_reason": self.retrieval_reason,
            "filtered_reason": getattr(self, "filtered_reason", None)
        }

class GraphTier:
    def __init__(self, engram_resolver: EngramResolver):
        self.resolver = engram_resolver

    def _is_blocked(self, engram: Engram) -> bool:
        if engram.governance:
            # We treat expired, vetoed, suppressed, deleted as blocked
            lf_state = engram.governance.lifecycle_state
            del_state = engram.governance.deletion_state
            if lf_state in {"archived", "expired"} or del_state in {"soft_deleted", "tombstone"}:
                return True
            # Also check if it's explicitly vetoed or suppressed by policy/conflict
            if engram.governance.conflict_status in {"suppressed", "vetoed"}:
                return True
        return False
        
    def _is_lineage_complete(self, engram: Engram) -> bool:
        lin = engram.lineage()
        # Must have both artifact_id and source_uri non-empty
        # and chunk_id
        if not lin.get("artifact_id") or not lin.get("source_uri") or not lin.get("chunk_id"):
            return False
        return True

    def expand_candidates(
        self,
        seed_candidates: List[SearchResult],
        query_embedding: Optional[np.ndarray] = None,
        max_depth: int = 1,
        max_neighbors_per_seed: int = 5,
        max_total_graph_candidates: int = 20,
        max_seed_candidates: int = 10,
        hub_degree_threshold: int = 5,
        score_threshold: float = 0.5
    ) -> Tuple[List[GraphCandidate], Dict[str, Any]]:
        
        start_time = time.perf_counter()
        
        # MG-Test-1 Constraint: max_depth > 1 rejected/ignored, fixed to 1
        if max_depth > 1:
            logger.warning(f"GraphTier max_depth {max_depth} requested but MG-Test-1 forces max_depth=1")
            max_depth = 1
            
        seeds_to_process = seed_candidates[:max_seed_candidates]
        
        import math
        import numpy as np

        discovered_candidates: List[GraphCandidate] = []
        ineligible_candidates: List[GraphCandidate] = []
        seen_ids = set()
        
        # Telemetry counters
        filtered_gov_count = 0
        filtered_lin_count = 0
        filtered_score_count = 0
        hub_candidates = 0
        repeated_candidate_counts = {}
        sum_graph_score = 0.0
        min_graph_score = float('inf')
        max_graph_score = -float('inf')
        
        # Pre-fill seen with seeds to avoid returning seeds as graph candidates
        for seed in seed_candidates:
            seen_ids.add(seed.engram.id)
            
        filtered_count = 0
        
        for seed_res in seeds_to_process:
            seed_engram = seed_res.engram
            
            # Check seed governance
            if self._is_blocked(seed_engram):
                # We skip expanding from blocked seeds
                filtered_count += len(seed_engram.edges)
                continue
                
            neighbors = seed_engram.edges[:max_neighbors_per_seed]
            
            for neighbor_id in neighbors:
                if len(discovered_candidates) >= max_total_graph_candidates:
                    break
                    
                if neighbor_id in seen_ids:
                    continue
                    
                neighbor_engram = self.resolver.get_by_id(neighbor_id)
                if not neighbor_engram:
                    continue
                    
                seen_ids.add(neighbor_id)
                
                # 2. Check neighbor governance
                if self._is_blocked(neighbor_engram):
                    filtered_gov_count += 1
                    continue
                    
                # 3. Check lineage
                lineage_ok = self._is_lineage_complete(neighbor_engram)
                gov_state = neighbor_engram.governance.lifecycle_state if neighbor_engram.governance else "active"
                
                # 4. Hub/Relevance Scoring
                degree = self.resolver.get_degree(neighbor_id)
                if degree > hub_degree_threshold:
                    hub_candidates += 1
                    
                hub_penalty = 1.0 / (1.0 + math.log1p(max(0, degree - hub_degree_threshold)))
                
                relevance_score = 1.0
                if query_embedding is not None and neighbor_engram.embedding is not None:
                    # simple cosine similarity
                    norm_q = np.linalg.norm(query_embedding)
                    norm_e = np.linalg.norm(neighbor_engram.embedding)
                    if norm_q > 0 and norm_e > 0:
                        relevance_score = np.dot(query_embedding, neighbor_engram.embedding) / (norm_q * norm_e)
                elif neighbor_engram.confidence is not None:
                    relevance_score = neighbor_engram.confidence
                
                graph_score = relevance_score * hub_penalty

                gc = GraphCandidate(
                    engram=neighbor_engram,
                    seed_id=seed_engram.id,
                    edge_path=[seed_engram.id, neighbor_id],
                    edge_depth=1,
                    graph_score=graph_score,
                    lineage_complete=lineage_ok,
                    governance_state=gov_state
                )
                
                # Track for telemetry
                gc.candidate_degree = degree
                gc.hub_penalty = hub_penalty
                gc.relevance_score = relevance_score
                gc.score_threshold = score_threshold
                
                if not lineage_ok:
                    gc.filtered_reason = "lineage_incomplete"
                    ineligible_candidates.append(gc)
                    filtered_lin_count += 1
                    continue
                    
                # 5. Score threshold filtering
                if graph_score < score_threshold:
                    gc.filtered_reason = "score_below_threshold"
                    ineligible_candidates.append(gc)
                    filtered_score_count += 1
                    continue

                repeated_candidate_counts[neighbor_id] = repeated_candidate_counts.get(neighbor_id, 0) + 1
                sum_graph_score += graph_score
                min_graph_score = min(min_graph_score, graph_score)
                max_graph_score = max(max_graph_score, graph_score)

                discovered_candidates.append(gc)
                    
            if len(discovered_candidates) >= max_total_graph_candidates:
                break
                
        # 6. Telemetry emission
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        avg_score = sum_graph_score / len(discovered_candidates) if discovered_candidates else 0.0
        min_score = min_graph_score if discovered_candidates else 0.0
        max_score = max_graph_score if discovered_candidates else 0.0
        
        top_repeated = sorted(repeated_candidate_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        telemetry = {
            "graph_candidate_count": len(discovered_candidates),
            "graph_governance_filtered_count": filtered_gov_count,
            "graph_lineage_filtered_count": filtered_lin_count,
            "graph_score_filtered_count": filtered_score_count,
            "graph_hub_penalty_filtered_count": sum(1 for c in ineligible_candidates if getattr(c, "hub_penalty", 1.0) < 1.0 and c.filtered_reason == "score_below_threshold"),
            "graph_latency_ms": round(latency_ms, 2),
            "graph_avg_graph_score": round(avg_score, 4),
            "graph_min_graph_score": round(min_score, 4),
            "graph_max_graph_score": round(max_score, 4),
            "graph_hub_candidate_count": hub_candidates,
            "graph_top_repeated_candidate_ids": [cid for cid, count in top_repeated if count > 1],
            "graph_score_threshold": score_threshold,
            "hub_degree_threshold": hub_degree_threshold,
            "ineligible_candidates": [c.to_dict() for c in ineligible_candidates]
        }
        
        return discovered_candidates, telemetry
