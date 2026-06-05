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

class InMemoryEngramResolver:
    def __init__(self, engrams: Dict[str, Engram] = None):
        self._store = engrams or {}
        
    def get_by_id(self, engram_id: str) -> Optional[Engram]:
        return self._store.get(engram_id)

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
            "retrieval_reason": self.retrieval_reason
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
        max_depth: int = 1,
        max_neighbors_per_seed: int = 5,
        max_total_graph_candidates: int = 20,
        max_seed_candidates: int = 10
    ) -> Tuple[List[GraphCandidate], Dict[str, Any]]:
        
        start_time = time.perf_counter()
        
        # MG-Test-1 Constraint: max_depth > 1 rejected/ignored, fixed to 1
        if max_depth > 1:
            logger.warning(f"GraphTier max_depth {max_depth} requested but MG-Test-1 forces max_depth=1")
            max_depth = 1
            
        seeds_to_process = seed_candidates[:max_seed_candidates]
        
        discovered_candidates: List[GraphCandidate] = []
        ineligible_candidates: List[GraphCandidate] = []
        seen_ids = set()
        
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
                
                # Check neighbor governance
                if self._is_blocked(neighbor_engram):
                    filtered_count += 1
                    continue
                    
                lineage_ok = self._is_lineage_complete(neighbor_engram)
                gov_state = neighbor_engram.governance.lifecycle_state if neighbor_engram.governance else "active"
                
                gc = GraphCandidate(
                    engram=neighbor_engram,
                    seed_id=seed_engram.id,
                    edge_path=[seed_engram.id, neighbor_id],
                    edge_depth=1,
                    lineage_complete=lineage_ok,
                    governance_state=gov_state
                )
                
                if lineage_ok:
                    discovered_candidates.append(gc)
                else:
                    ineligible_candidates.append(gc)
                    
            if len(discovered_candidates) >= max_total_graph_candidates:
                break
                
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        telemetry = {
            "graph_candidate_count": len(discovered_candidates),
            "graph_governance_filtered_count": filtered_count,
            "graph_lineage_filtered_count": len(ineligible_candidates),
            "graph_latency_ms": round(latency_ms, 2),
            "ineligible_candidates": [c.to_dict() for c in ineligible_candidates]
        }
        
        return discovered_candidates, telemetry
