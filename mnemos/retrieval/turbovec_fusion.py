"""Python-side Hybrid RRF Fusion for TurbovecTier."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from mnemos.retrieval.turbovec_tier import TurbovecTier

@dataclass(frozen=True)
class FusionHit:
    engram_uuid: str
    score: float
    dense_score: Optional[float]
    lexical_score: Optional[float]
    dense_rank: Optional[int]
    lexical_rank: Optional[int]
    source_uri: Optional[str]
    content: Optional[str]
    metadata: dict
    governance: dict
    explanation: dict

FUSION_POLICIES = {
    "semantic_dominant": {"dense_weight": 0.75, "lexical_weight": 0.25},
    "balanced": {"dense_weight": 0.50, "lexical_weight": 0.50},
    "lexical_dominant": {"dense_weight": 0.25, "lexical_weight": 0.75},
}

class TurbovecFusion:
    def __init__(self, tier: TurbovecTier, *, rrf_k: int = 60):
        self.tier = tier
        self.rrf_k = rrf_k

    def search(
        self,
        query_text: str,
        query_embedding: List[float],
        *,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        policy: str = "balanced",
        explain: bool = False,
    ) -> List[FusionHit]:
        if policy not in FUSION_POLICIES:
            raise ValueError(f"Unknown fusion policy: {policy}")
            
        policy_weights = FUSION_POLICIES[policy]
        dense_w = policy_weights["dense_weight"]
        lexical_w = policy_weights["lexical_weight"]
        
        # 1. Dense Search
        oversample = self.tier.config.oversample_factor
        dense_candidates = self.tier.search(query_embedding, top_k * oversample, filters)
        
        # 2. Lexical Search
        raw_lexical = self.tier.sidecar.lexical_search(query_text, top_k * oversample)
        if filters:
            lexical_candidates_dicts = self.tier.sidecar.filter_candidates(filters, [r["engram_uuid"] for r in raw_lexical])
            filtered_lexical_ids = {r["engram_uuid"] for r in lexical_candidates_dicts}
            lexical_candidates = [r for r in raw_lexical if r["engram_uuid"] in filtered_lexical_ids]
        else:
            lexical_candidates = raw_lexical
            
        # 3. Normalize to ranks
        union_uuids = set()
        dense_ranks = {}
        lexical_ranks = {}
        dense_scores = {}
        lexical_scores = {} 
        
        for rank, dh in enumerate(dense_candidates, start=1):
            union_uuids.add(dh.engram_uuid)
            dense_ranks[dh.engram_uuid] = rank
            dense_scores[dh.engram_uuid] = dh.score
            
        for rank, lh in enumerate(lexical_candidates, start=1):
            union_uuids.add(lh["engram_uuid"])
            lexical_ranks[lh["engram_uuid"]] = rank
            lexical_scores[lh["engram_uuid"]] = None  # FTS score not fetched explicitly
            
        # 4. Apply RRF
        scored_hits = []
        for uid in union_uuids:
            d_rank = dense_ranks.get(uid)
            l_rank = lexical_ranks.get(uid)
            
            score = 0.0
            if d_rank is not None:
                score += dense_w * (1.0 / (self.rrf_k + d_rank))
            if l_rank is not None:
                score += lexical_w * (1.0 / (self.rrf_k + l_rank))
                
            scored_hits.append((uid, score, d_rank, l_rank))
            
        scored_hits.sort(key=lambda x: x[1], reverse=True)
        top_scored = scored_hits[:top_k]
        
        if not top_scored:
            return []
            
        # 5. Fetch sidecar metadata
        final_uuids = [x[0] for x in top_scored]
        
        # 6. Apply deleted-row exclusion defensively
        final_rows = self.tier.sidecar.filter_candidates({}, final_uuids)
        row_map = {r["engram_uuid"]: r for r in final_rows}
        
        results = []
        for uid, score, d_rank, l_rank in top_scored:
            if uid not in row_map:
                continue 
            row = row_map[uid]
            
            explanation = {}
            if explain:
                explanation = {
                    "dense_rank": d_rank,
                    "lexical_rank": l_rank,
                    "dense_score": dense_scores.get(uid),
                    "lexical_score": lexical_scores.get(uid),
                    "policy": policy,
                    "dense_weight": dense_w,
                    "lexical_weight": lexical_w
                }
                
            results.append(FusionHit(
                engram_uuid=uid,
                score=score,
                dense_score=dense_scores.get(uid),
                lexical_score=lexical_scores.get(uid),
                dense_rank=d_rank,
                lexical_rank=l_rank,
                source_uri=row["source_uri"],
                content=row["content"],
                metadata=row["metadata_json"],
                governance=row["governance_json"],
                explanation=explanation
            ))
            
        return results
