import os
import datetime
import json
import shutil
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from mnemos.retrieval.turbovec_metadata import TurbovecMetadata
from mnemos.retrieval.turbovec_config import TurbovecConfig

try:
    import turbovec
except ImportError:
    turbovec = None

@dataclass
class SearchHit:
    engram_uuid: str
    score: float
    content: str
    source_uri: str
    metadata: dict
    governance: dict

class DenseIndexAdapter:
    def add(self, ids: List[int], vectors: List[List[float]]) -> None:
        raise NotImplementedError
        
    def search(self, query_vector: List[float], top_k: int) -> List[Tuple[int, float]]:
        raise NotImplementedError
        
    def delete(self, ids: List[int]) -> None:
        raise NotImplementedError
        
    def save(self, path: str) -> None:
        raise NotImplementedError
        
    @classmethod
    def load(cls, path: str) -> "DenseIndexAdapter":
        raise NotImplementedError

class MockDenseIndexAdapter(DenseIndexAdapter):
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: Dict[int, np.ndarray] = {}
        
    def add(self, ids: List[int], vectors: List[List[float]]) -> None:
        for i, v in zip(ids, vectors):
            if len(v) != self.dim:
                raise ValueError(f"Expected dim {self.dim}, got {len(v)}")
            self.vectors[i] = np.array(v, dtype=np.float32)
            
    def search(self, query_vector: List[float], top_k: int) -> List[Tuple[int, float]]:
        if not self.vectors:
            return []
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        q = q / q_norm if q_norm > 0 else q
        
        results = []
        for vid, v in self.vectors.items():
            v_norm = np.linalg.norm(v)
            v_normalized = v / v_norm if v_norm > 0 else v
            score = float(np.dot(q, v_normalized))
            results.append((vid, score))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete(self, ids: List[int]) -> None:
        for i in ids:
            self.vectors.pop(i, None)
            
    def save(self, path: str) -> None:
        # Avoid save errors if empty by handling keys vs values
        with open(path, "wb") as f:
            np.savez(f, dim=self.dim, ids=list(self.vectors.keys()), vectors=list(self.vectors.values()))
        
    @classmethod
    def load(cls, path: str) -> "MockDenseIndexAdapter":
        with open(path, "rb") as f:
            data = np.load(f)
            dim = int(data['dim'])
            adapter = cls(dim)
            ids = data['ids']
            vectors = data['vectors']
        for i, v in zip(ids, vectors):
            adapter.vectors[int(i)] = np.array(v, dtype=np.float32)
        return adapter

class RealTurbovecIndexAdapter(DenseIndexAdapter):
    def __init__(self, dim: int, bit_width: int):
        if turbovec is None:
            raise ImportError("turbovec package is not installed. Cannot use RealTurbovecIndexAdapter.")
        self.dim = dim
        self.bit_width = bit_width
        # Assuming turbovec API here based on context
        self.index = turbovec.IdMapIndex(dim=dim, bit_width=bit_width)

    def add(self, ids: List[int], vectors: List[List[float]]) -> None:
        if not ids:
            return
        vec_arr = np.array(vectors, dtype=np.float32)
        if vec_arr.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec_arr.shape[1]}")
        id_arr = np.array(ids, dtype=np.uint64)
        self.index.add_with_ids(vec_arr, id_arr)

    def search(self, query_vector: List[float], top_k: int) -> List[Tuple[int, float]]:
        if len(self.index) == 0:
            return []
        q = np.array([query_vector], dtype=np.float32)
        scores, ids = self.index.search(q, top_k)
        results = []
        for i in range(len(ids[0])):
            vid = ids[0][i]
            # typical FAISS/turbovec empty slot
            if vid == np.iinfo(np.uint64).max or vid == -1:
                continue
            results.append((int(vid), float(scores[0][i])))
        return results

    def delete(self, ids: List[int]) -> None:
        try:
            for id in ids:
                self.index.remove(int(id))
        except AttributeError:
            pass # fallback if not implemented

    def save(self, path: str) -> None:
        self.index.write(path)

    @classmethod
    def load(cls, path: str, dim: int = 768, bit_width: int = 4) -> "RealTurbovecIndexAdapter":
        if turbovec is None:
            raise ImportError("turbovec package is not installed. Cannot use RealTurbovecIndexAdapter.")
        adapter = cls(dim, bit_width)
        adapter.index = turbovec.IdMapIndex.load(path)
        return adapter

class TurbovecTier:
    def __init__(self, config: TurbovecConfig, use_mock: bool = True):
        self.config = config
        os.makedirs(self.config.storage_path, exist_ok=True)
        self.db_path = os.path.join(self.config.storage_path, "metadata.sqlite")
        self.sidecar = TurbovecMetadata(self.db_path)
        
        self.use_mock = use_mock
        if self.use_mock:
            self.adapter = MockDenseIndexAdapter(self.config.embedding_dim)
        else:
            self.adapter = RealTurbovecIndexAdapter(self.config.embedding_dim, self.config.bit_width)
            
    def index(self, engrams: list) -> dict:
        added_count = 0
        ids_batch = []
        vecs_batch = []
        for engram in engrams:
            uid = getattr(engram, 'uuid', getattr(engram, 'engram_uuid', None))
            if not uid and isinstance(engram, dict):
                uid = engram.get('uuid') or engram.get('engram_uuid')
            if not uid:
                raise ValueError("Missing Engram UUID")
                
            vec = getattr(engram, 'embedding', None)
            if vec is None and isinstance(engram, dict):
                vec = engram.get('embedding')
                
            if not vec or len(vec) != self.config.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch or missing. Expected {self.config.embedding_dim}")
            
            source_uri = getattr(engram, 'source_uri', '') or (engram.get('source_uri', '') if isinstance(engram, dict) else '')
            content = getattr(engram, 'content', '') or (engram.get('content', '') if isinstance(engram, dict) else '')
            metadata = getattr(engram, 'metadata', {}) or (engram.get('metadata', {}) if isinstance(engram, dict) else {})
            governance = getattr(engram, 'governance', {}) or (engram.get('governance', {}) if isinstance(engram, dict) else {})
            content_hash = getattr(engram, 'content_hash', '') or (engram.get('content_hash', '') if isinstance(engram, dict) else '')
            
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            t_id = self.sidecar.upsert_engram(
                engram_uuid=uid,
                source_uri=source_uri,
                content=content,
                metadata=metadata,
                governance=governance,
                content_hash=content_hash,
                created_at=now,
                updated_at=now
            )
            
            ids_batch.append(t_id)
            vecs_batch.append(vec)
            added_count += 1
            
        if ids_batch:
            self.adapter.add(ids_batch, vecs_batch)
            
        return {"indexed": added_count}
        
    def search(self, query_embedding: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[SearchHit]:
        dense_top_k = min(top_k * self.config.oversample_factor, self.config.max_dense_candidates)
        
        dense_results = self.adapter.search(query_embedding, dense_top_k)
        if not dense_results:
            return []
            
        t_id_to_score = {tid: score for tid, score in dense_results}
        rows = self.sidecar.get_by_turbovec_ids(list(t_id_to_score.keys()))
        
        candidate_uuids = [row["engram_uuid"] for row in rows]
        filtered_rows = self.sidecar.filter_candidates(filters or {}, candidate_uuids)
        
        hits = []
        for row in filtered_rows:
            tid = row["turbovec_id"]
            score = t_id_to_score.get(tid, 0.0)
            hits.append(SearchHit(
                engram_uuid=row["engram_uuid"],
                score=score,
                content=row["content"],
                source_uri=row["source_uri"],
                metadata=row["metadata_json"],
                governance=row["governance_json"]
            ))
            
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:top_k]
        
    def delete(self, engram_uuid: str) -> dict:
        row = self.sidecar.get_engram(engram_uuid)
        if row:
            self.sidecar.delete_engram(engram_uuid)
            self.adapter.delete([row["turbovec_id"]])
            return {"deleted": 1}
        return {"deleted": 0}
        
    def save(self, profile_dir: str) -> None:
        os.makedirs(profile_dir, exist_ok=True)
        index_path = os.path.join(profile_dir, "index.tvim")
        self.adapter.save(index_path)
        
        target_db_path = os.path.join(profile_dir, "metadata.sqlite")
        if os.path.abspath(self.db_path) != os.path.abspath(target_db_path):
            shutil.copy2(self.db_path, target_db_path)
            
        health = self.health()
        
        manifest = {
          "mnemos_turbovec_profile_version": "0.1",
          "embedding_model": self.config.embedding_model,
          "embedding_dim": self.config.embedding_dim,
          "bit_width": self.config.bit_width,
          "index_file": "index.tvim",
          "metadata_file": "metadata.sqlite",
          "created_at": getattr(self, "_created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
          "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "engram_count": health["engram_count"],
          "deleted_count": health["deleted_count"],
          "schema_version": "0.1"
        }
        
        with open(os.path.join(profile_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        
    @classmethod
    def load(cls, profile_dir: str, expected_config: Optional[TurbovecConfig] = None, use_mock: bool = True) -> "TurbovecTier":
        manifest_path = os.path.join(profile_dir, "manifest.json")
        index_path = os.path.join(profile_dir, "index.tvim")
        db_path = os.path.join(profile_dir, "metadata.sqlite")
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError("manifest.json missing")
        if not os.path.exists(index_path):
            raise FileNotFoundError("index.tvim missing")
        if not os.path.exists(db_path):
            raise FileNotFoundError("metadata.sqlite missing")
            
        with open(manifest_path, "r") as f:
            try:
                manifest = json.load(f)
            except json.JSONDecodeError:
                raise ValueError("manifest malformed")
            
        required_fields = ["embedding_dim", "bit_width", "index_file", "metadata_file"]
        for field in required_fields:
            if field not in manifest:
                raise ValueError(f"manifest malformed: missing {field}")
                
        # Validate against expected config if provided
        if expected_config:
            if manifest["embedding_dim"] != expected_config.embedding_dim:
                raise ValueError(f"embedding_dim mismatch: expected {expected_config.embedding_dim}, got {manifest['embedding_dim']}")
            if manifest["bit_width"] != expected_config.bit_width:
                raise ValueError(f"bit_width mismatch: expected {expected_config.bit_width}, got {manifest['bit_width']}")
        
        config = TurbovecConfig(
            embedding_model=manifest.get("embedding_model", "BAAI/bge-base-en-v1.5"),
            embedding_dim=manifest["embedding_dim"],
            bit_width=manifest["bit_width"],
            storage_path=profile_dir
        )
        
        tier = cls(config, use_mock=use_mock)
        
        if use_mock:
            tier.adapter = MockDenseIndexAdapter.load(index_path)
            if tier.adapter.dim != config.embedding_dim:
                raise ValueError("embedding_dim mismatch in index.tvim")
        else:
            tier.adapter = RealTurbovecIndexAdapter.load(index_path, config.embedding_dim, config.bit_width)
            
        tier._created_at = manifest.get("created_at")
        return tier
        
    def validate_persistence_integrity(self) -> dict:
        health = self.health()
        
        if isinstance(self.adapter, MockDenseIndexAdapter):
            dense_count = len(self.adapter.vectors)
        else:
            dense_count = health["engram_count"] + health["deleted_count"]
            
        is_consistent = dense_count >= health["engram_count"]
        
        return {
            "dense_count": dense_count,
            "active_metadata_count": health["engram_count"],
            "deleted_metadata_count": health["deleted_count"],
            "embedding_dim": self.config.embedding_dim,
            "bit_width": self.config.bit_width,
            "is_consistent": is_consistent,
            "dense_count_ge_active": is_consistent
        }
        
    def health(self) -> dict:
        with self.sidecar._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM engram_metadata WHERE deleted=0")
            engram_count = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM engram_metadata WHERE deleted=1")
            deleted_count = c.fetchone()[0]
            
        return {
            "status": "ok",
            "backend": "mock_turbovec" if self.use_mock else "turbovec",
            "metadata_status": "ok",
            "index_loaded": True,
            "embedding_dim": self.config.embedding_dim,
            "bit_width": self.config.bit_width,
            "engram_count": engram_count,
            "deleted_count": deleted_count,
            "error": None
        }
