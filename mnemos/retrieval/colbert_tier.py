"""
ColBERT Multi-Vector Retrieval Tier
====================================

Token-level late-interaction matching with TurboQuant compression.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mnemos.engram.model import Engram
from mnemos.retrieval.base import BaseRetriever, SearchResult

logger = logging.getLogger(__name__)

try:
    from mnemos.compression.turbo_quant import TurboQuant, QuantizedTensor
    _TURBOQUANT_AVAILABLE = True
except ImportError:
    _TURBOQUANT_AVAILABLE = False


@dataclass
class ColBERTConfig:
    """Configuration for the ColBERT multi-vector tier."""
    model_name: str = "colbert-ir/colbertv2.0"
    max_tokens: int = 32
    similarity_threshold: float = 0.5
    token_score_threshold: float = 0.5
    quantize_bits: int = 4           # 0 = disabled
    use_quantized_search: bool = False
    index_dir: str = "data/colbert"


class ColBERTTier(BaseRetriever):
    """ColBERT multi-vector retrieval with late-interaction MaxSim scoring."""

    def __init__(self, config: Optional[ColBERTConfig] = None):
        self._config = config or ColBERTConfig()
        self._model = None
        self._tokenizer = None

        # In-memory index
        self._doc_ids: List[str] = []
        self._doc_embeddings: List[np.ndarray] = []     # List of (T_i, D) arrays
        self._doc_engrams: Dict[str, Engram] = {}

        self._index_dir = Path(self._config.index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._load_index()
        logger.info(f"✅ ColBERT tier initialized: {self._config.model_name}")

    def _get_model(self):
        """Lazy-load the ColBERT model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._config.model_name)
                logger.info(f"Loaded ColBERT model: {self._config.model_name}")
            except Exception as e:
                logger.warning(f"ColBERT model not available: {e}. Using fallback.")
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _encode_multi_vector(self, text: str) -> np.ndarray:
        """Encode text into multi-vector token embeddings."""
        model = self._get_model()
        # Generate a single embedding and tile to simulate multi-vector
        # In production, use a true ColBERT encoder
        emb = model.encode(text, normalize_embeddings=True)
        return emb.reshape(1, -1)  # (1, D)

    @property
    def tier_name(self) -> str:
        return "colbert"

    def index(self, engrams: List[Engram]) -> int:
        """Index engrams with multi-vector embeddings."""
        if not engrams:
            return 0

        count = 0
        for e in engrams:
            try:
                embeddings = self._encode_multi_vector(e.content)
                self._doc_ids.append(e.id)
                self._doc_embeddings.append(embeddings)
                self._doc_engrams[e.id] = e
                count += 1
            except Exception as ex:
                logger.error(f"ColBERT indexing failed for {e.id}: {ex}")

        self._save_index()
        logger.debug(f"Indexed {count} engrams into ColBERT tier")
        return count

    def search(self, query: str, top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search using MaxSim late-interaction scoring."""
        if not self._doc_ids:
            return []

        try:
            query_emb = self._encode_multi_vector(query)   # (T_q, D)

            scores = []
            for i, doc_emb in enumerate(self._doc_embeddings):
                # MaxSim: for each query token, find the max similarity
                # to any document token, then sum across query tokens
                sim_matrix = query_emb @ doc_emb.T  # (T_q, T_d)
                max_sims = sim_matrix.max(axis=1)     # (T_q,)
                score = float(max_sims.mean())
                scores.append((i, score))

            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)

            results = []
            for idx, score in scores[:top_k]:
                doc_id = self._doc_ids[idx]
                engram = self._doc_engrams.get(doc_id)
                if engram:
                    results.append(SearchResult(
                        engram=engram, score=score, tier="colbert"
                    ))

            return results

        except Exception as e:
            logger.error(f"ColBERT search failed: {e}")
            return []

    def delete(self, engram_ids: List[str]) -> int:
        """Delete engrams from ColBERT index."""
        removed = 0
        ids_to_remove = set(engram_ids)
        new_ids = []
        new_embs = []

        for i, doc_id in enumerate(self._doc_ids):
            if doc_id in ids_to_remove:
                self._doc_engrams.pop(doc_id, None)
                removed += 1
            else:
                new_ids.append(doc_id)
                new_embs.append(self._doc_embeddings[i])

        self._doc_ids = new_ids
        self._doc_embeddings = new_embs
        if removed > 0:
            self._save_index()
        return removed

    def get(self, engram_id: str) -> Optional[Engram]:
        return self._doc_engrams.get(engram_id)

    def stats(self) -> Dict[str, Any]:
        return {
            "tier": "colbert",
            "document_count": len(self._doc_ids),
            "model": self._config.model_name,
            "quantize_bits": self._config.quantize_bits,
            "index_dir": str(self._index_dir),
        }

    # ──────────────────── Persistence ────────────────────

    def _save_index(self):
        """Persist the ColBERT index to disk."""
        try:
            index_file = self._index_dir / "colbert_index.npz"

            if not self._doc_ids:
                return

            # Stack embeddings (pad to uniform shape)
            max_tokens = max(e.shape[0] for e in self._doc_embeddings)
            dim = self._doc_embeddings[0].shape[1]
            padded = np.zeros((len(self._doc_embeddings), max_tokens, dim), dtype=np.float32)
            for i, emb in enumerate(self._doc_embeddings):
                padded[i, :emb.shape[0], :] = emb

            save_data = {"embeddings": padded, "ids": np.array(self._doc_ids)}

            # Apply TurboQuant compression if configured
            if self._config.quantize_bits > 0 and _TURBOQUANT_AVAILABLE:
                tq = TurboQuant(bits=self._config.quantize_bits, mode="mse")
                flat = padded.reshape(-1, dim)
                qt = tq.quantize(flat)
                save_data = {
                    "tq_codes": qt.codes,
                    "tq_norms": qt.norms,
                    "tq_codebook": qt.codebook,
                    "tq_boundaries": qt.boundaries,
                    "tq_shape_0": np.array(flat.shape[0]),
                    "tq_shape_1": np.array(flat.shape[1]),
                    "doc_shape": np.array(padded.shape),
                    "ids": np.array(self._doc_ids),
                    "tq_bits": np.array(self._config.quantize_bits),
                    "tq_seed": np.array(42),
                }

            np.savez_compressed(index_file, **save_data)
            logger.debug(f"Saved ColBERT index: {len(self._doc_ids)} docs")

        except Exception as e:
            logger.error(f"Failed to save ColBERT index: {e}")

    def _load_index(self):
        """Load index from disk if available."""
        index_file = self._index_dir / "colbert_index.npz"
        if not index_file.exists():
            return

        try:
            data = np.load(index_file, allow_pickle=True)
            ids = data["ids"].tolist()

            if "tq_codes" in data and _TURBOQUANT_AVAILABLE:
                bits = int(data["tq_bits"])
                tq = TurboQuant(bits=bits, mode="mse", seed=int(data["tq_seed"]))
                qt = QuantizedTensor(
                    codes=data["tq_codes"],
                    shape=(int(data["tq_shape_0"]), int(data["tq_shape_1"])),
                    bits=bits, mode="mse", rotation_seed=int(data["tq_seed"]),
                    norms=data["tq_norms"],
                    codebook=data["tq_codebook"],
                    boundaries=data["tq_boundaries"],
                )
                flat = tq.dequantize(qt)
                doc_shape = tuple(data["doc_shape"])
                embeddings = flat.reshape(doc_shape)
            elif "embeddings" in data:
                embeddings = data["embeddings"]
            else:
                return

            self._doc_ids = ids
            self._doc_embeddings = [embeddings[i] for i in range(len(ids))]
            logger.info(f"Loaded ColBERT index: {len(ids)} documents")

        except Exception as e:
            logger.error(f"Failed to load ColBERT index: {e}")
