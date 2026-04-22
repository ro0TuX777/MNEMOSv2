"""
Cross-Encoder Reranker Tier
===========================

Late-interaction precision rescoring using HuggingFace CrossEncoders.
"""

import logging
from typing import List, Optional

from mnemos.retrieval.base import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None
        self._initialized = False

    def _initialize(self):
        if not self._initialized:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                self._initialized = True
                logger.info(f"✅ Reranker initialized: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load reranker model '{self.model_name}': {e}")
                raise RuntimeError(f"Reranker unavailable: {e}") from e

    def rerank(
        self, query: str, results: List[SearchResult], top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank a list of SearchResults using the cross-encoder.
        """
        if not results:
            return []

        self._initialize()

        if self._model is None:
            logger.warning("Reranker model is not loaded. Returning un-ranked results.")
            return results[:top_k] if top_k else results

        # Prepare sentence pairs for the cross-encoder
        sentence_pairs = [[query, r.engram.content] for r in results]

        try:
            # Predict scores for each pair
            scores = self._model.predict(sentence_pairs)

            # Update the scores and re-sort
            for idx, r in enumerate(results):
                # Optionally add the original score to metadata before overwriting
                r.metadata["original_retrieval_score"] = r.score
                r.score = float(scores[idx])
                if "tiers" not in r.metadata:
                    r.metadata["tiers"] = [r.tier]
                if "reranker" not in r.metadata["tiers"]:
                    r.metadata["tiers"].append("reranker")

            # Sort by new cross-encoder score in descending order
            results.sort(key=lambda x: x.score, reverse=True)

            return results[:top_k] if top_k else results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original results if reranking fails
            return results[:top_k] if top_k else results
