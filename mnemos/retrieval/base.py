"""
Base Retriever Interface
=========================

Abstract interface that all MNEMOS retrieval tiers implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mnemos.engram.model import Engram


@dataclass
class SearchResult:
    """A single search result from a retrieval tier."""
    engram: Engram
    score: float = 0.0
    tier: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract interface for a MNEMOS retrieval tier."""

    @property
    @abstractmethod
    def tier_name(self) -> str:
        """Unique identifier for this tier (e.g. 'qdrant', 'lancedb', 'colbert')."""
        ...

    @abstractmethod
    def index(self, engrams: List[Engram]) -> int:
        """
        Index a batch of engrams.

        Args:
            engrams: List of engrams to store.

        Returns:
            Number of engrams successfully indexed.
        """
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for relevant engrams.

        Args:
            query: Search query text.
            top_k: Maximum number of results.
            filters: Optional metadata filters.

        Returns:
            Ranked list of SearchResult objects.
        """
        ...

    @abstractmethod
    def delete(self, engram_ids: List[str]) -> int:
        """
        Delete engrams by ID.

        Args:
            engram_ids: List of engram IDs to remove.

        Returns:
            Number of engrams successfully deleted.
        """
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """
        Get tier statistics.

        Returns:
            Dict with index size, document count, etc.
        """
        ...

    def get(self, engram_id: str) -> Optional[Engram]:
        """Retrieve a specific engram by ID. Default: search-based lookup."""
        results = self.search(engram_id, top_k=1)
        for r in results:
            if r.engram.id == engram_id:
                return r.engram
        return None
