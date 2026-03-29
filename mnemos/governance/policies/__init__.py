"""
BasePolicy — abstract interface all governance policies implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from mnemos.retrieval.base import SearchResult
    from mnemos.governance.models.governance_decision import GovernanceDecision


class BasePolicy(ABC):
    """Abstract base for all MNEMOS governance policies."""

    @property
    @abstractmethod
    def policy_name(self) -> str:
        """Unique name used to identify and disable this policy."""
        ...

    @abstractmethod
    def evaluate(
        self,
        result: "SearchResult",
        decision: "GovernanceDecision",
        context: Dict[str, Any],
    ) -> "GovernanceDecision":
        """
        Evaluate one search result and update its GovernanceDecision.

        Implementations should not raise — catch and log internally.

        Args:
            result:   Raw SearchResult from the retrieval tier.
            decision: Partially filled GovernanceDecision for this candidate.
            context:  Shared evaluation context dict.
                      Guaranteed keys: ``query``, ``all_candidate_ids``,
                      ``governance_mode``.

        Returns:
            Updated GovernanceDecision.
        """
        ...
