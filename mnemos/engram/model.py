"""
Engram — Universal Memory Unit
================================

The Engram is the atomic unit of knowledge in MNEMOS. It wraps a raw
document chunk with machine-generated enrichments that make retrieval
smarter and operations auditable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from mnemos.governance.models.memory_state import GovernanceMeta


@dataclass
class Engram:
    """
    Enriched memory fragment — the core storage unit of MNEMOS.

    Attributes:
        id: Unique identifier (UUID).
        content: Raw text content.
        embedding: Dense vector representation (quantised on disk).
        neuro_tags: Auto-generated semantic labels for retrieval boosting.
        source: Provenance URI — where this data came from.
        confidence: Quality signal (0.0–1.0) used for result ranking.
        created_at: ISO-format ingestion timestamp.
        metadata: Extensible application-specific key-value data.
        edges: IDs of related engrams (knowledge graph links).
        governance: Optional governance metadata (lifecycle, trust, lineage).
                    Populated by the governance layer; None for legacy engrams.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    neuro_tags: List[str] = field(default_factory=list)
    source: str = ""
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)
    edges: List[str] = field(default_factory=list)
    governance: Optional[GovernanceMeta] = field(default=None, repr=False)

    def to_dict(
        self,
        include_embedding: bool = False,
        include_governance: bool = False,
    ) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Args:
            include_embedding:  Include raw embedding vector (large; avoid in
                                API responses).
            include_governance: Include ``_governance`` block.  Off by default
                                to keep storage payloads lean; pass True when
                                explicitly persisting governance state.
        """
        d = {
            "id": self.id,
            "content": self.content,
            "neuro_tags": self.neuro_tags,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "edges": self.edges,
        }
        if include_embedding and self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        if include_governance and self.governance is not None:
            d["_governance"] = self.governance.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Engram":
        """Deserialise from a dictionary."""
        embedding = None
        if "embedding" in data and data["embedding"] is not None:
            embedding = np.array(data["embedding"], dtype=np.float32)

        governance = None
        if "_governance" in data and data["_governance"] is not None:
            governance = GovernanceMeta.from_dict(data["_governance"])

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            embedding=embedding,
            neuro_tags=data.get("neuro_tags", []),
            source=data.get("source", ""),
            confidence=data.get("confidence", 1.0),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
            metadata=data.get("metadata", {}),
            edges=data.get("edges", []),
            governance=governance,
        )

    def add_tag(self, tag: str) -> None:
        """Add a neuro-tag if not already present."""
        if tag not in self.neuro_tags:
            self.neuro_tags.append(tag)

    def link(self, other_id: str) -> None:
        """Create a relationship edge to another engram."""
        if other_id not in self.edges:
            self.edges.append(other_id)


@dataclass
class EngramBatch:
    """A batch of engrams for bulk operations."""

    engrams: List[Engram] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.engrams)

    def __iter__(self):
        return iter(self.engrams)

    def add(self, engram: Engram) -> None:
        self.engrams.append(engram)

    @property
    def ids(self) -> List[str]:
        return [e.id for e in self.engrams]

    @property
    def contents(self) -> List[str]:
        return [e.content for e in self.engrams]

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Stack all embeddings into a single array if available."""
        embs = [e.embedding for e in self.engrams if e.embedding is not None]
        if not embs:
            return None
        return np.vstack(embs)

    def to_dicts(self, include_embedding: bool = False) -> List[Dict[str, Any]]:
        return [e.to_dict(include_embedding=include_embedding) for e in self.engrams]
