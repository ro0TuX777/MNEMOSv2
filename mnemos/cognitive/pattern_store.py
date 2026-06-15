"""
PatternCandidateStore — Phase 18 of the PatternEngramCandidate extraction harness.

An in-memory advisory store for PatternEngramCandidate objects with optional
JSON persistence.  This store is intentionally separate from the main Qdrant /
pgvector engram index — candidates are advisory-only and must never be inserted
into the retrieval path without explicit governance promotion.

Advisory recall uses Jaccard similarity on the ``applies_when`` situation text
(set by PatternLearner from the SituationSummary).  No embedding calls.

Persistence is a single JSON file; the store is small by design (pattern
candidates accumulate slowly from offline evaluation runs).

Safety invariants
-----------------
- Candidates stored here are NEVER inserted into the main engram index.
- ``load()`` validates that no candidate has ``authoritative_for_retrieval=True``.
- ``find_relevant()`` only returns candidates with status in
  {PROMOTION_CANDIDATE, PROMOTION_RECOMMENDED}.  Approved / rejected
  candidates are excluded from advisory recall.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from mnemos.cognitive.pattern_engram import (
    PatternEngramCandidate,
    PROMOTION_CANDIDATE,
    PROMOTION_RECOMMENDED,
    PROMOTION_APPROVED,
    PROMOTION_REJECTED,
)
from mnemos.cognitive.pattern_learner import _jaccard
from mnemos.cognitive.promoted_pattern import PatternEngram


_ADVISORY_STATUSES = frozenset({PROMOTION_CANDIDATE, PROMOTION_RECOMMENDED})


def _token_set(text: str) -> frozenset:
    """Lowercase token set for Jaccard similarity."""
    return frozenset(text.lower().split())


class PatternCandidateStore:
    """
    Advisory-only store for PatternEngramCandidate objects.

    Usage::

        store = PatternCandidateStore(persist_path="pattern_candidates.json")
        store.load()                      # no-op if file absent
        store.add(candidate)
        relevant = store.find_relevant("a CLASS_B search cycle ...", top_k=3)
        store.save()

    Thread safety: not thread-safe.  Wrap with a lock if concurrent access
    is needed; the store is designed for single-process offline use.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._persist_path = persist_path
        self._candidates: Dict[str, PatternEngramCandidate] = {}
        # Pre-computed token sets for fast Jaccard recall
        self._token_sets: Dict[str, frozenset] = {}
        # Promoted engrams (Phase 20)
        self._promoted: Dict[str, PatternEngram] = {}

    # ── Write API ─────────────────────────────────────────────────────────────

    def add(self, candidate: PatternEngramCandidate) -> str:
        """
        Add a candidate to the store.  Returns the candidate_id.

        Raises ValueError if a candidate with the same ID already exists.
        Use update() to replace an existing entry.
        """
        cid = candidate.candidate_id
        if cid in self._candidates:
            raise ValueError(
                f"Candidate {cid!r} already exists. Use update() to replace."
            )
        self._candidates[cid] = candidate
        self._token_sets[cid] = _token_set(candidate.applies_when)
        return cid

    def update(self, candidate_id: str, candidate: PatternEngramCandidate) -> None:
        """
        Replace an existing candidate entry.

        Raises KeyError if candidate_id is not in the store.
        """
        if candidate_id not in self._candidates:
            raise KeyError(f"Candidate {candidate_id!r} not found in store.")
        self._candidates[candidate_id] = candidate
        self._token_sets[candidate_id] = _token_set(candidate.applies_when)

    # ── Read API ──────────────────────────────────────────────────────────────

    def get(self, candidate_id: str) -> Optional[PatternEngramCandidate]:
        """Return the candidate for the given ID, or None."""
        return self._candidates.get(candidate_id)

    def list_by_status(self, status: str) -> List[PatternEngramCandidate]:
        """Return all candidates matching the given promotion_status."""
        return [c for c in self._candidates.values() if c.promotion_status == status]

    def list_by_pattern_type(self, pattern_type: str) -> List[PatternEngramCandidate]:
        """Return all candidates with the given pattern_type."""
        return [c for c in self._candidates.values() if c.pattern_type == pattern_type]

    def list_all(self) -> List[PatternEngramCandidate]:
        """Return all candidates in insertion order."""
        return list(self._candidates.values())

    def count(self) -> int:
        """Total number of candidates in the store."""
        return len(self._candidates)

    def count_by_status(self) -> Dict[str, int]:
        """Count candidates grouped by promotion_status."""
        counts: Dict[str, int] = {
            PROMOTION_CANDIDATE: 0,
            PROMOTION_RECOMMENDED: 0,
            PROMOTION_APPROVED: 0,
            PROMOTION_REJECTED: 0,
        }
        for c in self._candidates.values():
            if c.promotion_status in counts:
                counts[c.promotion_status] += 1
        return counts

    # ── Promotion lifecycle (Phase 20) ───────────────────────────────────────

    def recommend(
        self,
        candidate_id: str,
        *,
        gate_id: str,
        confidence_threshold: float = 0.0,
    ) -> PatternEngramCandidate:
        """
        Mark a candidate as promotion_recommended.

        Calls ``candidate.recommend_promotion()`` in-place.

        Raises
        ------
        KeyError
            If ``candidate_id`` is not in the store.
        """
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(f"Candidate {candidate_id!r} not found.")
        candidate.recommend_promotion(gate_id=gate_id, confidence_threshold=confidence_threshold)
        return candidate

    def reject(self, candidate_id: str) -> PatternEngramCandidate:
        """
        Mark a candidate as rejected.

        Raises
        ------
        KeyError
            If ``candidate_id`` is not in the store.
        """
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(f"Candidate {candidate_id!r} not found.")
        candidate.promotion_status = PROMOTION_REJECTED
        return candidate

    def promote(
        self,
        candidate_id: str,
        *,
        governance_review_id: str,
        ledger_ref: Optional[str] = None,
    ) -> PatternEngram:
        """
        Approve a candidate and create the corresponding PatternEngram.

        The candidate must already be in ``promotion_recommended`` state.
        Calls ``approve_promotion()`` on the candidate, then constructs a
        PatternEngram via ``PatternEngram.from_approved_candidate()``.

        Parameters
        ----------
        candidate_id        : ID of the candidate to promote.
        governance_review_id: Governance review identifier (mandatory).
        ledger_ref          : Optional forensic ledger transaction ID to
                              record against the promotion.

        Raises
        ------
        KeyError
            If ``candidate_id`` is not in the store.
        PermissionError
            If the candidate is not in ``promotion_recommended`` state, or if
            ``PatternEngram.from_approved_candidate()`` rejects the candidate.
        """
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(f"Candidate {candidate_id!r} not found.")
        if candidate.promotion_status != PROMOTION_RECOMMENDED:
            raise PermissionError(
                f"Candidate {candidate_id!r} must be in promotion_recommended state "
                f"before approval (current: {candidate.promotion_status!r})."
            )
        candidate.approve_promotion(
            governance_review_id=governance_review_id,
            explicit_approval=True,
        )
        engram = PatternEngram.from_approved_candidate(
            candidate,
            governance_review_id=governance_review_id,
        )
        self._promoted[engram.pattern_id] = engram
        return engram

    def list_promoted(self) -> List[PatternEngram]:
        """Return all promoted PatternEngram objects in insertion order."""
        return list(self._promoted.values())

    def get_promoted(self, pattern_id: str) -> Optional[PatternEngram]:
        """Return a promoted PatternEngram by pattern_id, or None."""
        return self._promoted.get(pattern_id)

    # ── Advisory recall ───────────────────────────────────────────────────────

    def find_relevant(
        self,
        situation_text: str,
        *,
        top_k: int = 3,
        min_confidence: float = 0.0,
    ) -> List[PatternEngramCandidate]:
        """
        Return the top_k most relevant advisory candidates for situation_text.

        Relevance is Jaccard similarity between situation_text tokens and
        each candidate's applies_when tokens.  Only candidates with
        promotion_status in {candidate, promotion_recommended} are considered.
        Candidates below min_confidence are excluded.

        Returns an empty list when the store is empty or no candidates
        meet the filters.
        """
        if not self._candidates:
            return []

        query_tokens = _token_set(situation_text)
        scored: List[Tuple[float, PatternEngramCandidate]] = []

        for cid, candidate in self._candidates.items():
            if candidate.promotion_status not in _ADVISORY_STATUSES:
                continue
            if candidate.confidence_score < min_confidence:
                continue
            sim = _jaccard(query_tokens, self._token_sets.get(cid, frozenset()))
            scored.append((sim, candidate))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Persist the store to the configured persist_path as JSON.

        No-op if persist_path is None.
        """
        if not self._persist_path:
            return
        data = self.to_dict()
        tmp = self._persist_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, self._persist_path)

    def load(self) -> None:
        """
        Load candidates from the persist_path JSON file.

        No-op if the file does not exist.
        Raises ValueError if any loaded candidate has authoritative_for_retrieval=True
        (safety invariant violation — store file was corrupted or tampered).
        """
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        with open(self._persist_path, encoding="utf-8") as fh:
            data = json.load(fh)
        self._load_from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full store to a JSON-compatible dict."""
        return {
            "schema_version": "2",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "count": len(self._candidates),
            "candidates": [c.to_dict() for c in self._candidates.values()],
            "promoted_count": len(self._promoted),
            "promoted": [e.to_dict() for e in self._promoted.values()],
        }

    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        """Reconstruct candidates and promoted engrams from a to_dict() output."""
        self._candidates.clear()
        self._token_sets.clear()
        self._promoted.clear()
        for raw_engram in data.get("promoted", []):
            try:
                engram = PatternEngram.from_dict(raw_engram)
                self._promoted[engram.pattern_id] = engram
            except Exception:
                pass  # Tolerate bad promoted entries — candidates are primary
        for raw in data.get("candidates", []):
            # Safety check before reconstruction
            if raw.get("authoritative_for_retrieval") is True:
                raise ValueError(
                    f"Store integrity violation: candidate {raw.get('candidate_id')!r} "
                    "has authoritative_for_retrieval=True. Store file may be corrupted."
                )
            candidate = _candidate_from_dict(raw)
            self._candidates[candidate.candidate_id] = candidate
            self._token_sets[candidate.candidate_id] = _token_set(candidate.applies_when)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], persist_path: Optional[str] = None) -> "PatternCandidateStore":
        """Construct a store from a to_dict() output."""
        store = cls(persist_path=persist_path)
        store._load_from_dict(data)
        return store


# ── Candidate reconstruction ──────────────────────────────────────────────────


def _candidate_from_dict(d: Dict[str, Any]) -> PatternEngramCandidate:
    """Reconstruct a PatternEngramCandidate from a to_dict() output."""
    from mnemos.cognitive.learning_boundary import SEMANTIC_CANDIDATE_WRITE

    return PatternEngramCandidate(
        pattern_summary=d.get("pattern_summary", ""),
        supporting_cycle_ids=d.get("supporting_cycle_ids", []),
        supporting_engram_ids=d.get("supporting_engram_ids", []),
        contradicting_engram_ids=d.get("contradicting_engram_ids", []),
        confidence_score=float(d.get("confidence_score", 0.0)),
        support_score=float(d.get("support_score", 0.0)),
        contradiction_score=float(d.get("contradiction_score", 0.0)),
        pattern_type=d.get("pattern_type", "descriptive"),
        recommended_scope=d.get("recommended_scope", "local"),
        applies_when=d.get("applies_when", ""),
        does_not_apply_when=d.get("does_not_apply_when", ""),
        risk_if_wrong=d.get("risk_if_wrong", "unknown"),
        proposed_learning_class=d.get("proposed_learning_class", SEMANTIC_CANDIDATE_WRITE),
        candidate_id=d.get("candidate_id", ""),
        promotion_status=d.get("promotion_status", PROMOTION_CANDIDATE),
        governance_review_required=bool(d.get("governance_review_required", True)),
        first_seen_at=d.get("first_seen_at", ""),
        last_seen_at=d.get("last_seen_at", ""),
        governance_review_id=d.get("governance_review_id"),
    )
