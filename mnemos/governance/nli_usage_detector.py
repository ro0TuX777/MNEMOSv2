"""
NLIUsageDetector — entailment-grounded usage classification for the reflect path.

Phase 6 (W3 Reflect Precision): replaces the lexical word-overlap signal of
``UsageDetector`` with a stateless NLI cross-encoder. All higher-priority
signals (veto, contradiction state, explicit citation) are preserved unchanged
so read-path determinism guarantees are not affected.

Detection signals (in order of confidence)
-------------------------------------------
1. Explicit veto / contradiction state on the GovernanceDecision
   (VETOED / CONTRADICTED — identical to the lexical detector; an NLI
   contradiction signal NEVER overrides read-path conflict state)
2. Explicit citation — memory ID appears in the ``cited_ids`` set
3. Entailment — max P(entailment) over bidirectional
   (answer sentence ↔ memory content) pairs meets ``entailment_threshold``
4. Default — IGNORED when no signal fires

Direction note
--------------
Entailment is scored in both directions and the max is taken. A used memory
either restates an answer span (answer sentence entails memory) or is the
more-specific source the answer span was derived from (memory entails answer
sentence). Scoring only one direction systematically penalises one of the two
cases. Memories below ``min_memory_tokens_for_nli`` tokens are labeled IGNORED
deterministically — fragment/noun-phrase hypotheses are unreliable under NLI
(mirrors the lexical detector's token-floor guard).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from mnemos.retrieval.base import SearchResult
from mnemos.governance.models.governance_decision import GovernanceDecision
from mnemos.governance.usage_detector import UsageLabel

logger = logging.getLogger(__name__)

# Label order for cross-encoder/nli-deberta-v3-* checkpoints:
# index 0 = contradiction, 1 = entailment, 2 = neutral
_CONTRADICTION_INDEX = 0
_ENTAILMENT_INDEX = 1

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str, max_sentences: int, min_chars: int = 10) -> List[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) >= min_chars]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences[:max_sentences]


class NLIUsageDetector:
    """
    Entailment-grounded used-memory detector.

    Args:
        model_name:
            HuggingFace cross-encoder NLI checkpoint. Default is the smallest
            DeBERTa-v3 NLI model that shares the GPU runtime with the rerank lane.
        entailment_threshold:
            Minimum P(entailment) for a memory to be classified ``used``.
        max_answer_sentences:
            Cap on answer sentences scored per reflect cycle (bounds pair count).
        min_memory_tokens_for_nli:
            Memories with fewer alphanumeric tokens are labeled IGNORED without
            NLI inference (noun-phrase fragments are unreliable hypotheses).
        max_content_chars:
            Memory content is truncated to this length before pairing.
        device:
            Torch device override ("cuda" / "cpu"); auto-detected when None.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        entailment_threshold: float = 0.5,
        max_answer_sentences: int = 24,
        max_content_chars: int = 1000,
        min_memory_tokens_for_nli: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self._threshold = entailment_threshold
        self._max_answer_sentences = max(1, int(max_answer_sentences))
        self._max_content_chars = max(100, int(max_content_chars))
        self._min_memory_tokens = max(1, int(min_memory_tokens_for_nli))
        self._device = device
        self._model = None
        self.last_cycle_ms: Optional[float] = None

    # ── Model lifecycle ───────────────────────────────────────────────────

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            if self._device:
                self._model = CrossEncoder(self.model_name, device=self._device)
            else:
                self._model = CrossEncoder(self.model_name)
            logger.info(f"✅ NLI usage detector initialized: {self.model_name}")
        return self._model

    @property
    def resolved_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def health(self) -> bool:
        try:
            self._ensure_model()
            return True
        except Exception as exc:
            logger.error(f"NLI usage detector unavailable: {exc}")
            return False

    # ── Public API ────────────────────────────────────────────────────────

    def detect(
        self,
        query: str,
        answer: str,
        results: List[SearchResult],
        decisions: List[GovernanceDecision],
        cited_ids: Optional[List[str]] = None,
    ) -> Dict[str, UsageLabel]:
        labels, _ = self.detect_with_scores(query, answer, results, decisions, cited_ids)
        return labels

    def detect_with_scores(
        self,
        query: str,
        answer: str,
        results: List[SearchResult],
        decisions: List[GovernanceDecision],
        cited_ids: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, UsageLabel], Dict[str, Dict[str, float]]]:
        """
        Classify each candidate and return per-memory NLI probabilities.

        Returns:
            (labels, scores) where ``scores[engram_id]`` is
            ``{"entailment": float, "contradiction": float}`` for every
            candidate that reached the NLI stage. The contradiction
            probability is telemetry only — it does not change the label.
        """
        started = time.perf_counter()
        cited_set: frozenset = frozenset(cited_ids or [])
        decision_map = {d.engram_id: d for d in decisions}

        labels: Dict[str, UsageLabel] = {}
        scores: Dict[str, Dict[str, float]] = {}
        nli_candidates: List[SearchResult] = []

        for result in results:
            eid = result.engram.id
            decision = decision_map.get(eid)

            if decision is None:
                labels[eid] = UsageLabel.UNKNOWN
                continue
            if not decision.veto_pass:
                labels[eid] = UsageLabel.VETOED
                continue
            if decision.suppressed_by_contradiction:
                labels[eid] = UsageLabel.CONTRADICTED
                continue
            if eid in cited_set:
                labels[eid] = UsageLabel.USED
                continue
            if len(re.findall(r"\b[a-z0-9]+\b", result.engram.content.lower())) < self._min_memory_tokens:
                labels[eid] = UsageLabel.IGNORED
                scores[eid] = {"entailment": 0.0, "contradiction": 0.0}
                continue
            nli_candidates.append(result)

        if nli_candidates:
            sentences = _split_sentences(answer, self._max_answer_sentences)
            pairs: List[List[str]] = []
            spans: List[Tuple[str, int]] = []  # (engram_id, pair_count)
            for result in nli_candidates:
                content = result.engram.content[: self._max_content_chars]
                spans.append((result.engram.id, 2 * len(sentences)))
                for sentence in sentences:
                    pairs.append([sentence, content])
                    pairs.append([content, sentence])

            model = self._ensure_model()
            logits = np.asarray(model.predict(pairs, apply_softmax=False))
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)

            offset = 0
            for eid, count in spans:
                block = probs[offset : offset + count]
                offset += count
                entail = float(block[:, _ENTAILMENT_INDEX].max()) if len(block) else 0.0
                contra = float(block[:, _CONTRADICTION_INDEX].max()) if len(block) else 0.0
                scores[eid] = {"entailment": entail, "contradiction": contra}
                labels[eid] = UsageLabel.USED if entail >= self._threshold else UsageLabel.IGNORED

        self.last_cycle_ms = (time.perf_counter() - started) * 1000.0
        return labels, scores
