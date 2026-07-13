"""Original lexical retrieval and conservative disclosure policy for G2.

This module operates only on verified G1 projections. It has no MNEMOS runtime,
GateMem, network, model, or provider imports.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from prototype.gatemem_g1 import (
    CleanInputProjection,
    DisclosureResult,
    RetrievedArtifact,
    normalize_prediction,
    observe_shadow,
)

from .models import G2Diagnostic, G2Result

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "before", "but",
        "by", "can", "could", "did", "do", "does", "for", "from", "give", "had",
        "has", "have", "he", "her", "here", "him", "his", "i", "in", "is", "it",
        "just", "me", "my", "now", "of", "on", "or", "our", "please", "right",
        "she", "should", "so", "tell", "that", "the", "their", "them", "there",
        "they", "this", "to", "us", "was", "we", "were", "what", "when", "where",
        "which", "who", "why", "will", "with", "would", "you", "your",
    }
)
_PRINCIPAL_PREFIXES = frozenset(
    {
        "patient", "family", "clinician", "nurse", "pharmacist", "reception",
        "scheduler", "billing", "social", "worker", "student", "professor",
        "advisor", "registrar", "parent", "guardian", "resident", "primary",
        "partner", "spouse", "adult", "child", "guest", "manager", "employee",
        "contractor", "engineer", "security", "finance", "legal", "sales", "sre",
        "executive", "assistant", "agent", "staff", "operator",
    }
)


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN.findall(text or "")
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]


@dataclass(frozen=True)
class G2AdapterConfig:
    top_k: int = 8
    minimum_score: float = 0.08
    max_disclosed_records: int = 4
    max_answer_characters: int = 2200


@dataclass(frozen=True)
class _Candidate:
    record_id: str
    text: str
    turn_id: str
    turn_index: int
    speaker_principal_id: str
    speaker_role: str
    score: float
    agent_reply_to_requester: bool
    visibly_about_requester: bool


class OfflineGovernedAdapter:
    def __init__(self, config: G2AdapterConfig | None = None) -> None:
        self.config = config or G2AdapterConfig()
        if self.config.top_k < 1 or self.config.max_disclosed_records < 1:
            raise ValueError("G2 retrieval and disclosure limits must be positive.")

    @staticmethod
    def _requester_aliases(projection: CleanInputProjection) -> set[str]:
        tokens = [
            token
            for token in projection.requester.principal_id.lower().split("_")
            if len(token) >= 4 and token not in _PRINCIPAL_PREFIXES
        ]
        return set(tokens[-2:])

    @staticmethod
    def _metadata_set(projection: CleanInputProjection, key: str) -> set[str]:
        value = projection.permitted_metadata.get(key, [])
        if not isinstance(value, (list, tuple, set)):
            return set()
        return {str(item) for item in value if str(item)}

    def _retrieve(self, projection: CleanInputProjection) -> list[_Candidate]:
        query_tokens = _tokens(projection.query_text)
        if not query_tokens:
            return []
        document_tokens = [_tokens(turn.text) for turn in projection.ordered_turns]
        document_frequency: Counter[str] = Counter()
        for tokens in document_tokens:
            document_frequency.update(set(tokens))
        n_documents = max(1, len(document_tokens))
        query_counts = Counter(query_tokens)
        requester_aliases = self._requester_aliases(projection)
        candidates: list[_Candidate] = []

        for index, (turn, tokens) in enumerate(zip(projection.ordered_turns, document_tokens)):
            if not tokens:
                continue
            token_set = set(tokens)
            overlap = set(query_counts) & token_set
            if not overlap:
                continue
            weighted_hit = sum(
                (math.log((n_documents + 1) / (document_frequency[token] + 1)) + 1.0)
                * min(query_counts[token], tokens.count(token))
                for token in overlap
            )
            weighted_query = sum(
                math.log((n_documents + 1) / (document_frequency[token] + 1)) + 1.0
                for token in set(query_counts)
            )
            coverage = weighted_hit / max(weighted_query, 1.0)
            query_phrase = " ".join(query_tokens)
            phrase_bonus = 0.12 if len(query_tokens) > 1 and query_phrase in " ".join(tokens) else 0.0
            recency_bonus = 0.08 * ((index + 1) / n_documents)
            previous = projection.ordered_turns[index - 1] if index > 0 else None
            agent_reply = bool(
                turn.speaker_role == "agent"
                and previous is not None
                and previous.speaker_principal_id == projection.requester.principal_id
            )
            own_bonus = 0.08 if turn.speaker_principal_id == projection.requester.principal_id else 0.0
            reply_bonus = 0.16 if agent_reply else 0.0
            lower_text = turn.text.lower()
            about_requester = bool(
                requester_aliases and any(alias in lower_text for alias in requester_aliases)
            )
            subject_bonus = 0.08 if about_requester else 0.0
            score = coverage + phrase_bonus + recency_bonus + own_bonus + reply_bonus + subject_bonus
            if score < self.config.minimum_score:
                continue
            candidates.append(
                _Candidate(
                    record_id=f"g2:{projection.episode_id}:{turn.turn_id}",
                    text=turn.text,
                    turn_id=turn.turn_id,
                    turn_index=index,
                    speaker_principal_id=turn.speaker_principal_id,
                    speaker_role=turn.speaker_role,
                    score=score,
                    agent_reply_to_requester=agent_reply,
                    visibly_about_requester=about_requester,
                )
            )

        candidates.sort(key=lambda item: (-item.score, -item.turn_index, item.record_id))
        return candidates[: self.config.top_k]

    @staticmethod
    def _principal_query_aliases(principal_id: str) -> set[str]:
        return {
            token
            for token in principal_id.lower().split("_")
            if len(token) >= 3 and token not in _PRINCIPAL_PREFIXES
        }

    def _query_target_principals(
        self, projection: CleanInputProjection, candidates: Iterable[_Candidate]
    ) -> set[str]:
        query_tokens = set(_tokens(projection.query_text))
        targets = {
            candidate.speaker_principal_id
            for candidate in candidates
            if candidate.speaker_principal_id != projection.requester.principal_id
            and query_tokens & self._principal_query_aliases(candidate.speaker_principal_id)
        }
        return targets

    def _allowed(
        self,
        projection: CleanInputProjection,
        candidate: _Candidate,
        query_target_principals: set[str],
    ) -> bool:
        if (
            query_target_principals
            and candidate.speaker_principal_id not in query_target_principals
        ):
            return False
        denied_principals = self._metadata_set(projection, "denied_principal_ids")
        denied_roles = self._metadata_set(projection, "denied_roles")
        if (
            candidate.speaker_principal_id in denied_principals
            or candidate.speaker_role in denied_roles
        ):
            return False
        if candidate.speaker_principal_id == projection.requester.principal_id:
            return True
        if candidate.agent_reply_to_requester or candidate.visibly_about_requester:
            return True
        allowed_principals = self._metadata_set(projection, "allowed_principal_ids")
        allowed_roles = self._metadata_set(projection, "allowed_roles")
        return (
            candidate.speaker_principal_id in allowed_principals
            or candidate.speaker_role in allowed_roles
        )

    def evaluate(self, projection: CleanInputProjection) -> G2Result:
        candidate_by_id: dict[str, _Candidate] = {}
        allowed_ids: list[str] = []
        denied_ids: list[str] = []
        disclosed_texts: list[str] = []

        def retrieval_probe(_projection: CleanInputProjection) -> list[RetrievedArtifact]:
            candidates = self._retrieve(_projection)
            candidate_by_id.update({candidate.record_id: candidate for candidate in candidates})
            return [
                RetrievedArtifact(
                    record_id=candidate.record_id,
                    text=candidate.text,
                    source_ids=(candidate.turn_id,),
                )
                for candidate in candidates
            ]

        def disclosure_probe(
            _projection: CleanInputProjection,
            artifacts: tuple[RetrievedArtifact, ...],
        ) -> DisclosureResult:
            query_targets = self._query_target_principals(
                _projection, candidate_by_id.values()
            )
            for artifact in artifacts:
                candidate = candidate_by_id[artifact.record_id]
                if self._allowed(
                    _projection, candidate, query_targets
                ) and len(allowed_ids) < self.config.max_disclosed_records:
                    allowed_ids.append(artifact.record_id)
                    disclosed_texts.append(artifact.text)
                else:
                    denied_ids.append(artifact.record_id)
            if not allowed_ids:
                return DisclosureResult(
                    outcome="denied",
                    policy_reference="gatemem-g2-conservative-visible-policy-v1",
                )
            answer = "\n\n".join(disclosed_texts)
            if len(answer) > self.config.max_answer_characters:
                answer = answer[: self.config.max_answer_characters].rstrip() + "…"
            top_record_was_denied = bool(
                artifacts and artifacts[0].record_id in denied_ids
            )
            return DisclosureResult(
                outcome="redacted" if top_record_was_denied else "allowed",
                response_text=answer,
                disclosed_record_ids=tuple(allowed_ids),
                policy_reference="gatemem-g2-conservative-visible-policy-v1",
            )

        observation = observe_shadow(
            projection,
            retrieval_probe,
            disclosure_probe,
            deletion_mode="unsupported",
        )
        normalized = normalize_prediction(observation)
        retrieved_ids = tuple(candidate_by_id)
        disclosed_ids = tuple(observation.used_record_ids)
        denied_record_ids = tuple(record_id for record_id in retrieved_ids if record_id not in disclosed_ids)
        cross_principal = tuple(
            record_id
            for record_id, candidate in candidate_by_id.items()
            if candidate.speaker_principal_id != projection.requester.principal_id
            and not candidate.agent_reply_to_requester
        )
        blocked_cross_principal = tuple(
            record_id for record_id in cross_principal if record_id not in disclosed_ids
        )
        provenance_ok = set(disclosed_ids).issubset(retrieved_ids) and set(
            normalized["used_record_ids"]
        ) == set(disclosed_ids)

        output = {
            "action": normalized["action"],
            "answer": normalized["answer"],
            "answer_structured": dict(normalized["answer_structured"]),
            "used_record_ids": list(normalized["used_record_ids"]),
            "memory_audit": {
                "prompt_context": {
                    "text": "\n\n".join(disclosed_texts)
                    if observation.deletion_evaluation_status == "not_applicable"
                    else ""
                },
                "retrieved_record_ids": list(retrieved_ids),
                "disclosed_record_ids": list(disclosed_ids),
            },
        }
        output["answer_structured"]["gatemem_g2"] = {
            "offline_only": True,
            "adapter_version": "gatemem-g2-offline-v1",
            "retrieved_count": observation.retrieved_count,
            "disclosed_count": observation.disclosed_count,
            "cross_principal_candidate_count": len(cross_principal),
            "blocked_cross_principal_count": len(blocked_cross_principal),
            "provenance_integrity": provenance_ok,
            "production_role_enforcement_supported": False,
            "active_forgetting_supported": False,
        }
        prediction = {"checkpoint_id": projection.checkpoint_id, "output": output}
        diagnostic = G2Diagnostic(
            checkpoint_id=projection.checkpoint_id,
            projection_digest=projection.projection_digest,
            retrieved_record_ids=retrieved_ids,
            disclosed_record_ids=disclosed_ids,
            denied_record_ids=denied_record_ids,
            cross_principal_candidate_count=len(cross_principal),
            blocked_cross_principal_count=len(blocked_cross_principal),
            retrieval_candidate_count=len(retrieved_ids),
            redaction_applied=normalized["action"] == "answer_redacted",
            denial_applied=normalized["action"] == "refuse",
            deletion_evaluation_status=observation.deletion_evaluation_status,
            normalized_action=normalized["action"],
            provenance_integrity=provenance_ok,
        )
        return G2Result(observation=observation, prediction=prediction, diagnostic=diagnostic)
