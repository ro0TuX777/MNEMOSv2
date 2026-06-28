"""
Read-only associative routing query path.

Cue -> Tag -> Content. This module never returns content, governance state,
trust/utility scores, or authority decisions of any kind — only candidate
content ids, the cue/tag path that supports each candidate, and a
deterministically templated explanation string. Full content retrieval must
continue through the existing MNEMOS retrieval and governance path; this
projection is a non-authoritative routing aid only (see
``docs/...associative_routing_e0...`` design note for the full invariant
list).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import Abstention, Cue, RoutingPath, RoutingResponse, Tag
from .projection import Projection, build_projection

_WORD_RE = re.compile(r"[a-z0-9]+")

_EXPLANATION_TEMPLATES = {
    "blocked_by": "{cue} is blocked by {content}.",
    "depends_on": "{cue} depends on {content}.",
    "evidence_for": "{content} is cited as evidence for {cue}.",
    "supersedes": "{cue} supersedes {content}.",
    "superseded_by": "{cue} is superseded by {content}.",
    "contradicts": "{cue} contradicts {content}.",
    "unresolved_with": "{cue} is unresolved with {content}.",
    "frozen_baseline_of": "{cue} names {content} as a frozen baseline.",
    "handoff_for": "{content} is the handoff artifact for {cue}.",
    "current_state_of": "{content} is the current state of {cue}.",
    "preceded_by": "{cue} is preceded by {content}.",
    "replaced_by": "{cue} is replaced by {content}.",
}


#: Deterministic, ordered phrase -> tag_type detection. Order matters: more
#: specific multi-word phrases must be checked before their single-word
#: substrings (e.g. "superseded by" / "what superseded" before "supersede").
#: This is a controlled keyword lookup, not free-text relation extraction —
#: it only ever selects among the fixed E0 tag_type vocabulary.
_RELATIONSHIP_PHRASES = [
    ("superseded by", "superseded_by"),
    ("what superseded", "superseded_by"),
    ("who superseded", "superseded_by"),
    ("which superseded", "superseded_by"),
    ("replaced by", "replaced_by"),
    ("preceded by", "preceded_by"),
    ("blocked by", "blocked_by"),
    ("depends on", "depends_on"),
    ("frozen for regression", "frozen_baseline_of"),
    ("frozen regression baseline", "frozen_baseline_of"),
    ("frozen baseline", "frozen_baseline_of"),
    ("current state", "current_state_of"),
    ("current status", "current_state_of"),
    ("evidence for", "evidence_for"),
    ("unresolved with", "unresolved_with"),
    ("handoff for", "handoff_for"),
    ("contradicts", "contradicts"),
    ("contradict", "contradicts"),
    ("supersedes", "supersedes"),
    ("supersede", "supersedes"),
    ("blocks", "blocked_by"),
    ("block", "blocked_by"),
    ("frozen", "frozen_baseline_of"),
]


def _detect_relationship_filter(query_norm: str) -> Optional[str]:
    for phrase, tag_type in _RELATIONSHIP_PHRASES:
        if phrase in query_norm:
            return tag_type
    return None


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


@dataclass
class AssociativeRouter:
    """Deterministic Cue -> Tag -> Content router over a built projection."""

    projection: Projection

    @classmethod
    def from_fixtures(cls) -> "AssociativeRouter":
        return cls(projection=build_projection())

    def _match_cues(self, query: str) -> List[Cue]:
        """Match cues whose canonical value, normalized value, or any alias
        is wholly contained in the query (as a normalized phrase or as a set
        of tokens), or whose normalized phrase contains the query. This is a
        controlled, deterministic lexical match — no embedding or LLM
        scoring is involved in E0."""

        query_norm = _normalize(query)
        query_tokens = _tokens(query)
        matched: Dict[str, Cue] = {}

        for cue in self.projection.cues.values():
            candidates = [cue.normalized_value, _normalize(cue.canonical_value)] + [
                _normalize(a) for a in cue.aliases
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                candidate_tokens = _tokens(candidate)
                if not candidate_tokens:
                    continue
                if candidate in query_norm or candidate_tokens.issubset(query_tokens):
                    matched[cue.cue_id] = cue
                    break

        return list(matched.values())

    def _tags_from(self, cue_id: str) -> List[Tag]:
        return [t for t in self.projection.tags.values() if t.from_cue_id == cue_id]

    def route(self, query: str) -> RoutingResponse:
        matched_cues = self._match_cues(query)
        relationship_filter = _detect_relationship_filter(_normalize(query))

        if not matched_cues:
            return RoutingResponse(
                query=query,
                projection_snapshot=self.projection.snapshot,
                routing_result="abstained",
                matched_cues=[],
                routing_paths=[],
                candidate_content_ids=[],
                abstention=Abstention(
                    reason_code="NO_SUPPORTED_ASSOCIATIVE_PATH",
                    message="No Cue in the projection matches this query.",
                ),
            )

        paths: List[RoutingPath] = []
        candidate_content_ids: List[str] = []
        path_counter = 0

        for cue in matched_cues:
            tags = self._tags_from(cue.cue_id)
            if relationship_filter is not None:
                tags = [t for t in tags if t.tag_type == relationship_filter]
            for tag in tags:
                content = self.projection.content_index.get(tag.to_content_id)
                if content is None:
                    # Verification should already reject this; defensive skip only.
                    continue
                path_counter += 1
                template = _EXPLANATION_TEMPLATES.get(
                    tag.tag_type, "{cue} is related to {content} via {tag_type}."
                )
                explanation = template.format(
                    cue=cue.canonical_value, content=content.title, tag_type=tag.tag_type
                )
                paths.append(
                    RoutingPath(
                        path_id=f"path:{path_counter}",
                        cue_ids=[cue.cue_id],
                        tag_ids=[tag.tag_id],
                        content_ids=[tag.to_content_id],
                        explanation=explanation,
                    )
                )
                if tag.to_content_id not in candidate_content_ids:
                    candidate_content_ids.append(tag.to_content_id)

        if not paths:
            return RoutingResponse(
                query=query,
                projection_snapshot=self.projection.snapshot,
                routing_result="abstained",
                matched_cues=[c.cue_id for c in matched_cues],
                routing_paths=[],
                candidate_content_ids=[],
                abstention=Abstention(
                    reason_code="NO_SUPPORTED_ASSOCIATIVE_PATH",
                    message=(
                        "Matched cue(s) have no outgoing Tag of the relationship type "
                        "implied by this query."
                    ),
                ),
            )

        return RoutingResponse(
            query=query,
            projection_snapshot=self.projection.snapshot,
            routing_result="resolved",
            matched_cues=[c.cue_id for c in matched_cues],
            routing_paths=paths,
            candidate_content_ids=candidate_content_ids,
            abstention=None,
        )
