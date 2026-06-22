"""Deterministic extraction of candidate decision/source IDs.

Known prototype simplification (recorded in docs/session_context_assembler_phase_1_notes.md):
the frozen r0 corpus does not yet supply a separate `eligible_source_linked_engrams`
/ `prior_decision_artifacts` pool distinct from conversation content. This
extractor surfaces SCA-namespaced decision IDs (DEC-SCA-*) directly from
turn text as a stand-in for that pool. This is not the final selection
algorithm and is not benchmarked - it exists only to exercise the
provenance-preservation plumbing end to end.

Phase 2R addition: source IDs are recovered from the union of (a) regex
matches in turn text and (b) the turn's structured `linked_source_ids`
field. This removes the source-recall ceiling documented in
docs/session_context_assembler_phase_3_notes.md for cases whose required
source ID is never literally embedded in turn text. r0 turns carry no
`linked_source_ids`, so this is purely additive for r0: source recovery
there is unaffected.

`Turn.eligible` gates extraction: a turn marked ineligible contributes no
IDs (from either text or linked_source_ids), modeling the spec requirement
that blocked/ineligible artifacts must never be selected.
"""

from __future__ import annotations

import re
from typing import FrozenSet, Tuple

from .models import Turn

DECISION_ID_PATTERN = re.compile(r"\bDEC-SCA-[0-9]+[a-z]?\b")
SOURCE_ID_PATTERN = re.compile(r"\bSRC-SCA-[A-Za-z0-9\-]+\b")


def extract_ids_from_turn(turn: Turn) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Return (decision_ids, source_ids) mentioned in or linked from an
    eligible turn.

    Returns empty sets for ineligible turns regardless of content or links.
    """
    if not turn.eligible:
        return frozenset(), frozenset()
    decisions = frozenset(DECISION_ID_PATTERN.findall(turn.content))
    sources = frozenset(SOURCE_ID_PATTERN.findall(turn.content)) | frozenset(turn.linked_source_ids)
    return decisions, sources
