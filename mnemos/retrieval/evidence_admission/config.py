"""Configuration constants for Evidence Admission and Budgeting R0.

R0 is read-only and shadow-only (see
``docs/evidence_admission_and_budgeting_r0_design_note.md``). It never
suppresses, reorders, or injects into normal retrieval results, so — unlike
Associative Routing E1/E2 — it does not need a double opt-in to stay safe.
The global gate below exists purely as an operational on/off switch for the
additive shadow metadata block, not as a safety control over delivered
results.
"""

from __future__ import annotations

#: Request/body field name callers must set to ``true`` to receive a shadow
#: admission/sufficiency block. Absent or false => no R0 evaluation, no
#: change to the stable retrieval and response-contract fields.
EVIDENCE_ADMISSION_SHADOW_FLAG = "evidence_admission_shadow"

#: Global operational gate. Defaults to disabled: even when a caller sets
#: the request flag, R0 only executes and reports a real recommendation
#: when this is explicitly "true". When the request flag is set but this
#: gate is off, a bounded, neutral status block is still returned (so
#: callers can distinguish "R0 not requested" from "R0 requested but
#: operationally disabled").
EVIDENCE_ADMISSION_SHADOW_ENABLE_ENV = "MNEMOS_EVIDENCE_ADMISSION_SHADOW_ENABLED"

#: Reason code namespaces are kept distinct per the R0 authorization: a
#: pre-retrieval admission reason code must never be presented as if it
#: explained a post-retrieval sufficiency outcome, and vice versa.
ADMISSION_REASON_PREFIX = "ADMISSION_"
SUFFICIENCY_REASON_PREFIX = "SUFFICIENCY_"
