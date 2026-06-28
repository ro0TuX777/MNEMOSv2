"""GateMem G1 clean-input projection and normalization prototype.

Offline research only. This package is original MNEMOS code and imports
nothing from GateMem, ``mnemos``, ``service``, or ``mnemos_sdk``.
"""

from .models import (
    CleanInputProjection,
    DisclosureResult,
    RetrievedArtifact,
    ShadowObservation,
    shadow_observation_from_dict,
)
from .io import write_projections_jsonl
from .normalizer import normalize_prediction, write_predictions_jsonl
from .observer import observe_shadow
from .projector import ProjectionError, clean_projection_from_dict, project_clean_input

__all__ = [
    "CleanInputProjection",
    "DisclosureResult",
    "ProjectionError",
    "RetrievedArtifact",
    "ShadowObservation",
    "clean_projection_from_dict",
    "shadow_observation_from_dict",
    "normalize_prediction",
    "observe_shadow",
    "project_clean_input",
    "write_predictions_jsonl",
    "write_projections_jsonl",
]
