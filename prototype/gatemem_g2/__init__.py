"""GateMem G2 original offline retrieval/disclosure benchmark adapter."""

from .adapter import G2AdapterConfig, OfflineGovernedAdapter
from .harness import load_clean_projections_jsonl, run_offline_adapter
from .models import G2Diagnostic, G2Result

__all__ = [
    "G2AdapterConfig",
    "G2Diagnostic",
    "G2Result",
    "OfflineGovernedAdapter",
    "load_clean_projections_jsonl",
    "run_offline_adapter",
]

